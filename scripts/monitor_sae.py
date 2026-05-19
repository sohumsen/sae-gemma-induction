"""
Long-running monitor for the active SAE training run.
Polls W&B + local logs and prints ONE line whenever something actionable happens:
  - Each meaningful cossim milestone crossed (0.45, 0.55, 0.65, 0.75, 0.85)
  - Plateau detected (3 consecutive checks with <2% relative cossim growth)
  - Training completed (final checkpoint saved)
  - Process crashed (PID gone, no completion message)
  - Periodic heartbeat every ~15 min so silence ≠ stalled monitor

Each printed line is an event for the Monitor tool. Keep volume low.

Usage:
    python scripts/monitor_sae.py <run_name> <PID>
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

import wandb

RUN_NAME = sys.argv[1]
PID = int(sys.argv[2]) if len(sys.argv) > 2 else None

PROJECT = os.environ.get("WANDB_PROJECT", "sae-gemma-induction")
ENTITY = os.environ.get("WANDB_ENTITY", None)
LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / f"sae_main_{RUN_NAME.split('-')[-1]}.err"

POLL_INTERVAL = 90  # seconds between W&B polls
HEARTBEAT_EVERY = 10  # emit a heartbeat every N polls (~15 min)
MILESTONES = [0.45, 0.55, 0.65, 0.75, 0.85, 0.92]
PLATEAU_WINDOW = 3
PLATEAU_THRESHOLD = 0.02  # 2% relative growth required

PATH = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT

def emit(tag, **fields):
    rec = {"event": tag, "run": RUN_NAME, **fields}
    print(json.dumps(rec), flush=True)


def process_alive():
    if PID is None:
        return True
    # Use tasklist on Windows
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {PID}", "/NH"],
        capture_output=True, text=True
    )
    return str(PID) in out.stdout


def log_says_done():
    if not LOG_PATH.exists():
        return False
    try:
        text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")[-4000:]
        return "Done. SAE saved" in text or "100%|" in text.split("\n")[-3] if text else False
    except Exception:
        return False


def get_metrics():
    try:
        # Recreate api each call so wandb doesn't cache stale summary data
        api = wandb.Api(timeout=29)
        runs = api.runs(PATH, filters={"display_name": RUN_NAME}, order="-created_at")
        if not runs:
            return None
        run = runs[0]
        sm = run.summary
        def deep(key):
            v = sm.get(key, None)
            if v is None: return {}
            if isinstance(v, str):
                try: return json.loads(v.replace("'", '"'))
                except Exception: return {}
            # wandb SummarySubDict — supports .get()
            try: return dict(v)
            except Exception: return v
        recon = deep("reconstruction_quality")
        shrink = deep("shrinkage")
        perf = deep("model_performance_preservation")
        return {
            "step": sm.get("_step"),
            "n_tokens": sm.get("details/n_training_samples"),
            "cossim": recon.get("cossim"),
            "ev_legacy": sm.get("metrics/explained_variance_legacy"),
            "l2_ratio": shrink.get("l2_ratio"),
            "ce_score": perf.get("ce_loss_score"),
            "dead": sm.get("sparsity/dead_features"),
            "state": run.state,
        }
    except Exception as e:
        return {"error": str(e)[:120]}


# Initial emit so we know monitor is alive
emit("monitor_start", pid=PID, log=str(LOG_PATH))

reached = set()
cossim_history = []
poll = 0
last_heartbeat_metrics = None

while True:
    poll += 1
    m = get_metrics()

    # Process / state checks
    if not process_alive():
        if log_says_done():
            emit("completed", metrics=m)
        else:
            emit("crashed", metrics=m, last_log_tail=LOG_PATH.read_text(encoding="utf-8", errors="ignore")[-800:] if LOG_PATH.exists() else "")
        break

    if not m or m.get("error") or m.get("cossim") is None:
        if poll % HEARTBEAT_EVERY == 1:
            emit("waiting_for_metrics", note="training still in setup or no eval yet", metrics=m)
        time.sleep(POLL_INTERVAL)
        continue

    cossim = m["cossim"]
    cossim_history.append(cossim)

    # Milestone events
    for ms in MILESTONES:
        if ms not in reached and cossim >= ms:
            reached.add(ms)
            emit("milestone", threshold=ms, metrics=m)

    # Plateau detection
    if len(cossim_history) >= PLATEAU_WINDOW + 1:
        window = cossim_history[-(PLATEAU_WINDOW + 1):]
        rel_growth = (window[-1] - window[0]) / max(window[0], 0.01)
        if rel_growth < PLATEAU_THRESHOLD:
            emit("plateau", window=window, rel_growth=rel_growth, metrics=m)
            cossim_history = cossim_history[-1:]  # reset window so we don't re-emit immediately

    # Heartbeat
    if poll % HEARTBEAT_EVERY == 0:
        emit("heartbeat", poll=poll, metrics=m)

    time.sleep(POLL_INTERVAL)
