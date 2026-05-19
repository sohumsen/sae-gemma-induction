"""
Monitor for dictionary_learning SAE training (different W&B key schema than SAELens).
Tracks `frac_variance_explained` (= EV directly). Emits on milestones / plateau / crash / completion.

Usage: python scripts/monitor_dl.py <wandb_run_name> <PID> <log_path>
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
PID = int(sys.argv[2])
LOG_PATH = Path(sys.argv[3]) if len(sys.argv) > 3 else None

PROJECT = os.environ.get("WANDB_PROJECT", "sae-gemma-induction")
ENTITY = os.environ.get("WANDB_ENTITY", None)
PATH = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT

POLL_INTERVAL = 90
HEARTBEAT_EVERY = 10
EV_MILESTONES = [0.0, 0.2, 0.4, 0.6, 0.75, 0.85]
PLATEAU_WINDOW = 4
PLATEAU_THRESHOLD = 0.02


def emit(tag, **fields):
    rec = {"event": tag, "run": RUN_NAME, **fields}
    print(json.dumps(rec), flush=True)


def process_alive():
    out = subprocess.run(["tasklist", "/FI", f"PID eq {PID}", "/NH"], capture_output=True, text=True)
    return str(PID) in out.stdout


def log_says_done():
    if LOG_PATH is None or not LOG_PATH.exists():
        return False
    try:
        text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")[-4000:]
        return "Done." in text or "Training complete" in text or "[convert]" in text
    except Exception:
        return False


def get_metrics():
    try:
        api = wandb.Api(timeout=29)
        runs = api.runs(PATH, filters={"display_name": RUN_NAME}, order="-created_at")
        if not runs:
            return None
        run = runs[0]
        sm = run.summary
        return {
            "step": sm.get("_step"),
            "ev": sm.get("frac_variance_explained"),
            "l2_loss": sm.get("l2_loss"),
            "loss": sm.get("loss"),
            "auxk_loss": sm.get("auxk_loss"),
            "l0": sm.get("l0"),
            "effective_l0": sm.get("effective_l0"),
            "dead": sm.get("dead_features"),
            "state": run.state,
        }
    except Exception as e:
        return {"error": str(e)[:120]}


emit("monitor_start", pid=PID, log=str(LOG_PATH) if LOG_PATH else None)

reached = set()
ev_history = []  # list of (step, ev)
poll = 0
last_emit_step = -1

while True:
    poll += 1
    m = get_metrics()

    if not process_alive():
        if log_says_done():
            emit("completed", metrics=m)
        else:
            tail = ""
            if LOG_PATH and LOG_PATH.exists():
                try:
                    tail = LOG_PATH.read_text(encoding="utf-8", errors="ignore")[-1200:]
                except Exception:
                    tail = ""
            emit("crashed", metrics=m, last_log_tail=tail)
        break

    if not m or m.get("error") or m.get("ev") is None:
        if poll % HEARTBEAT_EVERY == 1:
            emit("waiting_for_metrics", note="no EV in W&B yet (still calibrating or pre-eval)", metrics=m)
        time.sleep(POLL_INTERVAL)
        continue

    ev = m["ev"]
    step = m["step"]

    # only update history when step has advanced (avoid same-eval duplicate plateau)
    if step != last_emit_step:
        ev_history.append((step, ev))
        last_emit_step = step

        for ms in EV_MILESTONES:
            if ms not in reached and ev >= ms:
                reached.add(ms)
                emit("milestone", threshold=ms, metrics=m)

        if len(ev_history) >= PLATEAU_WINDOW + 1:
            window = [v for _, v in ev_history[-(PLATEAU_WINDOW + 1):]]
            rel_growth = (window[-1] - window[0]) / max(abs(window[0]), 0.01)
            if rel_growth < PLATEAU_THRESHOLD:
                emit("plateau", window=window, rel_growth=rel_growth, metrics=m)
                ev_history = ev_history[-1:]

    if poll % HEARTBEAT_EVERY == 0:
        emit("heartbeat", poll=poll, metrics=m)

    time.sleep(POLL_INTERVAL)
