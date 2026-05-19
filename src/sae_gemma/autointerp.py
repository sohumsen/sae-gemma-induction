"""
Phase 4.4 — Auto-interpretation of SAE features via the Claude CLI.

For each feature, we send its top-20 activating text snippets to Claude
(Haiku 4.5 for bulk labelling, Sonnet 4.5 for the top-50 induction candidates)
and cache the label to results/feature_labels.json.

Key design decisions:
- Append-on-write after every call: a crash never loses work already done.
- _kill_proc_tree() on timeout: Windows stderr-pipe deadlocks if subprocess hangs.
- ThreadPoolExecutor(max_workers=CONCURRENCY) for rate-limit-safe parallelism.
- Labels already in the cache are skipped (idempotent re-runs).

Usage:
    python src/sae_gemma/autointerp.py \
        --snippets results/top_snippets.parquet \
        --features results/induction_candidate_ids.json \
        --model claude-haiku-4-5 \
        --workers 8

    # Re-label top induction features with Sonnet:
    python src/sae_gemma/autointerp.py \
        --snippets results/top_snippets.parquet \
        --features results/induction_candidate_ids.json \
        --model claude-sonnet-4-5 \
        --workers 4 \
        --overwrite
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd

from sae_gemma.paths import FEATURE_LABELS_PATH, REPO_ROOT, TOP_SNIPPETS_PATH

# ── Windows process-tree killer ───────────────────────────────────────────────
# Needed because subprocess.kill() on Windows only kills the top-level process;
# child processes (the claude CLI's Node runtime) keep running and hold the pipe open.

def _kill_proc_tree(pid: int, include_parent: bool = True) -> None:
    """Kill a process and all its descendants on Windows (or POSIX)."""
    try:
        import psutil
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        if include_parent:
            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass
    except ImportError:
        # Fallback: taskkill /F /T on Windows
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
            )
        else:
            os.kill(pid, signal.SIGKILL)


# ── Prompt template ───────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
Below are text snippets where a neural network feature activated most strongly. \
The token that most strongly activated the feature is marked in <<...>>. \
Describe in one sentence what linguistic pattern or concept this feature seems to detect. \
If the pattern is unclear from the snippets, say "unclear". \
Be specific and concrete — avoid vague terms like "language" or "text".

Snippets:
{snippets}
"""


def _format_snippets(rows: list[dict]) -> str:
    """Format top-activating snippets for the prompt."""
    parts = []
    for i, row in enumerate(rows[:20], 1):
        ctx = row.get("context", "")
        tok = row.get("token", "")
        # Bold the activating token
        marked = ctx.replace(tok, f"<<{tok}>>", 1) if tok else ctx
        parts.append(f"{i}. {marked.strip()}")
    return "\n".join(parts)


# ── Single-feature labeller ───────────────────────────────────────────────────

def label_feature(
    feature_id: int,
    snippets: list[dict],
    model: str = "claude-haiku-4-5",
    timeout: int = 60,
) -> str:
    """
    Call `claude --print --model <model> "<prompt>"` and return the label string.
    Returns "error: <msg>" on failure so callers can log and continue.
    """
    prompt = PROMPT_TEMPLATE.format(snippets=_format_snippets(snippets))

    try:
        proc = subprocess.Popen(
            ["claude", "--print", "--model", model, prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_proc_tree(proc.pid)
            return f"error: timeout after {timeout}s"

        if proc.returncode != 0:
            return f"error: returncode={proc.returncode} stderr={stderr[:200]!r}"

        label = stdout.strip()
        return label if label else "error: empty response"

    except FileNotFoundError:
        return "error: claude CLI not found on PATH"
    except Exception as exc:
        return f"error: {exc}"


# ── Cache helpers ─────────────────────────────────────────────────────────────

_cache_lock = Lock()


def _load_cache(path: Path) -> dict[int, str]:
    """Load feature_id → label from JSON cache. Returns {} if missing."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    except Exception:
        return {}


def _append_to_cache(path: Path, feature_id: int, label: str) -> None:
    """Thread-safe append of one feature label to the JSON cache."""
    with _cache_lock:
        cache = _load_cache(path)
        cache[feature_id] = label
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        tmp.replace(path)  # atomic on Windows (same drive)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_autointerp(
    snippets_path: Path,
    feature_ids: list[int] | None,
    model: str,
    workers: int,
    overwrite: bool,
    cache_path: Path,
    timeout: int = 60,
) -> None:
    df = pd.read_parquet(snippets_path)
    # Expected schema: feature_id (int), rank (int 0-19), context (str), token (str), activation (float)

    all_feature_ids = sorted(df["feature_id"].unique().tolist())
    target_ids = feature_ids if feature_ids is not None else all_feature_ids
    print(f"[autointerp] {len(target_ids)} features to label with {model}", flush=True)

    cache = _load_cache(cache_path)
    if not overwrite:
        target_ids = [fid for fid in target_ids if fid not in cache]
        print(f"[autointerp] {len(target_ids)} remain after skipping cached", flush=True)

    if not target_ids:
        print("[autointerp] Nothing to do — all labels already cached.", flush=True)
        return

    def _job(fid: int):
        rows = df[df["feature_id"] == fid].sort_values("rank").to_dict("records")
        if not rows:
            label = "error: no snippets found"
        else:
            label = label_feature(fid, rows, model=model, timeout=timeout)
        _append_to_cache(cache_path, fid, label)
        return fid, label

    done = 0
    errors = 0
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, fid): fid for fid in target_ids}
        for fut in as_completed(futures):
            fid, label = fut.result()
            done += 1
            if label.startswith("error:"):
                errors += 1
            elapsed = time.monotonic() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(target_ids) - done) / rate if rate > 0 else 0
            if done % 50 == 0 or done <= 5:
                print(
                    f"[autointerp] {done}/{len(target_ids)}  "
                    f"errors={errors}  rate={rate:.1f}/s  "
                    f"ETA={eta/3600:.1f}h  fid={fid}  label={label[:60]!r}",
                    flush=True,
                )

    print(f"[autointerp] Done. {done} labelled, {errors} errors. Cache: {cache_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Auto-interpret SAE features via Claude CLI")
    parser.add_argument("--snippets", type=Path, default=TOP_SNIPPETS_PATH)
    parser.add_argument("--cache", type=Path, default=FEATURE_LABELS_PATH)
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="JSON file with list of feature IDs to label (default: all features in snippets)",
    )
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-label even if already cached",
    )
    args = parser.parse_args()

    # Load env vars
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    feature_ids = None
    if args.features and args.features.exists():
        with args.features.open() as f:
            feature_ids = json.load(f)

    run_autointerp(
        snippets_path=args.snippets,
        feature_ids=feature_ids,
        model=args.model,
        workers=args.workers,
        overwrite=args.overwrite,
        cache_path=args.cache,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
