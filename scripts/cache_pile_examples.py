"""
Pre-cache pile examples to local JSONL so SAE training doesn't depend on
HuggingFace streaming staying connected. v9b crashed at step 18,808 because
the streaming client closed mid-training; this script removes that risk.

Output: data/pile_cache.jsonl (one {"text": "..."} per line)

Usage: python scripts/cache_pile_examples.py [--n 300000]
"""
import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = REPO_ROOT / "data" / "pile_cache.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300_000,
                   help="Number of examples to cache (~600 tok/example avg → 180M tok)")
    args = p.parse_args()

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If cache already has enough, skip.
    existing = 0
    if CACHE_PATH.exists():
        with CACHE_PATH.open(encoding="utf-8") as f:
            for _ in f:
                existing += 1
        if existing >= args.n:
            print(f"[cache] {CACHE_PATH} already has {existing} >= {args.n}, skipping")
            return
        print(f"[cache] {existing} examples already in cache, need {args.n - existing} more")

    # Robust streaming with retry on transient HF errors.
    saved = existing
    max_retries = 10
    retry = 0
    t0 = time.monotonic()

    while saved < args.n and retry < max_retries:
        try:
            ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
            it = iter(ds)
            # If we have existing, skip ahead. Streaming has no random access,
            # so we just take from the top; some duplication on retry is OK.
            with CACHE_PATH.open("a", encoding="utf-8") as f:
                for ex in it:
                    if saved >= args.n:
                        break
                    text = ex.get("text", "")
                    if not text:
                        continue
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    saved += 1
                    if saved % 5_000 == 0:
                        elapsed = time.monotonic() - t0
                        rate = saved / max(elapsed, 1)
                        eta = (args.n - saved) / max(rate, 1)
                        print(f"[cache] {saved}/{args.n}  rate={rate:.0f}/s  ETA={eta/60:.1f}m", flush=True)
            break  # full pass succeeded
        except Exception as e:
            retry += 1
            print(f"[cache] error: {type(e).__name__}: {str(e)[:160]}  retry={retry}/{max_retries}", flush=True)
            time.sleep(5)

    print(f"[cache] done. {saved} examples in {CACHE_PATH}  ({CACHE_PATH.stat().st_size/1024/1024:.0f} MB)")


if __name__ == "__main__":
    main()
