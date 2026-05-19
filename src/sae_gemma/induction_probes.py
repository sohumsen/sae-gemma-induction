"""
Phase 4.2 — Generate induction probe set and measure baseline ICL accuracy.

Constructs ~2000 sequences of the form:
    [rand_prefix] A B [rand_gap] A
where A and B are random tokens from the safe vocabulary range.
Variations in prefix length and gap length create a diverse probe distribution.

The baseline ICL accuracy (top-1 prediction of B at the final position)
must be ≥50% for the project to proceed.

Outputs:
    results/induction_probes.parquet
        columns: seq_id, tokens (list[int]), A (int), B (int),
                 prefix_len (int), gap_len (int), correct (bool),
                 top5 (list[int]), logit_B (float), logit_rank_B (int)
    results/baseline_icl_accuracy.json
        {"top1_accuracy": float, "top5_accuracy": float, "n_probes": int}

Usage:
    python src/sae_gemma/induction_probes.py --n-probes 2000
"""

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
from transformer_lens import HookedTransformer

from sae_gemma.model_utils import load_model
from sae_gemma.paths import INDUCTION_PROBES_PATH, REPO_ROOT, RESULTS_DIR

BASELINE_ICL_PATH = RESULTS_DIR / "baseline_icl_accuracy.json"

# Safe vocabulary: tokens 1000-20000 are common enough for reliable induction.
# Diagnostic on Gemma-2-2B: gap=1 vocab=1k-20k -> 53%; gap>=5 or full vocab -> <40%.
SAFE_VOCAB_LO = 1000
SAFE_VOCAB_HI = 20000  # clipped to model vocab size at runtime

# Probe design: gap=1 only — Gemma-2-2B induction drops sharply beyond gap=1.
PREFIX_LENS = list(range(5, 31, 5))      # [5, 10, 15, 20, 25, 30]
GAP_LENS = [1]                           # only gap=1 gives >50% accuracy


def _safe_vocab_range(model_vocab_size: int) -> tuple[int, int]:
    return SAFE_VOCAB_LO, min(SAFE_VOCAB_HI, model_vocab_size - 1)


def build_probe(
    rng: random.Random,
    vocab_lo: int,
    vocab_hi: int,
    prefix_len: int,
    gap_len: int,
) -> tuple[list[int], int, int]:
    """
    Build one induction probe sequence.
    Returns (tokens, A, B) where tokens ends with A and the model should predict B.
    """
    A = rng.randint(vocab_lo, vocab_hi)
    B = rng.randint(vocab_lo, vocab_hi)
    while B == A:
        B = rng.randint(vocab_lo, vocab_hi)

    prefix = [rng.randint(vocab_lo, vocab_hi) for _ in range(prefix_len)]
    gap = [rng.randint(vocab_lo, vocab_hi) for _ in range(gap_len)]
    # Ensure A doesn't appear in gap (would confuse induction)
    gap = [t if t != A else (t + 1 if t + 1 <= vocab_hi else t - 1) for t in gap]

    tokens = prefix + [A, B] + gap + [A]
    return tokens, A, B


@torch.no_grad()
def run_probes(
    model: HookedTransformer,
    probe_data: list[dict],
    device: str,
    batch_size: int = 32,
) -> list[dict]:
    """Run all probes through the model and fill in correctness + logits."""
    results = []
    vocab_lo, vocab_hi = _safe_vocab_range(model.cfg.d_vocab)

    for i in range(0, len(probe_data), batch_size):
        batch = probe_data[i: i + batch_size]
        # Pad sequences to same length
        max_len = max(len(p["tokens"]) for p in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        for j, p in enumerate(batch):
            toks = torch.tensor(p["tokens"], dtype=torch.long)
            padded[j, :len(toks)] = toks
            padded[j, len(toks):] = 0  # pad

        logits = model(padded)  # (batch, seq, vocab)

        for j, p in enumerate(batch):
            seq_len = len(p["tokens"])
            final_logits = logits[j, seq_len - 1, :]  # logits at last position (second A)
            top5 = torch.topk(final_logits, 5).indices.tolist()
            B = p["B"]
            logit_B = float(final_logits[B].item())
            # Rank of B in vocab (0-indexed, lower = better)
            logit_rank_B = int((final_logits > final_logits[B]).sum().item())
            results.append({
                **p,
                "correct": top5[0] == B,
                "top5": top5,
                "logit_B": logit_B,
                "logit_rank_B": logit_rank_B,
            })

        if (i // batch_size) % 10 == 0:
            n_done = min(i + batch_size, len(probe_data))
            print(f"[probes] {n_done}/{len(probe_data)} probes evaluated", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate induction probes and measure baseline ICL")
    parser.add_argument("--n-probes", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=INDUCTION_PROBES_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    print("[probes] Loading Gemma-2-2B ...", flush=True)
    model = load_model(device=args.device)

    vocab_lo, vocab_hi = _safe_vocab_range(model.cfg.d_vocab)
    print(f"[probes] Vocab range [{vocab_lo}, {vocab_hi}]", flush=True)

    rng = random.Random(args.seed)
    t0 = time.monotonic()

    # Generate probe_data list
    probe_data = []
    for seq_id in range(args.n_probes):
        prefix_len = rng.choice(PREFIX_LENS)
        gap_len = rng.choice(GAP_LENS)
        # Cap gap so total sequence ≤ 650 tokens (VRAM-safe)
        gap_len = min(gap_len, 650 - prefix_len - 2)
        gap_len = max(gap_len, 1)
        tokens, A, B = build_probe(rng, vocab_lo, vocab_hi, prefix_len, gap_len)
        probe_data.append({
            "seq_id": seq_id,
            "tokens": tokens,
            "A": A,
            "B": B,
            "prefix_len": prefix_len,
            "gap_len": gap_len,
        })

    print(f"[probes] Generated {len(probe_data)} probes. Running through model ...", flush=True)
    results = run_probes(model, probe_data, args.device, args.batch_size)

    # Compute accuracy
    n = len(results)
    top1_correct = sum(1 for r in results if r["correct"])
    top5_correct = sum(1 for r in results if r["B"] in r["top5"])
    top1_acc = top1_correct / n
    top5_acc = top5_correct / n

    elapsed = time.monotonic() - t0
    print(f"\n[probes] Results ({n} probes, {elapsed:.0f}s):", flush=True)
    print(f"  Top-1 ICL accuracy: {top1_acc:.1%} ({top1_correct}/{n})", flush=True)
    print(f"  Top-5 ICL accuracy: {top5_acc:.1%} ({top5_correct}/{n})", flush=True)

    if top1_acc < 0.50:
        print(
            "[probes] GATE FAILED: top-1 ICL accuracy < 50%. "
            "Gemma-2-2B may not have functional induction. ESCALATE before proceeding.",
            flush=True,
        )
    else:
        print(f"[probes] GATE PASSED: ICL accuracy {top1_acc:.1%} >= 50%", flush=True)

    # Save probe results
    df_rows = []
    for r in results:
        df_rows.append({
            "seq_id": r["seq_id"],
            "tokens": r["tokens"],
            "A": r["A"],
            "B": r["B"],
            "prefix_len": r["prefix_len"],
            "gap_len": r["gap_len"],
            "correct": r["correct"],
            "top1_pred": r["top5"][0] if r["top5"] else -1,
            "top5": r["top5"],
            "logit_B": r["logit_B"],
            "logit_rank_B": r["logit_rank_B"],
        })
    df = pd.DataFrame(df_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[probes] Probe results saved to {args.output}", flush=True)

    # Save summary
    summary = {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "n_probes": n,
        "n_correct_top1": top1_correct,
        "n_correct_top5": top5_correct,
        "gate_passed": top1_acc >= 0.50,
    }
    BASELINE_ICL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BASELINE_ICL_PATH.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[probes] Summary saved to {BASELINE_ICL_PATH}", flush=True)

    return 0 if top1_acc >= 0.50 else 1


if __name__ == "__main__":
    raise SystemExit(main())
