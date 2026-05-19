"""
Random-feature ablation control for the v9c SAE induction-feature ablation result.

The headline finding is: zeroing the top-50 induction-scoring features drops top-1
ICL accuracy from 57.75% -> 47.65% (-10.1pp). Reviewers will fairly ask: what if
50 *random* features did the same thing?

This script:
1. Loads v9c SAE + Gemma-2-2B.
2. Samples 50 *non-induction-cluster* features uniformly at random, for each of
   5 random seeds.
3. Runs the same feature ablation as ablations.py on each random sample.
4. Reports mean accuracy drop across the 5 random samples vs the 10.1pp top-50
   induction drop.

Output: results/random_feature_ablation.json.
"""
import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sae_gemma.model_utils import load_model
from sae_gemma.paths import (
    HOOK_NAME,
    INDUCTION_PROBES_PATH,
    RESULTS_DIR,
    SAE_MAIN_DIR,
    REPO_ROOT,
)

CANDIDATE_IDS_PATH = RESULTS_DIR / "induction_candidate_ids.json"
ABLATION_RESULTS_PATH = RESULTS_DIR / "ablation_results.json"


def load_sae_local(sae_path: Path, device: str):
    from sae_lens.saes.sae import SAE
    sae = SAE.load_from_disk(str(sae_path), device=device)
    sae.eval()
    return sae


@torch.no_grad()
def measure_icl_accuracy(
    model,
    sae,
    probes_df: pd.DataFrame,
    feature_ids_to_ablate: list[int],
    device: str,
    batch_size: int = 16,
) -> float:
    """Run model with SAE-patch + ablation; return top-1 accuracy on induction probes."""

    feature_mask = torch.zeros(sae.cfg.d_sae, dtype=torch.bool, device=device)
    if feature_ids_to_ablate:
        feature_mask[feature_ids_to_ablate] = True

    correct = 0
    total = 0
    probe_tokens = [torch.tensor(t, dtype=torch.long, device=device) for t in probes_df["tokens"]]
    answer_tokens = probes_df["B"].tolist()

    for i in range(0, len(probe_tokens), batch_size):
        batch = probe_tokens[i: i + batch_size]
        max_len = max(len(seq) for seq in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        seq_lens = []
        for j, seq in enumerate(batch):
            padded[j, : len(seq)] = seq
            seq_lens.append(len(seq))

        def patch_hook(value, hook):
            # value: [batch, seq, d_model]
            b, s, d = value.shape
            flat = value.reshape(b * s, d).float()
            z = sae.encode(flat)                          # [b*s, d_sae]
            z_ablated = z * (~feature_mask).float()       # zero target features
            recon = sae.decode(z_ablated)                 # [b*s, d_model]
            recon_orig = sae.decode(z)
            delta = (recon - recon_orig).reshape(b, s, d).to(value.dtype)
            return value + delta

        logits = model.run_with_hooks(padded, fwd_hooks=[(HOOK_NAME, patch_hook)])

        for j in range(len(batch)):
            final_pos = seq_lens[j] - 1
            pred = logits[j, final_pos].argmax().item()
            if pred == answer_tokens[i + j]:
                correct += 1
            total += 1

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ablate", type=int, default=50)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    print("[random_ab] Loading model + SAE ...", flush=True)
    model = load_model(device=args.device)
    sae = load_sae_local(SAE_MAIN_DIR, args.device)
    d_sae = sae.cfg.d_sae

    print(f"[random_ab] Loading induction probes + candidate IDs ...", flush=True)
    probes_df = pd.read_parquet(INDUCTION_PROBES_PATH)
    candidate_ids = set(json.loads(CANDIDATE_IDS_PATH.read_text(encoding="utf-8")))
    non_induction = [i for i in range(d_sae) if i not in candidate_ids]
    print(f"[random_ab] {len(non_induction)} non-induction features available", flush=True)

    # Reference: baseline accuracy + top-50 induction ablation from ablation_results.json
    ablation_results = json.loads(ABLATION_RESULTS_PATH.read_text(encoding="utf-8"))
    baseline_acc = ablation_results["baseline_accuracy"]
    induction_ns = ablation_results["feature_ablation"]["N"]
    induction_accs = ablation_results["feature_ablation"]["accuracy"]
    print(f"[random_ab] Reference baseline: {baseline_acc:.4f}", flush=True)
    print(f"[random_ab] Reference induction ablation: N={induction_ns}  acc={induction_accs}", flush=True)

    # Run random ablation
    results = []
    t0 = time.monotonic()
    for s in range(args.n_seeds):
        seed = args.seed_base + s
        rng = random.Random(seed)
        sample = rng.sample(non_induction, args.n_ablate)
        print(f"[random_ab] Seed {seed}: ablating {args.n_ablate} random non-induction features ...", flush=True)
        acc = measure_icl_accuracy(model, sae, probes_df, sample, args.device, args.batch_size)
        drop = baseline_acc - acc
        elapsed = time.monotonic() - t0
        print(f"[random_ab]   acc={acc:.4f}  drop={drop:+.4f} ({drop*100:+.2f}pp)  elapsed={elapsed/60:.1f}m", flush=True)
        results.append({"seed": seed, "n_ablated": args.n_ablate, "feature_ids": sample[:10], "accuracy": acc, "drop": drop})

    out = {
        "baseline_accuracy": baseline_acc,
        "induction_top50_accuracy": induction_accs[-1] if induction_accs else None,
        "induction_top50_drop": baseline_acc - (induction_accs[-1] if induction_accs else 0.0),
        "random_n_ablated": args.n_ablate,
        "random_n_seeds": args.n_seeds,
        "random_runs": results,
        "random_mean_acc": float(np.mean([r["accuracy"] for r in results])),
        "random_mean_drop": float(np.mean([r["drop"] for r in results])),
        "random_std_drop": float(np.std([r["drop"] for r in results])),
    }
    out_path = RESULTS_DIR / "random_feature_ablation.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[random_ab] === RESULTS ===", flush=True)
    print(f"  Top-50 induction features ablated: drop = {out['induction_top50_drop']*100:.2f}pp", flush=True)
    print(f"  50 RANDOM features (mean of {args.n_seeds} seeds): drop = {out['random_mean_drop']*100:.2f}pp ± {out['random_std_drop']*100:.2f}pp", flush=True)
    print(f"  Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
