"""
Phase 4.7 + 4.8 — Feature ablation and head ablation experiments.

4.7: Feature ablation (SAE-level)
    Clamp the top-N induction features to 0 during inference.
    Measure ICL accuracy drop on the probe set for N in {5, 10, 25, 50}.

4.8: Head ablation (attention-level, Olsson et al. baseline)
    Zero-out the output of each layer-12 attention head individually, then
    all "induction heads" together (those with high feature-head correlation).
    Measure ICL accuracy drop for comparison.

Outputs:
    results/ablation_results.json
        {
          "feature_ablation": {
            "N": [5, 10, 25, 50],
            "accuracy": [float, ...],
            "accuracy_drop": [float, ...],
            "ci_low": [...], "ci_high": [...]
          },
          "head_ablation": {
            "heads": [...],
            "accuracy": [...],
            "accuracy_drop": [...],
          },
          "baseline_accuracy": float
        }
    results/figures/ablation_curve.png

Usage:
    python src/sae_gemma/ablations.py \
        --sae-path models/sae_main \
        --candidates results/induction_candidate_ids.json \
        --scores results/induction_feature_scores.parquet \
        --probes results/induction_probes.parquet
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from sae_gemma.model_utils import load_model

from sae_gemma.paths import ABLATION_RESULTS_PATH, FIGURES_DIR, HOOK_NAME, INDUCTION_PROBES_PATH, REPO_ROOT, RESULTS_DIR

CANDIDATE_IDS_PATH = RESULTS_DIR / "induction_candidate_ids.json"
SCORES_PATH = RESULTS_DIR / "induction_feature_scores.parquet"
TARGET_LAYER = 12

FEATURE_ABLATION_NS = [5, 10, 25, 50]
N_BOOTSTRAP = 1000  # bootstrap samples for CIs


def load_sae_local(sae_path: Path, device: str):
    from sae_lens.saes.sae import SAE
    sae = SAE.load_from_disk(str(sae_path), device=device)
    sae.eval()
    return sae


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(correct: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05):
    """Return (mean, ci_low, ci_high) via non-parametric bootstrap."""
    rng = np.random.default_rng(0)
    n = len(correct)
    means = [correct[rng.integers(0, n, n)].mean() for _ in range(n_bootstrap)]
    means = np.array(means)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(correct.mean()), lo, hi


# ── Feature ablation ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_feature_ablation(
    model: HookedTransformer,
    sae,
    probe_seqs: list[tuple[list[int], int, int]],
    ablate_feature_ids: list[int],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run probes with the given SAE features clamped to 0.
    Returns boolean array of shape (n_probes,): True = correct top-1 prediction.

    Implementation: we hook residual stream at the SAE's layer, encode with SAE,
    zero out the ablated features, decode back, and subtract the original residual.
    This is equivalent to running the model with those SAE directions removed.
    """
    ablate_set = set(ablate_feature_ids)
    sae_dtype = torch.float32  # SAE trained in float32
    correct = np.zeros(len(probe_seqs), dtype=bool)

    for i in range(0, len(probe_seqs), batch_size):
        batch = probe_seqs[i: i + batch_size]
        max_len = max(len(p[0]) for p in batch)
        seq_lens = [len(p[0]) for p in batch]
        Bs = [p[2] for p in batch]

        padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        for j, (toks, A, B) in enumerate(batch):
            padded[j, :len(toks)] = torch.tensor(toks, dtype=torch.long)

        # Hook to intervene: encode → zero ablated features → decode → patch residual
        def make_ablation_hook(sae, ablate_ids):
            ablate_tensor = torch.tensor(list(ablate_ids), dtype=torch.long, device=device)

            def hook_fn(value, hook):
                # value: (batch, seq, d_model)
                B_sz, S, D = value.shape
                acts_flat = value.reshape(B_sz * S, D).to(sae_dtype)
                feat_acts = sae.encode(acts_flat)           # (B*S, n_features)
                feat_acts_zeroed = feat_acts.clone()
                feat_acts_zeroed[:, ablate_tensor] = 0.0
                reconstructed = sae.decode(feat_acts_zeroed)  # (B*S, d_model)
                recon_orig = sae.decode(feat_acts)            # (B*S, d_model)
                # Patch: value + (reconstructed - recon_orig) = ablated version
                delta = (reconstructed - recon_orig).reshape(B_sz, S, D).to(value.dtype)
                return value + delta

            return hook_fn

        hook_fn = make_ablation_hook(sae, ablate_set)
        logits = model.run_with_hooks(padded, fwd_hooks=[(HOOK_NAME, hook_fn)])

        for j, (slen, B_tok) in enumerate(zip(seq_lens, Bs)):
            pred = logits[j, slen - 1, :].argmax().item()
            correct[i + j] = (pred == B_tok)

    return correct


# ── Head ablation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_head_ablation(
    model: HookedTransformer,
    probe_seqs: list[tuple[list[int], int, int]],
    ablate_heads: list[int],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Zero out the output contribution of specified attention heads in layer TARGET_LAYER.
    Returns boolean correctness array.
    """
    # Hook on the attention output: blocks.{layer}.hook_attn_out
    # Or hook on hook_z (pre-projection head outputs) and zero them.
    hook_z_name = f"blocks.{TARGET_LAYER}.attn.hook_z"
    ablate_head_set = set(ablate_heads)
    correct = np.zeros(len(probe_seqs), dtype=bool)

    def make_head_hook(heads_to_zero):
        def hook_fn(value, hook):
            # value: (batch, seq, n_heads, d_head)
            for h in heads_to_zero:
                value[:, :, h, :] = 0.0
            return value
        return hook_fn

    hook_fn = make_head_hook(ablate_head_set)

    for i in range(0, len(probe_seqs), batch_size):
        batch = probe_seqs[i: i + batch_size]
        max_len = max(len(p[0]) for p in batch)
        seq_lens = [len(p[0]) for p in batch]
        Bs = [p[2] for p in batch]

        padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        for j, (toks, A, B) in enumerate(batch):
            padded[j, :len(toks)] = torch.tensor(toks, dtype=torch.long)

        logits = model.run_with_hooks(padded, fwd_hooks=[(hook_z_name, hook_fn)])

        for j, (slen, B_tok) in enumerate(zip(seq_lens, Bs)):
            pred = logits[j, slen - 1, :].argmax().item()
            correct[i + j] = (pred == B_tok)

    return correct


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ablation_curve(results: dict, output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Feature ablation curve
    ns = results["feature_ablation"]["N"]
    accs = results["feature_ablation"]["accuracy"]
    ci_lo = results["feature_ablation"]["ci_low"]
    ci_hi = results["feature_ablation"]["ci_high"]
    baseline = results["baseline_accuracy"]

    ax1.axhline(baseline, color="grey", linestyle="--", label=f"Baseline ({baseline:.1%})", alpha=0.7)
    ax1.plot(ns, accs, "o-", color="steelblue", label="With feature ablation")
    ax1.fill_between(ns, ci_lo, ci_hi, alpha=0.2, color="steelblue")
    ax1.set_xlabel("Number of top induction features ablated")
    ax1.set_ylabel("ICL top-1 accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(ns)
    ax1.legend()
    ax1.set_title("SAE Feature Ablation")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Head ablation bar chart
    head_labels = [str(h) for h in results["head_ablation"]["heads"]]
    head_accs = list(results["head_ablation"]["accuracy"])

    colours = ["salmon" if h != "All induction" else "firebrick" for h in head_labels]
    ax2.axhline(baseline, color="grey", linestyle="--", label=f"Baseline ({baseline:.1%})", alpha=0.7)
    ax2.bar(range(len(head_labels)), head_accs, color=colours)
    ax2.set_xticks(range(len(head_labels)))
    ax2.set_xticklabels(head_labels, rotation=45, fontsize=8)
    ax2.set_ylabel("ICL top-1 accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_title("Head Ablation (Olsson et al. baseline)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"[ablations] Figure saved to {output_path}", flush=True)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run feature and head ablation experiments")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip computation — just regenerate figures from cached ablation_results.json",
    )
    parser.add_argument("--sae-path", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, default=CANDIDATE_IDS_PATH)
    parser.add_argument("--scores", type=Path, default=SCORES_PATH)
    parser.add_argument("--probes", type=Path, default=INDUCTION_PROBES_PATH)
    parser.add_argument("--head-corr", type=Path, default=RESULTS_DIR / "head_correspondence.parquet")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    # --plot-only: just regenerate figures from cached results
    if args.plot_only:
        if not ABLATION_RESULTS_PATH.exists():
            print(f"[ablations] ERROR: {ABLATION_RESULTS_PATH} not found. Run full ablation first.")
            return 1
        with ABLATION_RESULTS_PATH.open() as f:
            results = json.load(f)
        plot_ablation_curve(results, FIGURES_DIR / "ablation_curve.png")
        print("[ablations] Plot-only mode complete.", flush=True)
        return 0

    if args.sae_path is None:
        parser.error("--sae-path is required unless --plot-only is set")

    print("[ablations] Loading Gemma-2-2B ...", flush=True)
    model = load_model(device=args.device)

    print(f"[ablations] Loading SAE from {args.sae_path} ...", flush=True)
    sae = load_sae_local(args.sae_path, args.device)

    # Load candidate feature IDs (ranked by induction score)
    with args.candidates.open() as f:
        candidate_ids = json.load(f)  # already sorted by induction_score desc
    print(f"[ablations] {len(candidate_ids)} candidate features", flush=True)

    # Load probe sequences directly from parquet
    df_probes = pd.read_parquet(args.probes)
    probe_seqs = [
        (list(row["tokens"]), int(row["A"]), int(row["B"]))
        for _, row in df_probes.iterrows()
    ]

    # Baseline (no ablation)
    print("[ablations] Computing baseline ICL accuracy ...", flush=True)
    baseline_correct = evaluate_feature_ablation(
        model, sae, probe_seqs, ablate_feature_ids=[], device=args.device, batch_size=args.batch_size
    )
    baseline_acc, _, _ = bootstrap_ci(baseline_correct)
    print(f"[ablations] Baseline top-1 accuracy: {baseline_acc:.1%}", flush=True)

    # Feature ablation
    feat_results = {"N": [], "accuracy": [], "accuracy_drop": [], "ci_low": [], "ci_high": []}
    for N in FEATURE_ABLATION_NS:
        top_n_ids = candidate_ids[:N]
        print(f"[ablations] Feature ablation N={N} ...", flush=True)
        correct = evaluate_feature_ablation(
            model, sae, probe_seqs, ablate_feature_ids=top_n_ids,
            device=args.device, batch_size=args.batch_size,
        )
        acc, ci_lo, ci_hi = bootstrap_ci(correct)
        feat_results["N"].append(N)
        feat_results["accuracy"].append(acc)
        feat_results["accuracy_drop"].append(float(baseline_acc - acc))
        feat_results["ci_low"].append(ci_lo)
        feat_results["ci_high"].append(ci_hi)
        print(f"  N={N}: acc={acc:.1%} (drop={baseline_acc - acc:.1%}) CI=[{ci_lo:.1%}, {ci_hi:.1%}]", flush=True)

    # Head ablation (using heads most correlated with induction features)
    df_corr = pd.read_parquet(args.head_corr)
    # Identify "induction heads": heads with max correlation > 0.2 across candidate features
    head_max_corr = df_corr.groupby("head_idx")["correlation"].max()
    induction_head_idxs = sorted(head_max_corr[head_max_corr > 0.2].index.tolist())
    all_head_idxs = sorted(df_corr["head_idx"].unique().tolist())

    print(f"[ablations] Induction heads (corr>0.2): {induction_head_idxs}", flush=True)

    head_results = {"heads": [], "accuracy": [], "accuracy_drop": []}

    # Ablate each head individually
    for h in all_head_idxs:
        print(f"[ablations] Head ablation h={h} ...", flush=True)
        correct = evaluate_head_ablation(
            model, probe_seqs, ablate_heads=[h], device=args.device, batch_size=args.batch_size
        )
        acc = float(correct.mean())
        head_results["heads"].append(h)
        head_results["accuracy"].append(acc)
        head_results["accuracy_drop"].append(float(baseline_acc - acc))
        print(f"  h={h}: acc={acc:.1%} (drop={baseline_acc - acc:.1%})", flush=True)

    # Ablate all induction heads together
    if induction_head_idxs:
        print(f"[ablations] Ablating all induction heads {induction_head_idxs} ...", flush=True)
        correct = evaluate_head_ablation(
            model, probe_seqs, ablate_heads=induction_head_idxs,
            device=args.device, batch_size=args.batch_size,
        )
        acc = float(correct.mean())
        head_results["heads"].append("all_induction")
        head_results["accuracy"].append(acc)
        head_results["accuracy_drop"].append(float(baseline_acc - acc))
        print(f"  all_induction={induction_head_idxs}: acc={acc:.1%} (drop={baseline_acc - acc:.1%})", flush=True)

    results = {
        "baseline_accuracy": float(baseline_acc),
        "feature_ablation": feat_results,
        "head_ablation": head_results,
        "induction_heads_identified": induction_head_idxs,
        "n_probes": len(probe_seqs),
        "n_bootstrap": N_BOOTSTRAP,
    }

    ABLATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ABLATION_RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[ablations] Results saved to {ABLATION_RESULTS_PATH}", flush=True)

    plot_ablation_curve(results, FIGURES_DIR / "ablation_curve.png")

    print("\n[ablations] Summary:")
    print(f"  Baseline ICL accuracy: {baseline_acc:.1%}")
    for N, acc, drop in zip(feat_results["N"], feat_results["accuracy"], feat_results["accuracy_drop"]):
        print(f"  Feature ablation N={N:>3}: {acc:.1%} (drop={drop:.1%})")


if __name__ == "__main__":
    main()
