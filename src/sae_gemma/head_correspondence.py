"""
Phase 4.6 — Feature-to-attention-head correspondence analysis.

For each top induction candidate feature, compute the Pearson correlation between:
    - The feature's activation at each token position across induction probes
    - The attention pattern (sum of attention received at the key position = pos of first A)
      for each layer-12 attention head

The Olsson et al. induction heads show a characteristic "shifted diagonal" attention
pattern: at the position of the second A, they attend back to the position of B
(i.e. the token immediately following the first A).

We also compute a simpler metric: the attention head's "induction score" (mean
attention to position (pos_of_second_A - prefix_len - gap_len - 1)) as a comparison.

Outputs:
    results/head_correspondence.parquet
        columns: feature_id, head_idx, layer, correlation, p_value, n_probes
    results/figures/head_correspondence.png
        Bar chart of top-N features × all heads, sorted by peak correlation

Usage:
    python src/sae_gemma/head_correspondence.py \
        --sae-path models/sae_main \
        --candidates results/induction_candidate_ids.json \
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
from scipy import stats
from transformer_lens import HookedTransformer

from sae_gemma.model_utils import load_model
from sae_gemma.paths import FIGURES_DIR, HEAD_CORR_PATH, HOOK_NAME, INDUCTION_PROBES_PATH, N_HEADS, REPO_ROOT, RESULTS_DIR

CANDIDATE_IDS_PATH = RESULTS_DIR / "induction_candidate_ids.json"
TARGET_LAYER = 12


def load_sae_local(sae_path: Path, device: str):
    from sae_lens.saes.sae import SAE
    sae = SAE.load_from_disk(str(sae_path), device=device)
    sae.eval()
    return sae


@torch.no_grad()
def compute_correlations(
    model: HookedTransformer,
    sae,
    probe_seqs: list[tuple[list[int], int, int, int]],  # (tokens, A, B, prefix_len)
    candidate_feature_ids: list[int],
    device: str,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    For each (feature, head) pair, compute Pearson r between:
        x = feature activation at final position
        y = attention weight from final position to position of B (the induction signal)

    Returns a DataFrame with columns: feature_id, head_idx, correlation, p_value, n_probes
    """
    n_heads = N_HEADS  # Gemma-2-2B: 8 query heads (GQA, 4 KV heads)
    n_probes = len(probe_seqs)

    # Store per-probe: feature activations + attention weights
    feature_acts_arr = np.zeros((n_probes, len(candidate_feature_ids)), dtype=np.float32)
    # attention_arr[probe, head] = attention weight from pos_second_A to pos_B
    attention_arr = np.zeros((n_probes, n_heads), dtype=np.float32)

    attn_hook_name = f"blocks.{TARGET_LAYER}.attn.hook_pattern"
    resid_hook_name = HOOK_NAME

    for i in range(0, n_probes, batch_size):
        batch = probe_seqs[i: i + batch_size]
        max_len = max(len(p[0]) for p in batch)
        seq_lens = [len(p[0]) for p in batch]
        # Position of B in each sequence: prefix_len + 1 (0-indexed)
        # Position of second A: len - 1
        # Position of B (= first A + 1): prefix_len + 1
        # We need to store these per-probe

        padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        for j, (toks, A, B, prefix_len) in enumerate(batch):
            padded[j, :len(toks)] = torch.tensor(toks, dtype=torch.long)

        residuals = {}
        attn_patterns = {}

        def hook_resid(value, hook):
            residuals[hook.name] = value.detach()

        def hook_attn(value, hook):
            attn_patterns[hook.name] = value.detach()

        model.run_with_hooks(
            padded,
            fwd_hooks=[
                (resid_hook_name, hook_resid),
                (attn_hook_name, hook_attn),
            ],
        )

        acts = residuals[resid_hook_name]  # (batch, seq, d_model)
        attn = attn_patterns[attn_hook_name]  # (batch, n_heads, seq_q, seq_k)

        for j, (toks, A, B, prefix_len) in enumerate(batch):
            slen = seq_lens[j]
            pos_second_A = slen - 1
            pos_B = prefix_len + 1  # probe format: [prefix] A B [gap] A

            # SAE feature activations at final position
            final_act = acts[j, pos_second_A, :].unsqueeze(0)
            feat_acts = sae.encode(
                final_act.float()  # SAE trained in float32
            ).squeeze(0).cpu().float().numpy()

            for fi, fid in enumerate(candidate_feature_ids):
                feature_acts_arr[i + j, fi] = feat_acts[fid]

            # Attention from second_A to B for each head
            for h in range(n_heads):
                attention_arr[i + j, h] = float(
                    attn[j, h, pos_second_A, pos_B].cpu().float().item()
                )

        if (i // batch_size) % 5 == 0:
            n_done = min(i + batch_size, n_probes)
            print(f"  {n_done}/{n_probes} probes", flush=True)

    # Compute Pearson correlations
    rows = []
    for fi, fid in enumerate(candidate_feature_ids):
        x = feature_acts_arr[:, fi]
        for h in range(n_heads):
            y = attention_arr[:, h]
            if x.std() < 1e-8 or y.std() < 1e-8:
                r, p = 0.0, 1.0
            else:
                r, p = stats.pearsonr(x, y)
            rows.append({
                "feature_id": fid,
                "head_idx": h,
                "layer": TARGET_LAYER,
                "correlation": float(r),
                "p_value": float(p),
                "n_probes": n_probes,
            })

    return pd.DataFrame(rows)


def plot_correspondence(df: pd.DataFrame, top_n_features: int = 20, output_path: Path = None):
    """Plot heatmap of feature × head correlation, for the top-N features by peak correlation."""
    top_features = (
        df.groupby("feature_id")["correlation"].max()
        .nlargest(top_n_features)
        .index.tolist()
    )
    sub = df[df["feature_id"].isin(top_features)]
    pivot = sub.pivot(index="feature_id", columns="head_idx", values="correlation")

    fig, ax = plt.subplots(figsize=(10, max(4, top_n_features * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"H{h}" for h in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"F{fid}" for fid in pivot.index], fontsize=7)
    ax.set_xlabel(f"Layer-{TARGET_LAYER} attention head")
    ax.set_ylabel("SAE feature (top candidates)")
    ax.set_title(f"Feature–Head correspondence (top-{top_n_features} induction features)")
    plt.colorbar(im, ax=ax, label="Pearson r (feature act vs attention to B)")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"[head_corr] Figure saved to {output_path}", flush=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute feature-to-head correspondence")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip computation — just regenerate figure from cached head_correspondence.parquet",
    )
    parser.add_argument("--sae-path", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, default=CANDIDATE_IDS_PATH)
    parser.add_argument("--probes", type=Path, default=INDUCTION_PROBES_PATH)
    parser.add_argument("--output", type=Path, default=HEAD_CORR_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    if args.plot_only:
        if not args.output.exists():
            print(f"[head_corr] ERROR: {args.output} not found. Run full analysis first.")
            return 1
        df_corr = pd.read_parquet(args.output)
        plot_correspondence(df_corr, top_n_features=20,
                            output_path=FIGURES_DIR / "head_correspondence.png")
        print("[head_corr] Plot-only mode complete.", flush=True)
        return 0

    if args.sae_path is None:
        parser.error("--sae-path is required unless --plot-only is set")

    print("[head_corr] Loading Gemma-2-2B ...", flush=True)
    model = load_model(device=args.device)

    print(f"[head_corr] Loading SAE from {args.sae_path} ...", flush=True)
    sae = load_sae_local(args.sae_path, args.device)

    with args.candidates.open() as f:
        candidate_ids = json.load(f)
    print(f"[head_corr] {len(candidate_ids)} candidate features", flush=True)

    # Load probe sequences directly from parquet
    df_probes = pd.read_parquet(args.probes)
    probe_seqs = [
        (list(row["tokens"]), int(row["A"]), int(row["B"]), int(row["prefix_len"]))
        for _, row in df_probes.iterrows()
    ]

    print("[head_corr] Computing feature-head correlations ...", flush=True)
    df_corr = compute_correlations(
        model, sae, probe_seqs, candidate_ids, args.device, args.batch_size
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_corr.to_parquet(args.output, index=False)
    print(f"[head_corr] Saved {len(df_corr)} rows to {args.output}", flush=True)

    # Print top findings
    top = df_corr.nlargest(20, "correlation")[["feature_id", "head_idx", "correlation", "p_value"]]
    print("\n[head_corr] Top-20 feature-head correlations:")
    print(top.to_string(index=False))

    # Plot
    plot_correspondence(df_corr, top_n_features=20,
                        output_path=FIGURES_DIR / "head_correspondence.png")


if __name__ == "__main__":
    main()
