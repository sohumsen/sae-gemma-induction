"""
Phase 6 — Activation patching: causal head-feature link.

For each induction probe, runs Gemma-2-2B twice:
1. Baseline — capture target features' activations at the final position.
2. Per-head ablation (h=0..7) — zero head h's hook_z output at all positions; capture again.

Outputs (results/activation_patching.json + figure) report the mean reduction in
each target feature's activation when each head is ablated.

Hypothesis: head-6 ablation should cause the largest reduction in F15289 (and the
other top induction features), upgrading the head-correspondence claim from
correlational (Pearson r=0.12) to causal (% of feature activation contributed by head 6).

Usage:
    python src/sae_gemma/activation_patching.py --n-probes 500
"""
import argparse
import json
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from sae_gemma.model_utils import load_model
from sae_gemma.paths import (
    HOOK_NAME,
    INDUCTION_PROBES_PATH,
    REPO_ROOT,
    RESULTS_DIR,
    SAE_MAIN_DIR,
)

ATTN_Z_HOOK = "blocks.12.attn.hook_z"
LAYER = 12
N_HEADS = 8
TARGET_FEATURES = [15289, 11606, 14740, 7467]  # top induction-clean features by score


def load_sae_local(sae_path: Path, device: str):
    from sae_lens.saes.sae import SAE
    sae = SAE.load_from_disk(str(sae_path), device=device)
    sae.eval()
    return sae


def zero_head_hook(value: torch.Tensor, hook, head_idx: int) -> torch.Tensor:
    """Zero out head_idx's z output at all positions."""
    value = value.clone()
    value[:, :, head_idx, :] = 0
    return value


def mean_head_hook(value: torch.Tensor, hook, head_idx: int, mean_z: torch.Tensor) -> torch.Tensor:
    """Replace head_idx's z output at all positions with the precomputed mean vector.

    `mean_z` is the per-head-dim mean over a representative distribution of probe positions,
    shape [d_head]. Broadcast-replace.
    """
    value = value.clone()
    value[:, :, head_idx, :] = mean_z.to(value.dtype).to(value.device)
    return value


@torch.no_grad()
def compute_head_z_mean(model, token_seqs, device, batch_size=8):
    """Compute per-head mean of hook_z across all positions of all probes.
    Returns tensor of shape [n_heads, d_head]."""
    accum = None
    n = 0
    for i in range(0, len(token_seqs), batch_size):
        batch = token_seqs[i: i + batch_size]
        max_len = max(len(seq) for seq in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        seq_lens = []
        for j, seq in enumerate(batch):
            padded[j, : len(seq)] = seq
            seq_lens.append(len(seq))

        captured = {}
        def cap_z(value, hook):
            captured["z"] = value  # [B, S, H, D_head]
            return value

        model.run_with_hooks(padded, fwd_hooks=[(ATTN_Z_HOOK, cap_z)])
        z = captured["z"]  # [B, S, H, D]
        # Sum over batch and seq dims (only over real positions, not pad)
        for j in range(len(batch)):
            zj = z[j, : seq_lens[j], :, :]  # [S_j, H, D]
            if accum is None:
                accum = zj.sum(dim=0).clone()  # [H, D]
            else:
                accum += zj.sum(dim=0)
            n += seq_lens[j]
    return (accum / n)  # [H, D]


@torch.no_grad()
def get_feature_activations(
    model: HookedTransformer,
    sae,
    token_seqs: list[torch.Tensor],
    target_features: list[int],
    device: str,
    head_ablate: int | None = None,
    ablation_mode: str = "zero",
    mean_z: torch.Tensor | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Run probes; return activations of target_features at each probe's final position. Shape [n_probes, n_features].

    ablation_mode='zero': zero head_ablate's z output (default, OOD).
    ablation_mode='mean': replace head_ablate's z with mean_z[head_ablate] at every position.
    """
    all_acts = []
    for i in range(0, len(token_seqs), batch_size):
        batch = token_seqs[i: i + batch_size]
        max_len = max(len(seq) for seq in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        seq_lens = []
        for j, seq in enumerate(batch):
            padded[j, : len(seq)] = seq
            seq_lens.append(len(seq))

        captured: dict = {}

        def capture_resid(value, hook):
            captured["resid"] = value
            return value

        fwd_hooks = [(HOOK_NAME, capture_resid)]
        if head_ablate is not None:
            if ablation_mode == "zero":
                fwd_hooks.append((ATTN_Z_HOOK, partial(zero_head_hook, head_idx=head_ablate)))
            elif ablation_mode == "mean":
                assert mean_z is not None, "mean ablation requires mean_z"
                fwd_hooks.append((ATTN_Z_HOOK, partial(mean_head_hook, head_idx=head_ablate, mean_z=mean_z[head_ablate])))
            else:
                raise ValueError(f"Unknown ablation_mode {ablation_mode}")

        model.run_with_hooks(padded, fwd_hooks=fwd_hooks)
        resid = captured["resid"]  # [batch, seq, d_model]

        # Activation at each probe's actual final (non-pad) position.
        final_resid = torch.stack(
            [resid[j, seq_lens[j] - 1, :] for j in range(len(batch))]
        )  # [batch, d_model]

        feature_acts = sae.encode(final_resid.float())  # [batch, d_sae]
        all_acts.append(feature_acts[:, target_features].cpu().numpy())

    return np.concatenate(all_acts, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Activation-patch each layer-12 head and measure effect on top induction features")
    parser.add_argument("--n-probes", type=int, default=500, help="Subsample of induction probes to use (max 2000)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sae-path", type=Path, default=SAE_MAIN_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "activation_patching.json")
    parser.add_argument("--ablation-mode", choices=["zero", "mean"], default="zero",
                        help="zero: replace head z with zeros (OOD). mean: replace with per-head mean over the probe distribution (more principled).")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    print(f"[ap] Loading model ...", flush=True)
    model = load_model(device=args.device)

    print(f"[ap] Loading SAE from {args.sae_path} ...", flush=True)
    sae = load_sae_local(args.sae_path, args.device)

    print(f"[ap] Loading {args.n_probes} induction probes ...", flush=True)
    probes_df = pd.read_parquet(INDUCTION_PROBES_PATH).head(args.n_probes)
    probe_tokens = [torch.tensor(t, dtype=torch.long, device=args.device) for t in probes_df["tokens"]]
    print(f"[ap] {len(probe_tokens)} probes loaded; first len={len(probe_tokens[0])}", flush=True)

    t0 = time.monotonic()
    print("[ap] Baseline (no ablation) ...", flush=True)
    baseline = get_feature_activations(model, sae, probe_tokens, TARGET_FEATURES, args.device, head_ablate=None, batch_size=args.batch_size)
    print(f"[ap]   baseline mean across probes: {baseline.mean(0)}", flush=True)

    mean_z = None
    if args.ablation_mode == "mean":
        print(f"[ap] Computing per-head z means across probes for mean-ablation ...", flush=True)
        mean_z = compute_head_z_mean(model, probe_tokens, args.device, args.batch_size)
        print(f"[ap]   mean_z shape: {tuple(mean_z.shape)}, head-norms: {mean_z.norm(dim=-1).tolist()}", flush=True)

    per_head: dict[int, np.ndarray] = {}
    for h in range(N_HEADS):
        print(f"[ap] {args.ablation_mode}-ablating head {h} ...", flush=True)
        per_head[h] = get_feature_activations(
            model, sae, probe_tokens, TARGET_FEATURES, args.device,
            head_ablate=h, ablation_mode=args.ablation_mode, mean_z=mean_z, batch_size=args.batch_size,
        )
        print(f"[ap]   head-{h}-ablated mean:    {per_head[h].mean(0)}", flush=True)

    elapsed = time.monotonic() - t0
    print(f"[ap] All 9 passes done in {elapsed/60:.1f} min", flush=True)

    # Assemble result
    result = {
        "target_features": TARGET_FEATURES,
        "n_probes": len(probe_tokens),
        "layer": LAYER,
        "n_heads": N_HEADS,
        "baseline_mean": baseline.mean(0).tolist(),
        "baseline_std": baseline.std(0).tolist(),
        "head_results": {},
    }
    for h in range(N_HEADS):
        ablated_mean = per_head[h].mean(0)
        reduction = baseline.mean(0) - ablated_mean
        reduction_pct = reduction / np.maximum(baseline.mean(0), 1e-6) * 100
        result["head_results"][str(h)] = {
            "ablated_mean": ablated_mean.tolist(),
            "ablated_std": per_head[h].std(0).tolist(),
            "reduction": reduction.tolist(),
            "reduction_pct": reduction_pct.tolist(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[ap] Saved results -> {args.output}", flush=True)

    # Figure
    fig, axes = plt.subplots(1, len(TARGET_FEATURES), figsize=(4 * len(TARGET_FEATURES), 3.5), sharey=True)
    if len(TARGET_FEATURES) == 1:
        axes = [axes]
    for i, fid in enumerate(TARGET_FEATURES):
        ax = axes[i]
        reductions_pct = [result["head_results"][str(h)]["reduction_pct"][i] for h in range(N_HEADS)]
        colors = ["crimson" if h == 6 else "steelblue" for h in range(N_HEADS)]
        ax.bar(range(N_HEADS), reductions_pct, color=colors, edgecolor="white")
        ax.set_title(f"F{fid}")
        ax.set_xlabel("Head index (layer 12)")
        if i == 0:
            ax.set_ylabel("% reduction in feature activation\nwhen head is ablated")
        ax.set_xticks(range(N_HEADS))
        ax.axhline(0, color="black", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.suptitle(f"Activation patching: how much each layer-12 head contributes to each induction feature\n(red = head 6, the head-level ablation winner)", y=1.02)
    plt.tight_layout()
    fig_path = RESULTS_DIR / "figures" / "activation_patching.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ap] Figure saved -> {fig_path}", flush=True)


if __name__ == "__main__":
    main()
