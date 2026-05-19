"""
Cross-seed replication analysis for a single SAE training run.

For a given trained SAE (from a non-default seed, e.g. seed=43 / seed=44):
1. Load it from `models/sae_main_dl_seed{N}/trainer_0/ae.pt`.
2. Score all features by induction_score on our 2,000 induction probes vs controls.
3. Identify the top-100 induction features.
4. Run feature ablation: zero the top-50 induction features and measure ICL accuracy drop.
5. Save results to `results/seed{N}_replication.json`.

Output is self-contained — does NOT overwrite v9c artifacts. Read by the
writeup's replication section to verify the qualitative findings hold.

    python scripts/replicate_seed.py --seed 43
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from sae_gemma.model_utils import load_model
from sae_gemma.paths import HOOK_NAME, INDUCTION_PROBES_PATH, REPO_ROOT, RESULTS_DIR
from sae_gemma.induction_probes import _safe_vocab_range


def load_dl_sae_as_saelens(seed: int, device: str):
    """Convert dl ae.pt into in-memory SAELens TopKSAE, no disk write."""
    dl_dir = REPO_ROOT / "models" / f"sae_main_dl_seed{seed}" / "trainer_0"
    ae_pt = dl_dir / "ae.pt"
    cfg_path = dl_dir / "config.json"
    if not ae_pt.exists():
        raise FileNotFoundError(f"{ae_pt} not found")

    sd = torch.load(ae_pt, map_location="cpu")
    dl_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    k = dl_cfg.get("trainer", {}).get("k", 100)
    d_sae, d_in = sd["encoder.weight"].shape

    # Build SAELens TopKSAE in memory
    from sae_lens.saes.topk_sae import TopKSAE, TopKSAEConfig

    cfg = TopKSAEConfig(
        d_in=int(d_in),
        d_sae=int(d_sae),
        dtype="float32",
        device=device,
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",
        k=int(k),
        rescale_acts_by_decoder_norm=False,
        reshape_activations="none",
    )
    sae = TopKSAE(cfg)
    state = {
        "W_enc": sd["encoder.weight"].T.contiguous().float(),
        "W_dec": sd["decoder.weight"].T.contiguous().float(),
        "b_enc": sd["encoder.bias"].float(),
        "b_dec": sd["b_dec"].float(),
    }
    sae.load_state_dict(state, strict=False, assign=True)
    sae = sae.to(device)
    sae.eval()
    return sae


@torch.no_grad()
def get_final_pos_features(model, sae, token_seqs, device, batch_size=16, return_logits=False):
    """Return per-probe SAE feature activations at final position. [n_probes, d_sae].
    If return_logits, also return final-position logits [n_probes, vocab]."""
    all_feats = []
    all_logits = []
    for i in range(0, len(token_seqs), batch_size):
        batch = token_seqs[i: i + batch_size]
        max_len = max(len(seq) for seq in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        seq_lens = []
        for j, seq in enumerate(batch):
            padded[j, : len(seq)] = seq
            seq_lens.append(len(seq))

        captured = {}
        def cap_resid(value, hook):
            captured["resid"] = value
            return value

        out = model.run_with_hooks(padded, fwd_hooks=[(HOOK_NAME, cap_resid)])
        resid = captured["resid"]  # [B, S, D]

        final_resid = torch.stack([resid[j, seq_lens[j] - 1, :] for j in range(len(batch))])
        z = sae.encode(final_resid.float())
        all_feats.append(z.cpu().numpy())
        if return_logits:
            final_logits = torch.stack([out[j, seq_lens[j] - 1, :] for j in range(len(batch))])
            all_logits.append(final_logits.cpu().numpy())

    feats = np.concatenate(all_feats, axis=0)
    if return_logits:
        logits = np.concatenate(all_logits, axis=0)
        return feats, logits
    return feats


@torch.no_grad()
def measure_icl_accuracy_with_ablation(model, sae, probes_df, ablate_ids: set, device, batch_size=16):
    """Run model with SAE-patch + ablation; return top-1 ICL accuracy."""
    mask = torch.zeros(sae.cfg.d_sae, dtype=torch.bool, device=device)
    if ablate_ids:
        mask[list(ablate_ids)] = True

    correct = 0
    total = 0
    tokens_list = [torch.tensor(t, dtype=torch.long, device=device) for t in probes_df["tokens"]]
    answers = probes_df["B"].tolist()

    for i in range(0, len(tokens_list), batch_size):
        batch = tokens_list[i: i + batch_size]
        max_len = max(len(seq) for seq in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        seq_lens = []
        for j, seq in enumerate(batch):
            padded[j, : len(seq)] = seq
            seq_lens.append(len(seq))

        def patch_hook(value, hook):
            B, S, D = value.shape
            flat = value.reshape(B * S, D).float()
            z = sae.encode(flat)
            z_abl = z * (~mask).float()
            recon = sae.decode(z_abl)
            recon_orig = sae.decode(z)
            delta = (recon - recon_orig).reshape(B, S, D).to(value.dtype)
            return value + delta

        logits = model.run_with_hooks(padded, fwd_hooks=[(HOOK_NAME, patch_hook)])
        for j in range(len(batch)):
            pred = logits[j, seq_lens[j] - 1].argmax().item()
            if pred == answers[i + j]:
                correct += 1
            total += 1
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Seed of the SAE training run (43 or 44).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--n-ablate", type=int, default=50)
    parser.add_argument("--n-controls", type=int, default=2000)
    parser.add_argument("--n-vocab-min", type=int, default=1000)
    parser.add_argument("--n-vocab-max", type=int, default=20000)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    print(f"[replicate] Loading model + seed-{args.seed} SAE ...", flush=True)
    model = load_model(device=args.device)
    sae = load_dl_sae_as_saelens(args.seed, args.device)

    # Probes + controls
    probes_df = pd.read_parquet(INDUCTION_PROBES_PATH)
    probe_tokens = [torch.tensor(t, dtype=torch.long, device=args.device) for t in probes_df["tokens"]]

    # Build matched control sequences (same length distribution, no A-B repeat)
    tokenizer = model.tokenizer
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    safe_lo, safe_hi = _safe_vocab_range(vocab_size)
    # Clip to user-specified range too
    safe_lo = max(safe_lo, args.n_vocab_min)
    safe_hi = min(safe_hi, args.n_vocab_max)
    rng = random.Random(123)
    control_tokens = []
    lengths = [len(t) for t in probe_tokens]
    for ln in lengths[: args.n_controls]:
        # uniformly random tokens of same length
        seq = torch.tensor([rng.randint(safe_lo, safe_hi - 1) for _ in range(ln)],
                           dtype=torch.long, device=args.device)
        control_tokens.append(seq)

    t0 = time.monotonic()
    print(f"[replicate] Computing feature activations on {len(probe_tokens)} induction + {len(control_tokens)} control probes ...", flush=True)
    feats_ind = get_final_pos_features(model, sae, probe_tokens, args.device, args.batch_size)
    feats_ctrl = get_final_pos_features(model, sae, control_tokens, args.device, args.batch_size)
    print(f"[replicate]   features: induction mean={feats_ind.mean():.4f}  control mean={feats_ctrl.mean():.4f}  ({(time.monotonic() - t0)/60:.1f}m)", flush=True)

    # Score
    induction_mean = feats_ind.mean(axis=0)        # [d_sae]
    control_mean = feats_ctrl.mean(axis=0)
    induction_score = induction_mean - control_mean
    ranking = np.argsort(-induction_score)         # descending
    top_n_ids = ranking[: args.top_n].tolist()

    # Ablation
    print(f"[replicate] Running ablation: zero top-{args.n_ablate} induction features ...", flush=True)
    baseline_acc = measure_icl_accuracy_with_ablation(model, sae, probes_df, set(), args.device, args.batch_size)
    ablate_ids = set(top_n_ids[: args.n_ablate])
    ablated_acc = measure_icl_accuracy_with_ablation(model, sae, probes_df, ablate_ids, args.device, args.batch_size)
    drop = baseline_acc - ablated_acc
    print(f"[replicate]   baseline acc = {baseline_acc:.4f}", flush=True)
    print(f"[replicate]   ablated acc  = {ablated_acc:.4f}  (drop = {drop:+.4f} = {drop*100:+.2f}pp)", flush=True)

    out = {
        "seed": args.seed,
        "n_probes": len(probe_tokens),
        "n_controls": len(control_tokens),
        "top_feature_id": int(top_n_ids[0]),
        "top_induction_score": float(induction_score[top_n_ids[0]]),
        "top_induction_mean": float(induction_mean[top_n_ids[0]]),
        "top_control_mean": float(control_mean[top_n_ids[0]]),
        "top20_ids": [int(x) for x in top_n_ids[:20]],
        "top20_scores": [float(induction_score[x]) for x in top_n_ids[:20]],
        "top20_mean_score": float(np.mean([induction_score[x] for x in top_n_ids[:20]])),
        "n_ablate": args.n_ablate,
        "baseline_accuracy": float(baseline_acc),
        "ablated_accuracy": float(ablated_acc),
        "drop_pp": float(drop * 100),
    }
    out_path = RESULTS_DIR / f"seed{args.seed}_replication.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[replicate] === SEED {args.seed} SUMMARY ===", flush=True)
    print(f"  Top induction feature: F{out['top_feature_id']}  (induction score = {out['top_induction_score']:.3f})", flush=True)
    print(f"  Top-20 mean induction score: {out['top20_mean_score']:.3f}", flush=True)
    print(f"  Top-50 ablation drop: {out['drop_pp']:.2f}pp  ({out['baseline_accuracy']*100:.2f}% -> {out['ablated_accuracy']*100:.2f}%)", flush=True)
    print(f"  Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
