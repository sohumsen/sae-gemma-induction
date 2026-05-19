"""
Phase 4.3 — Identify the induction feature cluster.

For each SAE feature, we compute:
    induction_score = mean_activation(induction probes, final position)
                    - mean_activation(control sequences, final position)

where control sequences are matched sequences without an A-B repetition structure.

Features are ranked by induction_score. The top ~50–200 form the candidate cluster.

Outputs:
    results/induction_feature_scores.parquet
        columns: feature_id, induction_mean, control_mean, induction_score, rank
    results/induction_candidate_ids.json
        List of feature IDs in the top-N candidates (default N=100)

Usage:
    python src/sae_gemma/find_induction_features.py \
        --sae-path models/sae_main \
        --probes results/induction_probes.parquet \
        --top-n 100
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from sae_gemma.model_utils import load_model

from sae_gemma.paths import HOOK_NAME, INDUCTION_PROBES_PATH, REPO_ROOT, RESULTS_DIR
from sae_gemma.induction_probes import _safe_vocab_range

SCORES_PATH = RESULTS_DIR / "induction_feature_scores.parquet"
CANDIDATE_IDS_PATH = RESULTS_DIR / "induction_candidate_ids.json"

def load_sae_local(sae_path: Path, device: str):
    from sae_lens.saes.sae import SAE
    sae = SAE.load_from_disk(str(sae_path), device=device)
    sae.eval()
    return sae


@torch.no_grad()
def _get_final_pos_features(
    model: HookedTransformer,
    sae,
    hook_name: str,
    token_seqs: list[list[int]],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    For each sequence in token_seqs, run Gemma + SAE and return the SAE feature
    activation vector at the FINAL position (where induction prediction occurs).

    Returns: np.ndarray of shape (n_seqs, n_features)
    """
    n_seqs = len(token_seqs)
    n_features = sae.cfg.d_sae
    result = np.zeros((n_seqs, n_features), dtype=np.float32)

    for i in range(0, n_seqs, batch_size):
        batch_seqs = token_seqs[i: i + batch_size]
        max_len = max(len(s) for s in batch_seqs)
        seq_lens = [len(s) for s in batch_seqs]

        padded = torch.zeros(len(batch_seqs), max_len, dtype=torch.long, device=device)
        for j, s in enumerate(batch_seqs):
            padded[j, :len(s)] = torch.tensor(s, dtype=torch.long)

        residuals = {}

        def hook_fn(value, hook):
            residuals[hook.name] = value.detach()

        model.run_with_hooks(padded, fwd_hooks=[(hook_name, hook_fn)])
        acts = residuals[hook_name]  # (batch, seq, d_model)

        for j, slen in enumerate(seq_lens):
            final_act = acts[j, slen - 1, :].unsqueeze(0)  # (1, d_model)
            feat_acts = sae.encode(final_act.float())  # SAE trained in float32
            result[i + j] = feat_acts.squeeze(0).cpu().float().numpy()

        if (i // batch_size) % 10 == 0:
            n_done = min(i + batch_size, n_seqs)
            print(f"  {n_done}/{n_seqs} sequences", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="Identify induction feature cluster")
    parser.add_argument("--sae-path", type=Path, required=True)
    parser.add_argument("--probes", type=Path, default=INDUCTION_PROBES_PATH)
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of top candidate features to export")
    parser.add_argument("--n-controls", type=int, default=2000,
                        help="Number of matched control sequences to generate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    print("[find_induction] Loading Gemma-2-2B ...", flush=True)
    model = load_model(device=args.device)

    print(f"[find_induction] Loading SAE from {args.sae_path} ...", flush=True)
    sae = load_sae_local(args.sae_path, args.device)
    n_features = sae.cfg.d_sae
    print(f"[find_induction] n_features={n_features}, hook={HOOK_NAME}", flush=True)

    # Load probe sequences directly from parquet (tokens column stores full sequences)
    df_probes = pd.read_parquet(args.probes)
    print(f"[find_induction] Loaded {len(df_probes)} induction probe sequences", flush=True)
    induction_seqs = [list(row) for row in df_probes["tokens"].tolist()]

    vocab_lo, vocab_hi = _safe_vocab_range(model.cfg.d_vocab)

    # Generate matched control sequences (same lengths, no A-B structure)
    print(f"[find_induction] Generating {args.n_controls} control sequences ...", flush=True)
    rng_ctrl = random.Random(args.seed)
    control_seqs = []
    for i in range(args.n_controls):
        row = df_probes.iloc[i % len(df_probes)]
        total_len = len(induction_seqs[i % len(induction_seqs)])
        toks = [rng_ctrl.randint(vocab_lo, vocab_hi) for _ in range(total_len)]
        control_seqs.append(toks)

    # Get final-position feature activations
    print("[find_induction] Computing feature activations for induction probes ...", flush=True)
    t0 = time.monotonic()
    induction_acts = _get_final_pos_features(
        model, sae, HOOK_NAME, induction_seqs, args.device, args.batch_size
    )  # (n_probes, n_features)

    print("[find_induction] Computing feature activations for control sequences ...", flush=True)
    control_acts = _get_final_pos_features(
        model, sae, HOOK_NAME, control_seqs, args.device, args.batch_size
    )  # (n_controls, n_features)

    elapsed = time.monotonic() - t0
    print(f"[find_induction] Activations computed in {elapsed:.0f}s", flush=True)

    # Compute induction scores
    induction_mean = induction_acts.mean(axis=0)   # (n_features,)
    control_mean = control_acts.mean(axis=0)        # (n_features,)
    induction_score = induction_mean - control_mean  # (n_features,)

    # Rank features
    ranked_ids = np.argsort(-induction_score)  # descending

    # Build and save scores DataFrame
    scores_df = pd.DataFrame({
        "feature_id": np.arange(n_features, dtype=np.int32),
        "induction_mean": induction_mean.astype(np.float32),
        "control_mean": control_mean.astype(np.float32),
        "induction_score": induction_score.astype(np.float32),
        "rank": np.argsort(np.argsort(-induction_score)).astype(np.int32),
    })
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(SCORES_PATH, index=False)
    print(f"[find_induction] Scores saved to {SCORES_PATH}", flush=True)

    # Export top-N candidate IDs
    top_ids = ranked_ids[:args.top_n].tolist()
    with CANDIDATE_IDS_PATH.open("w") as f:
        json.dump(top_ids, f, indent=2)
    print(f"[find_induction] Top-{args.top_n} candidates saved to {CANDIDATE_IDS_PATH}", flush=True)

    # Print top-20 for quick inspection
    print("\n[find_induction] Top-20 induction feature candidates:", flush=True)
    print(f"{'Rank':>5} {'FeatID':>8} {'Induction':>10} {'Control':>10} {'Score':>10}")
    for rank, fid in enumerate(ranked_ids[:20]):
        print(
            f"{rank:>5} {fid:>8} {induction_mean[fid]:>10.4f} "
            f"{control_mean[fid]:>10.4f} {induction_score[fid]:>10.4f}"
        )


if __name__ == "__main__":
    main()
