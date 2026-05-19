"""
Compare our v9c SAE's induction features against a public Gemma-Scope SAE.

Loads a pre-trained Gemma-Scope SAE (Google DeepMind release) for
google/gemma-2-2b layer 12 residual stream, scores its features by the same
induction_score we use for v9c (mean activation on induction probes - mean
activation on matched controls, at the final probe position), and reports
overlap between the top-20 induction features of the two SAEs.

Outputs:
    results/saebench_induction_scores.parquet
        columns: feature_id, induction_mean, control_mean, induction_score, rank
    results/saebench_candidate_ids.json
        Top-100 feature IDs

Usage:
    python scripts/sae_bench_comparison.py
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sae_gemma.model_utils import load_model
from sae_gemma.paths import (
    HOOK_NAME,
    INDUCTION_PROBES_PATH,
    REPO_ROOT,
    RESULTS_DIR,
)
from sae_gemma.induction_probes import _safe_vocab_range
from sae_gemma.find_induction_features import _get_final_pos_features

# v9c reference outputs (read-only — never overwritten by this script)
V9C_SCORES_PATH = RESULTS_DIR / "induction_feature_scores.parquet"
V9C_CANDIDATE_IDS_PATH = RESULTS_DIR / "induction_candidate_ids.json"

# New outputs for the public SAE
SAEBENCH_SCORES_PATH = RESULTS_DIR / "saebench_induction_scores.parquet"
SAEBENCH_CANDIDATE_IDS_PATH = RESULTS_DIR / "saebench_candidate_ids.json"

# Gemma-Scope release on HuggingFace (Google DeepMind)
DEFAULT_RELEASE = "gemma-scope-2b-pt-res-canonical"
DEFAULT_SAE_ID = "layer_12/width_16k/canonical"
# Fallback if canonical isn't registered: non-canonical release uses an L0 suffix
FALLBACK_RELEASE = "gemma-scope-2b-pt-res"
FALLBACK_SAE_ID_PREFIX = "layer_12/width_16k/average_l0_"


def load_public_sae(device: str):
    """
    Load the public Gemma-Scope SAE for layer 12, width 16k.

    Tries the canonical release first; falls back to picking any available
    average_l0_* variant from the non-canonical release if needed.
    """
    from sae_lens.saes.sae import SAE
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    directory = get_pretrained_saes_directory()

    # 1) Try the canonical release first
    if DEFAULT_RELEASE in directory and DEFAULT_SAE_ID in directory[DEFAULT_RELEASE].saes_map:
        release, sae_id = DEFAULT_RELEASE, DEFAULT_SAE_ID
        print(f"[saebench] Using canonical release: {release} / {sae_id}", flush=True)
    else:
        # 2) Fallback: pick the smallest-L0 width_16k variant from the non-canonical release
        if FALLBACK_RELEASE not in directory:
            raise RuntimeError(
                f"Neither {DEFAULT_RELEASE} nor {FALLBACK_RELEASE} found in sae_lens "
                f"pretrained_saes_directory. Update sae_lens or check release names."
            )
        candidates = [
            sid for sid in directory[FALLBACK_RELEASE].saes_map
            if sid.startswith(FALLBACK_SAE_ID_PREFIX)
        ]
        if not candidates:
            raise RuntimeError(
                f"No '{FALLBACK_SAE_ID_PREFIX}*' SAE found in release {FALLBACK_RELEASE}. "
                f"Available IDs starting with 'layer_12/width_16k': "
                f"{[s for s in directory[FALLBACK_RELEASE].saes_map if s.startswith('layer_12/width_16k')]}"
            )
        # Sort by the L0 number embedded in the id and take the median-ish one
        def _l0(s: str) -> int:
            try:
                return int(s.rsplit("_", 1)[-1])
            except ValueError:
                return 10**9
        candidates.sort(key=_l0)
        sae_id = candidates[len(candidates) // 2]
        release = FALLBACK_RELEASE
        print(f"[saebench] Canonical not available; using {release} / {sae_id}", flush=True)

    # SAE.from_pretrained returns (sae, cfg_dict, sparsity) in current sae_lens
    out = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    if isinstance(out, tuple):
        sae = out[0]
    else:
        sae = out
    sae.eval()
    print(
        f"[saebench] Loaded SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, "
        f"hook_name={sae.cfg.metadata.hook_name}",
        flush=True,
    )
    if sae.cfg.metadata.hook_name != HOOK_NAME:
        print(
            f"[saebench] WARNING: SAE hook_name {sae.cfg.metadata.hook_name} != project HOOK_NAME {HOOK_NAME}. "
            f"Continuing with project HOOK_NAME for activation extraction.",
            flush=True,
        )
    return sae, release, sae_id


def score_sae(model, sae, device: str, batch_size: int, n_controls: int, seed: int):
    """Run probes + controls through Gemma+SAE and return induction scores per feature."""
    df_probes = pd.read_parquet(INDUCTION_PROBES_PATH)
    print(f"[saebench] Loaded {len(df_probes)} induction probe sequences", flush=True)
    induction_seqs = [list(row) for row in df_probes["tokens"].tolist()]

    vocab_lo, vocab_hi = _safe_vocab_range(model.cfg.d_vocab)

    print(f"[saebench] Generating {n_controls} control sequences ...", flush=True)
    rng_ctrl = random.Random(seed)
    control_seqs = []
    for i in range(n_controls):
        total_len = len(induction_seqs[i % len(induction_seqs)])
        toks = [rng_ctrl.randint(vocab_lo, vocab_hi) for _ in range(total_len)]
        control_seqs.append(toks)

    print("[saebench] Computing feature activations for induction probes ...", flush=True)
    t0 = time.monotonic()
    induction_acts = _get_final_pos_features(
        model, sae, HOOK_NAME, induction_seqs, device, batch_size
    )

    print("[saebench] Computing feature activations for control sequences ...", flush=True)
    control_acts = _get_final_pos_features(
        model, sae, HOOK_NAME, control_seqs, device, batch_size
    )
    print(f"[saebench] Activations computed in {time.monotonic() - t0:.0f}s", flush=True)

    induction_mean = induction_acts.mean(axis=0)
    control_mean = control_acts.mean(axis=0)
    induction_score = induction_mean - control_mean
    return induction_mean, control_mean, induction_score


def try_fetch_hf_labels(release: str, sae_id: str):
    """Best-effort fetch of any feature labels / neuronpedia metadata for the SAE."""
    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
        directory = get_pretrained_saes_directory()
        info = directory.get(release)
        if info is None:
            return None
        # neuronpedia_id often present on Gemma-Scope releases
        npid = None
        if hasattr(info, "neuronpedia_id") and isinstance(info.neuronpedia_id, dict):
            npid = info.neuronpedia_id.get(sae_id)
        return {
            "release": release,
            "sae_id": sae_id,
            "repo_id": getattr(info, "repo_id", None),
            "model": getattr(info, "model", None),
            "neuronpedia_id": npid,
        }
    except Exception as e:
        print(f"[saebench] Could not fetch HF metadata: {e}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare v9c SAE vs public Gemma-Scope SAE")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--n-controls", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    print("[saebench] Loading Gemma-2-2B ...", flush=True)
    model = load_model(device=args.device)

    print("[saebench] Loading public Gemma-Scope SAE ...", flush=True)
    sae, release, sae_id = load_public_sae(args.device)
    n_features = sae.cfg.d_sae

    induction_mean, control_mean, induction_score = score_sae(
        model, sae, args.device, args.batch_size, args.n_controls, args.seed
    )

    ranked_ids = np.argsort(-induction_score)

    scores_df = pd.DataFrame({
        "feature_id": np.arange(n_features, dtype=np.int32),
        "induction_mean": induction_mean.astype(np.float32),
        "control_mean": control_mean.astype(np.float32),
        "induction_score": induction_score.astype(np.float32),
        "rank": np.argsort(np.argsort(-induction_score)).astype(np.int32),
    })
    SAEBENCH_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(SAEBENCH_SCORES_PATH, index=False)
    print(f"[saebench] Scores saved to {SAEBENCH_SCORES_PATH}", flush=True)

    top_ids = [int(x) for x in ranked_ids[:args.top_n].tolist()]
    with SAEBENCH_CANDIDATE_IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(top_ids, f, indent=2)
    print(f"[saebench] Top-{args.top_n} candidates saved to {SAEBENCH_CANDIDATE_IDS_PATH}", flush=True)

    # ── Comparison vs v9c ──────────────────────────────────────────────────────
    print("\n[saebench] === Comparison: v9c vs public Gemma-Scope SAE ===", flush=True)

    if V9C_CANDIDATE_IDS_PATH.exists():
        with V9C_CANDIDATE_IDS_PATH.open("r", encoding="utf-8") as f:
            v9c_top100 = json.load(f)
    else:
        v9c_top100 = []
        print(f"[saebench] WARNING: {V9C_CANDIDATE_IDS_PATH} not found.", flush=True)

    v9c_top20 = v9c_top100[:20]
    saebench_top20 = top_ids[:20]

    # Per-feature score view for the SAEBench top-20
    print("\n[saebench] Public SAE top-20 induction features:", flush=True)
    print(f"{'Rank':>5} {'FeatID':>8} {'Induction':>10} {'Control':>10} {'Score':>10}")
    for rank, fid in enumerate(saebench_top20):
        print(
            f"{rank:>5} {fid:>8} {induction_mean[fid]:>10.4f} "
            f"{control_mean[fid]:>10.4f} {induction_score[fid]:>10.4f}"
        )

    print("\n[saebench] v9c top-20 feature IDs:     ", v9c_top20, flush=True)
    print("[saebench] SAEBench top-20 feature IDs:", saebench_top20, flush=True)

    # Overlap is informational only — feature IDs are NOT comparable across SAEs
    # (different SAEs learn different feature bases). Reported for completeness.
    overlap = sorted(set(v9c_top20) & set(saebench_top20))
    print(
        f"\n[saebench] Top-20 ID overlap (note: feature IDs are not aligned across SAEs): "
        f"{len(overlap)} -> {overlap}",
        flush=True,
    )
    overlap100 = sorted(set(v9c_top100) & set(top_ids))
    print(f"[saebench] Top-100 ID overlap: {len(overlap100)}", flush=True)

    # Compare strength of the top induction signal across SAEs
    if V9C_SCORES_PATH.exists():
        v9c_scores = pd.read_parquet(V9C_SCORES_PATH)
        v9c_top_score = v9c_scores.sort_values("induction_score", ascending=False)["induction_score"].iloc[0]
        sae_top_score = float(induction_score[ranked_ids[0]])
        print(
            f"\n[saebench] Top-feature induction_score:  v9c={v9c_top_score:.4f}  "
            f"SAEBench={sae_top_score:.4f}",
            flush=True,
        )
        v9c_top20_mean = v9c_scores.sort_values("induction_score", ascending=False)["induction_score"].iloc[:20].mean()
        sae_top20_mean = float(induction_score[ranked_ids[:20]].mean())
        print(
            f"[saebench] Top-20 mean induction_score:   v9c={v9c_top20_mean:.4f}  "
            f"SAEBench={sae_top20_mean:.4f}",
            flush=True,
        )

    # HF metadata (neuronpedia link is the closest thing to "prior labels")
    meta = try_fetch_hf_labels(release, sae_id)
    if meta is not None:
        print("\n[saebench] Public SAE metadata (from sae_lens directory):", flush=True)
        for k, v in meta.items():
            print(f"  {k}: {v}")
        if meta.get("neuronpedia_id"):
            print(
                f"  -> Neuronpedia base URL: "
                f"https://neuronpedia.org/{meta['neuronpedia_id']}/<feature_id>",
                flush=True,
            )
            print("  Top-20 SAEBench feature Neuronpedia URLs:", flush=True)
            for fid in saebench_top20:
                print(f"    f{fid}: https://neuronpedia.org/{meta['neuronpedia_id']}/{fid}")


if __name__ == "__main__":
    main()
