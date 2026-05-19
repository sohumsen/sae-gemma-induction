"""
For a given seed (43 or 44), run capture_activations + autointerp on its top-5
induction features. Lets the writeup's multi-seed section claim qualitative
(not just quantitative) replication.

    python scripts/autointerp_seed.py --seed 43

Steps:
1. Convert models/sae_main_dl_seed{N}/trainer_0/ae.pt -> models/sae_seed{N}/ (SAELens format)
2. Run src/sae_gemma/capture_activations.py with --sae-path models/sae_seed{N}/
   and --output results/seed{N}_top_snippets.parquet (250k tokens, faster than 1M)
3. Build results/seed{N}_top5.json from results/seed{N}_replication.json
4. Run src/sae_gemma/autointerp.py with seed-specific snippets/features/cache
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def convert_dl_to_saelens(seed: int):
    """Inline conversion of seed's ae.pt -> SAELens format in models/sae_seed{N}/."""
    import torch
    from safetensors.torch import save_file

    dl_dir = REPO_ROOT / "models" / f"sae_main_dl_seed{seed}" / "trainer_0"
    out_dir = REPO_ROOT / "models" / f"sae_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = torch.load(dl_dir / "ae.pt", map_location="cpu", weights_only=False)
    dl_cfg = json.loads((dl_dir / "config.json").read_text(encoding="utf-8"))
    k = dl_cfg.get("trainer", {}).get("k", 100)
    d_sae, d_in = sd["encoder.weight"].shape

    out = {
        "W_enc": sd["encoder.weight"].T.contiguous().float(),
        "W_dec": sd["decoder.weight"].T.contiguous().float(),
        "b_enc": sd["encoder.bias"].float(),
        "b_dec": sd["b_dec"].float(),
    }
    save_file(out, str(out_dir / "sae_weights.safetensors"))

    cfg = {
        "apply_b_dec_to_input": True,
        "metadata": {
            "sae_lens_version": "6.43.0",
            "sae_lens_training_version": "6.43.0+dictionary_learning",
            "dataset_path": "local-pile-cache",
            "hook_name": "blocks.12.hook_resid_post",
            "model_name": "google/gemma-2-2b",
            "model_class_name": "HookedTransformer",
            "hook_head_index": None,
            "context_size": 1024,
            "seqpos_slice": [None],
            "model_from_pretrained_kwargs": {"dtype": "bfloat16"},
            "prepend_bos": True,
            "exclude_special_tokens": True,
            "sequence_separator_token": "bos",
            "disable_concat_sequences": False,
        },
        "dtype": "float32",
        "device": "cuda",
        "d_in": int(d_in),
        "normalize_activations": "expected_average_only_in",
        "k": int(k),
        "rescale_acts_by_decoder_norm": False,
        "d_sae": int(d_sae),
        "reshape_activations": "none",
        "architecture": "topk",
    }
    (out_dir / "cfg.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[seed{seed}] converted -> {out_dir}", flush=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--top-n", type=int, default=5, help="How many top induction features to label")
    parser.add_argument("--n-tokens", type=int, default=250_000)
    args = parser.parse_args()

    # 1. Convert
    sae_dir = convert_dl_to_saelens(args.seed)

    # 2. Capture activations
    snippets_path = REPO_ROOT / "results" / f"seed{args.seed}_top_snippets.parquet"
    print(f"[seed{args.seed}] capture_activations on {args.n_tokens:,} tokens ...", flush=True)
    r = subprocess.run(
        [sys.executable, "src/sae_gemma/capture_activations.py",
         "--sae-path", str(sae_dir),
         "--output", str(snippets_path),
         "--n-tokens", str(args.n_tokens),
         "--top-k", "20"],
        cwd=str(REPO_ROOT),
    )
    if r.returncode != 0:
        sys.exit(f"capture_activations failed (exit {r.returncode})")

    # 3. Build top-N ids file from replication.json
    rep = json.loads((REPO_ROOT / "results" / f"seed{args.seed}_replication.json").read_text(encoding="utf-8"))
    top_ids = rep["top20_ids"][: args.top_n]
    top_path = REPO_ROOT / "results" / f"seed{args.seed}_top{args.top_n}.json"
    top_path.write_text(json.dumps(top_ids), encoding="utf-8")
    print(f"[seed{args.seed}] top-{args.top_n} ids: {top_ids}", flush=True)

    # 4. Autointerp with seed-specific paths; ensure claude CLI is on PATH
    env = os.environ.copy()
    env["PATH"] = r"C:\Users\sohum\.local\bin;" + env.get("PATH", "")
    labels_path = REPO_ROOT / "results" / f"seed{args.seed}_labels.json"
    labels_path.write_text("{}", encoding="utf-8")
    print(f"[seed{args.seed}] autointerp Sonnet on top-{args.top_n} ...", flush=True)
    r = subprocess.run(
        [sys.executable, "src/sae_gemma/autointerp.py",
         "--snippets", str(snippets_path),
         "--cache", str(labels_path),
         "--features", str(top_path),
         "--model", "claude-sonnet-4-5",
         "--workers", "2",
         "--timeout", "90"],
        cwd=str(REPO_ROOT), env=env,
    )
    if r.returncode != 0:
        sys.exit(f"autointerp failed (exit {r.returncode})")

    # Print summary
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    print(f"\n=== seed {args.seed} top-{args.top_n} labels ===", flush=True)
    for fid in top_ids:
        label = labels.get(str(fid), "(missing)").split("\n")[0][:160]
        print(f"  F{fid}: {label}", flush=True)


if __name__ == "__main__":
    main()
