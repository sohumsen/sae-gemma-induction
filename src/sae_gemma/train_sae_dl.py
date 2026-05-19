"""
v8: Train TopK SAE using saprmarks/dictionary_learning (the library SAEBench used
to achieve cossim ~0.9, EV ~0.85 on the same Gemma-2-2B residual stream task).

Why switch from SAELens?  Empirically, SAELens TopK on Gemma-2-2B layer 12 residual
plateaus at cossim ~0.5, l2_ratio ~0.008 across every config we tried (v1-v7:
varying k, lr, normalize_activations, rescale_acts_by_decoder_norm, apply_b_dec).
SAEBench's published 16384 / 65536 width TopK SAEs use `dictionary_learning`
with `AutoEncoderTopK` and a different aux-loss formulation + EMA threshold.

Output: weights saved to models/sae_main_dl/trainer_0/ae.pt, then converted to
SAELens-compatible safetensors at models/sae_main/sae_weights.safetensors so
downstream scripts (find_induction_features.py, capture_activations.py,
head_correspondence.py, ablations.py) work without modification.
"""
import json
import os
from pathlib import Path

import torch
from dictionary_learning import ActivationBuffer
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.training import trainSAE
from nnsight import LanguageModel
from safetensors.torch import save_file

from sae_gemma.paths import REPO_ROOT, SAE_MAIN_DIR

PILE_CACHE = REPO_ROOT / "data" / "pile_cache.jsonl"

# Load .env (W&B API key, HF token)
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MODEL_NAME = "google/gemma-2-2b"
LAYER = 12
D_IN = 2304
D_SAE = 16_384   # v9b: reverted from 32k (made buffer refills 25× slower in tight VRAM, ETA 240+ hours)
K = 100          # kept: matches Gemma Scope canonical L0 for layer 12 (v8 was k=80; bumping should push EV from 0.87 → 0.90+)
NUM_TOKENS = 200_000_000
CTX_LEN = 1024
SAE_BS = 2048
LLM_BS = 4
N_CTXS = 244     # reverted to v8's value — was the safe-fast configuration

STEPS = NUM_TOKENS // SAE_BS    # ≈ 97,656 steps

DL_OUTPUT_DIR = REPO_ROOT / "models" / "sae_main_dl"
RUN_NAME = "topk_l12_dl_v9c"

# v9 checkpoints: save at 25/50/75/100% so a crash near the end isn't catastrophic.
# Intermediate saves go to trainer_0/checkpoints/ae_<step>.pt; final goes to trainer_0/ae.pt.
SAVE_STEPS = [int(STEPS * 0.25), int(STEPS * 0.50), int(STEPS * 0.75), STEPS]


def convert_to_safetensors(dl_save_dir: Path):
    """Convert dictionary_learning's ae.pt → SAELens-compatible sae_weights.safetensors."""
    ae_path = dl_save_dir / "trainer_0" / "ae.pt"
    if not ae_path.exists():
        print(f"[convert] ae.pt not found at {ae_path}", flush=True)
        return None

    sd = torch.load(ae_path, map_location="cpu")
    # AutoEncoderTopK uses Linear modules; encoder.weight shape = [d_sae, d_in]
    # SAELens expects W_enc [d_in, d_sae] and W_dec [d_sae, d_in]
    out = {
        "W_enc": sd["encoder.weight"].T.contiguous().float(),
        "b_enc": sd["encoder.bias"].float(),
        "W_dec": sd["decoder.weight"].T.contiguous().float(),
        "b_dec": sd["b_dec"].float(),
    }
    if "threshold" in sd:
        out["threshold"] = sd["threshold"].float()

    out_path = SAE_MAIN_DIR / "sae_weights_v8_dl.safetensors"
    save_file(out, str(out_path))
    print(f"[convert] saved -> {out_path}", flush=True)
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for SAE init. If set, also tags run name and output dir.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override RUN_NAME (e.g. for multi-seed replication runs).")
    parser.add_argument("--output-subdir", type=str, default=None,
                        help="Override output sub-dir (default: models/sae_main_dl). For seed replication, use e.g. models/sae_main_dl_seed43.")
    args = parser.parse_args()

    global RUN_NAME
    if args.run_name:
        RUN_NAME = args.run_name

    output_dir = DL_OUTPUT_DIR
    if args.output_subdir:
        output_dir = REPO_ROOT / args.output_subdir
    if args.seed is not None:
        import torch as _t
        import random as _r
        import numpy as _np
        _t.manual_seed(args.seed)
        _t.cuda.manual_seed_all(args.seed)
        _np.random.seed(args.seed)
        _r.seed(args.seed)
        print(f"[train_sae_dl] Seeded torch/numpy/python with seed={args.seed}", flush=True)

    print(f"[train_sae_dl] Loading {MODEL_NAME} via nnsight LanguageModel (bf16) ...", flush=True)
    model = LanguageModel(MODEL_NAME, device_map=DEVICE, dtype=DTYPE, dispatch=True)
    submodule = model.model.layers[LAYER]  # output = resid_post for that layer

    print(f"[train_sae_dl] Reading cached pile examples from {PILE_CACHE} ...", flush=True)
    if not PILE_CACHE.exists():
        raise FileNotFoundError(
            f"{PILE_CACHE} not found. Run `python scripts/cache_pile_examples.py` first. "
            "v9c trains from a local cache to avoid the HF streaming-client errors that killed v9b."
        )

    def cached_text_iterator():
        # Loop the cache file if training calls for more text than we cached.
        # 300k examples × ~600 tok ≈ 180M tok; loop gives safety margin past 200M tok target.
        while True:
            with PILE_CACHE.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    text = json.loads(line).get("text", "")
                    if text:
                        yield text
    data_gen = cached_text_iterator()

    buffer = ActivationBuffer(
        data=data_gen,
        model=model,
        submodule=submodule,
        d_submodule=D_IN,
        n_ctxs=N_CTXS,
        ctx_len=CTX_LEN,
        refresh_batch_size=LLM_BS,
        out_batch_size=SAE_BS,
        io="out",
        device=DEVICE,
        remove_bos=True,
    )

    trainer_cfg = {
        "trainer": TopKTrainer,
        "dict_class": AutoEncoderTopK,
        "activation_dim": D_IN,
        "dict_size": D_SAE,
        "k": K,
        "lr": 5e-5,
        "auxk_alpha": 1.0 / 32.0,
        "warmup_steps": 1000,
        "decay_start": int(STEPS * 0.8),
        "threshold_beta": 0.999,
        "threshold_start_step": 1000,
        "steps": STEPS,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "submodule_name": f"resid_post_layer_{LAYER}",
        "device": DEVICE,
        "wandb_name": RUN_NAME,
    }

    # Pre-create trainer_0 directory (dictionary_learning expects it during final save —
    # v8 crashed because this directory was deleted mid-training by a test script).
    (output_dir / "trainer_0").mkdir(parents=True, exist_ok=True)
    (output_dir / "trainer_0" / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[train_sae_dl] Starting training: {STEPS:,} steps, k={K}, d_sae={D_SAE}", flush=True)

    trainSAE(
        data=buffer,
        trainer_configs=[trainer_cfg],
        steps=STEPS,
        save_dir=str(output_dir),
        save_steps=SAVE_STEPS,               # v9: 25/50/75/100% checkpoints
        log_steps=100,                       # required (None default crashes)
        autocast_dtype=DTYPE,
        use_wandb=True,
        wandb_entity=os.environ.get("WANDB_ENTITY") or "",
        wandb_project=os.environ.get("WANDB_PROJECT", "sae-gemma-induction"),
        normalize_activations=True,          # constant-norm rescale (SAEBench convention)
    )

    print(f"[train_sae_dl] Training complete. Converting weights ...", flush=True)
    convert_to_safetensors(output_dir)
    print(f"[train_sae_dl] Done.", flush=True)


if __name__ == "__main__":
    main()
