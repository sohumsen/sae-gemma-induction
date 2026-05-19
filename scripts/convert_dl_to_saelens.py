"""
Convert dictionary_learning AutoEncoderTopK weights -> SAELens-loadable format.

Reads:  models/sae_main_dl/trainer_0/ae.pt          (and config.json)
Writes: models/sae_main/sae_weights.safetensors     (SAELens W_enc/W_dec/b_enc/b_dec layout)
        models/sae_main/cfg.json                    (SAELens TopK config)

Run after v8 training completes. v1 weights are already backed up at
models/sae_main/sae_weights_v1_backup.safetensors.

Usage:  python scripts/convert_dl_to_saelens.py
"""
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DL_DIR = REPO_ROOT / "models" / "sae_main_dl" / "trainer_0"
SAELENS_DIR = REPO_ROOT / "models" / "sae_main"
AE_PT = DL_DIR / "ae.pt"
DL_CFG = DL_DIR / "config.json"
OUT_WEIGHTS = SAELENS_DIR / "sae_weights.safetensors"
OUT_CFG = SAELENS_DIR / "cfg.json"


def main():
    if not AE_PT.exists():
        raise FileNotFoundError(f"{AE_PT} not found; v8 may still be training.")

    sd = torch.load(AE_PT, map_location="cpu")
    print(f"[convert] Loaded ae.pt with keys: {list(sd.keys())}")

    # AutoEncoderTopK uses Linear: encoder.weight [d_sae, d_in]; decoder.weight [d_in, d_sae]
    # SAELens convention: W_enc [d_in, d_sae]; W_dec [d_sae, d_in]
    W_enc = sd["encoder.weight"].T.contiguous().float()
    W_dec = sd["decoder.weight"].T.contiguous().float()
    b_enc = sd["encoder.bias"].float()
    b_dec = sd["b_dec"].float()
    threshold = sd.get("threshold", None)

    d_in, d_sae = W_enc.shape
    print(f"[convert] d_in={d_in}, d_sae={d_sae}")

    out = {"W_enc": W_enc, "W_dec": W_dec, "b_enc": b_enc, "b_dec": b_dec}
    # NOTE: dictionary_learning's `threshold` (EMA min-active-preact) is intentionally dropped —
    # SAELens TopKSAE doesn't accept it and selects top-k purely by encoder output magnitude.
    # The threshold value is preserved as metadata in the cfg below for reference only.
    if threshold is not None:
        threshold_val = float(threshold.item()) if hasattr(threshold, "item") else float(threshold)
    else:
        threshold_val = None

    SAELENS_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_WEIGHTS))
    print(f"[convert] saved weights -> {OUT_WEIGHTS}")

    # Build SAELens-compatible cfg.json
    dl_cfg = {}
    if DL_CFG.exists():
        with DL_CFG.open(encoding="utf-8") as f:
            dl_cfg = json.load(f)
        print(f"[convert] dl config: {dl_cfg}")

    k = dl_cfg.get("trainer", {}).get("k", 80)

    cfg = {
        "apply_b_dec_to_input": True,
        "metadata": {
            "sae_lens_version": "6.43.0",   # must be PEP 440 — SAELens loader rejects suffixed versions
            "sae_lens_training_version": "6.43.0+dictionary_learning",
            "dataset_path": "monology/pile-uncopyrighted",
            "hook_name": "blocks.12.hook_resid_post",
            "model_name": "google/gemma-2-2b",
            "model_class_name": "HookedTransformer",
            "hook_head_index": None,
            "context_size": 1024,
            "seqpos_slice": [None],
            "model_from_pretrained_kwargs": {"dtype": "bfloat16"},
            "prepend_bos": True,
            "exclude_special_tokens": True,           # dl removes BOS in buffer
            "sequence_separator_token": "bos",
            "disable_concat_sequences": False,
        },
        "dtype": "float32",
        "device": "cuda",
        "d_in": d_in,
        "normalize_activations": "expected_average_only_in",
        "k": int(k),
        "rescale_acts_by_decoder_norm": False,
        "d_sae": d_sae,
        "reshape_activations": "none",
        "architecture": "topk",
        "_dl_threshold": threshold_val,   # preserved for reference, not used by SAELens loader
    }
    with OUT_CFG.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[convert] wrote cfg -> {OUT_CFG}")
    print(f"[convert] done.")


if __name__ == "__main__":
    main()
