"""
Shared model loading utilities.

TransformerLens's default from_pretrained loads HF weights in fp32 on CPU first
(~9 GB RAM peak for Gemma-2-2B), which OOMs machines with <16 GB free RAM.
The RAM-efficient path: load HF model directly to CUDA in bf16, then wrap.
"""

import gc
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from sae_gemma.paths import MODEL_NAME


def load_model(device: str = "cuda", dtype: str = "bfloat16") -> HookedTransformer:
    """
    Load google/gemma-2-2b into a HookedTransformer with minimal RAM usage.

    Uses low_cpu_mem_usage=True + direct CUDA placement so the fp32 staging
    buffer never materialises in RAM.
    """
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    target_device = device if torch.cuda.is_available() else "cpu"

    # Load HF weights to CPU first so TL weight-processing (which can create
    # float32 intermediates) doesn't compete with the final CUDA model for VRAM.
    print(f"[model_utils] Loading {MODEL_NAME} HF weights -> CPU ({dtype}) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"[model_utils] Wrapping in HookedTransformer (fold_ln=False, CPU processing) ...", flush=True)
    # Keep model on CPU during TL weight processing — TL creates small temp tensors
    # that fragment CUDA memory if device="cuda" is passed here directly.
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        hf_model=hf_model,
        tokenizer=tokenizer,
        dtype=torch_dtype,
        device="cpu",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )

    # Free HF wrapper and clear any CUDA fragments before the big model transfer
    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if target_device != "cpu":
        model = model.to(target_device)
    model.eval()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        print(f"[model_utils] Model loaded. VRAM used: {used:.2f} GB", flush=True)

    return model
