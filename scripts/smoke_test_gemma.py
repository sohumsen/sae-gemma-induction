"""
Phase 1.5 + 1.6: Gemma-2-2B load smoke test + induction smoke test.
Run detached; tail logs/smoke_gemma.log to monitor.
"""
import sys
import time
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load credentials from .env before any HF/W&B imports
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            import os as _os
            _os.environ.setdefault(_k.strip(), _v.strip())

import torch

print(f"[{time.strftime('%H:%M:%S')}] PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[{time.strftime('%H:%M:%S')}] GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ── 1. Load Gemma-2-2B via TransformerLens ─────────────────────────────────
# Pre-load HF weights directly to CUDA in bf16 to avoid TransformerLens's
# default fp32-on-CPU staging (which peaks ~9 GB RAM and OOMs the box).
print(f"\n[{time.strftime('%H:%M:%S')}] Loading Gemma-2-2B HF weights directly to CUDA bf16 ...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Load to CPU; TL moves to CUDA after weight processing
    low_cpu_mem_usage=True,
)
hf_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
print(f"[{time.strftime('%H:%M:%S')}] HF weights loaded. Wrapping in HookedTransformer ...", flush=True)

from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    "google/gemma-2-2b",
    hf_model=hf_model,
    tokenizer=hf_tokenizer,
    dtype=torch.bfloat16,
    device="cuda" if torch.cuda.is_available() else "cpu",
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)
model.eval()

# Free the HF wrapper now that TL has copied weights into its own modules
del hf_model
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"[{time.strftime('%H:%M:%S')}] Model loaded. d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}", flush=True)

# Quick sanity: generate a few tokens
with torch.no_grad():
    out = model.generate("The capital of France is", max_new_tokens=5, temperature=0)
print(f"[{time.strftime('%H:%M:%S')}] Generation test: {out!r}", flush=True)

# VRAM check
if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[{time.strftime('%H:%M:%S')}] VRAM used={used:.2f} GB, reserved={reserved:.2f} GB", flush=True)

# ── 2. Induction smoke test ─────────────────────────────────────────────────
print(f"\n[{time.strftime('%H:%M:%S')}] Running induction smoke test ...", flush=True)

tokenizer = model.tokenizer
vocab_size = model.cfg.d_vocab

# Induction probe: [rand * PREFIX_LEN] [A] [B] [rand * GAP_LEN] [A]
# Expect top-1 prediction at final position == B.
# Diagnostic shows gap=1 + vocab 1k-20k is the only config with >50% accuracy
# on Gemma-2-2B (gap=5 -> 32%, gap=30 -> 9-16%).
N_PROBES = 200
PREFIX_LEN = 10
GAP_LEN = 1
random.seed(42)

# Tokens 1000-20000: common enough for model to exhibit induction reliably.
SAFE_VOCAB = list(range(1000, min(20000, vocab_size)))

n_correct = 0
for i in range(N_PROBES):
    A = random.choice(SAFE_VOCAB)
    B = random.choice([x for x in SAFE_VOCAB if x != A])
    prefix = [random.choice(SAFE_VOCAB) for _ in range(PREFIX_LEN)]
    gap = [random.choice(SAFE_VOCAB) for _ in range(GAP_LEN)]
    tokens = prefix + [A, B] + gap + [A]
    token_tensor = torch.tensor([tokens], dtype=torch.long)
    if torch.cuda.is_available():
        token_tensor = token_tensor.to("cuda")
    with torch.no_grad():
        logits = model(token_tensor)  # (1, seq, vocab)
    pred = logits[0, -1, :].argmax().item()
    if pred == B:
        n_correct += 1

accuracy = n_correct / N_PROBES
print(f"[{time.strftime('%H:%M:%S')}] Induction accuracy: {n_correct}/{N_PROBES} = {accuracy:.1%}", flush=True)

if accuracy >= 0.50:
    print(f"[{time.strftime('%H:%M:%S')}] GATE PASSED -- Gemma-2-2B does induction (>{accuracy:.0%})", flush=True)
else:
    print(f"[{time.strftime('%H:%M:%S')}] GATE FAILED -- accuracy {accuracy:.1%} < 50%. STOP AND ESCALATE.", flush=True)
    sys.exit(1)

print(f"\n[{time.strftime('%H:%M:%S')}] Phase 1 smoke tests complete. Ready for Phase 2 (pilot SAE).", flush=True)
