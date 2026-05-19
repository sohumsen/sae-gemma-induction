"""
Diagnostic: test induction accuracy under different probe configurations.
Identifies best (gap, vocab_range) before finalising smoke test threshold.
"""
import sys
import time
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            import os as _os
            _os.environ.setdefault(_k.strip(), _v.strip())

import torch
from sae_gemma.model_utils import load_model

print(f"[{time.strftime('%H:%M:%S')}] Loading model ...", flush=True)
model = load_model()
vocab_size = model.cfg.d_vocab
print(f"[{time.strftime('%H:%M:%S')}] Model ready. d_vocab={vocab_size}", flush=True)

N_PROBES = 100
PREFIX_LEN = 10  # fixed short prefix

CONFIGS = [
    ("gap=1  vocab=1k-20k",  1,  range(1000, 20000)),
    ("gap=5  vocab=1k-20k",  5,  range(1000, 20000)),
    ("gap=10 vocab=1k-20k", 10,  range(1000, 20000)),
    ("gap=30 vocab=1k-20k", 30,  range(1000, 20000)),
    ("gap=1  vocab=full",    1,  range(10, vocab_size)),
    ("gap=5  vocab=full",    5,  range(10, vocab_size)),
    ("gap=10 vocab=full",   10,  range(10, vocab_size)),
    ("gap=30 vocab=full",   30,  range(10, vocab_size)),
]

for label, gap, vocab_iter in CONFIGS:
    safe_vocab = list(vocab_iter)
    rng = random.Random(42)
    n_correct = 0
    for _ in range(N_PROBES):
        A = rng.choice(safe_vocab)
        B = rng.choice(safe_vocab)
        while B == A:
            B = rng.choice(safe_vocab)
        prefix = [rng.choice(safe_vocab) for _ in range(PREFIX_LEN)]
        middle = [rng.choice(safe_vocab) for _ in range(gap)]
        tokens = prefix + [A, B] + middle + [A]
        t = torch.tensor([tokens], dtype=torch.long, device="cuda")
        with torch.no_grad():
            logits = model(t)
        pred = logits[0, -1, :].argmax().item()
        if pred == B:
            n_correct += 1
    acc = n_correct / N_PROBES
    marker = " <-- PASS" if acc >= 0.50 else ""
    print(f"[{time.strftime('%H:%M:%S')}] {label:25s}  {n_correct}/{N_PROBES} = {acc:.0%}{marker}", flush=True)

print(f"\n[{time.strftime('%H:%M:%S')}] Diagnostic complete.", flush=True)
