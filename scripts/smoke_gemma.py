"""Gemma-2-2B load smoke test."""

import sys

import torch
from transformer_lens import HookedTransformer


def main() -> int:
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", dtype=torch.bfloat16)
    model.to("cuda")
    out = model.generate("The capital of France is", max_new_tokens=5)
    print(f"OUTPUT: {out!r}")

    free, total = torch.cuda.mem_get_info(0)
    used_gb = (total - free) / 1e9
    print(f"VRAM used after load: {used_gb:.2f} GB")
    if used_gb > 8.0:
        print(
            f"WARN: VRAM usage {used_gb:.2f} GB is higher than ~5 GB expected for bf16 Gemma-2-2B",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
