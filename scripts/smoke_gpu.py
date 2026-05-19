"""GPU smoke test — verify Blackwell SM_120 kernels work with the installed torch."""

import sys

import torch


def main() -> int:
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available", file=sys.stderr)
        return 1

    print(f"CUDA version (torch built against): {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"Device 0: {name} (SM_{cap[0]}{cap[1]})")

    # Real kernel test — pure metadata access can pass even when SM kernels are missing.
    try:
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        c = a @ b
        torch.cuda.synchronize()
        print(f"bf16 matmul OK, result norm: {c.float().norm().item():.4f}")
    except Exception as e:
        print(f"FAIL: bf16 matmul: {e}", file=sys.stderr)
        return 2

    free, total = torch.cuda.mem_get_info(0)
    print(f"VRAM free / total: {free / 1e9:.2f} GB / {total / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
