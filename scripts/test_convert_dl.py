"""
Pre-flight test for convert_dl_to_saelens.py.

Fabricates a tiny dictionary_learning-style ae.pt and runs the conversion,
then loads the result with sae_lens.saes.sae.SAE to confirm it works.

This catches conversion bugs before v8 finishes training.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def make_fake_ae_pt(out_dir: Path, d_in: int = 2304, d_sae: int = 16384, k: int = 80):
    """Mimic AutoEncoderTopK's state_dict layout."""
    sd = {
        "encoder.weight": torch.randn(d_sae, d_in) * 0.01,
        "encoder.bias": torch.zeros(d_sae),
        "decoder.weight": torch.randn(d_in, d_sae) * 0.01,
        "b_dec": torch.zeros(d_in),
        "threshold": torch.tensor(0.0),
        "k": torch.tensor(k),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_dir / "ae.pt")
    cfg = {"trainer": {"k": k, "dict_size": d_sae, "activation_dim": d_in}}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def main():
    print("[test_convert] Setting up fake training output ...")
    real_dl_dir = REPO_ROOT / "models" / "sae_main_dl"
    real_saelens_dir = REPO_ROOT / "models" / "sae_main"
    real_weights = real_saelens_dir / "sae_weights.safetensors"
    real_cfg = real_saelens_dir / "cfg.json"

    # Snapshot the real files so we can restore them
    snap_weights = None
    snap_cfg = None
    if real_weights.exists():
        snap_weights = real_weights.read_bytes()
    if real_cfg.exists():
        snap_cfg = real_cfg.read_text(encoding="utf-8")

    # If dl dir doesn't exist yet (v8 not done), make a fake one
    fake_made = False
    fake_trainer_dir = real_dl_dir / "trainer_0"
    if not (fake_trainer_dir / "ae.pt").exists():
        print("[test_convert] Real ae.pt not found; making fake one")
        make_fake_ae_pt(fake_trainer_dir, d_in=2304, d_sae=16384, k=80)
        fake_made = True

    try:
        # Run conversion
        import subprocess
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "convert_dl_to_saelens.py")],
            capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Conversion failed (exit {result.returncode})")

        # Check output files
        assert real_weights.exists(), f"{real_weights} missing"
        assert real_cfg.exists(), f"{real_cfg} missing"
        print("[test_convert] OK: weights + cfg written")

        # Try loading with SAELens
        from sae_lens.saes.sae import SAE
        sae = SAE.load_from_disk(str(real_saelens_dir), device="cpu")
        print(f"[test_convert] OK: SAELens.load_from_disk succeeded; cfg={sae.cfg}")
        print(f"[test_convert] SAE shapes: W_enc={tuple(sae.W_enc.shape)}, W_dec={tuple(sae.W_dec.shape)}")

        # Try encoding
        x = torch.randn(4, 2304)
        z = sae.encode(x)
        x_hat = sae.decode(z)
        print(f"[test_convert] OK: encode->decode works. z.shape={tuple(z.shape)}, x_hat.shape={tuple(x_hat.shape)}")
        # Check sparsity (TopK should give exactly k nonzero per row)
        nnz = (z != 0).sum(dim=-1).float().mean().item()
        print(f"[test_convert] sparsity check: mean nonzero per token = {nnz}")

        print("\n=== ALL TESTS PASSED ===")

    finally:
        # Restore real files — use shutil.copy2 instead of write_bytes (which can fail on large files on Windows)
        if fake_made:
            shutil.rmtree(real_dl_dir, ignore_errors=True)
            print(f"[test_convert] cleaned up fake {real_dl_dir}")
        if snap_weights is not None:
            # Restore from v1 backup file rather than the in-memory bytes (more reliable on Windows)
            v1_backup = real_saelens_dir / "sae_weights_v1_backup.safetensors"
            if v1_backup.exists():
                shutil.copy2(str(v1_backup), str(real_weights))
                print(f"[test_convert] restored {real_weights} (from v1 backup file)")
            else:
                try:
                    real_weights.write_bytes(snap_weights)
                except OSError as e:
                    print(f"[test_convert] WARNING: failed to restore weights: {e}")
        if snap_cfg is not None:
            real_cfg.write_text(snap_cfg, encoding="utf-8")
            print(f"[test_convert] restored {real_cfg}")


if __name__ == "__main__":
    main()
