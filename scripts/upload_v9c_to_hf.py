"""
Upload v9c SAE weights to Hugging Face Hub for reproducibility.

Creates (or updates) repo  sohumsen/sae-gemma2-2b-layer12-v9c  containing:
    - sae_weights.safetensors  (302 MB, SAELens-format)
    - cfg.json                  (SAE config: k=100, d_sae=16384, hook, etc.)
    - README.md                 (model card with EV, training, citation)

Run once. Requires HF_TOKEN in .env.

    python scripts/upload_v9c_to_hf.py
"""
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
SAE_DIR = REPO_ROOT / "models" / "sae_main"
REPO_ID = "senator1/sae-gemma2-2b-layer12-v9c"

MODEL_CARD = """---
license: mit
tags:
  - sparse-autoencoder
  - sae
  - mechanistic-interpretability
  - gemma-2-2b
  - induction-heads
base_model: google/gemma-2-2b
library_name: sae_lens
---

# SAE: Gemma-2-2B Layer 12 Residual Stream (v9c)

TopK Sparse Autoencoder trained on the residual stream after layer 12 of `google/gemma-2-2b`.
Used in *A sparse-feature audit of induction in Gemma-2-2B*:
[GitHub](https://github.com/sohumsen/sae-gemma-induction) ·
[interactive dashboard](https://sae-gemma.streamlit.app/).

## Quick facts

| | |
|---|---|
| Architecture | TopK SAE |
| Hook | `blocks.12.hook_resid_post` |
| `d_in` | 2,304 |
| `d_sae` | 16,384 |
| L0 / k | 100 |
| Training tokens | 200M |
| Dataset | `monology/pile-uncopyrighted` (BOS-excluded) |
| Library | `saprmarks/dictionary_learning` 0.1.0; converted to SAELens 6.43.0 format |
| Final explained variance | **0.85 (peak 0.893)** |
| Dead features | 0 |
| Hardware | Single RTX 5070 Ti (16 GB) |

## Loading

```python
from sae_lens.saes.sae import SAE

sae = SAE.load_from_disk(
    "sohumsen/sae-gemma2-2b-layer12-v9c",   # downloads from HF
    device="cuda",
)
```

Or download files manually with `huggingface_hub.snapshot_download` and pass the local
path to `SAE.load_from_disk`.

## What this SAE is for

It decomposes Gemma-2-2B's layer-12 residual stream into 16,384 named, monosemantic
features. Of those, ~100 are causally implicated in induction-style in-context learning
(predicting B after seeing `A B ... A`). The top induction feature, **F15289**, fires
on the second occurrence of a repeated word ("Never...Never", "Tier...Tier", ...).

For the full story — feature ranking, head-correspondence ablations, library-comparison
notes (SAELens TopK plateaus on this task; dictionary_learning does not) — see the
[GitHub repo](https://github.com/sohumsen/sae-gemma-induction).

## License

MIT, same as the source repository.
"""


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set in .env")

    api = HfApi(token=token)
    print(f"[hf] Ensuring repo {REPO_ID} exists ...")
    create_repo(REPO_ID, repo_type="model", exist_ok=True, token=token)

    readme_path = REPO_ROOT / ".hf_README_v9c.md"
    readme_path.write_text(MODEL_CARD, encoding="utf-8")

    files = [
        (SAE_DIR / "sae_weights.safetensors", "sae_weights.safetensors"),
        (SAE_DIR / "cfg.json", "cfg.json"),
        (readme_path, "README.md"),
    ]
    for local, remote in files:
        if not local.exists():
            print(f"[hf] WARNING: {local} not found, skipping")
            continue
        print(f"[hf] Uploading {local.name} -> {REPO_ID}/{remote} ({local.stat().st_size/1024/1024:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type="model",
            token=token,
        )

    readme_path.unlink(missing_ok=True)
    print(f"[hf] Done. https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
