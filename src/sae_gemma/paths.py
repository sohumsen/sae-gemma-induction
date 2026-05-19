"""Canonical project paths and model constants. Import from here — never hard-code elsewhere."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Gemma-2-2B architecture (from TransformerLens config) ─────────────────────
MODEL_NAME = "google/gemma-2-2b"
TARGET_LAYER = 12
HOOK_NAME = f"blocks.{TARGET_LAYER}.hook_resid_post"
D_MODEL = 2304
N_HEADS = 8       # query heads (Gemma-2-2B uses GQA: 8 query, 4 KV heads)
N_KV_HEADS = 4
N_LAYERS = 26

MODELS_DIR = REPO_ROOT / "models"
SAE_PILOT_DIR = MODELS_DIR / "sae_pilot"
SAE_MAIN_DIR = MODELS_DIR / "sae_main"

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FEATURE_LABELS_PATH = RESULTS_DIR / "feature_labels.json"
TOP_SNIPPETS_PATH = RESULTS_DIR / "top_snippets.parquet"
INDUCTION_PROBES_PATH = RESULTS_DIR / "induction_probes.parquet"
HEAD_CORR_PATH = RESULTS_DIR / "head_correspondence.parquet"
ABLATION_RESULTS_PATH = RESULTS_DIR / "ablation_results.json"

for _d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
