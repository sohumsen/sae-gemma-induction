#!/usr/bin/env bash
# run_all.sh — Regenerate all figures from cached intermediates.
# Does NOT re-train the SAE or re-run auto-interpretation.
# Requires: results/*.parquet + results/*.json produced by the analysis pipeline.
#
# Usage: bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Load .env
if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "[run_all] Regenerating figures from cached intermediates ..."

echo "[run_all] 1/3 training metrics ..."
python scripts/plot_training_metrics.py

echo "[run_all] 2/3 head correspondence ..."
python -m sae_gemma.head_correspondence --plot-only

echo "[run_all] 3/3 ablation curve ..."
python -m sae_gemma.ablations --plot-only

echo ""
echo "[run_all] Done. Figures written to results/figures/"
echo "  - results/figures/sae_training_metrics.png"
echo "  - results/figures/head_correspondence.png"
echo "  - results/figures/ablation_curve.png"
