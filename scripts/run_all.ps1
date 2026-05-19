# run_all.ps1 — Regenerate all figures from cached intermediates.
# Does NOT re-train the SAE or re-run auto-interpretation.
# Requires (produced by the analysis pipeline):
#   results/top_snippets.parquet
#   results/feature_labels.json
#   results/induction_probes.parquet
#   results/induction_candidate_ids.json
#   results/induction_feature_scores.parquet
#   results/head_correspondence.parquet
#   results/ablation_results.json
#
# Usage: .\scripts\run_all.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Load .env
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*([^#=\s][^=]*)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

Write-Host "[run_all] Regenerating figures from cached intermediates ..." -ForegroundColor Cyan

# 1. SAE training metrics (from W&B or local log)
Write-Host "[run_all] 1/3 training metrics ..."
python scripts\plot_training_metrics.py
if ($LASTEXITCODE -ne 0) { Write-Error "plot_training_metrics failed"; exit 1 }

# 2. Head correspondence heatmap (from results/head_correspondence.parquet)
Write-Host "[run_all] 2/3 head correspondence ..."
python -m sae_gemma.head_correspondence --plot-only
if ($LASTEXITCODE -ne 0) { Write-Error "head_correspondence failed"; exit 1 }

# 3. Ablation curve (from results/ablation_results.json)
Write-Host "[run_all] 3/3 ablation curve ..."
python -m sae_gemma.ablations --plot-only
if ($LASTEXITCODE -ne 0) { Write-Error "ablations failed"; exit 1 }

Write-Host ""
Write-Host "[run_all] Done. Figures written to results\figures\" -ForegroundColor Green
Write-Host "  - results\figures\sae_training_metrics.png"
Write-Host "  - results\figures\head_correspondence.png"
Write-Host "  - results\figures\ablation_curve.png"
