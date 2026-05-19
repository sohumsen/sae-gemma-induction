# Auto-run the full downstream pipeline on the v8 SAE.
# Waits for v8 training to complete, then runs:
#   1. convert_dl_to_saelens.py (writes models/sae_main/sae_weights.safetensors + cfg.json)
#   2. find_induction_features.py
#   3. head_correspondence.py
#   4. ablations.py
#   5. capture_activations.py
#   6. autointerp via Claude Sonnet (optional)
#
# Run after `python src/sae_gemma/train_sae_dl.py` finishes.
# Logs go to logs/pipeline_v8_<step>.{log,err}

$ErrorActionPreference = "Stop"
$ROOT = "C:\Users\sohum\CodeHome\sae-gemma-induction"
Set-Location $ROOT
$PY = ".venv\Scripts\python.exe"
$LOGDIR = "$ROOT\logs"

function Run-Step($name, $cmd) {
    Write-Host "`n=== [$name] starting ===" -ForegroundColor Cyan
    $logBase = Join-Path $LOGDIR "pipeline_v8_$name"
    $startTime = Get-Date
    $proc = Start-Process $PY -ArgumentList $cmd -WorkingDirectory $ROOT -RedirectStandardOutput "$logBase.log" -RedirectStandardError "$logBase.err" -NoNewWindow -Wait -PassThru
    $elapsed = ((Get-Date) - $startTime).TotalMinutes
    if ($proc.ExitCode -ne 0) {
        Write-Host "[$name] FAILED (exit $($proc.ExitCode)) after $($elapsed.ToString('F1'))m" -ForegroundColor Red
        Write-Host "Last 10 lines of stderr:"
        Get-Content "$logBase.err" -Encoding UTF8 -Tail 10
        exit 1
    }
    Write-Host "[$name] DONE in $($elapsed.ToString('F1'))m" -ForegroundColor Green
}

# 1. Convert dictionary_learning ae.pt -> SAELens format
Run-Step "convert" "scripts\convert_dl_to_saelens.py"

# 2 & 3. find_induction_features and head_correspondence are independent — parallel
Write-Host "`n=== [find_features + head_corr] starting in parallel ===" -ForegroundColor Cyan
$j1 = Start-Process $PY -ArgumentList "src/sae_gemma/find_induction_features.py --sae-path models/sae_main" -WorkingDirectory $ROOT -RedirectStandardOutput "$LOGDIR\pipeline_v8_find_features.log" -RedirectStandardError "$LOGDIR\pipeline_v8_find_features.err" -NoNewWindow -PassThru
$j2 = Start-Process $PY -ArgumentList "src/sae_gemma/head_correspondence.py --sae-path models/sae_main" -WorkingDirectory $ROOT -RedirectStandardOutput "$LOGDIR\pipeline_v8_head_corr.log" -RedirectStandardError "$LOGDIR\pipeline_v8_head_corr.err" -NoNewWindow -PassThru
$j1.WaitForExit()
$j2.WaitForExit()
if ($j1.ExitCode -ne 0 -or $j2.ExitCode -ne 0) {
    Write-Host "FAILED: find_features exit $($j1.ExitCode), head_corr exit $($j2.ExitCode)" -ForegroundColor Red
    exit 1
}
Write-Host "[find_features + head_corr] DONE" -ForegroundColor Green

# 4. Ablations (depends on find_features output)
Run-Step "ablations" "src/sae_gemma/ablations.py --sae-path models/sae_main"

# 5. Capture top-activating snippets (heavy — 1M tokens through SAE)
Run-Step "capture" "src/sae_gemma/capture_activations.py --sae-path models/sae_main --n-tokens 1000000 --top-k 20"

# 6. Autointerp via Claude Sonnet (top 20 induction candidates)
Run-Step "autointerp" "src/sae_gemma/autointerp.py --model claude-sonnet-4-5 --features top20"

Write-Host "`n*** PIPELINE COMPLETE ***" -ForegroundColor Green
Write-Host "Results in: $ROOT\results\"
Write-Host "Dashboard:  streamlit run src/sae_gemma/dashboard.py"
