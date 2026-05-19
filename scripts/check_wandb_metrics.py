"""
Poll W&B for the latest reconstruction-quality metrics of the v3 SAE training run.
Returns the most recent values of:
  - reconstruction_quality.explained_variance
  - reconstruction_quality.explained_variance_legacy   (the trustworthy one)
  - shrinkage.l2_ratio
  - sparsity.l0
  - mse losses
  - cossim (computed from sparsity_variance metrics if available)

Prints JSON to stdout for easy parsing from PowerShell.
"""
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

import wandb

PROJECT = os.environ.get("WANDB_PROJECT", "sae-gemma-induction")
ENTITY = os.environ.get("WANDB_ENTITY", None)
RUN_NAME_FILTER = sys.argv[1] if len(sys.argv) > 1 else "gemma2-2b-l12-main-16k-v3"

api = wandb.Api(timeout=29)
path = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT
runs = api.runs(path, filters={"display_name": RUN_NAME_FILTER}, order="-created_at")

if not runs:
    print(json.dumps({"error": f"No runs matching '{RUN_NAME_FILTER}'"}))
    sys.exit(1)

run = runs[0]
state = run.state
last_step = run.summary.get("_step", None)

# Pull the metrics we care about
keys_of_interest = [
    "metrics/explained_variance",
    "metrics/explained_variance_legacy",
    "metrics/explained_variance_legacy_std",
    "metrics/l0",
    "metrics/mean_log10_feature_sparsity",
    "losses/mse_loss",
    "losses/auxiliary_reconstruction_loss",
    "losses/overall_loss",
    "details/n_training_samples",
    "sparsity/dead_features",
    "sparsity/below_1e-5",
    # nested dicts surfaced as run.summary[key] returns a dict
    "reconstruction_quality",
    "shrinkage",
    "sparsity",
    "model_performance_preservation",
]

# Grab from summary (most recent eval) and from history (full curve)
out = {
    "run_name": run.name,
    "run_id": run.id,
    "state": state,
    "step": last_step,
    "url": run.url,
    "summary": {},
    "history_tail": {},
}

for key in keys_of_interest:
    val = run.summary.get(key, None)
    if val is not None:
        out["summary"][key] = val

# Also dump all summary keys so we see what's actually logged
out["all_summary_keys"] = sorted(
    [k for k in run.summary.keys() if not k.startswith("_")]
)

# Recent history (last 20 logged points) for trend
try:
    hist = run.history(keys=keys_of_interest, pandas=False, samples=20)
    if hist:
        out["history_tail"] = {
            key: [row.get(key) for row in hist if row.get(key) is not None][-5:]
            for key in keys_of_interest
        }
        out["history_tail"] = {k: v for k, v in out["history_tail"].items() if v}
except Exception as e:
    out["history_error"] = str(e)

print(json.dumps(out, indent=2, default=str))
