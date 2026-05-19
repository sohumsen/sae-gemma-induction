"""
Download SAE training metrics from W&B and save as results/figures/sae_training_metrics.png.
Falls back to parsing models/sae_main/log.txt if W&B is unavailable.

Run as part of scripts/run_all.ps1 / run_all.sh.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def plot_from_wandb() -> bool:
    """Fetch training history from W&B and plot. Returns True on success."""
    try:
        import wandb
        api = wandb.Api()
        project = os.environ.get("WANDB_PROJECT", "sae-gemma-induction")
        entity = os.environ.get("WANDB_ENTITY", None)
        path = f"{entity}/{project}" if entity else project

        # Find the most recent main training run (16k width)
        runs = api.runs(path, filters={"config.sae.d_sae": {"$gte": 8192}}, order="-created_at")
        if not runs:
            print("[plot_metrics] No W&B runs found — falling back to log file", flush=True)
            return False

        run = runs[0]
        print(f"[plot_metrics] W&B run: {run.name} ({run.id})", flush=True)

        history = run.history(
            keys=["metrics/l0", "losses/overall_loss", "metrics/explained_variance",
                  "sparsity/dead_features"],
            samples=500,
        )

        if history.empty:
            print("[plot_metrics] W&B history empty — trying log file", flush=True)
            return False

        # Get n_features for converting dead_features count → fraction
        sae_width = run.config.get("sae", {}).get("d_sae", None) if run.config else None

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle(f"SAE Training — {run.name}", fontsize=12)

        if sae_width and "sparsity/dead_features" in history.columns:
            history["dead_feature_fraction"] = history["sparsity/dead_features"] / sae_width
        else:
            history["dead_feature_fraction"] = float("nan")

        metrics = [
            ("metrics/l0", "L0 sparsity", axes[0, 0]),
            ("losses/overall_loss", "Overall loss", axes[0, 1]),
            ("metrics/explained_variance", "Explained variance", axes[1, 0]),
            ("dead_feature_fraction", "Dead feature fraction", axes[1, 1]),
        ]

        steps_col = "_step" if "_step" in history.columns else history.columns[0]
        for col, label, ax in metrics:
            if col in history.columns:
                ax.plot(history[steps_col], history[col], linewidth=1)
                ax.set_title(label)
                ax.set_xlabel("Training step")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        # Draw acceptance-criterion reference lines
        axes[0, 0].axhline(20, color="green", linestyle="--", alpha=0.5, label="L0=20 (target min)")
        axes[0, 0].axhline(100, color="red", linestyle="--", alpha=0.5, label="L0=100 (target max)")
        axes[0, 0].legend(fontsize=7)
        axes[1, 0].axhline(0.6, color="green", linestyle="--", alpha=0.5, label="EV=0.6 (target)")
        axes[1, 0].legend(fontsize=7)
        axes[1, 1].axhline(0.25, color="red", linestyle="--", alpha=0.5, label="25% dead (limit)")
        axes[1, 1].legend(fontsize=7)

        plt.tight_layout()
        out = FIGURES_DIR / "sae_training_metrics.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[plot_metrics] Saved to {out}", flush=True)
        return True

    except Exception as exc:
        print(f"[plot_metrics] W&B fetch failed: {exc}", flush=True)
        return False


def plot_from_logfile() -> bool:
    """Parse log.txt from the main training run and plot whatever metrics are available."""
    log_path = ROOT / "models" / "sae_main" / "log.txt"
    if not log_path.exists():
        print(f"[plot_metrics] Log file not found: {log_path}", flush=True)
        return False

    import re
    steps, l0s, losses = [], [], []
    step_re = re.compile(r"step[=:\s]+(\d+)", re.I)
    l0_re = re.compile(r"l0[=:\s]+([\d.]+)", re.I)
    loss_re = re.compile(r"(?:loss|l2)[=:\s]+([\d.]+)", re.I)

    with log_path.open() as f:
        for line in f:
            sm = step_re.search(line)
            lm = l0_re.search(line)
            rm = loss_re.search(line)
            if sm and lm:
                steps.append(int(sm.group(1)))
                l0s.append(float(lm.group(1)))
                losses.append(float(rm.group(1)) if rm else float("nan"))

    if not steps:
        print("[plot_metrics] Could not parse any metrics from log file", flush=True)
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(steps, l0s, linewidth=1, color="steelblue")
    ax1.axhline(20, color="green", linestyle="--", alpha=0.5, label="target min")
    ax1.axhline(100, color="red", linestyle="--", alpha=0.5, label="target max")
    ax1.set_title("L0 sparsity (from log)")
    ax1.set_xlabel("Step")
    ax1.legend(fontsize=8)

    ax2.plot(steps, losses, linewidth=1, color="salmon")
    ax2.set_title("Reconstruction loss (from log)")
    ax2.set_xlabel("Step")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "sae_training_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot_metrics] Saved to {out} (from log file)", flush=True)
    return True


if __name__ == "__main__":
    ok = plot_from_wandb() or plot_from_logfile()
    if not ok:
        print("[plot_metrics] Could not generate training metrics plot — skipping.", flush=True)
    sys.exit(0)  # non-fatal: run_all should not abort if W&B is unavailable
