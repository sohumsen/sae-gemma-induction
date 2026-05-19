"""
Phase 2 (pilot) and Phase 3 (main) SAE training.

Uses TopK activation (exact L0=TARGET_K guarantee) to avoid L1 coefficient tuning.

Usage:
    python src/sae_gemma/train_sae.py --mode pilot   # 4096-wide, 5M tokens
    python src/sae_gemma/train_sae.py --mode main    # 16384-wide, 200M tokens

Run detached for long runs:
    Start-Process python -ArgumentList "src/sae_gemma/train_sae.py --mode main" `
        -RedirectStandardOutput models/sae_main/log.txt `
        -RedirectStandardError models/sae_main/err.txt -NoNewWindow
"""

import argparse
import os
from pathlib import Path

from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner
from sae_lens.config import LoggingConfig
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig

from sae_gemma.model_utils import load_model
from sae_gemma.paths import D_MODEL, HOOK_NAME, MODEL_NAME, REPO_ROOT, SAE_MAIN_DIR, SAE_PILOT_DIR

DATASET_PATH = "monology/pile-uncopyrighted"
TARGET_K = 100   # v5: raised from 50; k=50 was too narrow a bottleneck for d_in=2304 (Gao et al., jbloom both use ≈100 at width 16k)


def make_config(mode: str) -> LanguageModelSAERunnerConfig:
    """Return a fully-specified runner config for pilot or main run."""
    assert mode in ("pilot", "main"), f"Unknown mode: {mode!r}"

    if mode == "pilot":
        d_sae = 4096
        training_tokens = 5_000_000
        output_path = str(SAE_PILOT_DIR)
        run_name = "gemma2-2b-l12-pilot-4k-v8"
        lr_warm_up_steps = 50
        n_checkpoints = 0
        store_batch_size_prompts = 16
        n_batches_in_buffer = 8
    else:
        d_sae = 16_384
        training_tokens = 200_000_000
        output_path = str(SAE_MAIN_DIR)
        run_name = "gemma2-2b-l12-main-16k-v7"
        lr_warm_up_steps = 2_000
        n_checkpoints = 5
        # 16384-feature SAE + Adam states use ~900 MB more than pilot;
        # halving buffer (16×8 → 8×4 = 300 MB vs 1.2 GB) gives Adam's
        # foreach ops room to allocate ~1.2 GB of temp tensors.
        store_batch_size_prompts = 8
        n_batches_in_buffer = 4

    sae_cfg = TopKTrainingSAEConfig(
        d_in=D_MODEL,
        d_sae=d_sae,
        dtype="float32",   # SAE and activations in float32; model stays bf16 via autocast_lm
        device="cuda",
        apply_b_dec_to_input=True,  # v7: revert, default SAELens behaviour (v5/v6 with False didn't help)
        normalize_activations="expected_average_only_in",  # v3: scales activations by sqrt(d_in)/E[||x||], folded into decoder at save. Gemma residuals have large norms; "none" gave cossim=0.42, EV_legacy=-0.5. See Anthropic April-2024 update; Gemma Scope methodology.
        k=TARGET_K,
        aux_loss_coefficient=1.0,   # v5 (revert v4): SAELens already internally scales aux by min(num_dead/k_aux, 1.0); coefficient=1.0 is the SAELens-canonical setting (Gao et al.'s 1/32 is for sparsify, different code path)
        rescale_acts_by_decoder_norm=True,  # v6: re-enable to fix shrinkage. v5 had l2_ratio=0.008 (decoder output ~1% of input norm). Letting decoder cols grow while rescaling activations should fix this.
    )

    logging_cfg = LoggingConfig(
        log_to_wandb=True,
        wandb_project=os.environ.get("WANDB_PROJECT", "sae-gemma-induction"),
        wandb_entity=os.environ.get("WANDB_ENTITY", None),
        run_name=run_name,
        wandb_log_frequency=100,
        eval_every_n_wandb_logs=10,
    )

    cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        # ── model ──────────────────────────────────────────────────────────────
        model_name=MODEL_NAME,
        model_class_name="HookedTransformer",
        hook_name=HOOK_NAME,
        model_from_pretrained_kwargs={"dtype": "bfloat16"},
        # ── data ───────────────────────────────────────────────────────────────
        dataset_path=DATASET_PATH,
        dataset_trust_remote_code=False,
        streaming=True,
        is_dataset_tokenized=False,
        context_size=1024,
        prepend_bos=True,
        use_cached_activations=False,
        # ── activation store ───────────────────────────────────────────────────
        store_batch_size_prompts=store_batch_size_prompts,
        n_batches_in_buffer=n_batches_in_buffer,
        train_batch_size_tokens=4096,  # SAE training step batch
        # ── training ───────────────────────────────────────────────────────────
        training_tokens=training_tokens,
        lr=3e-4,    # v7: SAEBench's actual working value (the 5e-5 jbloom figure was stale); higher lr after the encoder collapse seen in v6
        lr_scheduler_name="constant",   # v5: jbloom uses constant lr (was cosineannealing)
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=training_tokens // 4096,  # decay over full training run
        adam_beta1=0.9,
        adam_beta2=0.999,
        # ── hardware ───────────────────────────────────────────────────────────
        device="cuda",
        dtype="float32",    # activations in float32; model bf16 via autocast_lm
        autocast_lm=True,
        seed=42,
        # ── eval — batch_size=1 keeps logit tensor at 1.05 GB (256k vocab × 1024 ctx × fp32)
        eval_batch_size_prompts=1,
        n_eval_batches=2,
        # ── checkpointing ──────────────────────────────────────────────────────
        n_checkpoints=n_checkpoints,
        checkpoint_path=str(Path(output_path) / "checkpoints"),
        save_final_checkpoint=True,
        output_path=output_path,
        # ── logging ────────────────────────────────────────────────────────────
        logger=logging_cfg,
        verbose=True,
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train SAE on Gemma-2-2B layer 12 residual stream")
    parser.add_argument(
        "--mode",
        choices=["pilot", "main"],
        default="pilot",
        help="pilot = 4096-wide 5M tokens; main = 16384-wide 200M tokens",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from (e.g. models/sae_main/checkpoints/<run>/<step>)",
    )
    args = parser.parse_args()

    # Load env vars from .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    # Ensure output directory exists
    out_dir = SAE_PILOT_DIR if args.mode == "pilot" else SAE_MAIN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_config(args.mode)
    if args.resume_from:
        cfg.resume_from_checkpoint = str(args.resume_from)
        print(f"[train_sae] Resuming from checkpoint: {args.resume_from}", flush=True)
    print(f"[train_sae] mode={args.mode!r}  d_sae={cfg.sae.d_sae}  k={cfg.sae.k}  "
          f"tokens={cfg.training_tokens:,}  hook={cfg.hook_name}", flush=True)

    # Pre-load model via RAM-safe CPU-first path; pass as override so SAELens
    # doesn't reload it and OOM the 16 GB GPU during weight processing.
    model = load_model(device="cuda", dtype="bfloat16")

    runner = LanguageModelSAETrainingRunner(cfg, override_model=model)
    sae = runner.run()

    print(f"[train_sae] Done. SAE saved to {cfg.output_path}", flush=True)
    return sae


if __name__ == "__main__":
    main()
