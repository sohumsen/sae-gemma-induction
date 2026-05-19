# Why SAELens TopK plateaus on Gemma-2-2B residuals, and what `dictionary_learning` does differently

A focused technical note from our SAE training iteration on Gemma-2-2B layer 12
residual stream. We tried seven SAELens TopK configurations (v1–v7) and every
one plateaued at cossim ≈ 0.5, l2_ratio ≈ 0.008, EV_legacy ≈ −0.5. Switching to
`saprmarks/dictionary_learning` with similar hyperparameters reached EV > 0.85
in the first few hundred training steps.

This note distills what we learned. It's intended for anyone hitting the same
"my SAELens TopK SAE has negative EV no matter what I do" wall.

---

## The symptoms

Across SAELens v1–v7 (200M tokens each, varying k=50/100, lr=5e-5/7e-5/2e-4/3e-4,
`normalize_activations` ∈ {none, expected_average_only_in},
`rescale_acts_by_decoder_norm` ∈ {True, False},
`apply_b_dec_to_input` ∈ {True, False},
`aux_loss_coefficient` ∈ {0.03125, 1.0}):

- **cossim plateaus at 0.45–0.55** (target: >0.9)
- **l2_ratio ≈ 0.005–0.01** (target: ~1.0). Reconstruction norm is ~1% of input norm — the SAE is essentially outputting zero.
- **EV_legacy negative** (typically −0.4 to −0.7). The reconstruction is worse than predicting the activation mean.
- **CE-loss-with-SAE worse than CE-with-ablation** — patching the SAE's output back into the model hurts more than zeroing the residual entirely.

The SAE *does* find meaningful features at this quality (we successfully identified an induction-feature cluster with clean ablation effects), but the reconstruction quality fails any reviewer-bar test.

## What we ruled out

| Lever | Tried | Effect |
|---|---|---|
| k=50 → 100 | Yes | cossim 0.42 → 0.53. Helps a little, not enough. |
| `normalize_activations=expected_average_only_in` | Yes | cossim unchanged (0.41 → 0.41). |
| `rescale_acts_by_decoder_norm=True` | Yes | cossim unchanged; encoder collapsed (l1 = 30 → 2.87). |
| `apply_b_dec_to_input=True/False` | Both | No effect. |
| `aux_loss_coefficient=1.0` vs 1/32 | Both | 1.0 is the SAELens-canonical value (it already internally scales by `min(num_dead/k_aux, 1.0)`). 1/32 weakened aux 32× and *worsened* dead-feature accumulation. |
| `lr` ∈ {5e-5, 7e-5, 2e-4, 3e-4} | All | Higher lr killed features faster (dead features 33 → 85 in 700 steps at 3e-4); lower lr collapsed the encoder. |
| `lr_scheduler_name` ∈ {constant, cosineannealing} | Both | No difference. |

The plateau is **robust** to hyperparameters within SAELens's TopK trainer.

## What's actually different in `dictionary_learning`

Switching to `saprmarks/dictionary_learning` with `TopKTrainer` + `AutoEncoderTopK` (the library SAEBench uses for their EV-0.85 published SAEs), we hit EV = 0.41 at training step 5, EV = 0.60 at step 16, and EV = 0.87 at step 175 — same model, same data, same training token budget, same k. The differences (concrete, from the source):

### 1. Activation normalization lives in the buffer, not the SAE

SAELens's `normalize_activations="expected_average_only_in"`:
- Computes a global scaling factor at init (E[||x||]/√d).
- Applied via an `ActivationScaler` inside the SAE's forward pass.
- "Folded" into the SAE weights at the end of training via `fold_activation_norm_scaling_factor()` so the saved SAE works on raw activations at inference.

dictionary_learning's `normalize_activations=True`:
- Computed once on a fresh batch in the `ActivationBuffer`.
- The buffer pre-scales every activation handed to the trainer and post-unscales every reconstruction. Reconstruction loss is computed **in the original (unscaled) space**.

The key difference: in SAELens, the SAE itself sees pre-scaled inputs and produces pre-scaled outputs, with scale-folding deferred until save. In dictionary_learning, the SAE only ever sees a single fixed scale, and the loss is computed against the actual target scale. This matters because:
- The decoder unit-norm constraint applies in the scaled space.
- Optimizer dynamics depend on which space the gradient flows through.

### 2. Decoder unit-norm enforced by `ConstrainedAdam`, not a rescale flag

SAELens's `rescale_acts_by_decoder_norm`:
- True: multiplies feature activations by decoder column norm at TopK selection. Decoder norms can drift to ~2.67 (observed empirically).
- False: relies on the soft constraint baked into TopK's gradient flow. We observed decoder norms still growing/shrinking in degenerate ways depending on lr.

dictionary_learning's `ConstrainedAdam`:
- After every optimizer step, **projects** decoder columns to unit norm.
- **Removes the gradient component parallel to each decoder column** before the Adam update, so the optimizer doesn't keep "trying" to grow the magnitude.

The hard projection prevents the optimizer from finding degenerate "tiny encoder activations + larger decoder norms" or "encoder collapse" minima. We observed exactly these failure modes in SAELens v5 (l1=30, no projection) and v6 (l1=2.87, optimizer collapsed encoder side).

### 3. EMA threshold tracking on top of TopK

dictionary_learning's `AutoEncoderTopK` maintains an exponential moving average of the **minimum pre-activation that survives the top-k cut** across batches (parameterized by `threshold_beta=0.999`, `threshold_start_step=1000`). This threshold is used at inference as a JumpReLU-style gate.

SAELens TopK doesn't have this. The TopK selection at inference is purely magnitude-based. Why this matters during training: the EMA threshold gives a stable target for the encoder to grow into — features below the threshold get aux-loss gradient pressure to grow; features at the boundary are stabilized. Without it, our SAELens runs showed feature activations bouncing or collapsing.

### 4. Auxiliary loss formula difference

Both libraries cite Gao et al. 2024's `auxk` loss. Both have `aux_loss_coefficient` / `auxk_alpha`. But the **internal scaling differs**:

- SAELens: `aux_coef * min(num_dead / k_aux, 1.0) * loss`, where `k_aux = d_in // 2 = 1152`. With 32 dead features (our typical case), this scales loss by 32/1152 = 0.028. So `aux_coef=1.0` ≈ effective coefficient 0.028.
- dictionary_learning: `auxk_alpha * loss` (no internal scaling). With `auxk_alpha=1/32 = 0.03125`, the effective coefficient is 0.03125 directly.

These happen to land at similar effective magnitudes, but the difference is implementation-defined. We initially set `aux_coefficient=1/32` in SAELens after reading Gao et al., which was wrong for that library and weakened aux loss 32× too much. Always read the source, not the paper.

### 5. BOS token excluded from the activation buffer

dictionary_learning's `ActivationBuffer(remove_bos=True)` skips the BOS token entirely when filling its training buffer. SAELens by default includes BOS (with `prepend_bos=True`).

Gemma-2-2B's BOS token has anomalously large residual norms (the Gemma Scope team published a known issue thread about this). Including it in training makes the per-token MSE distribution heavily skewed — a single token type dominates the loss.

### 6. EV reporting

dictionary_learning's W&B logs `frac_variance_explained` — computed per-batch as the standard `1 - ||x - x_hat||^2 / ||x - mean(x)||^2`. This is the correct formula.

SAELens v6 has the EV-computation bug noted in their PR #665: `total_variance = E[||x||^2] - (E[||x||^2]/d)^2`, which is variance-about-zero rather than variance-about-mean. The trustworthy SAELens metric is `explained_variance_legacy` (named "legacy" but actually correct). This isn't the *cause* of the bad SAELens results (the legacy formula was also negative for us), but it's a separate gotcha worth knowing.

---

## What this means for someone using SAELens

If your SAELens TopK SAE on a residual-stream hook has:
- cossim < 0.6 after meaningful training
- l2_ratio < 0.1
- ce_loss_score negative
- Symptoms unchanged across reasonable hyperparameter sweeps

— it's worth trying `saprmarks/dictionary_learning` before deciding the data/model isn't amenable to SAE training. The five implementation differences above appear to compound into a meaningfully different optimization regime.

A direct port is straightforward (`pip install dictionary-learning`), but expects HuggingFace models wrapped with `nnsight.LanguageModel`. Weights save as a PyTorch `state_dict` to `ae.pt`; converting to a SAELens-loadable `cfg.json` + `sae_weights.safetensors` is ~30 lines of conversion (see `scripts/convert_dl_to_saelens.py` in this repo).

Caveat: I have *not* tried whether SAELens with a custom training loop that mimics dictionary_learning's `ConstrainedAdam` + buffer-side normalization could match dictionary_learning's results. It's possible the fix is just those two changes inside SAELens. I didn't have time to test, and switching to dictionary_learning was faster.

---

## References

- Gao et al. 2024, "Scaling and Evaluating Sparse Autoencoders" (the TopK + auxk paper): https://cdn.openai.com/papers/sparse-autoencoders.pdf
- Marks et al., `saprmarks/dictionary_learning`: https://github.com/saprmarks/dictionary_learning
- Karvonen et al., SAEBench: https://www.neuronpedia.org/sae-bench/info
- SAELens v6.43 source: https://github.com/jbloomAus/SAELens
- The EV-formula bug: https://github.com/jbloomAus/SAELens/pull/665
