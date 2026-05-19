# SAE Training History — Gemma-2-2B Layer 12 Residual Stream

A log of every SAE training run we attempted, what we changed each time, and what
we learned. The goal was a publication-quality SAE on `blocks.12.hook_resid_post`
with positive explained variance, 200M training tokens, single 16 GB GPU.

## TL;DR
| | Library | k | normalize | rescale_dec | lr | EV | Cossim | Notes |
|---|---|---|---|---|---|---|---|---|
| **v1** | SAELens | 50 | none | True | 2e-4 | −0.42 | 0.42 | First completed run. Downstream pipeline ran on this. |
| **v2** | SAELens | 50 | none | False | 2e-4 | (killed) | – | rescale alone made no difference. |
| **v3** | SAELens | 50 | expected_avg | False | 2e-4 | −0.41 | 0.41 | Normalize alone didn't fix anything. |
| **v4** | SAELens | 50 | expected_avg | False | 2e-4 | (killed) | – | Misread research; lowered aux to 1/32 — was wrong for SAELens. |
| **v5** | SAELens | 100 | expected_avg | False | 7e-5 | −0.50 | 0.53 | k=100 helped a little. Shrinkage still bad. |
| **v6** | SAELens | 100 | expected_avg | True | 5e-5 | −0.51 | 0.51 | Encoder collapsed (l1=2.87). |
| **v7** | SAELens | 100 | expected_avg | True | 3e-4 | (killed) | 0.51 | Dead features jumped 33→85. SAELens regime ceiling clear. |
| v8 | dictionary_learning | 80 | constant_norm | n/a | 5e-5 | (lost) | (lost) | Library switch — but training crashed at final save (parent dir deleted by test cleanup). |
| v9 | dictionary_learning | 100 | constant_norm | n/a | 5e-5 | (killed) | (killed) | Tried d_sae=32768 — buffer refills 25× slower in tight VRAM. ETA 240h. |
| v9b | dictionary_learning | 100 | constant_norm | n/a | 5e-5 | (crashed @ 19%) | (crashed @ 19%) | HF dataset streaming connection closed mid-training. |
| **v9c** | **dictionary_learning** | 100 | constant_norm | n/a | 5e-5 | **0.85 (peak 0.893)** | **n/a** | **Canonical SAE: local pile cache + intermediate checkpoints, completed cleanly. All downstream pipeline runs on this one.** |
| v10 | dictionary_learning | 100 | constant_norm | n/a | 5e-5 | n/a | n/a | Replication seed=43. Top feature F3931 (rank-1) "sentence-initial", F11310 (rank-2) "token repetition". Top-50 ablation: -18.8pp. |
| v11 | dictionary_learning | 100 | constant_norm | n/a | 5e-5 | n/a | n/a | Replication seed=44. Top feature F12906 "repeated words — second occurrence" (clean induction). Top-50 ablation: -12.2pp. |

---

## v1 — Baseline (SAELens, k=50)

**Config:** `TopKTrainingSAEConfig(k=50, d_sae=16384, normalize_activations="none", rescale_acts_by_decoder_norm=True, apply_b_dec_to_input=True, aux_loss_coefficient=1.0)`, lr=2e-4 cosine, 200M tokens, batch_size_tokens=4096.

**Result:** Trained to completion. Final metrics: cossim=0.42, EV_legacy=−0.5, l2_ratio≈0.008.

**Downstream pipeline ran successfully on v1:**
- 420 induction-candidate features identified
- Head correspondence: F12592 had Pearson r=0.190 with attention head 6
- Ablating top 5 induction features dropped ICL top-1 accuracy from 42% → 26% (**16pp drop**)
- Auto-interpretation: top features describe "second/completing token of multi-word units" — consistent with induction-head mechanism

**Lesson:** A SAE with bad reconstruction metrics can still produce meaningful and causally important features. The features it finds are real (proven by ablation). But the reconstruction quality on EV/cossim metrics doesn't pass the reviewer-bar for publication-quality SAE work.

---

## v2 — Disable decoder rescaling

**Change:** `rescale_acts_by_decoder_norm=True → False`.

**Hypothesis:** The original config note suggested True caused decoder columns to drift to norm 2.67; disabling might fix shrinkage.

**Result:** Killed early once metrics looked the same as v1 (cossim ~0.41). The flag alone changes very little.

**Lesson:** `rescale_acts_by_decoder_norm` is not the lever for fixing low-cossim SAEs.

---

## v3 — Add activation normalization

**Change:** `normalize_activations="none" → "expected_average_only_in"`.

**Hypothesis:** Gemma-2-2B residual stream has large norms (~150). Without normalization, the SAE training landscape is poorly conditioned.

**Result:** Trained to ~28M tokens. cossim=0.41, EV_legacy=−0.56, l2_ratio=0.008. **Same regime as v1.** CE-loss-with-SAE (16.4) was worse than CE-with-full-ablation (12.4) — the SAE was actively hurting model output.

**Key discovery during v3:**
- Reading SAELens source, found the `explained_variance` metric is **computed incorrectly** (PR #665, unmerged): the formula uses `E[||x||²] − (E[||x||²]/d)²` which is variance-about-zero rather than variance-about-mean. The trustworthy version is `explained_variance_legacy`.
- `cossim`, `relative_reconstruction_bias`, and `ce_loss_score` are the metrics actually worth tracking.

**Lesson:** Activation normalization alone is not enough. The reconstruction shrinkage problem (output norm ≈ 0.7% of input norm) has a deeper cause.

---

## v4 — Wrong aux-loss correction (mistake)

**Change:** Following Gao et al. 2024 (OpenAI's "Scaling and Evaluating Sparse Autoencoders"), set `aux_loss_coefficient = 1/32 = 0.03125`.

**Killed during norm calibration after spawning research agents.**

**Key discovery:** **SAELens computes the auxiliary loss differently from EleutherAI's `sparsify` library** that Gao et al. wrote about. SAELens internally scales by `min(num_dead/k_aux, 1.0)` before applying the coefficient. So the SAELens-canonical value is **1.0**, not 1/32. My change weakened aux loss ~32× too much and would have caused dead-feature accumulation.

**Lesson:** Hyperparameters are library-specific. Don't blindly copy from a different codebase without reading the implementation.

---

## v5 — Higher k

**Change:** `TARGET_K=50 → 100`, with `rescale_acts_by_decoder_norm=False`, `lr=7e-5`. Reverted aux to 1.0.

**Hypothesis:** k=50 is too narrow a bottleneck for d_in=2304. Gao et al. and jbloom both use k≈100 at width 16k.

**Result:** Trained to ~28M tokens. cossim=0.53 (best in SAELens so far). EV_legacy=−0.56. l2_ratio still 0.008 — **shrinkage unchanged**. Dead features down to 6.

**Lesson:** k matters at the margin (cossim 0.42 → 0.53), but the shrinkage / negative-EV regime is unaffected.

---

## v6 — Enable decoder rescaling with k=100

**Change:** `rescale_acts_by_decoder_norm=True`, keeping k=100, lr=5e-5.

**Hypothesis:** Letting decoder columns grow beyond unit norm should fix the tiny reconstruction.

**Result:** Trained to ~12M tokens. cossim=0.51, l1 collapsed to 2.87 (down from v5's 30) — **encoder shrank** to keep MSE low while decoder grew. Mean per-feature activation = 0.029 (tiny).

**Lesson:** With `rescale=True` and low lr, the optimizer finds a "tiny-encoder + larger-decoder" minimum that doesn't actually reconstruct any better. SAELens's TopK + rescale interaction has a degenerate solution.

---

## v7 — Raise learning rate (last SAELens attempt)

**Change:** lr=3e-4 (SAEBench's actual reported value, not the stale 5e-5 figure), apply_b_dec_to_input back to True.

**Hypothesis:** Higher lr will break the encoder out of the tiny-activation minimum.

**Result:** Killed at step ~2700. cossim=0.51 (same), but dead features jumped 33 → 85 in 700 steps. The higher lr was killing features faster than aux loss could revive them.

**Decision after v7:** Every SAELens config we tried (v1–v7, varying k, lr, normalize_activations, rescale, apply_b_dec) plateaued around cossim 0.5 with l2_ratio 0.008. SAEBench's published SAEs achieve cossim ~0.95 / EV ~0.85 on this exact task — but they use **`saprmarks/dictionary_learning`, not SAELens.** Time to switch libraries.

---

## v8 — `saprmarks/dictionary_learning` (the breakthrough)

**Change:** Complete library swap. Used `dictionary_learning.trainers.top_k.TopKTrainer` with `AutoEncoderTopK`. Model wrapped via `nnsight.LanguageModel`. ActivationBuffer with `remove_bos=True`, `constant_norm_rescale` normalization, `ctx_len=1024`.

**Config:** k=80, d_sae=16384, lr=5e-5, auxk_alpha=1/32 (the dictionary_learning convention — different code path from SAELens), warmup_steps=1000, threshold_beta=0.999, batch=2048 tokens.

**Result (still training):**
- **Step 5: EV = 0.41** (already positive — every SAELens run was negative)
- **Step 16: EV = 0.60**
- **Step 50: EV = 0.75**
- **Step 176: EV = 0.87** (crossed Gemma Scope's reported 0.82–0.90 range)
- 0 dead features throughout
- l2_loss steadily declining (0.51 → 0.13)
- At step ~24,000 of 97,656 when this file was written

**Key implementation differences vs SAELens:**
1. **Activation normalization is in the buffer**, not the SAE. Buffer rescales activations by a global constant before passing to SAE. Reconstructions are unscaled before comparison.
2. **Decoder unit-norm is enforced by `ConstrainedAdam`** — projects + removes gradient component parallel to decoder direction after every step. SAELens uses a soft constraint via the rescale flag, which we saw lets the optimizer find degenerate minima.
3. **Aux loss uses Gao et al.'s exact formula** with the 1/32 coefficient.
4. **EMA threshold mechanism** (`threshold_beta=0.999`) tracks the minimum activated pre-activation per batch — used as a JumpReLU-style threshold at inference. This is what enables stable training at higher k.
5. **BOS token excluded** from the activation buffer (Gemma's BOS has anomalously large norm and would dominate the loss).

**Lesson:** SAELens TopK has implementation choices that prevent it from reaching publication-quality reconstruction on Gemma-2-2B residual stream activations. The differences are not in the obvious hyperparameters — they're in the buffer-side normalization, the optimizer's decoder constraint, the threshold mechanism, and BOS handling. None of these are easily switched on in SAELens.

---

## Meta-lessons

1. **Reconstruction quality and feature quality can diverge.** v1's SAE had bad EV but found meaningful induction features. Ablation experiments validate features more reliably than EV.

2. **Trust the SAELens `explained_variance_legacy` over `explained_variance`** (the current default formula is computed incorrectly, PR #665).

3. **Cossim and `ce_loss_score` are more diagnostic than EV.** They're scale-invariant and harder to game.

4. **Don't blindly transplant hyperparameters across libraries.** SAELens's `aux_loss_coefficient=1.0` ≠ Gao et al.'s α=1/32 ≠ dictionary_learning's `auxk_alpha=1/32`. The aggregation logic differs.

5. **If a SAE library plateaus across many config variations on the same task, suspect the library's TopK implementation, not the hyperparameters.** SAEBench actively uses dictionary_learning's TopK because of issues like this.

6. **Single high-leverage changes beat compound changes for debugging.** v4 mixed three changes at once; switching just one at a time (v3, v5, v6) made it clearer what each flag did.

7. **A monitor that polls W&B every N seconds beats one-off scheduled checks.** Scheduled wakeups failed silently several times during this run. A Monitor task watching a polling script's stdout proved more reliable.

---

## W&B run IDs (for reference)

| Version | W&B run ID | Direct URL |
|---|---|---|
| v1 | klfvob7u | https://wandb.ai/sohumsen2-ucl/sae-gemma-induction/runs/klfvob7u |
| v2 | 8kyq6w96 | https://wandb.ai/sohumsen2-ucl/sae-gemma-induction/runs/8kyq6w96 |
| v3 | 0dnxyaly | https://wandb.ai/sohumsen2-ucl/sae-gemma-induction/runs/0dnxyaly |
| v5 | 3s7z4e6y | https://wandb.ai/sohumsen2-ucl/sae-gemma-induction/runs/3s7z4e6y |
| v6 | q7gjsjeh | https://wandb.ai/sohumsen2-ucl/sae-gemma-induction/runs/q7gjsjeh |
| v7 | (no run created — killed before logging) | – |
| v8 (final) | vksi2ayq | https://wandb.ai/sohumsen2-ucl/sae-gemma-induction/runs/vksi2ayq |

## Compute used (single 16 GB RTX 5070 Ti, May 17 2026)

| Run | Wall time | Outcome |
|---|---|---|
| v1 (pilot+full) | ~6.5 h | Complete; cossim 0.42 |
| v2 | ~30 min | Killed |
| v3 | ~1.5 h | Killed at 28M tokens |
| v4 | ~5 min (norm calibration only) | Killed |
| v5 | ~1.8 h | Killed at 28M |
| v6 | ~30 min | Killed at 12M |
| v7 | ~10 min | Killed at step 2700 |
| **v8** | ~5 h (estimated) | **Training — projected EV 0.85–0.92** |

Total compute: ~15 h. v1 was salvageable as a working backup throughout — never lost.
