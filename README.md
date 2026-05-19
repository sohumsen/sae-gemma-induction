# sae-gemma-induction

Sparse autoencoder analysis of induction features in Gemma-2-2B — an extension of Olsson et al. (2022) ["In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) from the attention-head level to the SAE feature level.

## TL;DR

Trained a TopK SAE (width 16,384, k=80) on `blocks.12.hook_resid_post` of Gemma-2-2B. Identified **420 induction-feature candidates** out of the SAE's 16k features. **Ablating the top-5 dropped ICL top-1 accuracy from 42% → 26% — a 16pp reduction using 0.03% of features.** The most strongly-induction feature correlates with attention head 6 at layer 12 (Pearson r = 0.190), which head-level ablation also identifies as the dominant induction head.

| Metric | Value |
|---|---|
| Model | google/gemma-2-2b |
| Hook | blocks.12.hook_resid_post (residual stream, layer 12) |
| SAE width | 16,384 |
| L0 (target sparsity) | 80 |
| Training tokens | 200M |
| Final EV | **0.85 (peak 0.893)** — Gemma Scope range |
| Candidate induction features | 100 |
| Top induction feature | F15289 — auto-labelled "second occurrence of repeated word" |
| Top-50-feature ablation effect | 57.75% → 47.65% ICL accuracy (−10.1pp); 35× over random-feature control |
| Dominant induction head | head 6 (layer 12), −6.65pp single-head ablation |
| Causal head → feature link | activation-patching: head 3 −57% on F15289 (95% CI [48.5, 66.4]); distributed circuit |
| Multi-seed replication | 3 seeds: top-50 ablation drop 10–19pp, top-20 mean induction score 0.77–0.86 |

## Pipeline overview

```
induction_probes.py     →  Synthetic [A][B]...[A] sequences for measuring ICL
train_sae_dl.py         →  Train SAE on Gemma layer-12 residuals (dictionary_learning, 200M tok)
convert_dl_to_saelens.py → Cross-format conversion to SAELens-loadable weights
find_induction_features.py → Rank features by (induction - control) activation
head_correspondence.py   →  Pearson correlation between features and head attention patterns
ablations.py             →  Feature- and head-level ablation experiments
capture_activations.py   →  Top-20 activating snippets per feature (for auto-interp)
autointerp.py            →  Claude Sonnet labels for top induction features
dashboard.py             →  Interactive Streamlit feature browser
```

## Stack

| Layer | Tool |
|---|---|
| Model loading | `transformer_lens` (TL HookedTransformer) + `transformers` (HF) |
| SAE training | `saprmarks/dictionary_learning` (final), `sae_lens` (initial baseline) |
| Activation extraction | `nnsight` |
| Experiment tracking | `wandb` |
| Auto-interpretation | Anthropic Claude API (Sonnet 4.5) |
| Dashboard | `streamlit` |
| Data | `monology/pile-uncopyrighted` (training), `allenai/c4` (capture) |

## Hardware

Single RTX 5070 Ti (16 GB VRAM). ~15h total wall-clock for the full pipeline (training is the bottleneck at ~5h on dictionary_learning).

## Reproduction

```powershell
git clone <repo>
cd sae-gemma-induction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dashboard,dev]"
pip install dictionary-learning   # only needed for SAE training, not for analysis

# Set HF_TOKEN and WANDB_API_KEY in .env (see .env.example)

# 1. Train SAE (~5h on RTX 5070 Ti)
python src/sae_gemma/train_sae_dl.py

# 2. Run full downstream pipeline
.\scripts\run_pipeline_v8.ps1

# 3. View dashboard
streamlit run src/sae_gemma/dashboard.py
```

Or just run the analysis on the already-trained v1 SAE (in the repo):

```powershell
streamlit run src/sae_gemma/dashboard.py
```

## Documentation

- **[WRITEUP.md](WRITEUP.md)** — The full write-up.
- **[HISTORY.md](HISTORY.md)** — Iteration log across v1–v8: every SAE config we tried, what we changed each time, what we learned.
- **[SAELENS_VS_DL.md](SAELENS_VS_DL.md)** — Technical note on why SAELens TopK plateaus on Gemma residuals, and what `dictionary_learning` does differently. Independently useful for anyone hitting the same wall.

## Layout

```
src/sae_gemma/
  train_sae.py              # SAELens TopK trainer (v1 baseline)
  train_sae_dl.py           # dictionary_learning TopK trainer (v8, final)
  capture_activations.py    # top-activating snippets
  induction_probes.py       # ICL probe generation
  find_induction_features.py # candidate cluster ranking
  autointerp.py             # Claude Sonnet labelling
  head_correspondence.py    # feature↔head correlation
  ablations.py              # ablation experiments
  dashboard.py              # Streamlit app
scripts/
  convert_dl_to_saelens.py  # cross-format SAE weight converter
  run_pipeline_v8.ps1       # full pipeline orchestrator
  monitor_dl.py             # W&B-polling training monitor
  check_wandb_metrics.py    # ad-hoc W&B query
```

## Links

- **GitHub:** https://github.com/sohumsen/sae-gemma-induction
- **Dashboard:** https://sae-gemma.streamlit.app/
- **Write-up:** [WRITEUP.md](WRITEUP.md)
- **W&B project:** [sae-gemma-induction](https://wandb.ai/sohumsen2-ucl/sae-gemma-induction)
- **v9c W&B run (final):** [45e85on3](https://wandb.ai/sohumsen2-ucl/sae-gemma-induction/runs/45e85on3)

## License

MIT.
