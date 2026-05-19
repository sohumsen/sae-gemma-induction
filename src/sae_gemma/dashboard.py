"""
Phase 5 — Streamlit feature browser. v9c data (k=100, EV 0.85, dictionary_learning).

Loads pre-computed artefacts (no live inference) and presents an interactive
feature explorer for the trained SAE on Gemma-2-2B layer 12.

Pre-requisites (must exist before deploying):
    results/top_snippets.parquet      — top-20 activating snippets per feature
    results/feature_labels.json       — auto-interp labels (Claude Haiku/Sonnet)
    results/induction_candidate_ids.json  — ranked induction cluster
    results/induction_feature_scores.parquet — per-feature induction scores
    results/head_correspondence.parquet — feature-head Pearson correlations
    results/ablation_results.json     — ablation experiment results

Run locally:
    streamlit run src/sae_gemma/dashboard.py

Deploy: push to Streamlit Community Cloud, point at this file.
"""

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def _fig_to_image(fig):
    """Convert a matplotlib figure to PNG bytes for st.image() — avoids st.pyplot() quirks."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()   # raw bytes — Streamlit's st.image handles bytes more reliably than BytesIO

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TOP_SNIPPETS_PATH = RESULTS_DIR / "top_snippets.parquet"
FEATURE_LABELS_PATH = RESULTS_DIR / "feature_labels.json"
CANDIDATE_IDS_PATH = RESULTS_DIR / "induction_candidate_ids.json"
SCORES_PATH = RESULTS_DIR / "induction_feature_scores.parquet"
HEAD_CORR_PATH = RESULTS_DIR / "head_correspondence.parquet"
ABLATION_PATH = RESULTS_DIR / "ablation_results.json"

# ── Cached data loaders ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading snippets …")
def load_snippets() -> pd.DataFrame:
    return pd.read_parquet(TOP_SNIPPETS_PATH)


@st.cache_data(show_spinner="Loading labels …")
def load_labels() -> dict[int, str]:
    with FEATURE_LABELS_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


@st.cache_data(show_spinner="Loading candidate IDs …")
def load_candidates() -> list[int]:
    with CANDIDATE_IDS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading feature scores …")
def load_scores() -> pd.DataFrame:
    return pd.read_parquet(SCORES_PATH)


@st.cache_data(show_spinner="Loading head correspondence …")
def load_head_corr() -> pd.DataFrame:
    return pd.read_parquet(HEAD_CORR_PATH)


@st.cache_data(show_spinner="Loading ablation results …")
def load_ablation() -> dict:
    with ABLATION_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# ── Helpers ────────────────────────────────────────────────────────────────────

def highlight_token(context: str, token: str) -> str:
    """Return HTML with every occurrence of the activating token highlighted.

    Tokenizer-decoded tokens carry semantic information in their leading space:
        ' Never' → word-START token (BPE first piece of a word)
        'words'  → CONTINUATION token (BPE middle/end piece of a word like 'Smashwords')
        '.'      → punctuation

    Four match strategies, chosen by token shape:
      1. Continuation token (no leading space, alphanumeric): case-sensitive substring
         match. Catches 'words' inside 'Smashwords', 'AC' inside 'HDAC3', 'nas' inside
         'pnas'. Case-sensitive to reduce false positives (e.g. 'AC' won't hit lowercase 'ac').
      2. Word-start uppercase token (' Cuc', ' Tier', ' Never'): word-boundary on the LEFT,
         then extend through \\w*. Highlights the full 'Cucurbita' for the BPE subtoken ' Cuc'.
      3. Word-start lowercase token (' to', ' smart', ' so'): word-boundary on BOTH sides,
         case-insensitive. Avoids 'to' matching 'tonnes', 'smart' matching 'smartphone'.
      4. Punctuation (e.g. '.', ',', '),'): plain case-insensitive match — no word boundaries.
    """
    import re

    escaped_ctx = context.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if not token:
        return escaped_ctx

    starts_with_space = token.startswith(" ")
    stripped = token.strip()
    if not stripped:
        return escaped_ctx
    escaped_tok = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    first = escaped_tok[0]

    if not first.isalnum():
        # Punctuation token: plain match anywhere.
        patterns = [re.compile(re.escape(escaped_tok), re.IGNORECASE)]
    elif not starts_with_space:
        # Continuation token (BPE middle-of-word piece): case-sensitive substring.
        patterns = [re.compile(re.escape(escaped_tok))]
    elif first.isupper():
        # Word-start uppercase: extend with \w* (catches 'Cuc' → 'Cucurbita').
        patterns = [re.compile(r"\b" + re.escape(escaped_tok) + r"\w*", re.IGNORECASE)]
    else:
        # Word-start lowercase: TRY strict word-bounds first (catches complete words
        # like 'to', 'smart' without over-extending to 'tonnes', 'smartphone'). If
        # nothing matches, FALL BACK to extension (catches BPE subtokens like 'fun'
        # → 'funnels', 'crystal' → 'crystals'/'crystallized'). This way the dashboard
        # never silently fails to highlight when the activating token is a real BPE
        # piece in the middle of a longer word.
        patterns = [
            re.compile(r"\b" + re.escape(escaped_tok) + r"\b", re.IGNORECASE),
            re.compile(r"\b" + re.escape(escaped_tok) + r"\w*", re.IGNORECASE),
        ]

    pattern = next((p for p in patterns if p.search(escaped_ctx)), None)
    if pattern is None:
        return escaped_ctx

    return pattern.sub(
        lambda m: f'<mark style="background:#ffe066;border-radius:3px;padding:0 2px">{m.group(0)}</mark>',
        escaped_ctx,
    )


def activation_histogram(feature_id: int, df: pd.DataFrame):
    """Plot activation value histogram for a feature."""
    rows = df[df["feature_id"] == feature_id]["activation"].dropna()
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.hist(rows, bins=20, color="steelblue", edgecolor="white")
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Count")
    ax.set_title(f"Feature {feature_id} — top-20 activations")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def head_corr_bar(feature_id: int, df_corr: pd.DataFrame):
    """Bar chart of Pearson r vs head index for a feature."""
    sub = df_corr[df_corr["feature_id"] == feature_id].sort_values("head_idx")
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(5, 2.5))
    colours = ["steelblue" if r >= 0 else "salmon" for r in sub["correlation"]]
    ax.bar(sub["head_idx"].astype(str), sub["correlation"], color=colours)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Attention head (layer 12)")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Feature {feature_id} — head correspondence")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ── About tab content ─────────────────────────────────────────────────────────

def _render_about_tab() -> None:
    """Plain-language explainer for the dashboard. Lives in the About tab."""
    st.markdown(
        """
### What is this dashboard?

This is an interactive feature browser for a **sparse autoencoder (SAE)** I trained on
**Gemma-2-2B layer 12**. Each of the 16,384 features below is a single named direction
the SAE discovered in the model's "thinking" at that layer. The dashboard lets you click
through them, see what each one fires on, and verify the analysis behind the write-up.

### What's a sparse autoencoder, in one paragraph?

A transformer like Gemma-2-2B carries a 2,304-dimensional vector through each layer — the
**residual stream**. Those 2,304 numbers describe "what the model is thinking" at each
token, but they're entangled: one number doesn't mean anything on its own. The SAE is a
small neural network that translates those 2,304 entangled numbers into **16,384 sparse
features**, where only ~100 are non-zero at any one token. To pull off the reconstruction
with such a tight bottleneck, each feature has to **specialise** — one feature ends up
representing "second occurrence of a repeated word", another "tokens inside a recipe
title", another "delimiter punctuation in a list", and so on. They become individually
interpretable in a way the raw 2,304 numbers are not.

### What's "induction"?

**Induction heads** (Olsson et al. 2022) are a known mechanism inside transformers: when
the model sees a pattern like `A B ... A` earlier in context, certain attention heads
attend back from the second `A` to right after the first `A` and promote `B` as the next
token. This is the simplest form of in-context learning ("if the model just saw A → B,
predict B after the next A"). For Gemma-2-2B layer 12, attention head 6 is the dominant
induction head; ablating it drops induction-style ICL accuracy by 6.7 percentage points.

### What this dashboard shows

For each of the 16,384 features, three things:

1. **Auto-interpretation label.** Generated by Claude Sonnet from the feature's top-20
   activating text snippets — a human-readable name like *"second occurrence of a repeated
   word or phrase within a short context window"*.
2. **Top activating snippets.** The 20 text snippets that fire this feature hardest, with
   the activating token highlighted. This is the ground truth — the SAE doesn't lie about
   what it activates on.
3. **Quantitative scores.** Induction score (how selectively the feature responds to
   A-B-A induction probes vs. random text), rank, and mean activation. Hover over any
   metric for a definition.

For features in the **induction cluster** (top 100 by induction score), there's also an
ablation curve showing what happens to the model's induction accuracy when we zero out
those features' contributions to the residual stream.

### What we learned

- **The SAE works.** Explained-variance 0.85 (peak 0.893), in DeepMind's published Gemma
  Scope range (0.82–0.90), zero dead features, on 200M training tokens — single 16 GB GPU.
- **The top induction feature is textbook-clean.** F15289 is auto-labelled as "second
  occurrence of a repeated word", and its top-activating snippets are *"Never...Never",
  "Tier...Tier", "so...so", "not...not", "I'm coming...coming"* — exactly the abstraction
  induction heads should encode. Pick it from the sidebar to see for yourself.
- **Ablation is causal.** Zeroing the top-50 induction features drops top-1
  token-copying accuracy from 57.8% → 47.6% — bigger than ablating the best single
  attention head (6.7pp).
- **The signal is distributed, not concentrated.** Earlier with a broken SAE (SAELens, EV
  −0.5) it *looked* like just 5 features mediated all of induction. With proper
  reconstruction it's spread across the top-50. **This is itself a methodological lesson**:
  be suspicious of "ultra-concentrated feature" claims from low-quality SAEs.
- **Library matters.** We hit a SAELens TopK implementation issue that took seven retrains
  to diagnose, then switched to `saprmarks/dictionary_learning` (the same library SAEBench
  uses) and got publication-quality reconstruction on the first attempt. There's a
  separate technical note (`SAELENS_VS_DL.md` in the repo) covering this.

### Trying it for yourself

- **Sidebar dropdown** → pick any feature ID (★ = induction cluster). F15289 is the
  cleanest induction example.
- **Search labels** → type "repeat", "delimiter", "punctuation", etc. to filter by the
  auto-interpretation label.
- **"Only induction cluster" checkbox** → restrict to the top 100 features.

### Where to read more

- **[Code & reproduction](https://github.com/sohumsen/sae-gemma-induction)** — full pipeline,
  scripts, hyperparameters, conversion utilities.
- **[Iteration history](https://github.com/sohumsen/sae-gemma-induction/blob/master/HISTORY.md)** —
  every SAE config v1–v9c we tried and what we learned from each.
- **[SAELens vs dictionary_learning](https://github.com/sohumsen/sae-gemma-induction/blob/master/SAELENS_VS_DL.md)** —
  technical note on why one library plateaus on Gemma residuals and the other doesn't.
- **[W&B project](https://wandb.ai/sohumsen2-ucl/sae-gemma-induction)** — live training curves.
"""
    )


# ── Page layout ────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="SAE Feature Browser — Gemma-2-2B Induction",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 SAE Feature Browser — Gemma-2-2B Layer 12")
    st.markdown(
        "Explore sparse autoencoder features from **Gemma-2-2B** (layer 12 residual stream). "
        "Features are labelled via Claude Haiku / Sonnet auto-interpretation. "
        "The **induction cluster** mediates in-context learning (token-copying). "
        "[GitHub](https://github.com/sohumsen/sae-gemma-induction) · "
        "[W&B runs](https://wandb.ai/sohumsen2-ucl/sae-gemma-induction)"
    )
    st.info(
        "🌟 **New here?** Open the **About** tab for a plain-language explainer, "
        "or jump straight to **F15289** in the sidebar — the cleanest induction "
        "feature: auto-labelled *\"second occurrence of a repeated word\"* with "
        "textbook-clean snippets like \"Never...**Never**\" and \"Tier...**Tier**\".",
        icon="ℹ️",
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    try:
        df_snippets = load_snippets()
        labels = load_labels()
        candidates = load_candidates()
        df_scores = load_scores()
        df_corr = load_head_corr() if HEAD_CORR_PATH.exists() else pd.DataFrame()
        ablation = load_ablation() if ABLATION_PATH.exists() else {}
    except Exception as exc:
        st.error(f"Failed to load pre-computed data: {exc}\n\nRun the analysis pipeline first.")
        return

    n_features = int(df_scores["feature_id"].max()) + 1
    candidate_set = set(candidates)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Feature selection")
        only_induction = st.checkbox("Only induction cluster", value=False)
        search_query = st.text_input("Search labels", placeholder="e.g. copy, repeat, induction")

        # Build feature list for selectbox
        if only_induction:
            feature_pool = candidates[:200]
        else:
            feature_pool = list(range(n_features))

        # Filter by label search
        if search_query:
            feature_pool = [
                fid for fid in feature_pool
                if search_query.lower() in labels.get(fid, "").lower()
            ]

        if not feature_pool:
            st.warning("No features match the filter.")
            return

        # Default to top induction feature (F15289 for v9c) when no filter applied,
        # otherwise default to the first item in the filtered list.
        default_fid = candidates[0] if candidates else (feature_pool[0] if feature_pool else 0)
        try:
            default_idx = feature_pool.index(default_fid) if default_fid in feature_pool else 0
        except Exception:
            default_idx = 0
        selected_fid = st.selectbox(
            "Feature ID",
            feature_pool,
            index=default_idx,
            format_func=lambda fid: f"F{fid}" + (" ★" if fid in candidate_set else ""),
        )

        st.divider()
        st.metric(
            "Total features",
            f"{n_features:,}",
            help="Number of features (named directions) the SAE was trained to discover. We use 16,384 — 7× the residual-stream dimension (2,304), so each feature can be highly specialised.",
        )
        st.metric(
            "Induction cluster size",
            len(candidates),
            help="Number of features whose 'induction score' (activation on induction probes minus activation on control sequences) was high enough to flag them as candidate induction features. Top-50 of these were tested in the ablation experiment.",
        )
        if ablation:
            st.metric(
                "Baseline ICL accuracy",
                f"{ablation.get('baseline_accuracy', 0):.1%}",
                help="Top-1 accuracy on 2,000 induction probes ([prefix] A B ... A → predict B) when the SAE is in the loop reconstructing the layer-12 residual stream, but no features are ablated. The 'ablation experiment' chart shows how this drops as we zero out the top induction features.",
            )

    # ── Tabs: Feature Browser + About ──────────────────────────────────────────
    tab_browser, tab_about = st.tabs(["🔍 Feature Browser", "ℹ️ About"])

    with tab_browser:
        fid = selected_fid
        label = labels.get(fid, "*(not yet labelled)*")
        is_induction = fid in candidate_set

        col_title, col_badge = st.columns([5, 1])
        with col_title:
            st.subheader(f"Feature {fid}")
        with col_badge:
            if is_induction:
                st.success("★ Induction cluster")

        st.markdown(f"**Auto-interpretation:** {label}")

        # Score info
        score_row = df_scores[df_scores["feature_id"] == fid]
        if not score_row.empty:
            row = score_row.iloc[0]
            rank_val = int(row["rank"])
            induction_score = float(row["induction_score"])
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Induction score",
                f"{induction_score:.4f}",
                help="How much MORE this feature activates on induction probes than on random controls. Computed as mean_activation_on_induction_probes − mean_activation_on_controls (at the final-token position). Larger = more specifically about induction. The top feature scores ~2.3; most features score near 0.",
            )
            c2.metric(
                "Rank",
                f"#{rank_val}",
                help="Where this feature sits when all 16,384 features are sorted by induction score (descending). #0 = most induction-specific. The top ~100 form the 'induction cluster'.",
            )
            c3.metric(
                "Mean activation (probes)",
                f"{row['induction_mean']:.4f}",
                help="Average activation of this feature across the 2,000 induction probes, measured at the final-token position (the second occurrence of 'A' in the [prefix] A B ... A probe). Higher = feature fires hard on the kind of pattern induction heads track.",
            )

        st.divider()

        # Top snippets
        st.subheader("Top activating snippets")
        st.caption(
            "All occurrences of the activating token are highlighted in yellow. "
            "For 'second occurrence' / 'repeated' features, you should see two matches in most snippets — "
            "the feature fires on one of them (typically the one implied by the auto-interpretation label)."
        )
        snippets = df_snippets[df_snippets["feature_id"] == fid].sort_values("rank")
        if snippets.empty:
            st.info("No snippets available for this feature.")
        else:
            for _, srow in snippets.iterrows():
                ctx = str(srow.get("context", ""))
                tok = str(srow.get("token", ""))
                act = float(srow.get("activation", 0))
                html = highlight_token(ctx, tok)
                st.markdown(
                    f'<div style="border-left:3px solid #ccc;padding:4px 10px;margin:4px 0;'
                    f'font-family:monospace;font-size:13px">'
                    f'<span style="color:#888;font-size:11px">act={act:.3f}</span><br>'
                    f"{html}</div>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # Charts side by side
        col_hist, col_heads = st.columns(2)
        with col_hist:
            st.subheader("Activation distribution")
            fig_hist = activation_histogram(fid, df_snippets)
            st.image(_fig_to_image(fig_hist), width="stretch")

        with col_heads:
            if not df_corr.empty:
                st.subheader("Head correspondence")
                fig_head = head_corr_bar(fid, df_corr)
                if fig_head:
                    st.image(_fig_to_image(fig_head), width="stretch")
                else:
                    st.info("No head correspondence data for this feature.")

        # Ablation summary (if induction feature)
        if is_induction and ablation:
            st.divider()
            st.subheader("Ablation experiment")
            fa = ablation.get("feature_ablation", {})
            if fa:
                ns = fa.get("N", [])
                accs = fa.get("accuracy", [])
                drops = fa.get("accuracy_drop", [])
                ci_lo = fa.get("ci_low", [])
                ci_hi = fa.get("ci_high", [])
                baseline = ablation.get("baseline_accuracy", None)

                fig_ab, ax = plt.subplots(figsize=(5, 3))
                if baseline is not None:
                    ax.axhline(baseline, color="grey", linestyle="--",
                               label=f"Baseline ({baseline:.1%})", alpha=0.7)
                ax.plot(ns, accs, "o-", color="steelblue", label="Ablated")
                if ci_lo and ci_hi:
                    ax.fill_between(ns, ci_lo, ci_hi, alpha=0.2, color="steelblue")
                ax.set_xlabel("N top features ablated")
                ax.set_ylabel("ICL top-1 accuracy")
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
                ax.legend(fontsize=9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.image(_fig_to_image(fig_ab), width="stretch")

                st.markdown(
                    f"Ablating the top **{max(ns)}** induction features drops ICL accuracy "
                    f"from **{baseline:.1%}** to **{min(accs):.1%}** "
                    f"(−{baseline - min(accs):.1%})."
                )

    with tab_about:
        _render_about_tab()


if __name__ == "__main__":
    main()
