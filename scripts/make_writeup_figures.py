"""Generate all figures referenced from WRITEUP.md.

Reads everything from results/*.json and results/*.parquet; writes PNGs to
results/figures/. Idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def load_json(name: str) -> dict:
    with (RESULTS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Ablation curve + head ablation + random control band
# ---------------------------------------------------------------------------
def fig_ablation_curve() -> None:
    abl = load_json("ablation_results.json")
    rnd = load_json("random_feature_ablation.json")

    baseline = abl["baseline_accuracy"] * 100
    Ns = abl["feature_ablation"]["N"]
    acc = [a * 100 for a in abl["feature_ablation"]["accuracy"]]
    ci_lo = [a * 100 for a in abl["feature_ablation"]["ci_low"]]
    ci_hi = [a * 100 for a in abl["feature_ablation"]["ci_high"]]
    heads = abl["head_ablation"]["heads"]
    head_acc = [a * 100 for a in abl["head_ablation"]["accuracy"]]

    rnd_mean_acc = rnd["random_mean_acc"] * 100
    rnd_std_drop = rnd["random_std_drop"] * 100

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: feature ablation curve with random control band
    ax_l.axhline(baseline, color="grey", linestyle="--", label=f"Baseline ({baseline:.1f}%)")
    ax_l.axhspan(
        rnd_mean_acc - rnd_std_drop,
        rnd_mean_acc + rnd_std_drop,
        color="tab:orange",
        alpha=0.25,
        label=f"Random 50 features (mean ±1σ across 5 seeds)",
    )
    ax_l.axhline(rnd_mean_acc, color="tab:orange", linestyle=":", linewidth=1)
    ax_l.fill_between(Ns, ci_lo, ci_hi, color="tab:blue", alpha=0.2)
    ax_l.plot(Ns, acc, "o-", color="tab:blue", label="Top-N induction features ablated")
    ax_l.set_xlabel("Number of top induction features ablated")
    ax_l.set_ylabel("ICL top-1 accuracy (%)")
    ax_l.set_title("SAE feature ablation vs random-feature control")
    ax_l.set_ylim(40, 65)
    ax_l.set_xticks(Ns)
    ax_l.legend(loc="lower left", fontsize=9)
    ax_l.grid(axis="y", alpha=0.3)

    # Right: head ablation bars, highlight head 6
    colors = ["tab:red" if h == 6 else "lightcoral" for h in heads]
    ax_r.bar(heads, head_acc, color=colors, edgecolor="black", linewidth=0.5)
    ax_r.axhline(baseline, color="grey", linestyle="--", label=f"Baseline ({baseline:.1f}%)")
    ax_r.set_xlabel("Layer-12 attention head")
    ax_r.set_ylabel("ICL top-1 accuracy (%)")
    ax_r.set_title("Head ablation (Olsson et al. baseline)")
    ax_r.set_xticks(heads)
    ax_r.set_ylim(40, 65)
    ax_r.legend(loc="lower left", fontsize=9)
    ax_r.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS / "ablation_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote ablation_curve.png")


# ---------------------------------------------------------------------------
# 2. Activation patching: paired zero vs mean ablation
# ---------------------------------------------------------------------------
def fig_activation_patching() -> None:
    zero = load_json("activation_patching.json")
    mean = load_json("activation_patching_mean.json")
    feat_ids_all = zero["target_features"]
    # Keep only the two features actually discussed in the new draft:
    # F15289 (the headline) and F14740 (the "head 6 still matters" feature).
    keep = [15289, 14740]
    keep_idx = [feat_ids_all.index(f) for f in keep]
    feat_ids = keep
    subtitles = {
        15289: "F15289 — rank-1 induction feature",
        14740: "F14740 — 'tokens in repeated/parallel structures'",
    }

    n_heads = 8
    heads = list(range(n_heads))

    def gather(payload: dict) -> np.ndarray:
        arr = np.zeros((len(feat_ids), n_heads))
        for h in heads:
            full = payload["head_results"][str(h)]["reduction_pct"]
            arr[:, h] = [full[i] for i in keep_idx]
        return arr

    z = gather(zero)
    m = gather(mean)

    fig, axes = plt.subplots(1, len(feat_ids), figsize=(6.5 * len(feat_ids), 4.6), sharey=False)
    width = 0.4
    x = np.arange(n_heads)

    for i, ax in enumerate(axes):
        ax.bar(x - width / 2, z[i], width, label="Zero-ablation (OOD)", color="tab:blue", edgecolor="black", linewidth=0.4)
        ax.bar(x + width / 2, m[i], width, label="Mean-ablation (in-distribution)", color="tab:orange", edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(subtitles[feat_ids[i]], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{h}" for h in heads])
        ax.set_xlabel("Layer-12 attention head")
        ax.set_ylabel("% reduction in feature activation when head is ablated")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)

        # Inline numeric labels above/below the *zero-ablation* bar for the
        # head with the largest absolute effect — no arrows, no overlap.
        max_h = int(np.argmax(np.abs(z[i])))
        zv = z[i][max_h]
        mv = m[i][max_h]
        ax.text(
            x[max_h] - width / 2,
            zv + (3 if zv > 0 else -3),
            f"{zv:+.0f}%",
            ha="center",
            va="bottom" if zv > 0 else "top",
            fontsize=9,
            fontweight="bold",
            color="tab:blue",
        )
        ax.text(
            x[max_h] + width / 2,
            mv + (3 if mv > 0 else -3),
            f"{mv:+.0f}%",
            ha="center",
            va="bottom" if mv > 0 else "top",
            fontsize=9,
            fontweight="bold",
            color="tab:orange",
        )

    fig.suptitle(
        "Zero vs mean ablation flips the 'which head matters' answer",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "activation_patching.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote activation_patching.png")


# ---------------------------------------------------------------------------
# 3. Induction-score distribution across all 16,384 features
# ---------------------------------------------------------------------------
def fig_induction_score_distribution() -> None:
    df = pd.read_parquet(RESULTS / "induction_feature_scores.parquet")
    scores = df["induction_score"].to_numpy()

    target_ids = [15289, 11606, 14740, 7467]
    target_scores = {fid: df.loc[df.feature_id == fid, "induction_score"].iloc[0] for fid in target_ids}

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.6))
    bins = np.linspace(scores.min(), scores.max(), 120)
    ax.hist(scores, bins=bins, color="lightgrey", edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("Induction score (mean act on induction probes − mean act on control)")
    ax.set_ylabel("Number of SAE features (log scale)")
    ax.set_title("Distribution of induction scores across all 16,384 v9c SAE features")

    colors = ["tab:red", "tab:orange", "tab:green", "tab:purple"]
    ymax = ax.get_ylim()[1]
    for (fid, sc), color in zip(target_scores.items(), colors):
        ax.axvline(sc, color=color, linewidth=1.5, linestyle="--")
        ax.text(
            sc,
            ymax * 0.4,
            f"F{fid}\n({sc:.2f})",
            color=color,
            fontsize=9,
            rotation=90,
            va="top",
            ha="right",
        )

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "induction_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote induction_score_distribution.png")


# ---------------------------------------------------------------------------
# 4. Multi-seed replication
# ---------------------------------------------------------------------------
def fig_multi_seed() -> None:
    seed43 = load_json("seed43_replication.json")
    seed44 = load_json("seed44_replication.json")
    # v9c is seed 42; numbers from WRITEUP table
    rows = [
        ("v9c (seed 42)", "F15289", 2.31, 0.79, 10.1),
        ("seed 43", f"F{seed43['top_feature_id']}", seed43["top_induction_score"], seed43["top20_mean_score"], seed43["drop_pp"]),
        ("seed 44", f"F{seed44['top_feature_id']}", seed44["top_induction_score"], seed44["top20_mean_score"], seed44["drop_pp"]),
    ]

    labels = [r[0] for r in rows]
    top_scores = [r[2] for r in rows]
    top20_means = [r[3] for r in rows]
    drops = [r[4] for r in rows]
    top_feat_labels = [r[1] for r in rows]

    fig, (ax_l, ax_m, ax_r) = plt.subplots(1, 3, figsize=(13, 4.2))

    x = np.arange(len(labels))

    # Panel A: top induction score
    ax_l.bar(x, top_scores, color="tab:blue", edgecolor="black")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels)
    ax_l.set_ylabel("Top induction score")
    ax_l.set_title("Rank-1 induction-score feature\n(IDs differ across seeds)")
    for xi, sc, lab in zip(x, top_scores, top_feat_labels):
        ax_l.text(xi, sc + 0.05, f"{lab}\n{sc:.2f}", ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, max(top_scores) * 1.3)
    ax_l.grid(axis="y", alpha=0.3)

    # Panel B: top-20 mean score
    ax_m.bar(x, top20_means, color="tab:green", edgecolor="black")
    mean20 = float(np.mean(top20_means))
    ax_m.axhline(mean20, color="black", linestyle="--", linewidth=1, label=f"Mean = {mean20:.2f}")
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(labels)
    ax_m.set_ylabel("Mean induction score of top-20 features")
    ax_m.set_title("Top-20 mean induction score\n(replicates within ±0.05)")
    ax_m.set_ylim(0, max(top20_means) * 1.3)
    for xi, sc in zip(x, top20_means):
        ax_m.text(xi, sc + 0.02, f"{sc:.2f}", ha="center", va="bottom", fontsize=9)
    ax_m.legend(loc="lower right", fontsize=9)
    ax_m.grid(axis="y", alpha=0.3)

    # Panel C: top-50 ablation drop
    ax_r.bar(x, drops, color="tab:red", edgecolor="black")
    mean_drop = float(np.mean(drops))
    std_drop = float(np.std(drops, ddof=1))
    ax_r.axhline(mean_drop, color="black", linestyle="--", linewidth=1, label=f"Mean = {mean_drop:.1f} ± {std_drop:.1f}pp")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels)
    ax_r.set_ylabel("Top-50 ablation ICL drop (pp)")
    ax_r.set_title("Top-50 ablation effect on ICL\n(replicates across seeds)")
    ax_r.set_ylim(0, max(drops) * 1.3)
    for xi, sc in zip(x, drops):
        ax_r.text(xi, sc + 0.3, f"{sc:.1f}pp", ha="center", va="bottom", fontsize=9)
    ax_r.legend(loc="lower right", fontsize=9)
    ax_r.grid(axis="y", alpha=0.3)

    fig.suptitle("Multi-seed replication of v9c SAE (seeds 42 / 43 / 44, identical training config)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "multi_seed_replication.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote multi_seed_replication.png")


# ---------------------------------------------------------------------------
# 5. Cross-SAE: v9c vs Gemma Scope
# ---------------------------------------------------------------------------
def fig_cross_sae() -> None:
    # Keep only the two score comparisons (selectivity is the cross-SAE claim;
    # raw activations are detail that belongs in the text).
    metrics = [
        ("Top induction score", 2.31, 1.72),
        ("Top-20 mean induction score", 0.79, 0.78),
    ]
    labels = [m[0] for m in metrics]
    v9c = [m[1] for m in metrics]
    scope = [m[2] for m in metrics]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.4))
    b1 = ax.bar(x - width / 2, v9c, width, label="v9c (mine, dictionary_learning)", color="tab:blue", edgecolor="black")
    b2 = ax.bar(x + width / 2, scope, width, label="Gemma Scope (DeepMind, SAEBench)", color="tab:gray", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Induction score")
    ax.set_title("Cross-SAE: same selectivity, different SAEs")
    for b, val in zip(b1, v9c):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.04, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    for b, val in zip(b2, scope):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.04, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(v9c) * 1.25)
    fig.tight_layout()
    fig.savefig(FIGS / "cross_sae_gemma_scope.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote cross_sae_gemma_scope.png")


# ---------------------------------------------------------------------------
# 6. MMLU negative finding
# ---------------------------------------------------------------------------
def fig_mmlu_negative() -> None:
    mmlu = load_json("mmlu_feature_activations.json")
    targets = mmlu["target_features"]
    idx_15289 = targets.index(15289)
    few_shot = mmlu["few_shot_mean"][idx_15289]
    shuffled = mmlu["shuffled_mean"][idx_15289]

    # Synthetic baseline from induction_feature_scores.parquet:
    df = pd.read_parquet(RESULTS / "induction_feature_scores.parquet")
    synth = df.loc[df.feature_id == 15289, "induction_mean"].iloc[0]
    synth_ctrl = df.loc[df.feature_id == 15289, "control_mean"].iloc[0]

    labels = [
        "Synthetic A-B-A\ninduction probe\n(final pos)",
        "Synthetic\ncontrol\n(final pos)",
        "MMLU 4-shot\nreal answers\n(final pos)",
        "MMLU 4-shot\nshuffled answers\n(final pos)",
    ]
    values = [synth, synth_ctrl, few_shot, shuffled]
    colors = ["tab:blue", "lightblue", "tab:red", "lightcoral"]

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.6))
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("F15289 mean activation at final position")
    ax.set_title(
        "F15289 fires on synthetic token-copying induction, not on natural few-shot ICL\n"
        f"(MMLU n={mmlu['n_questions']} questions, {mmlu['n_shots']}-shot)"
    )
    ax.set_ylim(0, max(values) * 1.3 + 0.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "mmlu_negative_finding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote mmlu_negative_finding.png")


# ---------------------------------------------------------------------------
# 7. Top-activating snippets with token highlights — F15289 and F14740
# ---------------------------------------------------------------------------
def fig_top_feature_snippets() -> None:
    """Two-panel: F15289 (second occurrence of repeated token) and F14740
    (tokens in parallel/repeated structures). Text is laid out via HPacker so
    segment widths come from the actual rendered glyphs — no overlap from
    bold-vs-normal width mismatches."""
    import re
    from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

    df = pd.read_parquet(RESULTS / "top_snippets.parquet")

    def extract_rows(feature_id: int, n_rows: int) -> list:
        sub = df[df.feature_id == feature_id].nsmallest(20, "rank").reset_index(drop=True)
        rows = []
        for _, row in sub.iterrows():
            token_clean = str(row["token"]).strip()
            if not token_clean:
                continue
            context = str(row["context"]).replace("\n", " · ")
            act = float(row["activation"])
            pattern = re.compile(r"\b" + re.escape(token_clean) + r"\b", re.IGNORECASE)
            matches = list(pattern.finditer(context))
            if len(matches) < 2:
                continue
            s0, e0 = matches[0].span()
            s1, e1 = matches[1].span()
            lo = max(0, s0 - 24)
            hi = min(len(context), e1 + 32)
            prefix = ("… " if lo > 0 else "") + context[lo:s0]
            first = context[s0:e0]
            middle = context[e0:s1]
            second = context[s1:e1]
            suffix = context[e1:hi] + (" …" if hi < len(context) else "")
            # Trim if too long
            max_chars = 100
            total = len(prefix) + len(first) + len(middle) + len(second) + len(suffix)
            if total > max_chars:
                overshoot = total - max_chars
                trim = overshoot // 2 + 1
                if len(prefix) > trim + 3:
                    prefix = "… " + prefix.lstrip("… ")[trim:]
                if len(suffix) > trim + 3:
                    suffix = suffix.rstrip(" …")[:-trim] + " …"
            rows.append((act, prefix, first, middle, second, suffix))
            if len(rows) >= n_rows:
                break
        return rows

    panels = [
        (
            15289,
            "F15289 — fires on the SECOND occurrence of a repeated token",
            extract_rows(15289, 6),
        ),
        (
            14740,
            "F14740 — fires on tokens in repeated / parallel structures",
            extract_rows(14740, 6),
        ),
    ]

    font_kwargs = {"family": "DejaVu Sans Mono", "size": 10}

    NBSP = " "

    def make_line(prefix, first, middle, second, suffix):
        boxes = []
        # NBSP in non-bold segments so HPacker preserves boundary whitespace
        # (TextArea otherwise strips trailing/leading regular spaces).
        for text, color, weight in [
            (prefix.replace(" ", NBSP), "#555555", "normal"),
            (first, "#222222", "bold"),
            (middle.replace(" ", NBSP), "#555555", "normal"),
            (second, "#c00000", "bold"),
            (suffix.replace(" ", NBSP), "#555555", "normal"),
        ]:
            if not text:
                continue
            boxes.append(
                TextArea(
                    text,
                    textprops={"color": color, "fontweight": weight, **font_kwargs},
                )
            )
        return HPacker(children=boxes, align="baseline", pad=0, sep=0)

    n_panels = len(panels)
    max_rows = max(len(p[2]) for p in panels)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(13, 0.55 * max_rows * n_panels + 1.6), squeeze=False
    )
    axes = axes.ravel()

    for ax, (fid, title, rows) in zip(axes, panels):
        n = len(rows)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n + 1)
        ax.invert_yaxis()
        ax.axis("off")
        ax.text(0.04, 0.35, "Act.", fontsize=10, fontweight="bold", ha="center")
        ax.text(0.09, 0.35, title, fontsize=11, fontweight="bold", ha="left")
        for i, (act, prefix, first, middle, second, suffix) in enumerate(rows):
            y = i + 1.0
            ax.text(0.04, y, f"{act:.1f}", fontsize=11, ha="center", va="center", fontweight="bold")
            packer = make_line(prefix, first, middle, second, suffix)
            ab = AnnotationBbox(
                packer,
                xy=(0.09, y),
                xycoords=("axes fraction", "data"),
                box_alignment=(0.0, 0.5),
                frameon=False,
                pad=0,
            )
            ax.add_artist(ab)

    fig.suptitle(
        "Top-activating C4 snippets — first occurrence in dark-bold, activating token in red-bold",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGS / "top_feature_snippets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote top_feature_snippets.png")


if __name__ == "__main__":
    fig_ablation_curve()
    fig_activation_patching()
    fig_induction_score_distribution()
    fig_multi_seed()
    fig_cross_sae()
    fig_mmlu_negative()
    fig_top_feature_snippets()
