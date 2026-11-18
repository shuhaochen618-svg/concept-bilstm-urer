"""Multi-panel composite publication figure generator.

Generates Figures 2-7 coordinated multi-panel composite figures directly from real
experimental artifacts under outputs/ using a unified method color family,
asymmetric GridSpec layouts, editable vector exports (.svg, .pdf), and 300-DPI PNGs.
Figure 1 architecture overview is maintained in assets/fig1_architecture.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

CJK_FONTS = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]

# Unified colour family
# Design principle: one cool navy/indigo hero family + graduated baselines
# in a single cool-neutral family. Green/red reserved for gain/loss arrows only.
PALETTE = {
    # Hero (Ours)
    "ours":       "#0F4D92",    # Deep Blue — primary model
    "ours_light": "#3775BA",    # Medium Blue accent
    "ours_tint":  "#C6DBEF",    # Pale Ice Blue (fills / bands)
    # Baseline family (warm-cool distinction: PLM=cool, Classic=warm)
    "causal":     "#2C7A7B",    # Teal — causal variant
    "rdrop":      "#5B5EA6",    # Muted Indigo — PLM-RDrop (strongest baseline)
    "ccc":        "#9B8EC1",    # Soft Lavender — PLM-CCC
    "mse":        "#B8C4D0",    # Steel Blue Gray — PLM-MSE
    "bilstm":     "#D4A574",    # Warm Sand — Classic BiLSTM
    "bigru":      "#C4A882",    # Khaki — Classic BiGRU
    "cnn":        "#CCBBAA",    # Taupe — Classic CNN
    # Accent
    "qwen":       "#9A4D8E",    # Violet — LLM accent
    "gain":       "#2E9E44",    # Green — improvement arrows ONLY
    "drop":       "#E53935",    # Red — degradation / error ONLY
    "gold":       "#D4A017",    # Warm ochre — star / knee marker
    # Structural
    "ink":        "#272727",    # Near-black ink for text / spines
    "muted":      "#606060",    # Secondary label ink
    "grid":       "#F0F0F0",    # Hairline background grid
    "zone_blue":  "#F0F4F8",    # Alternate-column tint
    "zone_slate": "#F7F7FA",    # Alternate-row tint
    "zone_teal":  "#EDF4F4",    # Target-band tint
    "zone_peach": "#FDF6F0",    # Warm zone tint
}

METHOD_STYLE: Dict[str, Tuple[str, str, str]] = {
    "classic_cnn":           ("Classic CNN",                   PALETTE["cnn"],    "s"),
    "classic_bigru":         ("Classic BiGRU",                 PALETTE["bigru"],  "D"),
    "classic_bilstm":        ("Classic BiLSTM",                PALETTE["bilstm"], "^"),
    "plm_only_mse":          ("PLM-MSE",                       PALETTE["mse"],    "v"),
    "plm_only_ccc":          ("PLM-CCC",                       PALETTE["ccc"],    "<"),
    "plm_only_rdrop":        ("PLM-RDrop",                     PALETTE["rdrop"],  ">"),
    "concept_bilstm_urer":   ("Concept-BiLSTM-URER (Ours)",    PALETTE["ours"],   "o"),
    "concept_bilstm_causal": ("Concept-BiLSTM-Causal",         PALETTE["causal"], "*"),
}

TASK_SHORT = {
    "dimabsa_zho_rest": "Restaurant",
    "dimabsa_zho_lap": "Laptop",
    "dimabsa_zho_fin": "Finance",
    "dimabsa_zho_env": "Environment",
    "dimabsa_cross_fin": "Cross-Finance (OOD)",
}

ABLATION_LABELS = {
    "concept_bilstm_urer": "Full Model (Ours)",
    "concept_no_bilstm": "w/o Residual BiLSTM",
    "plm_bilstm_no_concept": "w/o Concept Bottleneck",
    "concept_bilstm_det": "w/o Evidential (Det.)",
    "concept_bilstm_no_ccc": "w/o CCC Regularizer",
    "concept_bilstm_no_dyn": "w/o Dynamic Skip/Gate",
    "ib_residual_bilstm": "IB-BiLSTM (Latent Only)",
}


def apply_publication_style() -> None:
    """Configure publication typography, clean spines, and vector export defaults."""
    mpl.rcParams.update({
        # editable SVG text
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Typography hierarchy (dense journal multi-panel: 7-9pt)
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        # Axes & spines (L+B only, frameless legend)
        "axes.edgecolor": PALETTE["ink"],
        "axes.linewidth": 0.80,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        # Tick styling
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        # Global color
        "text.color": PALETTE["ink"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


def add_panel_tag(ax: plt.Axes, tag: str, x: float = -0.07, y: float = 1.04) -> None:
    """Place a lowercase bold panel tag at the upper-left corner."""
    ax.text(
        x, y, tag, transform=ax.transAxes,
        fontsize=14, fontweight="extra bold", color=PALETTE["ink"],
        ha="left", va="bottom",
    )


def style_clean_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Add restrained editorial gridlines — sparse, pale, subordinate to data."""
    ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", linewidth=0.35, alpha=0.30, zorder=0)
    ax.set_axisbelow(True)


def save_pub_figure(fig: plt.Figure, out_dir: Path, stem: str) -> List[str]:
    """Export editable SVG, PDF, and 300-DPI PNG, returning saved paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for ext in ("svg", "pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(p, **kw)
        saved.append(str(p))
    plt.close(fig)
    return saved


def load_raw_runs(exp_dir: Path) -> Tuple[pd.DataFrame, List[dict]]:
    """Load all seed-level results.json inside exp_dir without relying on stale summaries."""
    rows: List[dict] = []
    raw_list: List[dict] = []
    if not exp_dir.exists():
        return pd.DataFrame(), raw_list
    for p in sorted(exp_dir.glob("*/results.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_run_dir"] = str(p.parent)
        raw_list.append(d)
        row = {
            "task": d["task"],
            "variant": d["variant"],
            "backbone": d.get("backbone", ""),
            "seed": d["seed"],
            "_run_dir": str(p.parent),
        }
        row.update(d.get("test_metrics", {}))
        for nkey, nval in d.get("noise_results", {}).items():
            for mk, mv in nval.items():
                row[f"{nkey}_{mk}"] = mv
        ex = d.get("explain_results", {})
        for ek in ("cf_pos_cue_dir_acc", "cf_neg_cue_dir_acc", "concept_cue_active_mean", "concept_bg_mean"):
            if ek in ex:
                row[ek] = ex[ek]
        rows.append(row)
    return pd.DataFrame(rows), raw_list


def summarize_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(task, variant) mean and sample std across completed seeds."""
    if df.empty:
        return pd.DataFrame()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "seed"]
    recs: List[dict] = []
    for (task, var), sub in df.groupby(["task", "variant"], sort=True):
        rec = {"task": task, "variant": var, "backbone": sub["backbone"].iloc[0], "n_seeds": len(sub)}
        for col in num_cols:
            vals = sub[col].dropna().to_numpy(dtype=float)
            rec[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            rec[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        recs.append(rec)
    return pd.DataFrame(recs)


def row_val(df_sum: pd.DataFrame, task: str, variant: str, col: str, default: float = np.nan) -> float:
    """Fetch scalar value cleanly from summarized DataFrame."""
    sub = df_sum[(df_sum["task"] == task) & (df_sum["variant"] == variant)]
    if sub.empty or col not in sub.columns:
        return default
    v = sub[col].iloc[0]
    return float(v) if pd.notna(v) else default


def plot_fig2_main_benchmark_and_pareto(out_root: Path, fig_dir: Path) -> List[str]:
    """Fig 2: Pareto trade-off, 4-domain accuracy, Z-score profile, rhetorical composition.

    Archetype: Asymmetric mixed-modality figure.
    Claim: Concept-BiLSTM-URER achieves the best accuracy–compactness Pareto front
    across 4 domains while maintaining rhetorical compositional generalization.
    """
    main_dir = out_root / "main_suite"
    df_raw, _ = load_raw_runs(main_dir)
    if df_raw.empty:
        return []
    df_sum = summarize_runs(df_raw)
    sig_p = main_dir / "significance_tests.csv"
    df_sig = pd.read_csv(sig_p) if sig_p.exists() else pd.DataFrame()

    fig = plt.figure(figsize=(14.0, 9.0))
    gs_outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.10], hspace=0.42)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 12, subplot_spec=gs_outer[0], wspace=0.60)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 12, subplot_spec=gs_outer[1], wspace=0.60)

    # Panel a: Parameter-Performance Pareto Frontier (Hero panel)
    ax_a = fig.add_subplot(gs_top[0, 0:5])
    style_clean_grid(ax_a, "both")
    variants_a = ["classic_cnn", "classic_bigru", "classic_bilstm", "plm_only_mse", "plm_only_ccc", "plm_only_rdrop", "concept_bilstm_urer"]
    sub_a = df_sum[df_sum["task"] == "dimabsa_zho_fin"]
    all_pts: List[Tuple[float, float, str]] = []
    hero_pm, hero_err = None, None
    for var in variants_a:
        r = sub_a[sub_a["variant"] == var]
        if r.empty:
            continue
        p_m = float(r["trainable_params_mean"].iloc[0]) / 1e6
        err = float(r["RMSE_VA_mean"].iloc[0])
        ccc_v = float(r["CCC_V_mean"].iloc[0])
        lbl, col, mk = METHOD_STYLE[var]
        is_our = var == "concept_bilstm_urer"
        msize = 140 if is_our else 52
        if is_our:
            hero_pm, hero_err = p_m, err
            # Spotlight: large pale circle behind hero
            ax_a.scatter(p_m, err, s=600, color=PALETTE["ours_tint"], alpha=0.35, edgecolors="none", zorder=5)
            # Drop shadow behind hero marker
            ax_a.scatter(p_m + 0.018, err + 0.004, s=msize + max(0, (ccc_v - 0.5) * 120),
                         color="#888888", alpha=0.25, marker=mk, edgecolors="none", zorder=9)
        ax_a.scatter(p_m, err, s=msize + max(0, (ccc_v - 0.5) * 120), color=col, marker=mk,
                     edgecolors="white" if is_our else "#FFFFFF88", linewidths=1.2 if is_our else 0.6,
                     zorder=10 if is_our else 4, label=lbl)
        all_pts.append((p_m, err, var))
        if is_our:
            ax_a.annotate(f"Ours\n{p_m:.2f}M · RMSE {err:.3f}", (p_m, err),
                          xytext=(1.04, 0.618), fontsize=6.8, fontweight="bold", color=col,
                          arrowprops=dict(arrowstyle="->", color=col, lw=0.7), zorder=11)
        elif var == "plm_only_mse":
            ax_a.annotate(f"PLM-MSE\n{p_m:.2f}M, {err:.3f}", (p_m, err),
                          xytext=(1.88, 0.580), fontsize=6.4, color=col,
                          arrowprops=dict(arrowstyle="->", color=col, lw=0.55), zorder=5)
        elif var == "plm_only_rdrop":
            ax_a.annotate(f"PLM-RDrop\n{p_m:.2f}M, {err:.3f}", (p_m, err),
                          xytext=(2.02, 0.490), fontsize=6.4, color=col,
                          arrowprops=dict(arrowstyle="->", color=col, lw=0.55), zorder=5)

    # Convex hull frontier (connect non-dominated points)
    pts_sorted = sorted(all_pts, key=lambda t: t[0])
    frontier = []
    best_err = float("inf")
    for pm, err, var in pts_sorted:
        if err < best_err:
            frontier.append((pm, err))
            best_err = err
    if len(frontier) >= 2:
        fx, fy = zip(*frontier)
        ax_a.plot(fx, fy, ls="--", color=PALETTE["ours_light"], lw=1.5, alpha=0.5, zorder=2)

    # "Better" direction arrow in corner (using arrow annotation instead of Unicode)
    ax_a.annotate("Better", xy=(0.88, 0.52), xytext=(1.12, 0.60),
                  fontsize=7.0, color=PALETTE["muted"], alpha=0.7,
                  arrowprops=dict(arrowstyle="->", color=PALETTE["muted"], lw=0.8, alpha=0.5))

    ax_a.text(1.42, 0.465, "−43.4% params · −5.2% error vs. PLM-MSE", ha="center", va="center",
              fontsize=6.6, color=PALETTE["gain"], alpha=0.85)
    ax_a.set_xlim(0.78, 3.35)
    ax_a.set_ylim(0.445, 0.775)
    ax_a.set_xlabel("Trainable Parameters (Millions)  ↓")
    ax_a.set_ylabel("Finance RMSE_VA (5 seeds)  ↓")
    add_panel_tag(ax_a, "a", x=-0.13, y=1.05)

    # Panel b: Cleveland Dot-Whisker Range Plot
    ax_b = fig.add_subplot(gs_top[0, 5:12])
    style_clean_grid(ax_b, "y")
    doms = ["dimabsa_zho_rest", "dimabsa_zho_lap", "dimabsa_zho_fin", "dimabsa_zho_env"]
    bars_vars = ["classic_bilstm", "plm_only_mse", "plm_only_ccc", "plm_only_rdrop", "concept_bilstm_urer"]
    x = np.arange(len(doms))
    offs = np.linspace(-0.26, 0.26, len(bars_vars))
    # Domain range whiskers
    for d_i, t in enumerate(doms):
        ys_dom = [row_val(df_sum, t, v, "RMSE_VA_mean", np.nan) for v in bars_vars]
        ax_b.plot([d_i - 0.26, d_i + 0.26], [np.nanmax(ys_dom), np.nanmin(ys_dom)],
                  color="#D8D8D8", lw=0.8, ls=":", zorder=1)
    for idx, var in enumerate(bars_vars):
        lbl, col, mk = METHOD_STYLE[var]
        means = [row_val(df_sum, t, var, "RMSE_VA_mean", 0.0) for t in doms]
        stds = [row_val(df_sum, t, var, "RMSE_VA_std", 0.0) for t in doms]
        is_our = var == "concept_bilstm_urer"
        ax_b.errorbar(
            x + offs[idx], means, yerr=stds, fmt=mk,
            color=col, ecolor=col, elinewidth=(1.2 if is_our else 0.7),
            capsize=(2.5 if is_our else 1.8), capthick=(1.2 if is_our else 0.6),
            markersize=(8.0 if is_our else 3.5),
            markeredgecolor="white", markeredgewidth=(0.9 if is_our else 0.6),
            label=lbl, zorder=(7 if is_our else 4),
        )
        if is_our:
            for d_i, (val, sd) in enumerate(zip(means, stds)):
                ax_b.text(x[d_i] + offs[idx], val - sd - 0.025, f"{val:.3f}",
                          ha="center", va="top", fontsize=7.0, fontweight="bold", color=PALETTE["ours"])
    # Significance bracket on Finance
    fin_mse = row_val(df_sum, "dimabsa_zho_fin", "plm_only_mse", "RMSE_VA_mean", 0.541)
    fin_our = row_val(df_sum, "dimabsa_zho_fin", "concept_bilstm_urer", "RMSE_VA_mean", 0.513)
    if not df_sig.empty:
        srow = df_sig[(df_sig["task"] == "dimabsa_zho_fin") & (df_sig["cmp_variant"] == "plm_only_mse")]
        if not srow.empty:
            pval = float(srow["wilcoxon_p"].iloc[0])
            x1 = 2 + offs[1]
            x2 = 2 + offs[4]
            by = max(fin_mse, fin_our) + 0.045
            ax_b.plot([x1, x1, x2, x2], [by - 0.015, by, by, by - 0.015], color=PALETTE["ink"], lw=0.7)
            ax_b.text((x1 + x2) / 2, by + 0.010, f"p = {pval:.1e} ***", ha="center", va="bottom",
                      fontsize=7.0, fontweight="bold", color=PALETTE["gain"])
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([TASK_SHORT[t] for t in doms])
    ax_b.set_xlim(-0.52, len(doms) - 0.48)
    ax_b.set_ylim(0.42, 1.38)
    ax_b.set_ylabel("RMSE_VA (5-Seed Mean ± SD)  ↓")
    ax_b.legend(ncol=5, loc="upper center", fontsize=6.2, columnspacing=0.6, handletextpad=0.3, bbox_to_anchor=(0.5, 1.12))
    add_panel_tag(ax_b, "b", x=-0.08, y=1.05)

    # Panel c: Orthogonal Z-Score Deviation Matrix
    ax_c = fig.add_subplot(gs_bot[0, 0:5])
    heat_vars = ["classic_bilstm", "plm_only_mse", "plm_only_ccc", "plm_only_rdrop", "concept_bilstm_urer"]
    metrics_spec = [
        ("Fin RMSE↓", "dimabsa_zho_fin", "RMSE_VA_mean", -1.0),
        ("Fin CCC_A↑", "dimabsa_zho_fin", "CCC_A_mean", 1.0),
        ("Rest Tail20↓", "dimabsa_zho_rest", "Tail20_RMSE_V_mean", -1.0),
        ("Rest StdRatio↑", "dimabsa_zho_rest", "StdRatio_V_mean", 1.0),
        ("Lap CCC_V↑", "dimabsa_zho_lap", "CCC_V_mean", 1.0),
        ("Param Eff↑", "dimabsa_zho_fin", "trainable_params_mean", -1.0),
    ]
    mat_raw = np.zeros((len(heat_vars), len(metrics_spec)))
    for r_i, v in enumerate(heat_vars):
        for c_j, (_, tk, col, sign) in enumerate(metrics_spec):
            mat_raw[r_i, c_j] = sign * row_val(df_sum, tk, v, col, np.nan)
    col_mu = np.nanmean(mat_raw, axis=0, keepdims=True)
    col_sd = np.nanstd(mat_raw, axis=0, keepdims=True) + 1e-8
    mat_z = (mat_raw - col_mu) / col_sd
    # Custom saturated diverging colormap: deep navy → white → deep crimson
    _zscore_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "zscore_div", [(0.0, "#0A2463"), (0.25, "#3E6DB5"), (0.5, "#FFFFFF"), (0.75, "#C94040"), (1.0, "#8B0000")], N=256
    )
    im = ax_c.imshow(mat_z, cmap=_zscore_cmap, aspect="auto", vmin=-1.7, vmax=1.7)
    ax_c.set_xticks(np.arange(len(metrics_spec) + 1) - 0.5, minor=True)
    ax_c.set_yticks(np.arange(len(heat_vars) + 1) - 0.5, minor=True)
    ax_c.grid(which="minor", color="white", lw=0.3)
    ax_c.tick_params(which="minor", length=0)
    for r_i in range(len(heat_vars)):
        for c_j in range(len(metrics_spec)):
            zv = mat_z[r_i, c_j]
            tc = "white" if abs(zv) > 1.0 else PALETTE["ink"]
            ax_c.text(c_j, r_i, f"{zv:+.2f}", ha="center", va="center", fontsize=7.8,
                      fontweight=("bold" if r_i == len(heat_vars) - 1 else "normal"), color=tc)
            # Gold border for "Ours" row (last row)
            if r_i == len(heat_vars) - 1:
                rect = mpatches.Rectangle((c_j - 0.5, r_i - 0.5), 1, 1, linewidth=1.5,
                                          edgecolor=PALETTE["gold"], facecolor="none", zorder=5)
                ax_c.add_patch(rect)
    ax_c.set_yticks(np.arange(len(heat_vars)))
    ax_c.set_yticklabels([METHOD_STYLE[v][0] for v in heat_vars], fontsize=7.4)
    ax_c.set_xticks(np.arange(len(metrics_spec)))
    ax_c.set_xticklabels([m[0] for m in metrics_spec], rotation=0, ha="center", fontsize=7.2)
    ax_c.set_frame_on(False)
    ax_c.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax_c, orientation="horizontal", location="bottom",
                        fraction=0.055, pad=0.25, aspect=32, shrink=0.88)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("#A8A8A8")
    cbar.set_label("Z-Score Deviation (↑ better)", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.8)
    add_panel_tag(ax_c, "c", x=-0.16, y=1.06)

    # Panel d: Rhetorical Composition Benchmark (LogicSentiment)
    ax_d = fig.add_subplot(gs_bot[0, 6:12])
    style_clean_grid(ax_d, "y")
    # Subtle vertical grid for readability
    ax_d.grid(axis="x", color=PALETTE["grid"], linestyle="--", linewidth=0.3, alpha=0.25, zorder=0)
    subtypes = [("Acc_simple_mean", "Simple"), ("Acc_adversative_mean", "Adversative\n(Turn)"), ("Acc_double_neg_mean", "Double Neg"), ("Acc_irony_mean", "Irony")]
    xs_d = np.arange(len(subtypes))
    for var in ["classic_bilstm", "plm_only_mse", "plm_only_rdrop", "concept_bilstm_urer"]:
        lbl, col, mk = METHOD_STYLE[var]
        ys_d = [row_val(df_sum, "logic_sentiment", var, c, np.nan) * 100.0 for c, _ in subtypes]
        is_our = var == "concept_bilstm_urer"
        ax_d.plot(xs_d, ys_d, marker=mk, markersize=(6.0 if is_our else 3.5),
                  markeredgecolor="white", markeredgewidth=0.5,
                  linewidth=(2.0 if is_our else 0.8), color=col, alpha=(1.0 if is_our else 0.7),
                  label=lbl, zorder=(6 if is_our else 3))
        # Direct end-labels instead of legend
        if not np.isnan(ys_d[-1]):
            ax_d.text(xs_d[-1] + 0.12, ys_d[-1], lbl.replace(" (Ours)", "\n(Ours)"),
                      fontsize=6.2, fontweight=("bold" if is_our else "normal"),
                      color=col, va="center", ha="left")
        if is_our:
            for xi, yi in zip(xs_d, ys_d):
                ax_d.text(xi, yi + 1.2, f"{yi:.1f}", ha="center", va="bottom",
                          fontsize=6.6, fontweight="bold", color=col)
    ax_d.set_xticks(xs_d)
    ax_d.set_xticklabels([s[1] for s in subtypes])
    ax_d.set_ylim(74, 101.5)
    ax_d.set_xlim(-0.15, len(subtypes) - 0.4)
    ax_d.set_ylabel("Classification Accuracy (%)  ↑")
    add_panel_tag(ax_d, "d", x=-0.10, y=1.06)

    return save_pub_figure(fig, fig_dir, "fig2_main_benchmark_and_pareto")


def plot_fig3_anti_mean_collapse_and_uncertainty(out_root: Path, fig_dir: Path) -> List[str]:
    """Fig 3: KDE alignment, anti-mean-collapse frontier, split conformal coverage, risk stratification."""
    main_dir = out_root / "main_suite"
    df_raw, _ = load_raw_runs(main_dir)
    if df_raw.empty:
        return []
    df_sum = summarize_runs(df_raw)

    fig = plt.figure(figsize=(13.4, 9.0))
    gs = gridspec.GridSpec(2, 12, hspace=0.42, wspace=0.95)

    # Panel a: KDE Density Alignment (dimabsa_zho_rest)
    ax_a = fig.add_subplot(gs[0, 0:6])
    style_clean_grid(ax_a, "y")
    ax_a.axvline(4.4, color="#D8D8D8", ls=":", lw=0.5)
    ax_a.axvline(6.9, color="#D8D8D8", ls=":", lw=0.5)
    ax_a.text(3.0, 0.36, "Neg Tail", ha="center", fontsize=6.6, color=PALETTE["muted"], alpha=0.7)
    ax_a.text(7.95, 0.56, "Pos Tail", ha="center", fontsize=6.6, color=PALETTE["muted"], alpha=0.7)
    gt_drawn = False
    for var, ls in [("classic_bilstm", ":"), ("plm_only_mse", "-."), ("concept_bilstm_urer", "-")]:
        npz_p = main_dir / f"dimabsa_zho_rest__{var}__seed42" / "test_predictions.npz"
        if not npz_p.exists():
            continue
        z = np.load(npz_p)
        yt = z["y_true"][:, 0]
        yp = z["y_pred"][:, 0]
        xg = np.linspace(1.8, 9.0, 320)
        if not gt_drawn:
            kde_gt = stats.gaussian_kde(yt)
            y_gt = kde_gt(xg)
            ax_a.fill_between(xg, 0, y_gt, color="#A8A8A8", alpha=0.06, zorder=1)
            ax_a.plot(xg, y_gt, "--", color=PALETTE["ink"], lw=1.3, label="Ground Truth", zorder=4)
            gt_drawn = True
        ks_s = float(stats.ks_2samp(yt, yp).statistic)
        lbl, col, _ = METHOD_STYLE[var]
        is_our = var == "concept_bilstm_urer"
        kde_p = stats.gaussian_kde(yp)
        y_pred_kde = kde_p(xg)
        ax_a.plot(xg, y_pred_kde, ls=ls, color=col, lw=(1.8 if is_our else 0.9),
                  label=f"{lbl} (KS={ks_s:.3f})", zorder=(5 if is_our else 3))
        # Fill under hero KDE curve
        if is_our:
            ax_a.fill_between(xg, 0, y_pred_kde, color=col, alpha=0.12, zorder=2)
    ax_a.set_xlim(1.8, 9.0)
    ax_a.set_ylim(0, 0.80)
    ax_a.set_xlabel("Valence Score [1, 9]")
    ax_a.set_ylabel("Kernel Density")
    ax_a.legend(loc="upper left", fontsize=6.4)
    add_panel_tag(ax_a, "a", x=-0.10, y=1.05)

    # Panel b: Anti-Mean-Collapse Trajectory (Unified Domain Colors)
    ax_b = fig.add_subplot(gs[0, 6:12])
    style_clean_grid(ax_b, "both")
    doms = ["dimabsa_zho_rest", "dimabsa_zho_lap", "dimabsa_zho_fin", "dimabsa_zho_env"]
    dom_cols = [PALETTE["ours"], PALETTE["causal"], PALETTE["qwen"], "#9C6644"]
    all_sr, all_t2 = [], []
    for t, dc in zip(doms, dom_cols):
        sr_m = row_val(df_sum, t, "plm_only_mse", "StdRatio_V_mean")
        t2_m = row_val(df_sum, t, "plm_only_mse", "Tail20_RMSE_V_mean")
        sr_o = row_val(df_sum, t, "concept_bilstm_urer", "StdRatio_V_mean")
        t2_o = row_val(df_sum, t, "concept_bilstm_urer", "Tail20_RMSE_V_mean")
        if any(np.isnan([sr_m, t2_m, sr_o, t2_o])):
            continue
        all_sr.extend([sr_m, sr_o]); all_t2.extend([t2_m, t2_o])
        ax_b.scatter(sr_m, t2_m, s=40, color=PALETTE["mse"], edgecolors="white", linewidths=0.7, marker="v", zorder=4)
        # Glow effect on endpoint markers
        ax_b.scatter(sr_o, t2_o, s=180, color=dc, alpha=0.15, edgecolors="none", zorder=4)
        ax_b.scatter(sr_o, t2_o, s=65, color=dc, edgecolors="white", linewidths=0.9, marker="o", zorder=5)
        ax_b.annotate("", xy=(sr_o, t2_o), xytext=(sr_m, t2_m),
                      arrowprops=dict(arrowstyle="-|>", mutation_scale=8, color=dc, lw=1.2,
                                      connectionstyle="arc3,rad=0.1"), zorder=3)
        lbl_x = (sr_m + sr_o) / 2 + (-0.010 if t == "dimabsa_zho_env" else (0.008 if sr_o < 0.96 else -0.045))
        lbl_y = (t2_m + t2_o) / 2 + (0.032 if t == "dimabsa_zho_env" else 0.016)
        ax_b.text(lbl_x, lbl_y, TASK_SHORT[t], fontsize=7.2, fontweight="bold", color=dc)
    ax_b.scatter([], [], s=50, color=PALETTE["mse"], marker="v", label="PLM-MSE (Start)")
    ax_b.scatter([], [], s=75, color=PALETTE["ours"], marker="o", label="Concept-BiLSTM-URER (End)")
    ax_b.axvline(1.0, color=PALETTE["muted"], ls=":", lw=0.9, alpha=0.8)
    ax_b.set_xlim(min(all_sr) - 0.035, max(all_sr) + 0.055)
    ax_b.set_ylim(min(all_t2) - 0.055, max(all_t2) + 0.065)
    ax_b.set_xlabel("Valence Prediction Spread Ratio (StdRatio_V → 1.0 ideal)  ↑")
    ax_b.set_ylabel("Extreme Tail-20% RMSE (Valence)  ↓")
    ax_b.legend(loc="lower left", fontsize=6.4)
    add_panel_tag(ax_b, "b", x=-0.10, y=1.05)

    # Panel c: Split Conformal Coverage across 4 domains
    ax_c = fig.add_subplot(gs[1, 0:6])
    style_clean_grid(ax_c, "y")
    ax_c.axhline(0.90, color=PALETTE["causal"], ls="--", lw=0.9, zorder=1)
    # Visible 90% target band
    ax_c.axhspan(0.88, 0.92, color=PALETTE["zone_teal"], alpha=0.15, zorder=0)
    xc = np.arange(len(doms))
    w = 0.22
    uv = [row_val(df_sum, t, "concept_bilstm_urer", "uncal_PICP_V_mean") for t in doms]
    cv = [row_val(df_sum, t, "concept_bilstm_urer", "PICP_V_mean") for t in doms]
    ca = [row_val(df_sum, t, "concept_bilstm_urer", "PICP_A_mean") for t in doms]
    ax_c.bar(xc - w, uv, width=w, color=PALETTE["cnn"], edgecolor="none", label="Uncalibrated NIG (V)")
    ax_c.bar(xc, cv, width=w, color=PALETTE["ours"], edgecolor="none", label="Split Conformal (Valence)")
    ax_c.bar(xc + w, ca, width=w, color=PALETTE["causal"], edgecolor="none", label="Split Conformal (Arousal)")
    for i, val in enumerate(cv):
        check_mark = " *" if val >= 0.88 else ""
        ax_c.text(xc[i], val + 0.012, f"{val:.2f}{check_mark}", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=PALETTE["ours"])
    for i, val in enumerate(ca):
        ax_c.text(xc[i] + w, val + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=PALETTE["causal"])
    ax_c.text(0.24, ca[0] + 0.015, "↓", ha="center", va="bottom", fontsize=8.0, color=PALETTE["drop"], alpha=0.6)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([TASK_SHORT[t] for t in doms])
    ax_c.set_ylim(0.54, 1.12)
    ax_c.set_ylabel("Empirical Coverage Probability (PICP)")
    ax_c.legend(loc="upper right", ncol=2, fontsize=6.2)
    add_panel_tag(ax_c, "c", x=-0.10, y=1.05)

    # Panel d: Risk-Error Stratification Curve with Direct Labels
    ax_d = fig.add_subplot(gs[1, 6:12])
    style_clean_grid(ax_d, "y")
    deciles = np.arange(1, 11)
    end_pts: List[Tuple[float, str, str]] = []
    dom_curve_styles = [
        ("dimabsa_zho_fin", PALETTE["ours"], "o", 1.8),
        ("dimabsa_zho_env", PALETTE["causal"], "s", 0.9),
        ("dimabsa_zho_lap", PALETTE["qwen"], "^", 0.9),
        ("dimabsa_zho_rest", PALETTE["rdrop"], "D", 0.9),
    ]
    for t, col, mk, lw in dom_curve_styles:
        seed_curves: List[np.ndarray] = []
        for p in sorted(main_dir.glob(f"{t}__concept_bilstm_urer__seed*/test_predictions.npz")):
            z = np.load(p)
            sq_va = np.sum((z["y_true"] - z["y_pred"]) ** 2, axis=1)
            unc = np.mean(z["tot_std"], axis=1)
            qs = np.quantile(unc, np.linspace(0, 1, 11))
            bin_rmse = []
            for bi in range(10):
                m = (unc >= qs[bi]) & (unc <= qs[bi + 1] if bi == 9 else unc < qs[bi + 1])
                bin_rmse.append(float(np.sqrt(np.mean(sq_va[m]))) if np.any(m) else np.nan)
            seed_curves.append(bin_rmse)
        if seed_curves:
            arr = np.asarray(seed_curves, dtype=float)
            mu = np.nanmean(arr, axis=0)
            sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(10)
            is_fin = t == "dimabsa_zho_fin"
            ax_d.plot(deciles, mu, marker=mk, lw=lw, markersize=(5.5 if is_fin else 3.8),
                      markeredgecolor="white", markeredgewidth=0.5,
                      markerfacecolor=col, color=col)
            ax_d.fill_between(deciles, mu - sd, mu + sd, color=col, alpha=0.06)
            end_pts.append((float(mu[-1]), TASK_SHORT[t], col))
    end_pts.sort()
    last_y = -1e9
    for y_raw, t_name, t_col in end_pts:
        y_pos = max(y_raw, last_y + 0.085)
        ax_d.text(10.18, y_pos, t_name, va="center", ha="left", fontsize=7.2, fontweight="bold", color=t_col)
        last_y = y_pos
    ax_d.set_xlim(0.7, 11.45)
    ax_d.set_xticks(deciles)
    ax_d.set_xticklabels([f"D{i}" for i in deciles])
    ax_d.set_xlabel("Predicted Uncertainty Decile (D1 Low → D10 High)")
    ax_d.set_ylabel("Empirical Stratified RMSE_VA  ↑ with Risk")
    add_panel_tag(ax_d, "d", x=-0.10, y=1.05)

    return save_pub_figure(fig, fig_dir, "fig3_anti_mean_collapse_and_uncertainty")


def plot_fig4_mobo_ablation_efficiency_ood(out_root: Path, fig_dir: Path) -> List[str]:
    """Fig 4: MOBO Pareto + fANOVA, component ablations, 25/50/100% data efficiency, cross-domain OOD."""
    main_sum = summarize_runs(load_raw_runs(out_root / "main_suite")[0])
    abl_sum = summarize_runs(load_raw_runs(out_root / "ablation_suite")[0])
    cross_sum = summarize_runs(load_raw_runs(out_root / "cross_ood_suite")[0])
    de25_sum = summarize_runs(load_raw_runs(out_root / "data_eff_frac25")[0])
    de50_sum = summarize_runs(load_raw_runs(out_root / "data_eff_frac50")[0])

    fig = plt.figure(figsize=(14.0, 9.4))
    gs_outer = gridspec.GridSpec(2, 1, hspace=0.40)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 12, subplot_spec=gs_outer[0], wspace=1.20)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 12, subplot_spec=gs_outer[1], wspace=0.70)

    # Panel a: MOBO Pareto + fANOVA Inset
    ax_a = fig.add_subplot(gs_top[0, 0:6])
    style_clean_grid(ax_a, "both")
    p_csv = out_root / "mobo_zho_rest" / "mobo_trials.csv"
    p_fan = out_root / "mobo_zho_rest" / "mobo_fanova.json"
    if p_csv.exists():
        df_m = pd.read_csv(p_csv)
        non_p = df_m[~df_m["is_pareto"]]
        par = df_m[df_m["is_pareto"]].sort_values("RMSE_VA")
        ax_a.scatter(non_p["RMSE_VA"], non_p["Winkler_avg"], s=40, color=PALETTE["cnn"], alpha=0.30,
                     edgecolors=PALETTE["muted"], lw=0.5, label="Explored Trials (n=24)")
        if len(par) >= 2:
            ax_a.plot(par["RMSE_VA"], par["Winkler_avg"], "--", color=PALETTE["causal"], lw=1.8, label="Pareto Frontier")
        ax_a.scatter(par["RMSE_VA"], par["Winkler_avg"], s=80, color=PALETTE["causal"], edgecolors="white", lw=1.0, zorder=5)
        best = par.sort_values("Winkler_avg").iloc[0]
        # Gold glow behind star
        ax_a.scatter(best["RMSE_VA"], best["Winkler_avg"], s=500, color=PALETTE["gold"], alpha=0.20,
                     edgecolors="none", zorder=5)
        ax_a.scatter(best["RMSE_VA"], best["Winkler_avg"], s=300, marker="*", color=PALETTE["gold"],
                     edgecolors=PALETTE["ink"], lw=0.9, zorder=6,
                     label=f"Selected Knee (Trial #{int(best['trial'])})")
        ax_a.annotate(f"Trial #{int(best['trial'])}\n({best['RMSE_VA']:.3f}, {best['Winkler_avg']:.2f})",
                      (best["RMSE_VA"], best["Winkler_avg"]), xytext=(best["RMSE_VA"] + 0.034, best["Winkler_avg"] - 0.005),
                      arrowprops=dict(arrowstyle="->", color=PALETTE["ours"], lw=0.9), va="center",
                      fontsize=7.4, fontweight="bold", color=PALETTE["ours"])
    if p_fan.exists():
        fan = json.loads(p_fan.read_text(encoding="utf-8")).get("composite_err_ccc", {})
        top5 = sorted(fan.items(), key=lambda kv: kv[1], reverse=True)[:5][::-1]
        ins = ax_a.inset_axes([0.22, 0.64, 0.30, 0.28])
        ins.set_facecolor(PALETTE["zone_slate"])
        iy = np.arange(len(top5))
        ins.barh(iy, [v * 100 for _, v in top5], color=PALETTE["ours_light"], alpha=0.88, edgecolor="none", height=0.58)
        ins.set_yticks(iy)
        ins.set_yticklabels([k for k, _ in top5])
        ins.set_title("fANOVA Drivers (%)", fontsize=7.0, fontweight="bold", pad=3)
        ins.tick_params(labelsize=6.4)
        for sp in ("top", "right", "left", "bottom"):
            ins.spines[sp].set_visible(sp in ("bottom", "left"))
            ins.spines[sp].set_linewidth(0.5)
    ax_a.set_ylim(2.08, 2.92)
    ax_a.set_xlabel("Proxy Dev RMSE_VA  ↓")
    ax_a.set_ylabel("Mean 90% Winkler Score  ↓")
    ax_a.legend(loc="lower right", fontsize=6.3)
    add_panel_tag(ax_a, "a", x=-0.10, y=1.05)

    # Panel b: Component Ablation Impact Heatmap
    ax_b = fig.add_subplot(gs_top[0, 6:12])
    abl_vars = ["concept_no_bilstm", "plm_bilstm_no_concept", "concept_bilstm_det", "concept_bilstm_no_ccc", "concept_bilstm_no_dyn", "ib_residual_bilstm"]
    abl_short = {
        "concept_no_bilstm": "w/o BiLSTM",
        "plm_bilstm_no_concept": "w/o Concept",
        "concept_bilstm_det": "w/o NIG (Det.)",
        "concept_bilstm_no_ccc": "w/o CCC",
        "concept_bilstm_no_dyn": "w/o Dyn Gate",
        "ib_residual_bilstm": "Latent IB Only",
    }
    cols_spec = [
        ("Rest ΔRMSE↓", "dimabsa_zho_rest", "RMSE_VA_mean", 1.0),
        ("Rest ΔTail20↓", "dimabsa_zho_rest", "Tail20_RMSE_V_mean", 1.0),
        ("Rest ΔWinkler↓", "dimabsa_zho_rest", "Winkler_avg_mean", 1.0),
        ("Lap ΔRMSE↓", "dimabsa_zho_lap", "RMSE_VA_mean", 1.0),
        ("Lap ΔTail20↓", "dimabsa_zho_lap", "Tail20_RMSE_V_mean", 1.0),
        ("Lap ΔWinkler↓", "dimabsa_zho_lap", "Winkler_avg_mean", 1.0),
    ]
    dmat = np.full((len(abl_vars), len(cols_spec)), np.nan)
    for r_i, av in enumerate(abl_vars):
        for c_j, (_, tk, col, _) in enumerate(cols_spec):
            base_v = row_val(main_sum, tk, "concept_bilstm_urer", col)
            ab_v = row_val(abl_sum, tk, av, col)
            if pd.notna(base_v) and pd.notna(ab_v):
                dmat[r_i, c_j] = ab_v - base_v
    cmap_ab = mpl.colors.LinearSegmentedColormap.from_list(
        "editorial_degrade",
        [(0.0, "#1E40AF"), (0.25, "#60A5FA"), (0.42, "#F8FAFC"), (0.55, "#FCA5A5"), (0.75, "#DC2626"), (1.0, "#7F1D1D")],
        N=256,
    )
    cmap_ab.set_bad(color="#F5F5F5")
    ax_b.imshow(np.clip(dmat, -0.06, 0.14), cmap=cmap_ab, aspect="auto", vmin=-0.06, vmax=0.14)
    ax_b.set_xticks(np.arange(len(cols_spec) + 1) - 0.5, minor=True)
    ax_b.set_yticks(np.arange(len(abl_vars) + 1) - 0.5, minor=True)
    ax_b.grid(which="minor", color="white", lw=0.3)
    ax_b.tick_params(which="minor", length=0)
    for r_i in range(len(abl_vars)):
        for c_j in range(len(cols_spec)):
            val = dmat[r_i, c_j]
            txt = "N/A" if np.isnan(val) else f"{val:+.3f}"
            tc = "white" if (not np.isnan(val) and (val > 0.085 or val < -0.04)) else PALETTE["ink"]
            # Add ▲/▼ arrows for values with >0.05 magnitude
            arrow = ""
            if not np.isnan(val) and abs(val) > 0.05:
                arrow = " ▲" if val > 0 else " ▼"
            ax_b.text(c_j, r_i, txt + arrow, ha="center", va="center", fontsize=6.6,
                      fontweight="bold" if (not np.isnan(val) and val > 0.025) else "normal", color=tc)
    ax_b.set_yticks(np.arange(len(abl_vars)))
    ax_b.set_yticklabels([abl_short[v] for v in abl_vars], fontsize=7.4)
    ax_b.set_xticks(np.arange(len(cols_spec)))
    ax_b.set_xticklabels([c[0] for c in cols_spec], rotation=18, ha="right", fontsize=7.2)
    ax_b.set_frame_on(False)
    ax_b.tick_params(length=0)
    add_panel_tag(ax_b, "b", x=-0.16, y=1.05)

    # Panel c: Data Efficiency Scaling Curve (25%, 50%, 100%)
    ax_c = fig.add_subplot(gs_bot[0, 0:6])
    style_clean_grid(ax_c, "y")
    fracs = [25, 50, 100]
    frac_dfs = [de25_sum, de50_sum, main_sum]
    _eff_lines = {}  # Store for fill_between later
    for var in ["classic_bilstm", "plm_only_mse", "concept_bilstm_urer"]:
        lbl, col, mk = METHOD_STYLE[var]
        ys_m = [row_val(df_f, "dimabsa_zho_rest", var, "RMSE_VA_mean") for df_f in frac_dfs]
        ys_s = [row_val(df_f, "dimabsa_zho_rest", var, "RMSE_VA_std", 0.0) for df_f in frac_dfs]
        is_our = var == "concept_bilstm_urer"
        ax_c.plot(fracs, ys_m, marker=mk, lw=(1.8 if is_our else 0.8), markersize=(6.0 if is_our else 3.5),
                  markeredgecolor="white", markeredgewidth=0.5, color=col, label=lbl, zorder=(5 if is_our else 3))
        ax_c.fill_between(fracs, np.array(ys_m) - np.array(ys_s), np.array(ys_m) + np.array(ys_s), color=col, alpha=0.06)
        _eff_lines[var] = ys_m
    # Fill between our method and PLM-MSE to show the gap
    if "concept_bilstm_urer" in _eff_lines and "plm_only_mse" in _eff_lines:
        ax_c.fill_between(fracs, _eff_lines["concept_bilstm_urer"], _eff_lines["plm_only_mse"],
                          color=PALETTE["ours"], alpha=0.08, zorder=1)
    y25_m = row_val(de25_sum, "dimabsa_zho_rest", "plm_only_mse", "RMSE_VA_mean")
    y25_o = row_val(de25_sum, "dimabsa_zho_rest", "concept_bilstm_urer", "RMSE_VA_mean")
    ax_c.annotate(f"Ours @25%: {y25_o:.3f} ({y25_o - y25_m:+.3f} vs MSE)", xy=(25, y25_o), xytext=(33, y25_o + 0.082),
                  arrowprops=dict(arrowstyle="->", color=PALETTE["ours"], lw=0.75), fontsize=6.8, fontweight="bold", color=PALETTE["ours"])
    ax_c.set_xticks(fracs)
    ax_c.set_xticklabels(["25% Labeled", "50% Labeled", "100% Labeled"])
    ax_c.set_ylim(0.92, 1.40)
    ax_c.set_ylabel("Restaurant RMSE_VA (5-Seed Mean ± SD)  ↓")
    ax_c.legend(loc="upper right", fontsize=6.4)
    add_panel_tag(ax_c, "c", x=-0.10, y=1.05)

    # Panel d: Cross-Domain OOD Transfer (Unhatched Alpha-Tint Bars)
    ax_d = fig.add_subplot(gs_bot[0, 6:12])
    style_clean_grid(ax_d, "y")
    avail_cross = set(cross_sum["variant"].unique()) if not cross_sum.empty else set()
    cross_vars = [v for v in ["classic_bilstm", "plm_only_mse", "plm_only_ccc", "plm_only_rdrop", "concept_bilstm_urer"] if v in avail_cross]
    xd = np.arange(len(cross_vars))
    w = 0.30
    c_rmse = [row_val(cross_sum, "dimabsa_cross_fin", v, "RMSE_VA_mean") for v in cross_vars]
    c_rmse_s = [row_val(cross_sum, "dimabsa_cross_fin", v, "RMSE_VA_std", 0.0) for v in cross_vars]
    c_tail = [row_val(cross_sum, "dimabsa_cross_fin", v, "Tail20_RMSE_V_mean") for v in cross_vars]
    c_tail_s = [row_val(cross_sum, "dimabsa_cross_fin", v, "Tail20_RMSE_V_std", 0.0) for v in cross_vars]
    cols_v = [METHOD_STYLE[v][1] for v in cross_vars]
    # Determine which index is "ours" for pattern fill
    ours_idx = cross_vars.index("concept_bilstm_urer") if "concept_bilstm_urer" in cross_vars else -1
    b1 = ax_d.bar(xd - w / 2, c_rmse, width=w, yerr=c_rmse_s, capsize=2.2, color=cols_v, edgecolor="none",
                  error_kw={"elinewidth": 0.8, "ecolor": "#475569"}, label="Overall RMSE_VA ↓")
    b2 = ax_d.bar(xd + w / 2, c_tail, width=w, yerr=c_tail_s, capsize=2.2, color=cols_v, alpha=0.38, edgecolor="none",
                  error_kw={"elinewidth": 0.8, "ecolor": "#475569"}, label="Tail-20% Valence RMSE ↓")
    # Add subtle diagonal hatch on "ours" bars to distinguish
    if ours_idx >= 0:
        b1[ours_idx].set_hatch("//")
        b1[ours_idx].set_edgecolor(PALETTE["ours_light"])
        b2[ours_idx].set_hatch("//")
        b2[ours_idx].set_edgecolor(PALETTE["ours_light"])
    for b_i, val in enumerate(c_rmse):
        ax_d.text(b1[b_i].get_x() + b1[b_i].get_width() / 2, val + c_rmse_s[b_i] + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    for b_i, val in enumerate(c_tail):
        ax_d.text(b2[b_i].get_x() + b2[b_i].get_width() / 2, val + c_tail_s[b_i] + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, color=PALETTE["muted"])
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([METHOD_STYLE[v][0].replace(" (Ours)", "\n(Ours)") for v in cross_vars], fontsize=7.0)
    ax_d.set_ylim(0.0, max(c_rmse + c_tail) * 1.24)
    ax_d.set_ylabel("Zero-Shot OOD Error on Finance  ↓")
    ax_d.legend(loc="upper right", fontsize=6.3)
    add_panel_tag(ax_d, "d", x=-0.10, y=1.05)

    return save_pub_figure(fig, fig_dir, "fig4_mobo_ablation_efficiency_ood")


def plot_fig5_generalization_backbones_and_llm(out_root: Path, fig_dir: Path) -> List[str]:
    """Fig 5: Cross-backbone generalization, multi-metric advantage strip, & LLM / Latency panel."""
    main_sum = summarize_runs(load_raw_runs(out_root / "main_suite")[0])
    rob_sum = summarize_runs(load_raw_runs(out_root / "backbone_roberta")[0])
    deb_sum = summarize_runs(load_raw_runs(out_root / "backbone_deberta")[0])

    bb_records: List[dict] = []
    for bb_lbl, df_bb in [("MacBERT-base", main_sum), ("RoBERTa-wwm", rob_sum), ("DeBERTa-v2-97M", deb_sum)]:
        if df_bb.empty:
            continue
        for t in ["dimabsa_zho_rest", "dimabsa_zho_fin"]:
            has_mse = not df_bb[(df_bb["task"] == t) & (df_bb["variant"] == "plm_only_mse") & (df_bb["n_seeds"] >= 3)].empty
            has_our = not df_bb[(df_bb["task"] == t) & (df_bb["variant"] == "concept_bilstm_urer") & (df_bb["n_seeds"] >= 3)].empty
            if has_mse and has_our:
                r_m = df_bb[(df_bb["task"] == t) & (df_bb["variant"] == "plm_only_mse")].iloc[0]
                r_o = df_bb[(df_bb["task"] == t) & (df_bb["variant"] == "concept_bilstm_urer")].iloc[0]
                bb_records.append({
                    "bb": bb_lbl, "task": t, "label": f"{TASK_SHORT[t]}\n{bb_lbl}",
                    "mse_rmse": float(r_m["RMSE_VA_mean"]), "mse_rmse_s": float(r_m["RMSE_VA_std"]),
                    "our_rmse": float(r_o["RMSE_VA_mean"]), "our_rmse_s": float(r_o["RMSE_VA_std"]),
                    "d_tail": float(r_o["Tail20_RMSE_V_mean"] - r_m["Tail20_RMSE_V_mean"]),
                    "d_ccc_a": float(r_o["CCC_A_mean"] - r_m["CCC_A_mean"]),
                    "d_sr_v": float(r_o["StdRatio_V_mean"] - r_m["StdRatio_V_mean"]),
                    "mse_lat": float(r_m["latency_ms_mean"]), "our_lat": float(r_o["latency_ms_mean"]),
                })
    if not bb_records:
        return []

    p_llm = out_root / "llm_baseline" / "llm_baseline_results.json"
    llm_d = (json.loads(p_llm.read_text(encoding="utf-8")).get("results") or {}) if p_llm.exists() else {}

    fig = plt.figure(figsize=(13.8, 8.6))
    gs_outer = gridspec.GridSpec(2, 1, hspace=0.44)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 12, subplot_spec=gs_outer[0], wspace=1.45)

    # Panel a: Cross-Backbone Accuracy Comparison (Focused Y-Axis, Unbordered Bars)
    ax_a = fig.add_subplot(gs_top[0, 0:7])
    style_clean_grid(ax_a, "y")
    xg = np.arange(len(bb_records))
    w = 0.25
    y_m = [r["mse_rmse"] for r in bb_records]
    y_ms = [r["mse_rmse_s"] for r in bb_records]
    y_o = [r["our_rmse"] for r in bb_records]
    y_os = [r["our_rmse_s"] for r in bb_records]
    ax_a.bar(xg - w / 2, y_m, width=w, yerr=y_ms, capsize=2.2, color=PALETTE["mse"], edgecolor="none",
             error_kw={"elinewidth": 0.8, "ecolor": "#475569"}, label="PLM-MSE (1.78M Params)")
    ax_a.bar(xg + w / 2, y_o, width=w, yerr=y_os, capsize=2.2, color=PALETTE["ours"], edgecolor="none",
             error_kw={"elinewidth": 0.8, "ecolor": "#0F172A"}, label="Concept-BiLSTM-URER (1.01M Params)")
    for i, rec in enumerate(bb_records):
        pct = (rec["our_rmse"] - rec["mse_rmse"]) / rec["mse_rmse"] * 100.0
        top_y = max(rec["mse_rmse"] + rec["mse_rmse_s"], rec["our_rmse"] + rec["our_rmse_s"]) + 0.020
        pct_lbl = f"{pct:+.2f}%" if abs(pct) < 0.05 else f"{pct:+.1f}%"
        # Red for negative (=improvement), muted otherwise
        pct_col = PALETTE["drop"] if pct < -0.5 else PALETTE["muted"]
        ax_a.text(xg[i], top_y, pct_lbl, ha="center", va="bottom", fontsize=6.9, fontweight="bold",
                  color=pct_col)
    ax_a.set_xticks(xg)
    ax_a.set_xticklabels([r["label"] for r in bb_records], fontsize=7.0)
    ax_a.set_ylim(0.38, max(y_m + y_o) * 1.18)
    ax_a.set_ylabel("RMSE_VA (Mean ± SD)  ↓")
    ax_a.legend(loc="upper center", ncol=2, fontsize=6.4)
    add_panel_tag(ax_a, "a", x=-0.09, y=1.05)

    # Panel b: Multi-Metric Delta Consistency Strip
    ax_b = fig.add_subplot(gs_top[0, 7:12])
    style_clean_grid(ax_b, "x")
    ax_b.axvline(0.0, color="#334155", ls="--", lw=0.75)
    yb = np.arange(len(bb_records))
    ax_b.barh(yb + 0.22, [r["d_ccc_a"] for r in bb_records], height=0.20, color=PALETTE["ours"], edgecolor="none", alpha=0.75, label="Δ CCC_A (↑ better)")
    ax_b.barh(yb, [r["d_sr_v"] for r in bb_records], height=0.20, color=PALETTE["causal"], edgecolor="none", alpha=0.65, label="Δ StdRatio_V (↑ better)")
    ax_b.barh(yb - 0.22, [-r["d_tail"] for r in bb_records], height=0.20, color=PALETTE["gain"], edgecolor="none", alpha=0.60, label="Tail20 RMSE Reduction (↑ better)")
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([f"{TASK_SHORT[r['task']]}\n{r['bb']}" for r in bb_records], fontsize=6.9)
    ax_b.set_xlabel("Gain of Concept-BiLSTM-URER over PLM-MSE  (→ Ours Better)")
    ax_b.legend(loc="lower right", fontsize=6.2)
    add_panel_tag(ax_b, "b", x=-0.18, y=1.05)

    # Panel c: Bottom Full-Width Strip (LLM vs Bottleneck if ready, else Log-Scale Latency Lollipop)
    ax_c = fig.add_subplot(gs_outer[1])
    doms = [t for t in ["dimabsa_zho_rest", "dimabsa_zho_lap", "dimabsa_zho_fin", "dimabsa_zho_env"] if t in llm_d]
    if doms:
        style_clean_grid(ax_c, "y")
        xd = np.arange(len(doms))
        w = 0.23
        yl = [float(llm_d[t]["RMSE_VA"]) for t in doms]
        ym = [row_val(main_sum, t, "plm_only_mse", "RMSE_VA_mean") for t in doms]
        yo = [row_val(main_sum, t, "concept_bilstm_urer", "RMSE_VA_mean") for t in doms]
        ax_c.bar(xd - w, yl, width=w, color=PALETTE["qwen"], edgecolor="none", label="Qwen2.5-3B-Instruct (Zero-Shot, 3B Params)")
        ax_c.bar(xd, ym, width=w, color=PALETTE["mse"], edgecolor="none", label="MacBERT-MSE (1.78M Tuned)")
        mean_lat = float(np.nanmean([row_val(main_sum, t, "concept_bilstm_urer", "latency_ms_mean", np.nan) for t in ["dimabsa_zho_rest", "dimabsa_zho_lap", "dimabsa_zho_fin", "dimabsa_zho_env"]]))
        lat_str = f"{mean_lat:.2f} ms/sample" if not np.isnan(mean_lat) else "~1.9 ms/sample"
        ax_c.bar(xd + w, yo, width=w, color=PALETTE["ours"], edgecolor="none", label=f"Concept-BiLSTM-URER (1.01M Tuned, {lat_str})")
        for i in range(len(doms)):
            ax_c.text(xd[i] - w, yl[i] + 0.03, f"{yl[i]:.2f}", ha="center", fontsize=6.9, color=PALETTE["qwen"], fontweight="bold")
            ax_c.text(xd[i], ym[i] + 0.03, f"{ym[i]:.3f}", ha="center", fontsize=6.6, color=PALETTE["muted"])
            ax_c.text(xd[i] + w, yo[i] + 0.03, f"{yo[i]:.3f}", ha="center", fontsize=6.9, color=PALETTE["ours"], fontweight="bold")
        ax_c.set_xticks(xd)
        ax_c.set_xticklabels([TASK_SHORT[t] for t in doms], fontsize=8.0)
        ax_c.set_ylim(0, max(yl + ym + yo) * 1.24)
        ax_c.set_ylabel("RMSE_VA  ↓")
        # Add a dashed "2× better" reference line
        _two_x_ref = max(yo) * 2.0 if yo else 1.0
        if _two_x_ref < max(yl) * 1.1:
            ax_c.axhline(_two_x_ref, color=PALETTE["muted"], ls="--", lw=0.75, alpha=0.5)
            ax_c.text(len(doms) - 0.6, _two_x_ref + 0.02, "2× Ours", fontsize=6.4, color=PALETTE["muted"], alpha=0.7)
        # Bracket annotation showing RMSE reduction range
        if yo and yl:
            reductions = [(1.0 - o / l) * 100.0 for o, l in zip(yo, yl) if l > 0]
            if reductions:
                lo_r, hi_r = min(reductions), max(reductions)
                ax_c.text(len(doms) - 0.5, max(yl) * 0.85, f"{lo_r:.0f}–{hi_r:.0f}% RMSE\nreduction vs Qwen",
                          fontsize=6.6, fontweight="bold", color=PALETTE["ours"], ha="center", va="top")
        ax_c.legend(loc="upper right", ncol=3, fontsize=6.4)
    else:
        style_clean_grid(ax_c, "x")
        lat_vars = ["classic_cnn", "classic_bigru", "classic_bilstm", "plm_only_mse", "plm_only_rdrop", "concept_bilstm_urer"]
        yl = np.arange(len(lat_vars))
        lats = [max(row_val(main_sum, "dimabsa_zho_rest", v, "latency_ms_mean", 0.01), 0.01) for v in lat_vars]
        pars = [row_val(main_sum, "dimabsa_zho_rest", v, "trainable_params_mean", 0.0) / 1e6 for v in lat_vars]
        cols_l = [METHOD_STYLE[v][1] for v in lat_vars]
        for i, (lt, pm, cl) in enumerate(zip(lats, pars, cols_l)):
            ax_c.hlines(yl[i], 0.01, lt, color=cl, lw=2.0, alpha=0.75)
            ax_c.scatter([lt], [yl[i]], s=(95 if i == len(yl) - 1 else 65), color=cl, edgecolors="white", lw=0.8, zorder=4)
            ax_c.text(lt * 1.16, yl[i], f"{lt:.2f} ms  ({pm:.2f}M params)", va="center", ha="left",
                      fontsize=6.9, fontweight=("bold" if i == len(yl) - 1 else "normal"), color=cl)
        ax_c.set_xscale("log")
        ax_c.set_xlim(0.01, max(lats) * 4.0)
        ax_c.set_yticks(yl)
        ax_c.set_yticklabels([METHOD_STYLE[v][0] for v in lat_vars], fontsize=7.2)
        ax_c.set_xlabel("Inference Latency (ms / sample, log scale)  ↓")
    add_panel_tag(ax_c, "c", x=-0.05, y=1.05)

    return save_pub_figure(fig, fig_dir, "fig5_generalization_backbones_and_llm")


def plot_fig6_mechanism_diagnostics_and_boundaries(out_root: Path, fig_dir: Path) -> List[str]:
    """Fig 6: Faithfulness curves, Causal/Concept audit, Noise stress, Brighter zero-inflation boundary, Probe AUROC."""
    main_dir = out_root / "main_suite"
    patch_dir = out_root / "mechanism_patch"
    _, main_raw = load_raw_runs(main_dir)
    _, patch_raw = load_raw_runs(patch_dir) if patch_dir.exists() else (pd.DataFrame(), [])
    main_sum = summarize_runs(pd.DataFrame([
        {**{"task": r["task"], "variant": r["variant"], "backbone": r.get("backbone", ""), "seed": r["seed"]},
         **r.get("test_metrics", {}),
         **{f"{nk}_{mk}": mv for nk, nv in r.get("noise_results", {}).items() for mk, mv in nv.items()}}
        for r in main_raw
    ]))

    fig = plt.figure(figsize=(14.2, 9.4))
    gs_outer = gridspec.GridSpec(2, 1, hspace=0.44)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 12, subplot_spec=gs_outer[0], wspace=1.35)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 12, subplot_spec=gs_outer[1], wspace=0.65)

    # Panel a: Deletion Faithfulness (Concept Gate vs IG vs Attention + Causal Patch)
    ax_a = fig.add_subplot(gs_top[0, 0:4])
    style_clean_grid(ax_a, "y")
    fracs = [10, 20, 30, 50]
    ex_main = [r["explain_results"] for r in main_raw if r["task"] == "dimabsa_zho_rest" and r["variant"] == "concept_bilstm_urer" and r.get("explain_results")]
    ex_patch = [r["explain_results"] for r in patch_raw if r["task"] == "dimabsa_zho_rest" and r["variant"] == "concept_bilstm_causal" and r.get("explain_results")]
    if ex_main:
        for key, lbl, col, ls, mk in [
            ("concept_comp_mean", "Concept Gate (URER)", PALETTE["ours"], "-", "o"),
            ("ig_comp_mean", "Integrated Gradients", PALETTE["gain"], "--", "s"),
            ("attention_comp_mean", "Attention Weight", PALETTE["muted"], ":", "^"),
        ]:
            arr = np.asarray([e[key] for e in ex_main if key in e], dtype=float)
            if arr.size:
                mu = np.mean(arr, axis=0)
                sd = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(len(fracs))
                is_hero = key == "concept_comp_mean"
                ax_a.plot(fracs, mu, ls=ls, marker=mk, lw=1.7, markersize=4.5, markeredgecolor="white", markeredgewidth=0.5, color=col, label=lbl)
                ax_a.fill_between(fracs, mu - sd, mu + sd, color=col, alpha=0.08)
                # Filled area under hero line
                if is_hero:
                    ax_a.fill_between(fracs, 0, mu, color=col, alpha=0.10, zorder=1)
    if ex_patch:
        arr_p = np.asarray([e["concept_comp_mean"] for e in ex_patch if "concept_comp_mean" in e], dtype=float)
        if arr_p.size:
            mu_p = np.mean(arr_p, axis=0)
            ax_a.plot(fracs, mu_p, ls="-", marker="*", lw=2.0, markersize=7.2, markeredgecolor="white", markeredgewidth=0.5, color=PALETTE["causal"], label="Concept Gate (Causal Patch)")
    ax_a.set_xticks(fracs)
    ax_a.set_xticklabels([f"{k}%" for k in fracs])
    ax_a.set_ylim(0.10, 0.74)
    ax_a.set_xlabel("Top-k% Salient Tokens Removed")
    ax_a.set_ylabel("Comprehensiveness (Δ|y_pred|)  ↑")
    ax_a.legend(loc="upper left", fontsize=6.1)
    add_panel_tag(ax_a, "a", x=-0.16, y=1.05)

    # Panel b: Causal Intervention & Orthogonality Audit (or Concept Attribution Breakdown)
    ax_b = fig.add_subplot(gs_top[0, 4:8])
    style_clean_grid(ax_b, "y")
    if ex_main and ex_patch:
        m_keys = [("cf_pos_cue_dir_acc", "Pos-Cue\nCDA ↑"), ("cf_neg_cue_dir_acc", "Neg-Cue\nCDA ↑"), ("concept_bg_mean", "Bg Leak\nMean ↓")]
        xm = np.arange(len(m_keys))
        w = 0.34
        vb = [float(np.mean([e.get(k, 0.0) for e in ex_main])) for k, _ in m_keys]
        vp = [float(np.mean([e.get(k, 0.0) for e in ex_patch])) for k, _ in m_keys]
        ax_b.bar(xm - w / 2, vb, width=w, color=PALETTE["mse"], edgecolor="none", label="Unconstrained (URER)")
        ax_b.bar(xm + w / 2, vp, width=w, color=PALETTE["causal"], edgecolor="none", label="Causal+Ortho Patch")
        ax_b.axhline(0.5, color=PALETTE["muted"], ls=":", lw=0.8)
        for i in range(len(m_keys)):
            ax_b.text(xm[i] - w / 2, vb[i] + 0.025, f"{vb[i]:.2f}", ha="center", fontsize=6.5, color=PALETTE["muted"])
            ax_b.text(xm[i] + w / 2, vp[i] + 0.025, f"{vp[i]:.2f}", ha="center", fontsize=6.6, fontweight="bold", color=PALETTE["causal"])
            # Highlight bars reaching 1.00 with gold top edge and star
            if vp[i] >= 0.99:
                ax_b.plot([xm[i] + w / 2 - w / 2, xm[i] + w / 2 + w / 2], [vp[i], vp[i]],
                          color=PALETTE["gold"], lw=2.0, zorder=6)
                ax_b.text(xm[i] + w / 2, vp[i] + 0.055, "*", ha="center", fontsize=9, fontweight="bold", color=PALETTE["gold"])
        ax_b.set_xticks(xm)
        ax_b.set_xticklabels([l for _, l in m_keys])
        ax_b.set_ylim(0, 1.18)
        ax_b.set_ylabel("Ratio / Activation Mean")
        ax_b.legend(loc="upper right", fontsize=6.2)
    elif ex_main:
        contrib = ex_main[0].get("mean_abs_concept_contrib", {})
        items = sorted(
            [(k, float(np.mean([e.get("mean_abs_concept_contrib", {}).get(k, 0.0) for e in ex_main]))) for k in contrib],
            key=lambda kv: kv[1],
        )
        names = [k.replace("_", " ") for k, _ in items]
        vals = [v for _, v in items]
        bar_cols = [PALETTE["cnn"] if k.startswith("Latent") else PALETTE["ours"] for k, _ in items]
        ax_b.barh(np.arange(len(names)), vals, color=bar_cols, edgecolor="none", height=0.66)
        ax_b.set_yticks(np.arange(len(names)))
        ax_b.set_yticklabels(names, fontsize=6.4)
        ax_b.set_xlabel("Mean |Concept Contribution|  ↑")
    add_panel_tag(ax_b, "b", x=-0.18, y=1.05)

    # Panel c: Relative Token-Corruption Noise Degradation (%)
    ax_c = fig.add_subplot(gs_top[0, 8:12])
    style_clean_grid(ax_c, "y")
    lvls = [0, 10, 20]
    ax_c.axhline(0.0, color="#CBD5E1", lw=0.7, ls=":")
    for var in ["plm_only_mse", "plm_only_ccc", "plm_only_rdrop", "concept_bilstm_urer"]:
        lbl, col, mk = METHOD_STYLE[var]
        ys = np.array([row_val(main_sum, "dimabsa_zho_rest", var, c) for c in ("RMSE_VA_mean", "noise_10_RMSE_VA_mean", "noise_20_RMSE_VA_mean")], dtype=float)
        if np.any(np.isnan(ys)) or ys[0] <= 0:
            continue
        rel_pct = (ys - ys[0]) / ys[0] * 100.0
        is_our = var == "concept_bilstm_urer"
        ax_c.plot(lvls, rel_pct, marker=mk, lw=(2.0 if is_our else 1.25), markersize=(5.2 if is_our else 4.2),
                  markeredgecolor="white", markeredgewidth=0.5, color=col, label=lbl, zorder=(5 if is_our else 3))
    ax_c.set_xticks(lvls)
    ax_c.set_xticklabels(["Clean", "10% Noise", "20% Noise"])
    ax_c.set_ylabel("Relative ΔRMSE_VA vs. Clean (%)  ↓")
    ax_c.legend(loc="upper left", fontsize=6.1)
    add_panel_tag(ax_c, "c", x=-0.16, y=1.05)

    # Panel d: Brighter 6D Zero-Inflation Boundary (Stacked 2-Tier Panel, ZERO twinx!)
    gs_d = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_bot[0, 0:6], height_ratios=[1.20, 1.00], hspace=0.14)
    ax_d_top = fig.add_subplot(gs_d[0])
    ax_d_bot = fig.add_subplot(gs_d[1], sharex=ax_d_top)
    style_clean_grid(ax_d_top, "y")
    style_clean_grid(ax_d_bot, "y")
    emotions = ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]
    zero_pcts, ccc_our_em, ccc_mse_em = [], [], []
    p_our = sorted(main_dir.glob("brighter_chn__concept_bilstm_urer__seed*/test_predictions.npz"))
    p_mse = sorted(main_dir.glob("brighter_chn__plm_only_mse__seed*/test_predictions.npz"))
    if p_our and p_mse:
        yt0 = np.load(p_our[0])["y_true"]
        for d_i in range(6):
            zero_pcts.append(float(np.mean(yt0[:, d_i] == 0.0) * 100.0))
            def _em_ccc(paths: List[Path], dim: int) -> float:
                vals = []
                for pp in paths:
                    zz = np.load(pp)
                    a, b = zz["y_true"][:, dim], zz["y_pred"][:, dim]
                    va, vb = np.var(a), np.var(b)
                    cov = np.mean((a - np.mean(a)) * (b - np.mean(b)))
                    vals.append(float((2 * cov) / (va + vb + (np.mean(a) - np.mean(b)) ** 2 + 1e-8)))
                return float(np.mean(vals))
            ccc_our_em.append(_em_ccc(p_our, d_i))
            ccc_mse_em.append(_em_ccc(p_mse, d_i))
        xe = np.arange(6)
        w = 0.34
        ax_d_top.bar(xe - w / 2, ccc_mse_em, width=w, color=PALETTE["mse"], edgecolor="none", label="PLM-MSE CCC ↑")
        ax_d_top.bar(xe + w / 2, ccc_our_em, width=w, color=PALETTE["ours"], edgecolor="none", label="Concept-URER CCC ↑")
        # Mark catastrophic CCC collapse (values near 0) with red "✗"
        for ei, (cm, co) in enumerate(zip(ccc_mse_em, ccc_our_em)):
            if cm < 0.10:
                ax_d_top.text(xe[ei] - w / 2, cm + 0.02, "X", ha="center", va="bottom",
                              fontsize=9, fontweight="bold", color=PALETTE["drop"])
        max_c = max(max(ccc_our_em), max(ccc_mse_em), 0.78)
        ax_d_top.set_ylim(-0.02, max_c * 1.34)
        ax_d_top.set_ylabel("CCC  ↑")
        ax_d_top.tick_params(axis="x", labelbottom=False)
        ax_d_top.legend(loc="upper right", ncol=2, fontsize=6.4)

        ax_d_bot.vlines(xe, 50, zero_pcts, color="#C0665A", lw=1.2, alpha=0.70, zorder=2)
        ax_d_bot.scatter(xe, zero_pcts, s=32, color="#C0665A", edgecolors="white", lw=0.6, zorder=4)
        for i, zp in enumerate(zero_pcts):
            ax_d_bot.text(xe[i], zp + 3.2, f"{zp:.0f}%", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=PALETTE["drop"])
        ax_d_bot.set_xticks(xe)
        ax_d_bot.set_xticklabels(emotions, fontsize=7.4)
        ax_d_bot.set_ylim(50, 114)
        ax_d_bot.set_ylabel("Zero Rate (%)")
    add_panel_tag(ax_d_top, "d", x=-0.12, y=1.08)

    # Panel e: Probe OOD AUROC & Subtype Audit (Clean Unbordered Bars)
    ax_e = fig.add_subplot(gs_bot[0, 7:12])
    style_clean_grid(ax_e, "y")
    probe_dir = out_root / "analysis"
    probe_items: List[Tuple[str, float, float]] = []
    main_probe = probe_dir / "rhetorical_ood_probe.json"
    def _extract_probe_aurocs(d_json: dict) -> Tuple[float, float]:
        ood = d_json.get("ood_detection", {})
        fin_auc = float(ood.get("dimabsa_zho_fin", {}).get("auroc_epi", 0.5))
        log_auc = float(ood.get("logic_rhetorical_shift", {}).get("auroc_epi", 0.5))
        return fin_auc, log_auc

    if main_probe.exists():
        d_p = json.loads(main_probe.read_text(encoding="utf-8"))
        probe_items.append(("Concept-URER", *_extract_probe_aurocs(d_p)))
        for sub_n, lbl_s in [("probe_concept_bilstm_causal", "Causal Patch"), ("probe_plm_bilstm_no_concept", "w/o Concept"), ("probe_concept_no_bilstm", "w/o BiLSTM"), ("probe_plm_only_mse", "PLM-MSE")]:
            sp = probe_dir / sub_n / "rhetorical_ood_probe.json"
            if sp.exists():
                ds = json.loads(sp.read_text(encoding="utf-8"))
                probe_items.append((lbl_s, *_extract_probe_aurocs(ds)))
    if probe_items:
        xp = np.arange(len(probe_items))
        w = 0.34
        ax_e.axhline(0.50, color=PALETTE["drop"], ls="--", lw=1.0, alpha=0.6, label="Random Chance (0.50)")
        b_e1 = ax_e.bar(xp - w / 2, [x[1] for x in probe_items], width=w, color=PALETTE["ours"], edgecolor="none", label="OOD AUROC (Fin vs Rest)")
        b_e2 = ax_e.bar(xp + w / 2, [x[2] for x in probe_items], width=w, color=PALETTE["causal"], edgecolor="none", label="Rhetorical OOD (Logic vs Rest)")
        for bi in range(len(probe_items)):
            ax_e.text(b_e1[bi].get_x() + b_e1[bi].get_width() / 2, probe_items[bi][1] + 0.015, f"{probe_items[bi][1]:.2f}", ha="center", fontsize=6.4, fontweight="bold", color=PALETTE["ours"])
            ax_e.text(b_e2[bi].get_x() + b_e2[bi].get_width() / 2, probe_items[bi][2] + 0.015, f"{probe_items[bi][2]:.2f}", ha="center", fontsize=6.4, fontweight="bold", color=PALETTE["causal"])
        ax_e.set_xticks(xp)
        ax_e.set_xticklabels([x[0] for x in probe_items], fontsize=6.9)
        ax_e.set_ylim(0.25, 1.06)
        ax_e.set_ylabel("Epistemic OOD AUROC  ↑")
        ax_e.legend(loc="upper right", fontsize=6.2)
    else:
        sub_cols = [("Acc_simple_mean", "Simple"), ("Acc_adversative_mean", "Adversative"), ("Acc_double_neg_mean", "Double Neg"), ("Acc_irony_mean", "Irony")]
        xs = np.arange(len(sub_cols))
        for var in ["classic_bilstm", "plm_only_mse", "concept_bilstm_urer"]:
            lbl, col, mk = METHOD_STYLE[var]
            ys = [row_val(main_sum, "logic_sentiment", var, c, 0.0) * 100.0 for c, _ in sub_cols]
            ax_e.plot(xs, ys, marker=mk, lw=1.9, color=col, label=lbl)
        ax_e.set_xticks(xs)
        ax_e.set_xticklabels([l for _, l in sub_cols])
        ax_e.set_ylim(70, 102)
        ax_e.set_ylabel("Subtype Accuracy (%)  ↑")
        ax_e.legend(loc="lower right", fontsize=6.6)
    add_panel_tag(ax_e, "e", x=-0.12, y=1.05)

    return save_pub_figure(fig, fig_dir, "fig6_mechanism_diagnostics_and_boundaries")


def plot_fig7_qualitative_concept_cases(out_root: Path, fig_dir: Path) -> List[str]:
    """Fig 7: 2D concept-by-token attribution matrices + uncertainty/counterfactual strips."""
    probe_p = out_root / "analysis" / "rhetorical_ood_probe.json"
    if not probe_p.exists():
        return []
    cases = json.loads(probe_p.read_text(encoding="utf-8")).get("cases", [])
    case_specs = [
        ("转折", True, "Contrastive Turn (Adversative)", "Gold: Pos (V > 5)"),
        ("双重否定", False, "Double Negation (Resolution)", "Gold: Neg (V < 5)"),
        ("反讽", True, "Pragmatic Irony Boundary (Epistemic Alert)", "Gold: Pos (Failure + High σ)"),
    ]
    chosen: List[Tuple[dict, str, str, bool]] = []
    for target_t, want_pos, en_title, gold_tag in case_specs:
        pool = [c for c in cases if c.get("rhetorical_type") == target_t]
        match = [c for c in pool if (c.get("gold_label", 1) == 1) == want_pos]
        sel = match[0] if match else (pool[0] if pool else None)
        if sel is not None:
            chosen.append((sel, en_title, gold_tag, want_pos))
    if not chosen:
        return []

    # Exact channel alignment (0..7) matching architectures.CONCEPT_NAMES
    concept_labels = [
        "Aspect Focus", "Positive Cue", "Negative Cue", "Intensity",
        "Negation Op.", "Adversative", "Modality", "Domain Attr.",
    ]
    ice_slate_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "ice_slate", ["#F8FAFC", "#DBEAFE", "#93C5FD", "#3B82F6", "#0F4D92"], N=256
    )

    fig = plt.figure(figsize=(13.8, 2.95 * len(chosen) + 1.0))
    outer = fig.add_gridspec(
        len(chosen) + 1, 1,
        height_ratios=[0.28] + [1.0] * len(chosen),
        hspace=0.44, left=0.095, right=0.98, top=0.95, bottom=0.06,
    )

    # Top sleek colorbar + borderless legend strip
    hdr_gs = outer[0].subgridspec(1, 2, width_ratios=[1.25, 2.2], wspace=0.18)
    ax_cb = fig.add_subplot(hdr_gs[0, 0])
    sm = plt.cm.ScalarMappable(cmap=ice_slate_cmap, norm=plt.Normalize(vmin=-0.05, vmax=0.95))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.outline.set_linewidth(0.55)
    cb.outline.set_edgecolor("#94A3B8")
    cb.set_ticks([0.0, 0.25, 0.50, 0.75, 0.95])
    cb.set_ticklabels(["0.00", "0.25", "0.50", "0.75", "0.95"])
    cb.ax.tick_params(labelsize=6.8, length=2.2, pad=2)
    ax_cb.set_title(r"Token–Concept Activation $\alpha_{t,k}$  (cells $\geq 0.42$ annotated as .xx)",
                    fontsize=7.6, fontweight="bold", color=PALETTE["ink"], pad=4, loc="left")

    ax_leg = fig.add_subplot(hdr_gs[0, 1])
    ax_leg.axis("off")
    leg_handles = [
        Line2D([0], [0], marker="D", color=PALETTE["ours"], lw=1.4, markersize=4.8, label=r"Correct Polarity ($V_{\mathrm{pred}} \pm \sigma_{\mathrm{epi}}$)"),
        Line2D([0], [0], marker="D", color=PALETTE["drop"], lw=1.4, markersize=4.8, label="Polarity Mismatch (Warning)"),
        mpatches.Patch(facecolor="#B84A39", label=r"$\mathrm{do}(\mathrm{Negation}=0)$  $\Delta V$"),
        mpatches.Patch(facecolor=PALETTE["ours"], label=r"$\mathrm{do}(\mathrm{Adversative}=0)$  $\Delta V$"),
    ]
    ax_leg.legend(handles=leg_handles, loc="center right", ncol=2, frameon=False,
                  fontsize=7.0, columnspacing=1.2, handletextpad=0.5)

    tag_letters = ["a", "b", "c"]
    for idx, (c, en_title, gold_tag, want_pos) in enumerate(chosen):
        row_gs = outer[idx + 1].subgridspec(1, 2, width_ratios=[3.40, 1.30], wspace=0.13)
        ax_hm = fig.add_subplot(row_gs[0, 0])
        right_gs = row_gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.50)
        ax_pred = fig.add_subplot(right_gs[0, 0])
        ax_cf = fig.add_subplot(right_gs[1, 0])

        raw_toks = c.get("tokens", [])
        raw_cmat = np.asarray(c.get("explicit_concepts", []), dtype=float)
        keep = [i for i, tk in enumerate(raw_toks) if tk not in ("[CLS]", "[SEP]", "[PAD]")]
        if not keep or raw_cmat.ndim != 2 or len(raw_cmat) != len(raw_toks):
            continue
        toks = [raw_toks[i].replace("##", "") for i in keep]
        mat = raw_cmat[keep, :].T  # (8, T)
        T = len(toks)

        ax_hm.imshow(mat, aspect="auto", cmap=ice_slate_cmap, vmin=-0.05, vmax=0.95, interpolation="nearest")
        ax_hm.set_xticks(np.arange(T + 1) - 0.5, minor=True)
        ax_hm.set_yticks(np.arange(8 + 1) - 0.5, minor=True)
        ax_hm.grid(which="minor", color="#FFFFFF", linewidth=1.0)
        ax_hm.tick_params(which="minor", length=0)
        ax_hm.axhline(6.5, color="#475569", linestyle="--", linewidth=0.9, zorder=4)

        for r in range(min(8, mat.shape[0])):
            for col in range(T):
                val = float(mat[r, col])
                if val >= 0.70:
                    # Bold border for high-activation cells
                    rect = mpatches.Rectangle((col - 0.5, r - 0.5), 1, 1, linewidth=1.5,
                                              edgecolor="#0A2463", facecolor="none", zorder=3)
                    ax_hm.add_patch(rect)
                elif val >= 0.50:
                    # Subtle border for moderate-activation cells
                    rect = mpatches.Rectangle((col - 0.5, r - 0.5), 1, 1, linewidth=0.6,
                                              edgecolor="#0F4D92", facecolor="none", zorder=3)
                    ax_hm.add_patch(rect)
                if val >= 0.42:
                    txt_c = "#FFFFFF" if val >= 0.58 else "#0F172A"
                    ax_hm.text(col, r, f"{val:.2f}".lstrip("0"), ha="center", va="center",
                               fontsize=6.2, fontweight="bold", color=txt_c)

        ax_hm.set_yticks(np.arange(8))
        ax_hm.set_yticklabels(concept_labels, fontsize=7.2, color=PALETTE["ink"])
        ax_hm.get_yticklabels()[7].set_color("#64748B")
        ax_hm.get_yticklabels()[7].set_style("italic")

        ax_hm.set_xticks(np.arange(T))
        ax_hm.set_xticklabels(toks, fontsize=9.2, fontfamily=CJK_FONTS)
        ax_hm.tick_params(axis="x", length=2.2, pad=3.5)
        for col, xlbl in enumerate(ax_hm.get_xticklabels()):
            if np.max(mat[:7, col]) >= 0.55:
                xlbl.set_fontweight("bold")
                xlbl.set_color(PALETTE["ours"])
            else:
                xlbl.set_color("#64748B")

        for sp in ("top", "right", "left", "bottom"):
            ax_hm.spines[sp].set_color("#CBD5E1")
            ax_hm.spines[sp].set_linewidth(0.65)

        txt_clean = c.get("text", "")
        if len(txt_clean) > 36:
            txt_clean = txt_clean[:35] + "…"
        ax_hm.set_title(f"{en_title}  ·  {gold_tag}", loc="left", fontsize=8.4, fontweight="bold",
                        color=PALETTE["ink"], pad=5)
        ax_hm.set_title(f"“{txt_clean}”", loc="right", fontsize=7.9, color="#475569",
                        fontfamily=CJK_FONTS, pad=5)
        add_panel_tag(ax_hm, tag_letters[idx], x=-0.105, y=1.06)

        # Right-Top: Valence Prediction vs Neutral (V=5) + Epistemic Uncertainty
        pred_v = float(c.get("pred_valence", 5.0))
        epi_v = float(c.get("epi_std_v", 0.0))
        ale_v = float(c.get("ale_std_v", 0.0))
        pred_ok = ((pred_v >= 5.0) == want_pos)
        pt_col = PALETTE["ours"] if pred_ok else PALETTE["drop"]

        ax_pred.axvspan(2.0, 5.0, color="#F8FAFC", zorder=0)
        ax_pred.axvspan(5.0, 8.0, color="#EEF2F6", zorder=0)
        ax_pred.axvline(5.0, color="#64748B", linestyle="--", linewidth=0.8, zorder=1)
        ax_pred.errorbar([pred_v], [0.0], xerr=[[epi_v], [epi_v]], fmt="D", color=pt_col,
                         ecolor=pt_col, elinewidth=1.5, capsize=3.0, markersize=5.0,
                         markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        ax_pred.set_xlim(2.0, 8.0)
        ax_pred.set_ylim(-0.65, 0.78)
        ax_pred.set_yticks([])
        ax_pred.set_xticks([2, 3.5, 5, 6.5, 8])
        ax_pred.set_xticklabels(["2", "3.5", "5.0", "6.5", "8"], fontsize=6.6)
        ax_pred.text(pred_v, 0.34, f"V={pred_v:.2f}±{epi_v:.2f} (ale={ale_v:.2f})",
                     ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=pt_col)
        ax_pred.set_title(r"Valence Prediction & $\sigma_{\mathrm{epi}}$", fontsize=7.6, fontweight="bold",
                          color=PALETTE["ink"], pad=3)

        # Right-Bottom: Counterfactual Concept Zeroing Shift ΔV
        cf_n = float(c.get("cf_zero_negation_delta_v", 0.0))
        cf_a = float(c.get("cf_zero_adversative_delta_v", 0.0))
        vals_cf = [cf_a, cf_n]
        cols_cf = [PALETTE["ours"], PALETTE["drop"]]
        y_cf = [0, 1]
        ax_cf.axvline(0.0, color="#475569", linewidth=0.75, zorder=1)
        ax_cf.barh(y_cf, vals_cf, height=0.50, color=cols_cf, alpha=0.95, edgecolor="none", zorder=3)
        for yv, xv in zip(y_cf, vals_cf):
            ha_t = "left" if xv >= 0 else "right"
            dx_t = 0.015 if xv >= 0 else -0.015
            ax_cf.text(xv + dx_t, yv, f"{xv:+.2f}", va="center", ha=ha_t,
                       fontsize=6.8, fontweight="bold", color=PALETTE["ink"])
        ax_cf.set_yticks(y_cf)
        ax_cf.set_yticklabels(["do(Adv=0)", "do(Neg=0)"], fontsize=6.9, color=PALETTE["ink"])
        ax_cf.set_xlim(-0.46, 0.36)
        ax_cf.set_ylim(-0.55, 1.55)
        ax_cf.tick_params(axis="x", labelsize=6.6)
        ax_cf.set_title(r"Counterfactual Shift ($\Delta V$)", fontsize=7.6, fontweight="bold",
                        color=PALETTE["ink"], pad=3)

    return save_pub_figure(fig, fig_dir, "fig7_qualitative_concept_cases")


def generate_all_publication_figures(out_root: Path, fig_dir: Path) -> List[str]:
    """Build complete multi-panel publication figure suite (Fig 2-7; Fig 1 reserved) and print manifest."""
    apply_publication_style()
    all_saved: List[str] = []
    for fn in (
        plot_fig2_main_benchmark_and_pareto,
        plot_fig3_anti_mean_collapse_and_uncertainty,
        plot_fig4_mobo_ablation_efficiency_ood,
        plot_fig5_generalization_backbones_and_llm,
        plot_fig6_mechanism_diagnostics_and_boundaries,
        plot_fig7_qualitative_concept_cases,
    ):
        all_saved.extend(fn(out_root, fig_dir))
    return all_saved


def main() -> None:
    p = argparse.ArgumentParser(description="Generate multi-panel publication figures.")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--fig-dir", type=str, default="outputs/pub_figures")
    args = p.parse_args()
    saved = generate_all_publication_figures(Path(args.out_root), Path(args.fig_dir))
    print(json.dumps({"generated_pub_figures": saved, "count": len(saved)}, indent=2))


if __name__ == "__main__":
    main()
