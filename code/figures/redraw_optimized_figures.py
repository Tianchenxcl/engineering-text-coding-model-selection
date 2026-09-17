from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DATA_ROOT = ROOT / "results"
OUT = ROOT / "reproduced" / "figures"
SOURCE_OUT = OUT / "source_data"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_OUT.mkdir(parents=True, exist_ok=True)


COLORS = {
    "pale_apricot": "#FFE7CC",
    "soft_orange": "#FFB66D",
    "burnt_orange": "#F27C2B",
    "warm_sand": "#E7C9A0",
    "cool_mist": "#DCE6F2",
    "sky_blue": "#A7CFE2",
    "cerulean": "#3B82D6",
    "deep_cobalt": "#0D47A1",
    "ink": "#172033",
    "muted": "#697386",
    "gray": "#9AA5B1",
    "light_gray": "#D9DEE5",
    "grid": "#E7EBF0",
    "white": "#FFFFFF",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 9.1,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.8,
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "text.color": COLORS["ink"],
        "figure.facecolor": COLORS["white"],
        "axes.facecolor": COLORS["white"],
        "savefig.facecolor": COLORS["white"],
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)


def assert_close(actual: float, expected: float, tol: float = 1e-9, label: str = "value") -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, obtained {actual}")


def clean_axis(ax: mpl.axes.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.75, pad=2.5)
    if grid:
        ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11.2,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def save_all(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=400)
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def copy_source(path: Path, target_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    shutil.copy2(path, SOURCE_OUT / target_name)
    return pd.read_csv(path)


def plot_figure1() -> dict:
    overall = copy_source(
        DATA_ROOT / "step2_descriptive_reliability" / "figure_source_data" / "figure1_overall_units.csv",
        "figure1_overall_units.csv",
    )
    bands = copy_source(
        DATA_ROOT / "step2_descriptive_reliability" / "figure_source_data" / "figure1_reliability_bands.csv",
        "figure1_reliability_bands.csv",
    )
    order = ["SVM", "DeBERTa", "LLM zero-shot", "LLM few-shot", "LLM codebook"]
    display = {
        "SVM": "SVM",
        "DeBERTa": "DeBERTa",
        "LLM zero-shot": "LLM\nzero-shot",
        "LLM few-shot": "LLM\nfew-shot",
        "LLM codebook": "LLM\ncodebook",
    }
    colors = {
        "SVM": COLORS["burnt_orange"],
        "DeBERTa": COLORS["cerulean"],
        "LLM zero-shot": COLORS["gray"],
        "LLM few-shot": COLORS["soft_orange"],
        "LLM codebook": COLORS["deep_cobalt"],
    }
    counts = overall.groupby("candidate").size().to_dict()
    assert counts == {"DeBERTa": 240, "LLM codebook": 40, "LLM few-shot": 40, "LLM zero-shot": 40, "SVM": 240}
    means = overall.groupby("candidate")["kappa"].mean().to_dict()
    assert_close(means["SVM"], 0.5309004983532282, label="Figure 1 SVM mean")
    assert_close(means["DeBERTa"], 0.5372490617015548, label="Figure 1 DeBERTa mean")
    assert_close(means["LLM codebook"], 0.5177688111616436, label="Figure 1 codebook mean")
    band_sums = bands.groupby("candidate")["share"].sum()
    if not np.allclose(band_sums.to_numpy(float), 1.0):
        raise AssertionError("Figure 1 reliability-band shares do not sum to one")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.35, 3.05),
        gridspec_kw={"width_ratios": [1.28, 1.0]},
    )
    ax = axes[0]
    rng = np.random.default_rng(20260725)
    vals = [overall.loc[overall["candidate"] == c, "kappa"].to_numpy(float) for c in order]
    box = ax.boxplot(
        vals,
        positions=np.arange(len(order)),
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["ink"], "linewidth": 1.4},
        whiskerprops={"color": COLORS["muted"], "linewidth": 0.9},
        capprops={"color": COLORS["muted"], "linewidth": 0.9},
        boxprops={"edgecolor": COLORS["muted"], "linewidth": 0.9},
    )
    for patch, candidate in zip(box["boxes"], order):
        patch.set_facecolor(colors[candidate])
        patch.set_alpha(0.22)
    for i, candidate in enumerate(order):
        y = overall.loc[overall["candidate"] == candidate, "kappa"].to_numpy(float)
        jitter = rng.normal(0, 0.070 if len(y) > 100 else 0.055, size=len(y))
        ax.scatter(
            i + jitter,
            y,
            s=8.5 if len(y) > 100 else 13,
            color=colors[candidate],
            alpha=0.30 if len(y) > 100 else 0.48,
            edgecolors="none",
            rasterized=False,
            zorder=2,
        )
    ax.axhline(0.60, color=COLORS["muted"], linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    ax.set_xticks(np.arange(len(order)), [display[c] for c in order])
    ax.set_ylabel("Cohen's kappa")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("Task-level reliability distributions", pad=7)
    clean_axis(ax)
    panel_label(ax, "A", x=-0.12)

    ax = axes[1]
    band_order = ["<0.20", "0.20-<0.40", "0.40-<0.60", "0.60-<0.80", ">=0.80"]
    band_colors = [
        COLORS["pale_apricot"],
        COLORS["soft_orange"],
        COLORS["light_gray"],
        COLORS["sky_blue"],
        COLORS["deep_cobalt"],
    ]
    pivot = bands.pivot(index="candidate", columns="kappa_band", values="share").reindex(order).fillna(0.0)
    left = np.zeros(len(order))
    for band, color in zip(band_order, band_colors):
        shares = pivot[band].to_numpy(float)
        ax.barh(np.arange(len(order)), shares, left=left, height=0.54, color=color, edgecolor="white", linewidth=0.55)
        for i, (x0, share) in enumerate(zip(left, shares)):
            if share >= 0.16:
                txt_color = COLORS["white"] if band == ">=0.80" else COLORS["ink"]
                ax.text(x0 + share / 2, i, f"{share:.0%}", ha="center", va="center", fontsize=6.4, color=txt_color)
        left += shares
    ax.set_yticks(np.arange(len(order)), [display[c].replace("\n", " ") for c in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of analysis units")
    ax.set_title("Distribution across kappa bands", pad=7)
    clean_axis(ax, grid="x")
    panel_label(ax, "B", x=-0.24)
    handles = [mpl.patches.Patch(facecolor=c, edgecolor="none", label=b.replace(">=", "≥").replace("-<", "–<")) for b, c in zip(band_order, band_colors)]
    ax.legend(handles=handles, title="Kappa", title_fontsize=7.2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False, columnspacing=1.1, handlelength=1.4)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.85, bottom=0.27, wspace=0.40)
    save_all(fig, "Figure1_reliability_distributions")
    return {"n": int(len(overall)), "means": means}


def plot_figure2() -> dict:
    grid = copy_source(
        DATA_ROOT / "step3_supervised_complexity" / "predictions" / "adjusted_kappa_prediction_grid.csv",
        "figure2_adjusted_kappa_prediction_grid.csv",
    )
    diff = copy_source(
        DATA_ROOT / "step3_supervised_complexity" / "predictions" / "adjusted_model_difference_grid.csv",
        "figure2_adjusted_model_difference_grid.csv",
    )
    key = grid[(grid["dimension"] == "label_number") & (grid["model_code"] == "SVM")].set_index("x_original")
    assert_close(key.loc[8.0, "predicted_kappa"], 0.4354476018510821, label="Figure 2 SVM 8-label prediction")
    dkey = diff[(diff["dimension"] == "log2_training_size_z")].set_index("x_original")
    assert_close(dkey.loc[3000.0, "predicted_difference_deberta_minus_svm"], 0.0447389, tol=1e-6, label="Figure 2 3000-sample contrast")

    model_order = ["SVM", "DeBERTa"]
    model_colors = {"SVM": COLORS["burnt_orange"], "DeBERTa": COLORS["cerulean"]}
    model_markers = {"SVM": "o", "DeBERTa": "s"}
    model_styles = {"SVM": "-", "DeBERTa": "--"}
    specs = [
        ("label_number", "Label number", "Label number"),
        ("class_imbalance_z", "Class imbalance index", "Class imbalance"),
        ("annotation_disagreement_z", "Human coding\ndisagreement rate", "Human coding\ndisagreement rate"),
        ("log2_training_size_z", "Training sample size", "Training sample size"),
        ("semantic_similarity_z", "Label semantic similarity", "Label semantic similarity"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 5.12), sharey=False)
    for panel, (ax, (dim, xlabel, title)) in enumerate(zip(axes.flat[:5], specs)):
        subset = grid[grid["dimension"] == dim]
        for model in model_order:
            g = subset[subset["model_code"] == model].sort_values("x_original")
            x = g["x_original"].to_numpy(float)
            y = g["predicted_kappa"].to_numpy(float)
            lo = g["ci95_low"].to_numpy(float)
            hi = g["ci95_high"].to_numpy(float)
            ax.fill_between(x, lo, hi, color=model_colors[model], alpha=0.11, linewidth=0, zorder=1)
            ax.plot(x, y, color=model_colors[model], linestyle=model_styles[model], linewidth=1.65, marker=model_markers[model] if len(x) <= 10 else None, markersize=4.1, markeredgecolor="white", markeredgewidth=0.45, label=model, zorder=3)
        ax.set_title(title, pad=6)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.0)
        if panel not in (0, 3):
            ax.tick_params(labelleft=False)
        clean_axis(ax)
        panel_label(ax, chr(ord("A") + panel))
        if dim == "label_number":
            ax.set_xticks([3, 5, 8, 11], ["3", "5", "8", "11"])
        if dim == "log2_training_size_z":
            ticks = [300, 3000, 6000, 9000, 12000, 15000]
            ax.set_xticks(ticks, ["0.3k", "3k", "6k", "9k", "12k", "15k"])
    axes[0, 0].set_ylabel("Adjusted predicted kappa")
    axes[1, 0].set_ylabel("Adjusted predicted kappa")
    handles = [
        Line2D([0], [0], color=model_colors[m], linestyle=model_styles[m], marker=model_markers[m], markersize=4.5, linewidth=1.6, label=m)
        for m in model_order
    ]
    axes[0, 0].legend(handles=handles, loc="upper right", frameon=False)

    ax = axes[1, 2]
    g = diff[diff["dimension"] == "log2_training_size_z"].sort_values("x_original")
    x = g["x_original"].to_numpy(float)
    y = g["predicted_difference_deberta_minus_svm"].to_numpy(float)
    lo = g["ci95_low"].to_numpy(float)
    hi = g["ci95_high"].to_numpy(float)
    ax.fill_between(x, lo, hi, color=COLORS["deep_cobalt"], alpha=0.12, linewidth=0)
    ax.plot(x, y, color=COLORS["deep_cobalt"], linewidth=1.7, marker="D", markersize=4.1, markeredgecolor="white", markeredgewidth=0.45)
    ax.axhline(0, color=COLORS["muted"], linewidth=0.9, linestyle=(0, (3, 2)))
    ax.set_title("DeBERTa minus SVM", pad=6)
    ax.set_xlabel("Training sample size")
    ax.set_ylabel("Adjusted kappa difference")
    ax.set_xticks([300, 3000, 6000, 9000, 12000, 15000], ["0.3k", "3k", "6k", "9k", "12k", "15k"])
    ax.set_ylim(-0.065, 0.14)
    clean_axis(ax)
    panel_label(ax, "F")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.10, wspace=0.34, hspace=0.42)
    save_all(fig, "Figure2_supervised_complexity_response")
    return {"grid_rows": int(len(grid)), "difference_rows": int(len(diff))}


def plot_figure3() -> dict:
    curves = copy_source(
        DATA_ROOT / "step4_llm_prompt_analysis" / "predictions" / "llm_adjusted_performance_grid.csv",
        "figure3_llm_adjusted_performance_grid.csv",
    )
    key = curves[(curves["dimension"] == "label_number") & (curves["prompt_strategy"] == "codebook_guided")].set_index("x_original")
    assert_close(key.loc[3.0, "predicted_performance"], 0.561811, tol=1e-5, label="Figure 3 codebook 3-label prediction")
    prompts = ["zero_shot", "few_shot", "codebook_guided"]
    labels = {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "codebook_guided": "Codebook-guided"}
    colors = {"zero_shot": COLORS["gray"], "few_shot": COLORS["burnt_orange"], "codebook_guided": COLORS["deep_cobalt"]}
    markers = {"zero_shot": "o", "few_shot": "^", "codebook_guided": "s"}
    styles = {"zero_shot": ":", "few_shot": "--", "codebook_guided": "-"}
    specs = [
        ("label_number", "Label number", "Label number"),
        ("annotation_disagreement_z", "Human coding\ndisagreement rate", "Human coding\ndisagreement rate"),
        ("semantic_similarity_z", "Label semantic similarity", "Label semantic similarity"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.62), sharey=True)
    for panel, (ax, (dim, xlabel, title)) in enumerate(zip(axes, specs)):
        subset = curves[curves["dimension"] == dim]
        for prompt in prompts:
            g = subset[subset["prompt_strategy"] == prompt].sort_values("x_original")
            x = g["x_original"].to_numpy(float)
            y = g["predicted_performance"].to_numpy(float)
            lo = g["ci95_low"].to_numpy(float)
            hi = g["ci95_high"].to_numpy(float)
            ax.fill_between(x, lo, hi, color=colors[prompt], alpha=0.10, linewidth=0)
            ax.plot(x, y, color=colors[prompt], linestyle=styles[prompt], linewidth=1.55, marker=markers[prompt] if len(x) <= 10 else None, markersize=4.0, markeredgecolor="white", markeredgewidth=0.45, label=labels[prompt])
        ax.set_title(title, pad=6)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.20, 0.70)
        clean_axis(ax)
        panel_label(ax, chr(ord("A") + panel))
        if dim == "label_number":
            ax.set_xticks([3, 5, 8, 11], ["3", "5", "8", "11"])
    axes[0].set_ylabel("Adjusted predicted kappa")
    handles = [Line2D([0], [0], color=colors[p], linestyle=styles[p], marker=markers[p], markersize=4.5, linewidth=1.6, label=labels[p]) for p in prompts]
    axes[0].legend(handles=handles, loc="upper right", frameon=False)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.22, wspace=0.30)
    save_all(fig, "Figure3_llm_prompt_complexity_response")
    return {"rows": int(len(curves))}


def plot_figure4() -> dict:
    data = copy_source(
        DATA_ROOT / "step5_common24_cross_model" / "diagnostics" / "common24_validated_analysis_records.csv",
        "figure4_common24_records.csv",
    )
    summaries = copy_source(
        DATA_ROOT / "step5_common24_cross_model" / "tables" / "common24_model_summary.csv",
        "figure4_common24_model_summary.csv",
    )
    pairwise = copy_source(
        DATA_ROOT / "step5_common24_cross_model" / "tables" / "common24_pairwise_comparisons.csv",
        "figure4_common24_pairwise_comparisons.csv",
    )
    if len(data) != 72 or data["combination_id"].nunique() != 24:
        raise AssertionError("Figure 4 must contain 24 matched combinations and 72 records")
    ksum = summaries[summaries["outcome"] == "kappa"].set_index("candidate_short")
    assert_close(ksum.loc["SVM", "mean"], 0.672144860225933, label="Figure 4 SVM mean")
    assert_close(ksum.loc["DeBERTa", "mean"], 0.7061543915592385, label="Figure 4 DeBERTa mean")
    assert_close(ksum.loc["LLM codebook", "mean"], 0.5029438446207436, label="Figure 4 LLM mean")

    order = ["SVM", "DeBERTa", "LLM codebook"]
    colors = {"SVM": COLORS["burnt_orange"], "DeBERTa": COLORS["cerulean"], "LLM codebook": COLORS["deep_cobalt"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.25), gridspec_kw={"width_ratios": [0.78, 1.72]})
    ax = axes[0]
    wide = data.pivot(index="combination_id", columns="candidate_short", values="kappa").reindex(columns=order)
    for _, row in wide.iterrows():
        ax.plot(np.arange(3), row.to_numpy(float), color=COLORS["light_gray"], linewidth=0.55, alpha=0.55, zorder=1)
    rng = np.random.default_rng(20260725)
    for i, candidate in enumerate(order):
        values = wide[candidate].to_numpy(float)
        ax.scatter(i + rng.normal(0, 0.035, len(values)), values, s=14, color=colors[candidate], alpha=0.48, edgecolors="none", zorder=2)
        row = ksum.loc[candidate]
        ax.errorbar(i, row["mean"], yerr=[[row["mean"] - row["bootstrap_ci95_low"]], [row["bootstrap_ci95_high"] - row["mean"]]], fmt="o", color=COLORS["ink"], markerfacecolor=colors[candidate], markeredgecolor="white", markeredgewidth=0.7, markersize=6.8, capsize=3, linewidth=1.1, zorder=4)
    ax.set_xticks([0, 1, 2], ["SVM", "DeBERTa", "LLM\ncodebook"])
    ax.set_ylabel("Cohen's kappa")
    ax.set_ylim(0.22, 0.94)
    ax.set_title("Common24 matched tasks", pad=7)
    clean_axis(ax)
    panel_label(ax, "A", x=-0.16)
    ax.text(0.02, 0.03, "Lines pair tasks; large markers show mean and 95% CI", transform=ax.transAxes, fontsize=5.5, color=COLORS["muted"])

    ax = axes[1]
    outcome_order = ["kappa", "macro_f1", "mcc"]
    contrast_order = ["deberta_minus_svm", "llm_minus_svm", "deberta_minus_llm"]
    contrast_labels = {
        "deberta_minus_svm": "DeBERTa − SVM",
        "llm_minus_svm": "LLM codebook − SVM",
        "deberta_minus_llm": "DeBERTa − LLM codebook",
    }
    outcome_labels = {"kappa": "κ", "macro_f1": "F1", "mcc": "MCC"}
    contrast_colors = {
        "deberta_minus_svm": COLORS["cerulean"],
        "llm_minus_svm": COLORS["burnt_orange"],
        "deberta_minus_llm": COLORS["deep_cobalt"],
    }
    ys, labels_y = [], []
    y = 0.0
    for oi, outcome in enumerate(outcome_order):
        for contrast in contrast_order:
            row = pairwise[(pairwise["outcome"] == outcome) & (pairwise["contrast"] == contrast)].iloc[0]
            ax.errorbar(row["mean_difference"], y, xerr=[[row["mean_difference"] - row["bootstrap_ci95_low"]], [row["bootstrap_ci95_high"] - row["mean_difference"]]], fmt="o", color=contrast_colors[contrast], markeredgecolor="white", markeredgewidth=0.45, markersize=5.0, capsize=2.5, linewidth=1.2, zorder=3)
            ys.append(y)
            labels_y.append(f"{outcome_labels[outcome]} · {contrast_labels[contrast].replace('LLM codebook', 'LLM')}")
            y += 1.0
        if oi < len(outcome_order) - 1:
            ax.axhline(y - 0.48, color=COLORS["grid"], linewidth=0.8)
            y += 0.45
    ax.axvline(0, color=COLORS["muted"], linewidth=0.9, linestyle=(0, (3, 2)))
    ax.set_yticks(ys, labels_y)
    ax.tick_params(axis="y", labelsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(-0.245, 0.285)
    ax.set_xlabel("Mean paired performance difference")
    ax.set_title("Paired gaps with 95% task-bootstrap CIs", pad=7)
    clean_axis(ax, grid="x")
    panel_label(ax, "B", x=-0.30)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.17, wspace=0.62)
    save_all(fig, "Figure4_common24_model_comparison")
    return {"records": int(len(data)), "combinations": int(data["combination_id"].nunique())}


def regression_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    actual = frame["actual"].to_numpy(float)
    prediction = frame["prediction"].to_numpy(float)
    rmse = float(np.sqrt(np.mean((actual - prediction) ** 2)))
    denom = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - np.sum((actual - prediction) ** 2) / denom)
    return rmse, r2


def calibration_bins(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    temp = frame[["actual", "prediction"]].copy()
    temp["bin"] = pd.qcut(temp["actual"], q=bins, duplicates="drop")
    return temp.groupby("bin", observed=True).agg(actual=("actual", "mean"), prediction=("prediction", "mean")).reset_index(drop=True)


def calibration_panel(ax: mpl.axes.Axes, frame: pd.DataFrame, color: str, title: str, label: str, lim: tuple[float, float], show_ylabel: bool = True) -> tuple[float, float]:
    rmse, r2 = regression_metrics(frame)
    binned = calibration_bins(frame)
    ax.scatter(frame["actual"], frame["prediction"], s=8.5, color=color, alpha=0.20, edgecolors="none", zorder=1)
    ax.plot(lim, lim, color=COLORS["muted"], linestyle=(0, (3, 2)), linewidth=0.9, zorder=0)
    ax.plot(binned["actual"], binned["prediction"], color=COLORS["ink"], linewidth=1.45, marker="o", markersize=4.0, markerfacecolor=COLORS["white"], markeredgewidth=0.9, zorder=3)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed kappa")
    if show_ylabel:
        ax.set_ylabel("Out-of-fold predicted kappa")
    ax.set_title(title, pad=6)
    ax.text(0.04, 0.95, f"RMSE = {rmse:.3f}\n$R^2$ = {r2:.3f}\n$n$ = {len(frame)}", transform=ax.transAxes, ha="left", va="top", fontsize=7.0)
    clean_axis(ax)
    panel_label(ax, label, x=-0.15)
    return rmse, r2


def prediction_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    record = copy_source(
        DATA_ROOT / "step6_record_stratified_10fold" / "predictions" / "record_10fold_oof_predictions.csv",
        "figures5_6_record_stratified_oof_predictions.csv",
    )
    grouped = copy_source(
        DATA_ROOT / "step6_performance_prediction" / "predictions" / "prediction_oof_records.csv",
        "figures5_6_grouped_oof_predictions.csv",
    )
    algorithms = ["candidate_mean", "ridge", "gradient_boosting", "mlp"]
    for name, frame in [("record", record), ("grouped", grouped)]:
        counts = frame.groupby(["outcome", "algorithm"]).size()
        if set(frame["outcome"]) != {"kappa", "macro_f1", "mcc"} or set(frame["algorithm"]) != set(algorithms):
            raise AssertionError(f"Unexpected {name} prediction structure")
        if not (counts == 600).all():
            raise AssertionError(f"Each {name} outcome-algorithm cell must contain 600 out-of-fold records")
    return record, grouped


def plot_figure5(record: pd.DataFrame, grouped: pd.DataFrame) -> dict:
    record_k = record[record["outcome"] == "kappa"].copy()
    grouped_k = grouped[grouped["outcome"] == "kappa"].copy()
    algorithms = ["candidate_mean", "ridge", "gradient_boosting", "mlp"]
    labels = ["Configuration\nmean", "Ridge", "Gradient\nboosting", "MLP"]
    metrics_rows = []
    for validation, frame in [("Record-stratified", record_k), ("Grouped by label combination", grouped_k)]:
        for algorithm in algorithms:
            rmse, r2 = regression_metrics(frame[frame["algorithm"] == algorithm])
            metrics_rows.append({"validation": validation, "algorithm": algorithm, "rmse": rmse, "r2": r2})
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(SOURCE_OUT / "figure5_kappa_validation_metrics.csv", index=False)
    lookup = metrics.set_index(["validation", "algorithm"])
    assert_close(lookup.loc[("Record-stratified", "gradient_boosting"), "rmse"], 0.0841310, tol=1e-6, label="Figure 5 record gradient RMSE")
    assert_close(lookup.loc[("Grouped by label combination", "ridge"), "rmse"], 0.1104136, tol=1e-6, label="Figure 5 grouped ridge RMSE")

    fig = plt.figure(figsize=(7.35, 6.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.45], hspace=0.48, wspace=0.31)
    ax_rmse = fig.add_subplot(gs[0, 0])
    ax_r2 = fig.add_subplot(gs[0, 1])
    validation_colors = {"Record-stratified": COLORS["cerulean"], "Grouped by label combination": COLORS["burnt_orange"]}
    validation_markers = {"Record-stratified": "o", "Grouped by label combination": "s"}
    x = np.arange(len(algorithms))
    for validation in validation_colors:
        sub = metrics[metrics["validation"] == validation].set_index("algorithm").loc[algorithms]
        ax_rmse.plot(x, sub["rmse"], color=validation_colors[validation], marker=validation_markers[validation], markersize=5.0, markeredgecolor="white", markeredgewidth=0.55, linewidth=1.65, label=validation)
        ax_r2.plot(x, sub["r2"], color=validation_colors[validation], marker=validation_markers[validation], markersize=5.0, markeredgecolor="white", markeredgewidth=0.55, linewidth=1.65, label=validation)
    for ax, ylabel, title, label in [(ax_rmse, "RMSE (lower is better)", "Prediction error", "A"), (ax_r2, "$R^2$ (higher is better)", "Explained variation", "B")]:
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=6)
        clean_axis(ax)
        panel_label(ax, label)
    ax_rmse.set_ylim(0.07, 0.18)
    ax_r2.set_ylim(-0.05, 0.82)
    ax_r2.legend(loc="lower right", frameon=False)

    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    lim = (-0.02, 1.12)
    record_best = record_k[record_k["algorithm"] == "gradient_boosting"]
    grouped_best = grouped_k[grouped_k["algorithm"] == "ridge"]
    calibration_panel(ax_c, record_best, COLORS["cerulean"], "Best interpolation: gradient boosting", "C", lim, True)
    calibration_panel(ax_d, grouped_best, COLORS["burnt_orange"], "Best transfer: ridge", "D", lim, False)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.95, bottom=0.08)
    save_all(fig, "Figure5_model_performance_prediction")
    return {"metrics": metrics_rows}


def plot_figure6(grouped: pd.DataFrame) -> dict:
    grouped_k = grouped[grouped["outcome"] == "kappa"].copy()
    algorithms = ["candidate_mean", "ridge", "gradient_boosting", "mlp"]
    titles = ["Configuration mean", "Ridge", "Gradient boosting", "MLP (supplementary)"]
    colors = [COLORS["gray"], COLORS["deep_cobalt"], COLORS["burnt_orange"], COLORS["warm_sand"]]
    fig, axes = plt.subplots(1, 4, figsize=(7.35, 2.46), sharex=True, sharey=True)
    metrics = []
    lim = (-0.02, 1.05)
    for i, (ax, algorithm, title, color) in enumerate(zip(axes, algorithms, titles, colors)):
        frame = grouped_k[grouped_k["algorithm"] == algorithm]
        rmse, r2 = calibration_panel(ax, frame, color, title, chr(ord("A") + i), lim, show_ylabel=(i == 0))
        if i > 0:
            ax.set_ylabel("")
        metrics.append({"algorithm": algorithm, "rmse": rmse, "r2": r2})
    fig.subplots_adjust(left=0.075, right=0.995, top=0.84, bottom=0.22, wspace=0.16)
    save_all(fig, "Figure6_grouped_holdout_calibration")
    return {"metrics": metrics}


def main() -> None:
    qa = {
        "palette": COLORS,
        "figure1": plot_figure1(),
        "figure2": plot_figure2(),
        "figure3": plot_figure3(),
        "figure4": plot_figure4(),
    }
    record, grouped = prediction_sources()
    qa["figure5"] = plot_figure5(record, grouped)
    qa["figure6"] = plot_figure6(grouped)
    outputs = sorted(p.name for p in OUT.iterdir() if p.is_file())
    expected = [f"Figure{i}_{suffix}" for i, suffix in [
        (1, "reliability_distributions"),
        (2, "supervised_complexity_response"),
        (3, "llm_prompt_complexity_response"),
        (4, "common24_model_comparison"),
        (5, "model_performance_prediction"),
        (6, "grouped_holdout_calibration"),
    ]]
    for stem in expected:
        for ext in ["svg", "pdf", "png", "tiff"]:
            path = OUT / f"{stem}.{ext}"
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"Missing output: {path}")
    qa["output_files"] = outputs
    (OUT / "qa_summary.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created {len(expected)} optimized figures in {OUT}")


if __name__ == "__main__":
    main()
