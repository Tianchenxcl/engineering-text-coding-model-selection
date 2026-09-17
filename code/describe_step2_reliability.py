from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_FILES = [
    "supervised_analysis_ready.csv",
    "llm_analysis_ready.csv",
    "supervised_task_pairs.csv",
    "llm_combination_prompt_means.csv",
    "common24_primary_analysis_ready.csv",
]

METRICS = ["kappa", "macro_f1", "mcc"]
MODEL_ORDER = [
    "SVM",
    "DeBERTa",
    "LLM zero-shot",
    "LLM few-shot",
    "LLM codebook",
]
SUPERVISED_MODEL_ORDER = ["SVM", "DeBERTa"]
PROMPT_ORDER = ["zero_shot", "few_shot", "codebook_guided"]
PROMPT_DISPLAY = {
    "zero_shot": "LLM zero-shot",
    "few_shot": "LLM few-shot",
    "codebook_guided": "LLM codebook",
}
MODEL_DISPLAY = {
    "TF-IDF + LinearSVC": "SVM",
    "DeBERTa-v3-base": "DeBERTa",
}
PALETTE = {
    "SVM": "#4F6B5B",
    "DeBERTa": "#2F6B9A",
    "LLM zero-shot": "#8B8B8B",
    "LLM few-shot": "#D28B3C",
    "LLM codebook": "#B55245",
}
RELIABILITY_BANDS = ["<0.20", "0.20-<0.40", "0.40-<0.60", "0.60-<0.80", ">=0.80"]
RELIABILITY_COLORS = {
    "<0.20": "#D9D9D9",
    "0.20-<0.40": "#B8C4CC",
    "0.40-<0.60": "#8FA9B8",
    "0.60-<0.80": "#5F879E",
    ">=0.80": "#315F78",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: descriptive performance and reliability landscape"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def read_inputs(input_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Step 1 files in {input_dir}: {missing}")
    return {name.removesuffix(".csv"): pd.read_csv(input_dir / name, low_memory=False) for name in REQUIRED_FILES}


def stable_seed(base_seed: int, *parts: Any) -> int:
    text = "|".join(map(str, parts))
    offset = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + offset) % (2**32 - 1)


def cluster_bootstrap_mean_ci(
    frame: pd.DataFrame,
    value_column: str,
    cluster_column: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    clean = frame[[value_column, cluster_column]].dropna().copy()
    if clean.empty or clean[cluster_column].nunique() < 2 or repetitions <= 0:
        return np.nan, np.nan
    grouped = clean.groupby(cluster_column)[value_column].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    cluster_count = len(grouped)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        estimates[index] = sums[sampled].sum() / counts[sampled].sum()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def summarize_groups(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
    cluster_column: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_columns, dropna=False, sort=False)
    for group_key, group in grouped:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_columns, keys))
        row["analysis_unit_count"] = len(group)
        row["cluster_count"] = int(group[cluster_column].nunique())
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            ci_low, ci_high = cluster_bootstrap_mean_ci(
                group,
                metric,
                cluster_column,
                repetitions,
                stable_seed(seed, *keys, metric),
            )
            row.update(
                {
                    f"{metric}_mean": float(values.mean()),
                    f"{metric}_median": float(values.median()),
                    f"{metric}_sd": float(values.std(ddof=1)),
                    f"{metric}_q25": float(values.quantile(0.25)),
                    f"{metric}_q75": float(values.quantile(0.75)),
                    f"{metric}_min": float(values.min()),
                    f"{metric}_max": float(values.max()),
                    f"{metric}_cluster_ci95_low": ci_low,
                    f"{metric}_cluster_ci95_high": ci_high,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_name_from_common(value: str) -> str:
    if value in MODEL_DISPLAY:
        return MODEL_DISPLAY[value]
    text = str(value)
    if "codebook_guided" in text:
        return "LLM codebook"
    if "few_shot" in text:
        return "LLM few-shot"
    if "zero_shot" in text:
        return "LLM zero-shot"
    return text


def prepare_overall_units(
    supervised: pd.DataFrame,
    llm_prompt_means: pd.DataFrame,
) -> pd.DataFrame:
    supervised_units = supervised[
        ["original_task_id", "combination_id", "model_name", "kappa", "macro_f1", "mcc"]
    ].copy()
    supervised_units["candidate"] = supervised_units["model_name"].map(MODEL_DISPLAY)
    supervised_units["analysis_unit_id"] = supervised_units["original_task_id"].astype(str)

    llm_units = llm_prompt_means[
        ["combination_id", "prompt_strategy", "kappa_mean", "macro_f1_mean", "mcc_mean"]
    ].copy()
    llm_units = llm_units.rename(
        columns={"kappa_mean": "kappa", "macro_f1_mean": "macro_f1", "mcc_mean": "mcc"}
    )
    llm_units["candidate"] = llm_units["prompt_strategy"].map(PROMPT_DISPLAY)
    llm_units["analysis_unit_id"] = llm_units["combination_id"].astype(str)
    return pd.concat(
        [
            supervised_units[["analysis_unit_id", "combination_id", "candidate", *METRICS]],
            llm_units[["analysis_unit_id", "combination_id", "candidate", *METRICS]],
        ],
        ignore_index=True,
    )


def reliability_band(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf],
        labels=RELIABILITY_BANDS,
        right=False,
        ordered=True,
    )


def reliability_distribution(overall_units: pd.DataFrame) -> pd.DataFrame:
    frame = overall_units.copy()
    frame["kappa_band"] = reliability_band(frame["kappa"])
    counts = (
        frame.groupby(["candidate", "kappa_band"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby("candidate")["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    return counts


def add_tertile_bins(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    specifications: list[tuple[str, str]],
    reference_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    rows = []
    for source, destination in specifications:
        values = pd.to_numeric(reference[source], errors="coerce").dropna()
        lower = float(values.quantile(1 / 3))
        upper = float(values.quantile(2 / 3))
        numeric = pd.to_numeric(output[source], errors="coerce")
        output[destination] = pd.Categorical(
            np.where(numeric <= lower, "Low", np.where(numeric <= upper, "Middle", "High")),
            categories=["Low", "Middle", "High"],
            ordered=True,
        )
        rows.append(
            {
                "reference": reference_name,
                "source_variable": source,
                "bin_variable": destination,
                "lower_tertile_cutpoint": lower,
                "upper_tertile_cutpoint": upper,
                "reference_rows": len(values),
            }
        )
    return output, pd.DataFrame(rows)


def supervised_descriptive_tables(
    supervised: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    frame = supervised.copy()
    frame["candidate"] = frame["model_name"].map(MODEL_DISPLAY)
    reference = frame.drop_duplicates("original_task_id")
    frame, cutpoints = add_tertile_bins(
        frame,
        reference,
        [
            ("class_imbalance_complexity", "class_imbalance_bin"),
            ("annotation_disagreement_complexity", "annotation_disagreement_bin"),
            ("semantic_similarity_complexity", "semantic_similarity_bin"),
        ],
        "unique_supervised_tasks",
    )
    tables = {
        "supervised_by_training_size": summarize_groups(
            frame,
            ["candidate", "training_sample_size"],
            METRICS,
            "combination_id",
            repetitions,
            seed,
        ),
        "supervised_by_label_number": summarize_groups(
            frame,
            ["candidate", "label_number"],
            METRICS,
            "combination_id",
            repetitions,
            seed,
        ),
        "supervised_by_balance_design": summarize_groups(
            frame,
            ["candidate", "class_balance_design"],
            METRICS,
            "combination_id",
            repetitions,
            seed,
        ),
    }
    bin_tables = []
    for bin_column in [
        "class_imbalance_bin",
        "annotation_disagreement_bin",
        "semantic_similarity_bin",
    ]:
        summary = summarize_groups(
            frame,
            ["candidate", bin_column],
            METRICS,
            "combination_id",
            repetitions,
            stable_seed(seed, bin_column),
        )
        summary = summary.rename(columns={bin_column: "complexity_level"})
        summary.insert(1, "complexity_dimension", bin_column.removesuffix("_bin"))
        bin_tables.append(summary)
    tables["supervised_by_complexity_tertile"] = pd.concat(bin_tables, ignore_index=True)
    return tables, frame, cutpoints


def llm_descriptive_tables(
    llm_prompt_means: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    frame = llm_prompt_means.copy()
    frame["candidate"] = frame["prompt_strategy"].map(PROMPT_DISPLAY)
    frame = frame.rename(
        columns={"kappa_mean": "kappa", "macro_f1_mean": "macro_f1", "mcc_mean": "mcc"}
    )
    reference = frame.drop_duplicates("combination_id")
    frame, cutpoints = add_tertile_bins(
        frame,
        reference,
        [
            ("annotation_disagreement_complexity", "annotation_disagreement_bin"),
            ("semantic_similarity_complexity", "semantic_similarity_bin"),
        ],
        "unique_llm_combinations",
    )
    tables = {
        "llm_prompt_summary": summarize_groups(
            frame,
            ["candidate"],
            METRICS,
            "combination_id",
            repetitions,
            seed,
        ),
        "llm_by_label_number": summarize_groups(
            frame,
            ["candidate", "label_number"],
            METRICS,
            "combination_id",
            repetitions,
            seed,
        ),
        "llm_repeat_stability": summarize_groups(
            frame,
            ["candidate"],
            ["kappa_sd", "macro_f1_sd", "mcc_sd"],
            "combination_id",
            repetitions,
            stable_seed(seed, "repeat_stability"),
        ),
    }
    bin_tables = []
    for bin_column in ["annotation_disagreement_bin", "semantic_similarity_bin"]:
        summary = summarize_groups(
            frame,
            ["candidate", bin_column],
            METRICS,
            "combination_id",
            repetitions,
            stable_seed(seed, bin_column),
        )
        summary = summary.rename(columns={bin_column: "complexity_level"})
        summary.insert(1, "complexity_dimension", bin_column.removesuffix("_bin"))
        bin_tables.append(summary)
    tables["llm_by_complexity_tertile"] = pd.concat(bin_tables, ignore_index=True)

    pivot = frame.pivot(index="combination_id", columns="prompt_strategy", values="kappa")
    prompt_deltas = pd.DataFrame(index=pivot.index)
    prompt_deltas["codebook_minus_zero"] = pivot["codebook_guided"] - pivot["zero_shot"]
    prompt_deltas["few_minus_zero"] = pivot["few_shot"] - pivot["zero_shot"]
    prompt_deltas["codebook_minus_few"] = pivot["codebook_guided"] - pivot["few_shot"]
    prompt_deltas = prompt_deltas.reset_index()
    delta_rows = []
    for column in ["codebook_minus_zero", "few_minus_zero", "codebook_minus_few"]:
        values = prompt_deltas[column].dropna()
        ci_low, ci_high = cluster_bootstrap_mean_ci(
            prompt_deltas,
            column,
            "combination_id",
            repetitions,
            stable_seed(seed, column),
        )
        delta_rows.append(
            {
                "contrast": column,
                "combination_count": len(values),
                "mean_delta_kappa": values.mean(),
                "median_delta_kappa": values.median(),
                "sd_delta_kappa": values.std(ddof=1),
                "minimum_delta_kappa": values.min(),
                "maximum_delta_kappa": values.max(),
                "positive_difference_share": (values > 0).mean(),
                "cluster_ci95_low": ci_low,
                "cluster_ci95_high": ci_high,
            }
        )
    return tables, frame, cutpoints, pd.DataFrame(delta_rows), prompt_deltas


def supervised_pair_differences(
    pairs: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    group_specs = [
        ("overall", None),
        ("training_sample_size", "training_sample_size"),
        ("label_number", "label_number"),
    ]
    for grouping_name, column in group_specs:
        grouped = [("all", pairs)] if column is None else list(pairs.groupby(column, dropna=False))
        for level, group in grouped:
            values = group["kappa_delta_deberta_minus_svm"].dropna()
            ci_low, ci_high = cluster_bootstrap_mean_ci(
                group,
                "kappa_delta_deberta_minus_svm",
                "combination_id",
                repetitions,
                stable_seed(seed, grouping_name, level),
            )
            rows.append(
                {
                    "grouping": grouping_name,
                    "level": level,
                    "task_count": len(values),
                    "combination_count": group["combination_id"].nunique(),
                    "mean_delta_kappa": values.mean(),
                    "median_delta_kappa": values.median(),
                    "sd_delta_kappa": values.std(ddof=1),
                    "deberta_win_share": (values > 0).mean(),
                    "cluster_ci95_low": ci_low,
                    "cluster_ci95_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def prepare_common_summary(
    common_primary: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = common_primary.copy()
    frame["candidate"] = frame["candidate_model"].map(candidate_name_from_common)
    summary = summarize_groups(
        frame,
        ["candidate"],
        METRICS,
        "combination_id",
        repetitions,
        seed,
    )
    quadrant = summarize_groups(
        frame,
        ["candidate", "common_complexity_quadrant"],
        METRICS,
        "combination_id",
        repetitions,
        stable_seed(seed, "quadrant"),
    )
    return summary, quadrant


def configure_matplotlib() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def save_figure(fig: Any, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")


def panel_label(ax: Any, label: str) -> None:
    ax.text(-0.12, 1.07, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def draw_box_jitter(
    ax: Any,
    frame: pd.DataFrame,
    category: str,
    value: str,
    order: list[str],
    seed: int,
    reference_line: float | None = 0.60,
) -> None:
    data = [frame.loc[frame[category] == item, value].dropna().to_numpy() for item in order]
    box = ax.boxplot(
        data,
        positions=np.arange(len(order)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.2},
        whiskerprops={"color": "#5A5A5A", "linewidth": 0.8},
        capprops={"color": "#5A5A5A", "linewidth": 0.8},
    )
    rng = np.random.default_rng(seed)
    for index, (item, values) in enumerate(zip(order, data)):
        box["boxes"][index].set_facecolor(PALETTE[item])
        box["boxes"][index].set_alpha(0.55)
        jitter = rng.normal(0, 0.055, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=9,
            alpha=0.45,
            color=PALETTE[item],
            edgecolors="none",
            rasterized=True,
        )
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_ylim(-0.05, 1.0)
    if reference_line is not None:
        ax.axhline(reference_line, color="#B0B0B0", linewidth=0.7, linestyle="--")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)


def plot_overall_landscape(
    plt: Any,
    overall_units: pd.DataFrame,
    reliability: pd.DataFrame,
    figure_dir: Path,
    seed: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.25, 1.0]})
    draw_box_jitter(axes[0], overall_units, "candidate", "kappa", MODEL_ORDER, seed)
    axes[0].set_ylabel("Cohen's kappa")
    axes[0].set_title("Reliability distributions across constructed tasks")
    panel_label(axes[0], "A")

    pivot = reliability.pivot(index="candidate", columns="kappa_band", values="share").reindex(
        index=MODEL_ORDER, columns=RELIABILITY_BANDS
    ).fillna(0)
    left = np.zeros(len(pivot))
    for band in RELIABILITY_BANDS:
        values = pivot[band].to_numpy()
        axes[1].barh(
            np.arange(len(pivot)),
            values,
            left=left,
            height=0.62,
            color=RELIABILITY_COLORS[band],
            label=band,
        )
        left += values
    axes[1].set_yticks(np.arange(len(pivot)))
    axes[1].set_yticklabels(MODEL_ORDER)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Share of analysis units")
    axes[1].set_title("Distribution across numeric kappa bands")
    axes[1].legend(title="Kappa", ncol=3, fontsize=6.5, title_fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    axes[1].grid(axis="x", color="#E6E6E6", linewidth=0.6)
    panel_label(axes[1], "B")
    fig.tight_layout(w_pad=1.4, rect=(0, 0.12, 1, 1))
    save_figure(fig, figure_dir, "figure1_overall_reliability_landscape")
    plt.close(fig)


def draw_summary_lines(
    ax: Any,
    summary: pd.DataFrame,
    x_column: str,
    order: list[Any],
    candidates: list[str],
    xlabel: str,
) -> None:
    positions = np.arange(len(order))
    for candidate in candidates:
        group = summary[summary["candidate"] == candidate].set_index(x_column).reindex(order)
        mean = group["kappa_mean"].to_numpy(dtype=float)
        low = group["kappa_cluster_ci95_low"].to_numpy(dtype=float)
        high = group["kappa_cluster_ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([mean - low, high - mean])
        yerr = np.where(np.isfinite(yerr), np.maximum(yerr, 0), np.nan)
        ax.errorbar(
            positions,
            mean,
            yerr=yerr,
            marker="o",
            markersize=4,
            linewidth=1.3,
            capsize=2,
            color=PALETTE[candidate],
            label=candidate,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([str(value) for value in order])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean kappa")
    ax.set_ylim(-0.05, 1.0)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)


def plot_supervised_landscape(
    plt: Any,
    tables: dict[str, pd.DataFrame],
    pair_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.1))
    training = tables["supervised_by_training_size"]
    sizes = sorted(training["training_sample_size"].dropna().astype(int).unique())
    draw_summary_lines(axes[0, 0], training, "training_sample_size", sizes, SUPERVISED_MODEL_ORDER, "Training samples")
    axes[0, 0].set_title("Learning curves")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].tick_params(axis="x", labelrotation=45, labelsize=6)
    for label_item in axes[0, 0].get_xticklabels():
        label_item.set_ha("right")
    panel_label(axes[0, 0], "A")

    label = tables["supervised_by_label_number"]
    labels = sorted(label["label_number"].dropna().astype(int).unique())
    draw_summary_lines(axes[0, 1], label, "label_number", labels, SUPERVISED_MODEL_ORDER, "Number of labels")
    axes[0, 1].set_title("Label-space complexity")
    panel_label(axes[0, 1], "B")

    bins = tables["supervised_by_complexity_tertile"]
    for ax, dimension, title, panel in [
        (axes[0, 2], "class_imbalance", "Class imbalance", "C"),
        (axes[1, 0], "annotation_disagreement", "Annotation disagreement", "D"),
        (axes[1, 1], "semantic_similarity", "Label semantic similarity", "E"),
    ]:
        subset = bins[bins["complexity_dimension"] == dimension]
        draw_summary_lines(ax, subset, "complexity_level", ["Low", "Middle", "High"], SUPERVISED_MODEL_ORDER, "Complexity tertile")
        ax.set_title(title)
        panel_label(ax, panel)

    delta = pair_summary[pair_summary["grouping"] == "training_sample_size"].copy()
    delta["level_numeric"] = pd.to_numeric(delta["level"], errors="coerce")
    delta = delta.sort_values("level_numeric")
    positions = np.arange(len(delta))
    means = delta["mean_delta_kappa"].to_numpy(dtype=float)
    low = delta["cluster_ci95_low"].to_numpy(dtype=float)
    high = delta["cluster_ci95_high"].to_numpy(dtype=float)
    axes[1, 2].errorbar(
        positions,
        means,
        yerr=np.vstack([means - low, high - means]),
        marker="o",
        color="#5B4A42",
        linewidth=1.3,
        capsize=2,
    )
    axes[1, 2].axhline(0, color="#555555", linewidth=0.8)
    axes[1, 2].set_xticks(positions)
    axes[1, 2].set_xticklabels(delta["level_numeric"].astype(int).astype(str))
    axes[1, 2].tick_params(axis="x", labelrotation=45, labelsize=6)
    for label_item in axes[1, 2].get_xticklabels():
        label_item.set_ha("right")
    axes[1, 2].set_xlabel("Training samples")
    axes[1, 2].set_ylabel("Kappa difference")
    axes[1, 2].set_title("DeBERTa minus SVM")
    axes[1, 2].grid(axis="y", color="#E6E6E6", linewidth=0.6)
    panel_label(axes[1, 2], "F")
    fig.tight_layout(w_pad=1.1, h_pad=1.5)
    save_figure(fig, figure_dir, "figure2_supervised_complexity_landscape")
    plt.close(fig)


def plot_llm_landscape(
    plt: Any,
    llm_frame: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    prompt_deltas: pd.DataFrame,
    figure_dir: Path,
    seed: int,
) -> None:
    candidates = [PROMPT_DISPLAY[item] for item in PROMPT_ORDER]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    draw_box_jitter(axes[0, 0], llm_frame, "candidate", "kappa", candidates, seed)
    axes[0, 0].set_ylabel("Mean kappa across repeats")
    axes[0, 0].set_title("Prompt reliability distributions")
    panel_label(axes[0, 0], "A")

    label = tables["llm_by_label_number"]
    labels = sorted(label["label_number"].dropna().astype(int).unique())
    draw_summary_lines(axes[0, 1], label, "label_number", labels, candidates, "Number of labels")
    axes[0, 1].set_title("Performance by label number")
    axes[0, 1].legend(fontsize=6.5)
    panel_label(axes[0, 1], "B")

    bins = tables["llm_by_complexity_tertile"]
    disagreement = bins[bins["complexity_dimension"] == "annotation_disagreement"]
    draw_summary_lines(axes[0, 2], disagreement, "complexity_level", ["Low", "Middle", "High"], candidates, "Disagreement tertile")
    axes[0, 2].set_title("Annotation disagreement")
    panel_label(axes[0, 2], "C")

    similarity = bins[bins["complexity_dimension"] == "semantic_similarity"]
    draw_summary_lines(axes[1, 0], similarity, "complexity_level", ["Low", "Middle", "High"], candidates, "Similarity tertile")
    axes[1, 0].set_title("Label semantic similarity")
    panel_label(axes[1, 0], "D")

    sorted_delta = prompt_deltas.sort_values("codebook_minus_zero").reset_index(drop=True)
    axes[1, 1].scatter(
        np.arange(len(sorted_delta)),
        sorted_delta["codebook_minus_zero"],
        s=18,
        color=PALETTE["LLM codebook"],
        alpha=0.8,
    )
    axes[1, 1].axhline(0, color="#555555", linewidth=0.8)
    axes[1, 1].set_xlabel("Label combinations, ordered by improvement")
    axes[1, 1].set_ylabel("Codebook minus zero-shot kappa")
    axes[1, 1].set_title("Paired codebook improvement")
    axes[1, 1].grid(axis="y", color="#E6E6E6", linewidth=0.6)
    panel_label(axes[1, 1], "E")

    draw_box_jitter(
        axes[1, 2],
        llm_frame,
        "candidate",
        "kappa_sd",
        candidates,
        stable_seed(seed, "repeat_sd"),
        reference_line=None,
    )
    axes[1, 2].set_ylim(-0.005, max(0.12, float(llm_frame["kappa_sd"].max()) * 1.15))
    axes[1, 2].set_ylabel("Kappa SD across repeats")
    axes[1, 2].set_title("Repeat-to-repeat variability")
    panel_label(axes[1, 2], "F")
    fig.tight_layout(w_pad=1.3, h_pad=1.5)
    save_figure(fig, figure_dir, "figure3_llm_prompt_reliability")
    plt.close(fig)


def plot_common24(
    plt: Any,
    common_frame: pd.DataFrame,
    quadrant_summary: pd.DataFrame,
    figure_dir: Path,
    seed: int,
) -> None:
    order = ["SVM", "DeBERTa", "LLM codebook"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [0.9, 1.35]})
    draw_box_jitter(axes[0], common_frame, "candidate", "kappa", order, seed)
    axes[0].set_ylabel("Cohen's kappa")
    axes[0].set_title("Matched Common24 performance")
    panel_label(axes[0], "A")

    quadrant_order = [
        "low_semantic_low_disagreement",
        "low_semantic_high_disagreement",
        "high_semantic_low_disagreement",
        "high_semantic_high_disagreement",
    ]
    quadrant_labels = ["Low S / Low D", "Low S / High D", "High S / Low D", "High S / High D"]
    draw_summary_lines(
        axes[1],
        quadrant_summary,
        "common_complexity_quadrant",
        quadrant_order,
        order,
        "Common24 complexity quadrant",
    )
    axes[1].set_xticklabels(quadrant_labels, rotation=15, ha="right")
    axes[1].set_title("Model-task fit across complexity quadrants")
    axes[1].legend(fontsize=7)
    panel_label(axes[1], "B")
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, figure_dir, "figure4_common24_matched_comparison")
    plt.close(fig)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    source_dir = output_dir / "figure_source_data"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    inputs = read_inputs(input_dir)
    supervised = inputs["supervised_analysis_ready"]
    llm_prompt_means = inputs["llm_combination_prompt_means"]
    supervised_pairs = inputs["supervised_task_pairs"]
    common_primary = inputs["common24_primary_analysis_ready"]

    overall_units = prepare_overall_units(supervised, llm_prompt_means)
    overall_summary = summarize_groups(
        overall_units,
        ["candidate"],
        METRICS,
        "combination_id",
        args.bootstrap_reps,
        args.seed,
    )
    reliability = reliability_distribution(overall_units)

    supervised_tables, supervised_binned, supervised_cutpoints = supervised_descriptive_tables(
        supervised, args.bootstrap_reps, args.seed
    )
    llm_tables, llm_binned, llm_cutpoints, llm_delta_summary, prompt_deltas = llm_descriptive_tables(
        llm_prompt_means, args.bootstrap_reps, args.seed
    )
    pair_summary = supervised_pair_differences(
        supervised_pairs, args.bootstrap_reps, args.seed
    )
    common_summary, common_quadrant = prepare_common_summary(
        common_primary, args.bootstrap_reps, args.seed
    )
    common_plot_frame = common_primary.copy()
    common_plot_frame["candidate"] = common_plot_frame["candidate_model"].map(candidate_name_from_common)

    cutpoints = pd.concat([supervised_cutpoints, llm_cutpoints], ignore_index=True)
    outputs = {
        "overall_model_performance": overall_summary,
        "reliability_band_distribution": reliability,
        "supervised_pair_differences": pair_summary,
        "llm_prompt_paired_differences": llm_delta_summary,
        "common24_primary_summary": common_summary,
        "common24_by_complexity_quadrant": common_quadrant,
        "complexity_bin_cutpoints": cutpoints,
        **supervised_tables,
        **llm_tables,
    }
    for name, frame in outputs.items():
        save_csv(frame, table_dir / f"{name}.csv")

    save_csv(overall_units, source_dir / "figure1_overall_units.csv")
    save_csv(reliability, source_dir / "figure1_reliability_bands.csv")
    save_csv(supervised_binned, source_dir / "figure2_supervised_units.csv")
    save_csv(pair_summary, source_dir / "figure2_supervised_pair_differences.csv")
    save_csv(llm_binned, source_dir / "figure3_llm_prompt_units.csv")
    save_csv(prompt_deltas, source_dir / "figure3_llm_prompt_deltas.csv")
    save_csv(common_plot_frame, source_dir / "figure4_common24_units.csv")
    save_csv(common_quadrant, source_dir / "figure4_common24_quadrants.csv")

    figure_status = "skipped"
    if not args.skip_figures:
        plt = configure_matplotlib()
        plot_overall_landscape(plt, overall_units, reliability, figure_dir, args.seed)
        plot_supervised_landscape(plt, supervised_tables, pair_summary, figure_dir)
        plot_llm_landscape(plt, llm_binned, llm_tables, prompt_deltas, figure_dir, args.seed)
        plot_common24(plt, common_plot_frame, common_quadrant, figure_dir, args.seed)
        figure_status = "rendered"

    overall_lookup = overall_summary.set_index("candidate")["kappa_mean"].to_dict()
    common_lookup = common_summary.set_index("candidate")["kappa_mean"].to_dict()
    delta_lookup = llm_delta_summary.set_index("contrast")["mean_delta_kappa"].to_dict()
    report = {
        "generated_at_utc": utc_now(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "bootstrap_repetitions": args.bootstrap_reps,
        "bootstrap_cluster": "label combination",
        "figure_status": figure_status,
        "table_count": len(outputs),
        "figure_count": 4 if figure_status == "rendered" else 0,
        "descriptive_findings": {
            "overall_mean_kappa": overall_lookup,
            "common24_mean_kappa": common_lookup,
            "llm_mean_prompt_differences": delta_lookup,
        },
        "output_checks": {
            "supervised_analysis_units": len(supervised),
            "supervised_matched_task_pairs": len(supervised_pairs),
            "llm_combination_prompt_units": len(llm_prompt_means),
            "common24_primary_units": len(common_primary),
            "overall_missing_metric_cells": int(overall_units[METRICS].isna().sum().sum()),
            "overall_candidate_count": int(overall_units["candidate"].nunique()),
            "common24_candidate_count": int(common_plot_frame["candidate"].nunique()),
        },
        "interpretation_boundary": (
            "These are descriptive distributions and cluster-bootstrap intervals. "
            "Statistical interaction claims require the mixed-effects models in Step 3."
        ),
        "comparison_boundary": (
            "Overall candidate distributions come from each candidate's available task set. "
            "Use the matched Common24 results for direct cross-family comparisons."
        ),
    }
    write_json(output_dir / "step2_descriptive_report.json", report)
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    print(f"\nStep 2 outputs: {output_dir}")


if __name__ == "__main__":
    main()
