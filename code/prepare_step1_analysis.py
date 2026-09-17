from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_FILES = {
    "supervised": "supervised_performance_long.csv",
    "llm": "llm_performance_long.csv",
    "common24": "common24_cross_model_long.csv",
    "all_runs": "all_performance_long.csv",
}

PRIMARY_METRICS = ["kappa", "macro_f1", "mcc"]
BOUNDED_ZERO_ONE = ["macro_f1", "accuracy", "weighted_f1"]
BOUNDED_MINUS_ONE_ONE = ["kappa", "mcc"]


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
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: variable processing and data-quality checks for the contract coding study"
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-deep-test-checks", action="store_true")
    return parser.parse_args()


def contains_required_files(path: Path) -> bool:
    return path.exists() and all((path / filename).exists() for filename in REQUIRED_FILES.values())


def detect_package_root(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path("/hy-tmp/hy_tmp_experiment_package"),
            Path.cwd() / "hy_tmp_experiment_package",
            Path.cwd(),
            Path.cwd().parent / "hy_tmp_experiment_package",
        ]
    )
    try:
        candidates.append(Path(__file__).resolve().parents[1])
    except NameError:
        pass
    for candidate in candidates:
        if (candidate / "code").exists() and (
            (candidate / "supervised_data").exists() or (candidate / "llm_data").exists()
        ):
            return candidate.resolve()
    return (explicit or Path.cwd()).resolve()


def detect_input_dir(package_root: Path, explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path("/hy-tmp"),
            Path("/hy-tmp/hy_tmp_experiment_package"),
            Path("/hy-tmp/hy_tmp_experiment_package/outputs/analysis_ready"),
            Path.cwd(),
            Path.cwd().parent,
            package_root,
            package_root / "outputs" / "analysis_ready",
            package_root.parent,
        ]
    )
    for candidate in candidates:
        if contains_required_files(candidate):
            return candidate.resolve()
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find all four analysis CSV files. Searched:\n{searched}")


def read_inputs(input_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(input_dir / filename, low_memory=False)
        for name, filename in REQUIRED_FILES.items()
    }


def observed_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)
    return str(value)


def add_check(
    checks: list[dict[str, Any]],
    dataset: str,
    check_id: str,
    passed: bool | None,
    severity_if_failed: str,
    observed: Any,
    expected: Any,
    implication: str,
) -> None:
    status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    checks.append(
        {
            "dataset": dataset,
            "check_id": check_id,
            "status": status,
            "severity_if_failed": severity_if_failed,
            "observed": observed_text(observed),
            "expected": observed_text(expected),
            "implication": implication,
        }
    )


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    dataset: str,
    checks: list[dict[str, Any]],
) -> bool:
    missing = [column for column in columns if column not in frame.columns]
    add_check(
        checks,
        dataset,
        "required_columns",
        not missing,
        "critical",
        {"missing": missing},
        "no required columns missing",
        "Missing required columns would make the planned statistical model invalid.",
    )
    return not missing


def dataset_profile(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "dataset": name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "completed_rows": int((frame.get("status") == "completed").sum())
        if "status" in frame.columns
        else None,
        "unique_performance_units": int(frame["performance_unit_id"].nunique())
        if "performance_unit_id" in frame.columns
        else None,
        "unique_combinations": int(frame["combination_id"].nunique())
        if "combination_id" in frame.columns
        else None,
    }


def check_metric_columns(
    frame: pd.DataFrame,
    dataset: str,
    checks: list[dict[str, Any]],
) -> None:
    for metric in BOUNDED_MINUS_ONE_ONE + BOUNDED_ZERO_ONE:
        if metric not in frame.columns:
            add_check(
                checks,
                dataset,
                f"metric_{metric}_available",
                False,
                "critical",
                "missing",
                "column available",
                "The metric cannot be used as an outcome or robustness measure.",
            )
            continue
        values = pd.to_numeric(frame[metric], errors="coerce")
        lower, upper = (-1.0, 1.0) if metric in BOUNDED_MINUS_ONE_ONE else (0.0, 1.0)
        invalid = values.isna() | (values < lower) | (values > upper)
        add_check(
            checks,
            dataset,
            f"metric_{metric}_valid",
            not invalid.any(),
            "critical",
            {"invalid_rows": int(invalid.sum()), "min": values.min(), "max": values.max()},
            f"all finite values in [{lower}, {upper}]",
            "Invalid outcome values would bias descriptive and regression results.",
        )

        low_name = f"{metric}_ci95_low"
        high_name = f"{metric}_ci95_high"
        if low_name in frame.columns and high_name in frame.columns:
            low = pd.to_numeric(frame[low_name], errors="coerce")
            high = pd.to_numeric(frame[high_name], errors="coerce")
            reversed_ci = low.notna() & high.notna() & (low > high)
            add_check(
                checks,
                dataset,
                f"metric_{metric}_ci_order",
                not reversed_ci.any(),
                "high",
                {"reversed_intervals": int(reversed_ci.sum())},
                "lower bound <= upper bound",
                "Reversed confidence limits indicate corrupted bootstrap summaries.",
            )


def numeric_equal(left: pd.Series, right: pd.Series, tolerance: float = 1e-10) -> bool:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return bool(np.allclose(left_values, right_values, atol=tolerance, rtol=tolerance, equal_nan=True))


def run_summary_quality_checks(
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    supervised = tables["supervised"]
    llm = tables["llm"]
    common24 = tables["common24"]
    all_runs = tables["all_runs"]
    checks: list[dict[str, Any]] = []

    required = {
        "supervised": [
            "performance_unit_id",
            "original_task_id",
            "combination_id",
            "model_family",
            "label_number",
            "class_imbalance_complexity",
            "annotation_disagreement_complexity",
            "training_size_complexity",
            "semantic_similarity_complexity",
            "kappa",
            "macro_f1",
            "mcc",
        ],
        "llm": [
            "performance_unit_id",
            "combination_id",
            "prompt_strategy",
            "repeat_id",
            "label_number",
            "annotation_disagreement_complexity",
            "semantic_similarity_complexity",
            "kappa",
            "macro_f1",
            "mcc",
        ],
        "common24": ["performance_unit_id", "combination_id", "model_family", "kappa"],
        "all_runs": ["performance_unit_id", "combination_id", "model_family", "kappa"],
    }
    for name, frame in tables.items():
        require_columns(frame, required[name], name, checks)
        check_metric_columns(frame, name, checks)
        if "status" in frame.columns:
            incomplete = frame["status"].astype(str).ne("completed")
            add_check(
                checks,
                name,
                "all_runs_completed",
                not incomplete.any(),
                "critical",
                {"incomplete_rows": int(incomplete.sum())},
                "all rows completed",
                "Incomplete runs must not enter performance models.",
            )
        duplicate_ids = int(frame["performance_unit_id"].duplicated().sum())
        add_check(
            checks,
            name,
            "performance_unit_unique",
            duplicate_ids == 0,
            "critical",
            {"duplicated_ids": duplicate_ids},
            "zero duplicated performance_unit_id values",
            "Duplicated runs would overweight selected tasks or model configurations.",
        )

    task_counts = supervised.groupby("original_task_id").size()
    model_counts = supervised.groupby("original_task_id")["model_family"].nunique()
    add_check(
        checks,
        "supervised",
        "paired_model_coverage",
        bool((task_counts == 2).all() and (model_counts == 2).all()),
        "critical",
        {
            "unique_tasks": int(supervised["original_task_id"].nunique()),
            "rows_per_task": task_counts.value_counts().sort_index().to_dict(),
            "model_families_per_task": model_counts.value_counts().sort_index().to_dict(),
        },
        "exactly two model-family rows for every supervised task",
        "The SVM-DeBERTa comparison relies on complete task-level pairing.",
    )

    if "test_sha256" in supervised.columns:
        test_hashes = supervised.groupby("original_task_id")["test_sha256"].nunique(dropna=False)
        add_check(
            checks,
            "supervised",
            "paired_test_file_hash",
            bool((test_hashes == 1).all()),
            "critical",
            {"tasks_with_multiple_hashes": int((test_hashes > 1).sum())},
            "one test-file hash within every paired task",
            "Different test files inside a paired task would invalidate direct model comparisons.",
        )

    complexity_columns = [
        "label_number_complexity",
        "class_imbalance_complexity",
        "annotation_disagreement_complexity",
        "training_size_complexity",
        "semantic_similarity_complexity",
    ]
    varying = {}
    for column in complexity_columns:
        if column in supervised.columns:
            counts = supervised.groupby("original_task_id")[column].nunique(dropna=False)
            varying[column] = int((counts > 1).sum())
    add_check(
        checks,
        "supervised",
        "paired_complexity_metadata",
        all(value == 0 for value in varying.values()),
        "critical",
        varying,
        "zero task pairs with inconsistent complexity metadata",
        "Both supervised models must be evaluated under identical task complexity conditions.",
    )

    source_pairs = [
        ("label_number_complexity", "label_number"),
        ("class_imbalance_complexity", "realized_class_imbalance_index"),
        ("annotation_disagreement_complexity", "sampled_annotation_disagreement_rate"),
        ("training_size_complexity", "final_training_n"),
        ("semantic_similarity_complexity", "avg_pairwise_semantic_similarity"),
    ]
    source_consistency = {}
    for standardized, source in source_pairs:
        if standardized in supervised.columns and source in supervised.columns:
            source_consistency[f"{standardized}={source}"] = numeric_equal(
                supervised[standardized], supervised[source]
            )
    add_check(
        checks,
        "supervised",
        "complexity_source_consistency",
        all(source_consistency.values()),
        "critical",
        source_consistency,
        "all standardized complexity fields equal their source fields before z-scaling",
        "A mismatch would mean the statistical predictor is not the intended complexity measure.",
    )

    llm_cells = llm.groupby(["combination_id", "prompt_strategy", "repeat_id"]).size()
    prompt_sets = llm.groupby("combination_id")["prompt_strategy"].nunique()
    repeat_sets = llm.groupby(["combination_id", "prompt_strategy"])["repeat_id"].nunique()
    add_check(
        checks,
        "llm",
        "prompt_repeat_grid_complete",
        bool((llm_cells == 1).all() and (prompt_sets == 3).all() and (repeat_sets == 3).all()),
        "critical",
        {
            "duplicate_or_missing_cells": int((llm_cells != 1).sum()),
            "combinations_not_three_prompts": int((prompt_sets != 3).sum()),
            "prompt_cells_not_three_repeats": int((repeat_sets != 3).sum()),
        },
        "one row per combination-prompt-repeat and complete 3x3 grid",
        "Prompt effects require the complete within-combination repeated design.",
    )

    llm_source_pairs = [
        ("label_number_complexity", "label_number"),
        ("annotation_disagreement_complexity", "annotation_disagreement_rate"),
        ("semantic_similarity_complexity", "avg_pairwise_semantic_similarity"),
    ]
    llm_source_consistency = {}
    for standardized, source in llm_source_pairs:
        if standardized in llm.columns and source in llm.columns:
            llm_source_consistency[f"{standardized}={source}"] = numeric_equal(llm[standardized], llm[source])
    add_check(
        checks,
        "llm",
        "complexity_source_consistency",
        all(llm_source_consistency.values()),
        "critical",
        llm_source_consistency,
        "all applicable LLM complexity fields equal their source fields before z-scaling",
        "LLM training-size and training-imbalance fields remain not applicable by design.",
    )

    common_counts = common24.groupby("combination_id").size()
    add_check(
        checks,
        "common24",
        "shared_task_grid_complete",
        bool(len(common_counts) == 24 and (common_counts == 11).all()),
        "critical",
        {
            "combinations": len(common_counts),
            "rows_per_combination": common_counts.value_counts().sort_index().to_dict(),
        },
        "24 combinations and 11 rows per combination (2 supervised + 3 prompts x 3 repeats)",
        "The predeclared cross-model comparison requires a complete matched grid.",
    )

    common_test_n = common24.groupby("combination_id")["test_n"].nunique(dropna=False)
    add_check(
        checks,
        "common24",
        "shared_test_size_consistency",
        bool((common_test_n == 1).all()),
        "critical",
        {"combinations_with_multiple_test_sizes": int((common_test_n > 1).sum())},
        "one test_n value within every shared combination",
        "Different test sizes inside a shared combination would weaken paired comparisons.",
    )

    add_check(
        checks,
        "all_runs",
        "row_reconciliation",
        len(all_runs) == len(supervised) + len(llm),
        "critical",
        {"all_rows": len(all_runs), "supervised_plus_llm": len(supervised) + len(llm)},
        "all_performance rows equal supervised rows plus LLM rows",
        "A mismatch indicates missing or duplicated rows during table assembly.",
    )

    class_imbalance_missing = int(llm["class_imbalance_complexity"].isna().sum())
    training_size_missing = int(llm["training_size_complexity"].isna().sum())
    add_check(
        checks,
        "llm",
        "nonapplicable_training_dimensions",
        class_imbalance_missing == len(llm) and training_size_missing == len(llm),
        "high",
        {
            "class_imbalance_missing": class_imbalance_missing,
            "training_size_missing": training_size_missing,
            "rows": len(llm),
        },
        "training imbalance and training size are missing for every LLM run",
        "These dimensions must not be imputed or included in the LLM-specific model.",
    )

    profiles = pd.DataFrame([dataset_profile(name, frame) for name, frame in tables.items()])
    return pd.DataFrame(checks), profiles


def standardization_parameters(
    reference: pd.DataFrame,
    dataset: str,
    specifications: list[tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    for source, output in specifications:
        values = pd.to_numeric(reference[source], errors="coerce")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        rows.append(
            {
                "dataset": dataset,
                "source_variable": source,
                "standardized_variable": output,
                "mean": mean,
                "std_ddof0": std,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "reference_rows": int(values.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def apply_standardization(frame: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for row in parameters.itertuples(index=False):
        values = pd.to_numeric(output[row.source_variable], errors="coerce")
        if not np.isfinite(row.std_ddof0) or row.std_ddof0 <= 0:
            output[row.standardized_variable] = np.nan
        else:
            output[row.standardized_variable] = (values - row.mean) / row.std_ddof0
    return output


def prepare_supervised(supervised: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = supervised.copy()
    output["log2_training_size"] = np.log2(pd.to_numeric(output["training_size_complexity"], errors="coerce"))
    output["label_number_factor"] = output["label_number"].astype("Int64").astype(str)
    output["model_variant"] = output["model_name"].fillna(output["model_family"])
    output["task_cluster_id"] = output["original_task_id"].astype(str)
    output["combination_cluster_id"] = output["combination_id"].astype(str)

    reference = output.drop_duplicates("original_task_id").copy()
    specs = [
        ("class_imbalance_complexity", "class_imbalance_z"),
        ("annotation_disagreement_complexity", "annotation_disagreement_z"),
        ("log2_training_size", "log2_training_size_z"),
        ("semantic_similarity_complexity", "semantic_similarity_z"),
    ]
    if "max_pairwise_semantic_similarity" in output.columns:
        specs.append(("max_pairwise_semantic_similarity", "max_semantic_similarity_z"))
    parameters = standardization_parameters(reference, "supervised_task_reference", specs)
    output = apply_standardization(output, parameters)

    pair_metrics = output.pivot_table(
        index="original_task_id",
        columns="model_variant",
        values=PRIMARY_METRICS,
        aggfunc="first",
    )
    pair_metrics.columns = [f"{metric}__{model}" for metric, model in pair_metrics.columns]
    pair_metrics = pair_metrics.reset_index()
    task_metadata = reference[
        [
            "original_task_id",
            "combination_id",
            "design_tier",
            "label_number",
            "class_balance_design",
            "training_sample_size",
            "class_imbalance_complexity",
            "annotation_disagreement_complexity",
            "semantic_similarity_complexity",
        ]
    ]
    pairs = task_metadata.merge(pair_metrics, on="original_task_id", how="left")
    if "kappa__DeBERTa-v3-base" in pairs.columns and "kappa__TF-IDF + LinearSVC" in pairs.columns:
        pairs["kappa_delta_deberta_minus_svm"] = (
            pairs["kappa__DeBERTa-v3-base"] - pairs["kappa__TF-IDF + LinearSVC"]
        )
    return output, parameters, pairs


def prepare_llm(llm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = llm.copy()
    output["label_number_factor"] = output["label_number"].astype("Int64").astype(str)
    output["prompt_strategy_factor"] = output["prompt_strategy"].astype(str)
    output["repeat_id_factor"] = output["repeat_id"].astype("Int64").astype(str)
    output["model_variant"] = output["request_model"].astype(str) + "__" + output["prompt_strategy"].astype(str)
    output["combination_cluster_id"] = output["combination_id"].astype(str)

    reference = output.drop_duplicates("combination_id").copy()
    specs = [
        ("annotation_disagreement_complexity", "annotation_disagreement_z"),
        ("semantic_similarity_complexity", "semantic_similarity_z"),
    ]
    parameters = standardization_parameters(reference, "llm_combination_reference", specs)
    output = apply_standardization(output, parameters)

    group_columns = ["combination_id", "prompt_strategy"]
    means = (
        output.groupby(group_columns, as_index=False)
        .agg(
            label_number=("label_number", "first"),
            selected_label_ids=("selected_label_ids", "first"),
            selected_labels=("selected_labels", "first"),
            annotation_disagreement_complexity=("annotation_disagreement_complexity", "first"),
            semantic_similarity_complexity=("semantic_similarity_complexity", "first"),
            complexity_quadrant=("complexity_quadrant", "first"),
            test_n=("test_n", "first"),
            repeat_count=("repeat_id", "nunique"),
            kappa_mean=("kappa", "mean"),
            kappa_sd=("kappa", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_sd=("mcc", "std"),
        )
    )
    return output, parameters, means


def common_metadata(llm_ready: pd.DataFrame, combinations: list[str]) -> pd.DataFrame:
    metadata = (
        llm_ready[llm_ready["combination_id"].isin(combinations)]
        .drop_duplicates("combination_id")
        [
            [
                "combination_id",
                "label_number",
                "selected_label_ids",
                "selected_labels",
                "annotation_disagreement_complexity",
                "semantic_similarity_complexity",
                "complexity_quadrant",
                "test_n",
            ]
        ]
        .copy()
    )
    metadata = metadata.rename(
        columns={
            "label_number": "common_label_number",
            "selected_label_ids": "common_selected_label_ids",
            "selected_labels": "common_selected_labels",
            "annotation_disagreement_complexity": "common_annotation_disagreement",
            "semantic_similarity_complexity": "common_semantic_similarity",
            "complexity_quadrant": "common_complexity_quadrant",
            "test_n": "common_test_n",
        }
    )
    metadata["common_label_number_factor"] = metadata["common_label_number"].astype("Int64").astype(str)
    for source, output in [
        ("common_annotation_disagreement", "common_annotation_disagreement_z"),
        ("common_semantic_similarity", "common_semantic_similarity_z"),
    ]:
        values = pd.to_numeric(metadata[source], errors="coerce")
        std = values.std(ddof=0)
        metadata[output] = (values - values.mean()) / std if std > 0 else np.nan
    return metadata


def prepare_common24(
    common24: pd.DataFrame,
    llm_ready: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combinations = sorted(common24["combination_id"].dropna().astype(str).unique())
    metadata = common_metadata(llm_ready, combinations)

    supervised = common24[common24["model_family"] != "large_language_model"].copy()
    supervised_primary = pd.DataFrame(
        {
            "combination_id": supervised["combination_id"].astype(str),
            "candidate_model": supervised["model_name"].astype(str),
            "candidate_family": supervised["model_family"].astype(str),
            "prompt_strategy": "not_applicable",
            "repeat_count": 1,
            "training_sample_size": supervised["training_sample_size"],
            "training_class_imbalance": supervised["realized_class_imbalance_index"],
            "kappa": supervised["kappa"],
            "macro_f1": supervised["macro_f1"],
            "mcc": supervised["mcc"],
            "accuracy": supervised["accuracy"],
        }
    )

    llm_rows = common24[common24["model_family"] == "large_language_model"].copy()
    llm_aggregated = (
        llm_rows.groupby(["combination_id", "prompt_strategy"], as_index=False)
        .agg(
            repeat_count=("repeat_id", "nunique"),
            kappa=("kappa", "mean"),
            macro_f1=("macro_f1", "mean"),
            mcc=("mcc", "mean"),
            accuracy=("accuracy", "mean"),
        )
    )
    llm_aggregated["candidate_model"] = "DeepSeek V4 Flash (" + llm_aggregated["prompt_strategy"] + ")"
    llm_aggregated["candidate_family"] = "large_language_model"
    llm_aggregated["training_sample_size"] = np.nan
    llm_aggregated["training_class_imbalance"] = np.nan

    all_strategies = pd.concat([supervised_primary, llm_aggregated], ignore_index=True, sort=False)
    all_strategies = all_strategies.merge(metadata, on="combination_id", how="left", validate="many_to_one")

    llm_primary = llm_aggregated[llm_aggregated["prompt_strategy"] == "codebook_guided"].copy()
    primary = pd.concat([supervised_primary, llm_primary], ignore_index=True, sort=False)
    primary = primary.merge(metadata, on="combination_id", how="left", validate="many_to_one")
    return primary, all_strategies


def vif_table(reference: pd.DataFrame, features: list[str], dataset: str) -> pd.DataFrame:
    clean = reference[features].apply(pd.to_numeric, errors="coerce").dropna()
    rows = []
    for feature in features:
        y = clean[feature].to_numpy(dtype=float)
        other_features = [name for name in features if name != feature]
        if np.nanstd(y) <= 0:
            vif = math.inf
            r_squared = 1.0
        elif not other_features:
            vif = 1.0
            r_squared = 0.0
        else:
            x = clean[other_features].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(x)), x])
            coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
            residual = y - x @ coefficients
            sse = float(np.sum(residual**2))
            sst = float(np.sum((y - y.mean()) ** 2))
            r_squared = max(0.0, min(1.0, 1.0 - sse / sst)) if sst > 0 else 1.0
            vif = math.inf if r_squared >= 1.0 - 1e-12 else 1.0 / (1.0 - r_squared)
        rows.append(
            {
                "dataset": dataset,
                "feature": feature,
                "vif": vif,
                "r_squared_against_other_features": r_squared,
                "reference_rows": len(clean),
                "screening_note": "VIF is a screening diagnostic; categorical label-number effects are modeled as factors later.",
            }
        )
    return pd.DataFrame(rows)


def normalized_text(value: Any) -> str:
    return " ".join(str(value).split())


def stable_test_signature(frame: pd.DataFrame) -> str:
    columns = ["record_id", "text", "original_label"]
    prepared = frame[columns].copy()
    prepared["text"] = prepared["text"].map(normalized_text)
    prepared = prepared.sort_values(columns).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(prepared, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def deep_test_checks(package_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mapping_path = package_root / "task_file_mapping.xlsx"
    supervised_dir = package_root / "supervised_data"
    llm_dir = package_root / "llm_data"
    if not mapping_path.exists() or not supervised_dir.exists():
        return pd.DataFrame(), pd.DataFrame(), {"available": False, "reason": "package data files not found"}

    mapping = pd.read_excel(mapping_path)
    fixed_rows = []
    for combination_id, group in mapping.groupby("combination_id"):
        signatures = []
        test_sizes = []
        for row in group.itertuples(index=False):
            test = pd.read_excel(supervised_dir / row.new_test_file)
            signatures.append(stable_test_signature(test))
            test_sizes.append(len(test))
        fixed_rows.append(
            {
                "combination_id": combination_id,
                "task_count": len(group),
                "substantive_test_signatures": len(set(signatures)),
                "test_size_values": ";".join(map(str, sorted(set(test_sizes)))),
                "fixed_test_pass": len(set(signatures)) == 1 and len(set(test_sizes)) == 1,
            }
        )
    fixed_frame = pd.DataFrame(fixed_rows)

    alignment_rows = []
    if llm_dir.exists():
        anchors = mapping[mapping["comparison_anchor"] == 1]
        for row in anchors.itertuples(index=False):
            llm_path = llm_dir / f"LLM_{row.combination_id}.json"
            if not llm_path.exists():
                alignment_rows.append(
                    {
                        "combination_id": row.combination_id,
                        "alignment_pass": False,
                        "reason": "LLM task JSON missing",
                    }
                )
                continue
            supervised_test = pd.read_excel(supervised_dir / row.new_test_file).sort_values("record_id").reset_index(drop=True)
            llm_payload = json.loads(llm_path.read_text(encoding="utf-8"))
            llm_test = pd.DataFrame(llm_payload["fixed_test"]).sort_values("record_id").reset_index(drop=True)
            same_size = len(supervised_test) == len(llm_test)
            same_ids = same_size and supervised_test["record_id"].astype(str).equals(llm_test["record_id"].astype(str))
            same_labels = same_size and supervised_test["original_label"].astype(str).equals(
                llm_test["label"].astype(str)
            )
            if same_size:
                supervised_text = supervised_test["text"].map(normalized_text)
                llm_text = llm_test["text"].map(normalized_text)
                exact_share = float((supervised_text == llm_text).mean())
                similarity = float(
                    np.mean(
                        [
                            difflib.SequenceMatcher(None, left, right).ratio()
                            for left, right in zip(supervised_text, llm_text)
                        ]
                    )
                )
            else:
                exact_share = np.nan
                similarity = np.nan
            alignment_rows.append(
                {
                    "combination_id": row.combination_id,
                    "supervised_test_n": len(supervised_test),
                    "llm_test_n": len(llm_test),
                    "same_record_ids": same_ids,
                    "same_labels": same_labels,
                    "normalized_text_exact_share": exact_share,
                    "mean_text_similarity": similarity,
                    "alignment_pass": bool(same_size and same_ids and same_labels),
                    "reason": "",
                }
            )
    alignment_frame = pd.DataFrame(alignment_rows)
    summary = {
        "available": True,
        "fixed_test_combinations": len(fixed_frame),
        "fixed_test_failures": int((~fixed_frame["fixed_test_pass"]).sum()),
        "common24_alignments": len(alignment_frame),
        "common24_alignment_failures": int((~alignment_frame["alignment_pass"]).sum())
        if not alignment_frame.empty
        else None,
        "weighted_normalized_text_exact_share": float(
            np.average(
                alignment_frame["normalized_text_exact_share"],
                weights=alignment_frame["supervised_test_n"],
            )
        )
        if not alignment_frame.empty
        else None,
        "weighted_mean_text_similarity": float(
            np.average(
                alignment_frame["mean_text_similarity"],
                weights=alignment_frame["supervised_test_n"],
            )
        )
        if not alignment_frame.empty
        else None,
    }
    return fixed_frame, alignment_frame, summary


def data_dictionary() -> pd.DataFrame:
    rows = [
        ("supervised_analysis_ready.csv", "kappa", "Primary reliability outcome", "original"),
        ("supervised_analysis_ready.csv", "label_number_factor", "Label-number factor: 3, 5, 8, or 11", "derived"),
        ("supervised_analysis_ready.csv", "class_imbalance_z", "Standardized realized class-imbalance index", "derived"),
        ("supervised_analysis_ready.csv", "annotation_disagreement_z", "Standardized sampled annotation-disagreement rate", "derived"),
        ("supervised_analysis_ready.csv", "log2_training_size_z", "Standardized base-2 log training size", "derived"),
        ("supervised_analysis_ready.csv", "semantic_similarity_z", "Standardized average pairwise label-semantic similarity", "derived"),
        ("supervised_analysis_ready.csv", "task_cluster_id", "Paired SVM-DeBERTa task cluster", "derived"),
        ("supervised_analysis_ready.csv", "combination_cluster_id", "Label-combination cluster", "derived"),
        ("llm_analysis_ready.csv", "prompt_strategy_factor", "Zero-shot, few-shot, or codebook-guided prompt", "derived"),
        ("llm_analysis_ready.csv", "repeat_id_factor", "Repeated API run identifier", "derived"),
        ("llm_analysis_ready.csv", "annotation_disagreement_z", "Combination-level standardized disagreement", "derived"),
        ("llm_analysis_ready.csv", "semantic_similarity_z", "Combination-level standardized semantic similarity", "derived"),
        ("common24_primary_analysis_ready.csv", "candidate_model", "SVM, DeBERTa, or codebook-guided DeepSeek", "derived"),
        ("common24_primary_analysis_ready.csv", "common_annotation_disagreement_z", "Common24 standardized disagreement", "derived"),
        ("common24_primary_analysis_ready.csv", "common_semantic_similarity_z", "Common24 standardized semantic similarity", "derived"),
    ]
    return pd.DataFrame(rows, columns=["file", "variable", "definition", "status"])


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    package_root = detect_package_root(args.package_root)
    input_dir = detect_input_dir(package_root, args.input_dir)
    output_dir = (args.output_dir or package_root / "outputs" / "statistical_analysis" / "step1_data_quality").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = read_inputs(input_dir)
    checks, profiles = run_summary_quality_checks(tables)
    supervised_ready, supervised_parameters, supervised_pairs = prepare_supervised(tables["supervised"])
    llm_ready, llm_parameters, llm_prompt_means = prepare_llm(tables["llm"])
    common_primary, common_all_strategies = prepare_common24(tables["common24"], llm_ready)

    supervised_reference = supervised_ready.drop_duplicates("original_task_id")
    llm_reference = llm_ready.drop_duplicates("combination_id")
    supervised_features = [
        "label_number",
        "class_imbalance_complexity",
        "annotation_disagreement_complexity",
        "log2_training_size",
        "semantic_similarity_complexity",
    ]
    llm_features = [
        "label_number",
        "annotation_disagreement_complexity",
        "semantic_similarity_complexity",
    ]
    supervised_correlation = supervised_reference[supervised_features].corr(method="spearman")
    llm_correlation = llm_reference[llm_features].corr(method="spearman")
    supervised_vif = vif_table(supervised_reference, supervised_features, "supervised_task_reference")
    llm_vif = vif_table(llm_reference, llm_features, "llm_combination_reference")

    supervised_support = (
        supervised_reference.groupby(
            ["label_number", "training_sample_size", "design_tier", "class_balance_design"],
            dropna=False,
        )
        .size()
        .rename("unique_task_count")
        .reset_index()
    )
    llm_support = (
        llm_reference.groupby(["label_number", "complexity_quadrant"], dropna=False)
        .size()
        .rename("unique_combination_count")
        .reset_index()
    )

    fixed_tests = pd.DataFrame()
    common_alignment = pd.DataFrame()
    deep_summary: dict[str, Any]
    if args.skip_deep_test_checks:
        deep_summary = {"available": False, "reason": "skipped by argument"}
    else:
        fixed_tests, common_alignment, deep_summary = deep_test_checks(package_root)

    if deep_summary.get("available"):
        add_check(
            checks_records := checks.to_dict("records"),
            "source_test_data",
            "fixed_test_substantive_consistency",
            deep_summary["fixed_test_failures"] == 0,
            "critical",
            {"failed_combinations": deep_summary["fixed_test_failures"]},
            "one substantive fixed-test signature per label combination",
            "Training-size and imbalance effects require a fixed test set within each label combination.",
        )
        add_check(
            checks_records,
            "source_test_data",
            "common24_record_label_alignment",
            deep_summary["common24_alignment_failures"] == 0,
            "critical",
            {
                "failed_combinations": deep_summary["common24_alignment_failures"],
                "normalized_text_exact_share": deep_summary["weighted_normalized_text_exact_share"],
                "mean_text_similarity": deep_summary["weighted_mean_text_similarity"],
            },
            "identical record IDs and target labels for supervised and LLM Common24 tests",
            "Cross-family comparisons require the same clauses and target labels.",
        )
        checks = pd.DataFrame(checks_records)
    else:
        checks_records = checks.to_dict("records")
        add_check(
            checks_records,
            "source_test_data",
            "deep_fixed_test_checks",
            None,
            "high",
            deep_summary.get("reason", "not available"),
            "package source data available",
            "Summary-level checks still run, but sample-level test consistency was not rechecked.",
        )
        checks = pd.DataFrame(checks_records)

    parameters = pd.concat([supervised_parameters, llm_parameters], ignore_index=True)
    dictionary = data_dictionary()

    save_csv(supervised_ready, output_dir / "supervised_analysis_ready.csv")
    save_csv(llm_ready, output_dir / "llm_analysis_ready.csv")
    save_csv(supervised_pairs, output_dir / "supervised_task_pairs.csv")
    save_csv(llm_prompt_means, output_dir / "llm_combination_prompt_means.csv")
    save_csv(common_primary, output_dir / "common24_primary_analysis_ready.csv")
    save_csv(common_all_strategies, output_dir / "common24_all_strategies_analysis_ready.csv")
    save_csv(checks, output_dir / "quality_checks.csv")
    save_csv(profiles, output_dir / "dataset_profiles.csv")
    save_csv(parameters, output_dir / "standardization_parameters.csv")
    save_csv(supervised_support, output_dir / "supervised_complexity_support.csv")
    save_csv(llm_support, output_dir / "llm_complexity_support.csv")
    supervised_correlation.to_csv(output_dir / "supervised_complexity_spearman.csv", encoding="utf-8-sig")
    llm_correlation.to_csv(output_dir / "llm_complexity_spearman.csv", encoding="utf-8-sig")
    save_csv(supervised_vif, output_dir / "supervised_vif_screening.csv")
    save_csv(llm_vif, output_dir / "llm_vif_screening.csv")
    save_csv(dictionary, output_dir / "analysis_data_dictionary.csv")
    if not fixed_tests.empty:
        save_csv(fixed_tests, output_dir / "fixed_test_consistency.csv")
    if not common_alignment.empty:
        save_csv(common_alignment, output_dir / "common24_test_alignment.csv")

    failures = checks[checks["status"] == "FAIL"]
    report = {
        "generated_at_utc": utc_now(),
        "input_dir": input_dir,
        "package_root": package_root,
        "output_dir": output_dir,
        "input_rows": {name: len(frame) for name, frame in tables.items()},
        "analysis_ready_rows": {
            "supervised": len(supervised_ready),
            "llm": len(llm_ready),
            "supervised_pairs": len(supervised_pairs),
            "llm_prompt_means": len(llm_prompt_means),
            "common24_primary": len(common_primary),
            "common24_all_strategies": len(common_all_strategies),
        },
        "quality_check_counts": checks["status"].value_counts().to_dict(),
        "failed_checks": failures[
            ["dataset", "check_id", "severity_if_failed", "observed", "implication"]
        ].to_dict("records"),
        "deep_test_check_summary": deep_summary,
        "analysis_rules": {
            "primary_outcome": "kappa",
            "robustness_outcomes": ["macro_f1", "mcc"],
            "label_number_encoding": "categorical factor",
            "training_size_transformation": "log2 then z-standardized on unique supervised tasks",
            "continuous_standardization": "population SD (ddof=0) on unique task or combination reference units",
            "llm_nonapplicable_dimensions": ["training_size", "training_class_imbalance"],
            "common24_primary_llm_strategy": "codebook_guided averaged over three repeats",
        },
    }
    write_json(output_dir / "step1_validation_report.json", report)

    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    print(f"\nStep 1 outputs: {output_dir}")


if __name__ == "__main__":
    main()
