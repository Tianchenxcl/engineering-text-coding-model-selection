from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTCOMES = ["kappa", "macro_f1", "mcc"]
OUTCOME_DISPLAY = {
    "kappa": "Cohen's kappa",
    "macro_f1": "Macro-F1",
    "mcc": "MCC",
}
MODEL_VARIANTS = [
    "SVM",
    "DeBERTa",
    "LLM_zero_shot",
    "LLM_few_shot",
    "LLM_codebook_guided",
]
VARIANT_DISPLAY = {
    "SVM": "SVM",
    "DeBERTa": "DeBERTa",
    "LLM_zero_shot": "LLM zero-shot",
    "LLM_few_shot": "LLM few-shot",
    "LLM_codebook_guided": "LLM codebook",
}
ALGORITHMS = ["candidate_mean", "ridge", "gradient_boosting", "mlp"]
ALGORITHM_DISPLAY = {
    "candidate_mean": "Candidate mean",
    "ridge": "Ridge interactions",
    "gradient_boosting": "Gradient boosting",
    "mlp": "MLP (supplementary)",
}
ALGORITHM_COLORS = {
    "candidate_mean": "#777B80",
    "ridge": "#247B7B",
    "gradient_boosting": "#D28B3C",
    "mlp": "#B55245",
}
NUMERIC_BASE_FEATURES = [
    "label_number_log2",
    "class_imbalance_filled",
    "annotation_disagreement",
    "training_size_log2_filled",
    "semantic_similarity",
]
OUTER_FOLDS = 8
INNER_FOLDS = 5
BOOTSTRAP_REPETITIONS = 20000


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
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 6: model-performance prediction")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], dataset: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")


def prepare_unified_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    supervised_path = input_dir / "supervised_analysis_ready.csv"
    llm_path = input_dir / "llm_combination_prompt_means.csv"
    common_path = input_dir / "common24_primary_analysis_ready.csv"
    for path in [supervised_path, llm_path, common_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing Step 1 input: {path}")

    supervised = pd.read_csv(supervised_path, low_memory=False)
    llm = pd.read_csv(llm_path, low_memory=False)
    common = pd.read_csv(common_path, low_memory=False)
    require_columns(
        supervised,
        [
            "performance_unit_id",
            "combination_id",
            "model_variant",
            "comparison_anchor",
            "label_number_complexity",
            "class_imbalance_complexity",
            "annotation_disagreement_complexity",
            "training_sample_size",
            "semantic_similarity_complexity",
            *OUTCOMES,
        ],
        "supervised_analysis_ready",
    )
    require_columns(
        llm,
        [
            "combination_id",
            "prompt_strategy",
            "label_number",
            "annotation_disagreement_complexity",
            "semantic_similarity_complexity",
            "kappa_mean",
            "macro_f1_mean",
            "mcc_mean",
            "repeat_count",
        ],
        "llm_combination_prompt_means",
    )
    require_columns(common, ["combination_id", "candidate_model", *OUTCOMES], "common24_primary")
    common_combinations = set(common["combination_id"].astype(str))

    supervised_variant_map = {
        "TF-IDF + LinearSVC": "SVM",
        "DeBERTa-v3-base": "DeBERTa",
    }
    supervised_output = pd.DataFrame(
        {
            "record_id": supervised["performance_unit_id"].astype(str),
            "source": "supervised",
            "combination_id": supervised["combination_id"].astype(str),
            "model_variant": supervised["model_variant"].map(supervised_variant_map),
            "model_family": supervised["model_variant"].map(
                {
                    "TF-IDF + LinearSVC": "traditional_ml",
                    "DeBERTa-v3-base": "pretrained_language_model",
                }
            ),
            "label_number": supervised["label_number_complexity"],
            "class_imbalance": supervised["class_imbalance_complexity"],
            "annotation_disagreement": supervised["annotation_disagreement_complexity"],
            "training_sample_size": supervised["training_sample_size"],
            "semantic_similarity": supervised["semantic_similarity_complexity"],
            "kappa": supervised["kappa"],
            "macro_f1": supervised["macro_f1"],
            "mcc": supervised["mcc"],
            "repeat_count": 1,
            "is_common24_primary": supervised["comparison_anchor"].astype(int).eq(1),
        }
    )

    llm_variant_map = {
        "zero_shot": "LLM_zero_shot",
        "few_shot": "LLM_few_shot",
        "codebook_guided": "LLM_codebook_guided",
    }
    llm_output = pd.DataFrame(
        {
            "record_id": (
                "llmmean::"
                + llm["combination_id"].astype(str)
                + "::"
                + llm["prompt_strategy"].astype(str)
            ),
            "source": "llm",
            "combination_id": llm["combination_id"].astype(str),
            "model_variant": llm["prompt_strategy"].map(llm_variant_map),
            "model_family": "large_language_model",
            "label_number": llm["label_number"],
            "class_imbalance": np.nan,
            "annotation_disagreement": llm["annotation_disagreement_complexity"],
            "training_sample_size": np.nan,
            "semantic_similarity": llm["semantic_similarity_complexity"],
            "kappa": llm["kappa_mean"],
            "macro_f1": llm["macro_f1_mean"],
            "mcc": llm["mcc_mean"],
            "repeat_count": llm["repeat_count"],
            "is_common24_primary": (
                llm["combination_id"].astype(str).isin(common_combinations)
                & llm["prompt_strategy"].astype(str).eq("codebook_guided")
            ),
        }
    )
    data = pd.concat([supervised_output, llm_output], ignore_index=True)
    numeric_columns = [
        "label_number",
        "class_imbalance",
        "annotation_disagreement",
        "training_sample_size",
        "semantic_similarity",
        *OUTCOMES,
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    universally_required = [
        "record_id",
        "combination_id",
        "model_variant",
        "model_family",
        "label_number",
        "annotation_disagreement",
        "semantic_similarity",
        *OUTCOMES,
    ]
    if data[universally_required].isna().any().any():
        missing = data[universally_required].isna().sum()
        raise ValueError(f"Unified prediction data contain missing values: {missing[missing > 0].to_dict()}")
    supervised_mask = data["source"].eq("supervised")
    if data.loc[supervised_mask, ["class_imbalance", "training_sample_size"]].isna().any().any():
        raise ValueError("Supervised prediction records are missing training-design predictors")
    if data.loc[~supervised_mask, ["class_imbalance", "training_sample_size"]].notna().any().any():
        raise ValueError("LLM training size and imbalance must remain not applicable")
    if data["record_id"].duplicated().any():
        raise ValueError("Prediction record IDs are not unique")
    if set(data["model_variant"].unique()) != set(MODEL_VARIANTS):
        raise ValueError(f"Unexpected model variants: {sorted(data['model_variant'].unique())}")
    if not np.isfinite(data[["label_number", "annotation_disagreement", "semantic_similarity", *OUTCOMES]]).all().all():
        raise ValueError("Unified prediction data contain non-finite values")
    if not data[OUTCOMES].apply(lambda column: column.between(-1, 1).all()).all():
        raise ValueError("Performance values fall outside [-1, 1]")
    if data["combination_id"].nunique() != 40:
        raise ValueError("Expected 40 unique label combinations")
    if len(data) != 600:
        raise ValueError(f"Expected 600 model-task records, found {len(data)}")
    if data["is_common24_primary"].sum() != 72:
        raise ValueError("Expected 72 Common24 primary records")

    data["label_number_log2"] = np.log2(data["label_number"].astype(float))
    data["class_imbalance_filled"] = data["class_imbalance"].fillna(0.0)
    data["training_size_log2_filled"] = np.where(
        data["training_sample_size"].notna(),
        np.log2(data["training_sample_size"].fillna(1.0)),
        0.0,
    )
    data["model_variant"] = pd.Categorical(
        data["model_variant"], categories=MODEL_VARIANTS, ordered=True
    )
    data = data.sort_values(["combination_id", "model_variant", "record_id"]).reset_index(drop=True)

    support = (
        data.groupby("model_variant", observed=True)
        .agg(
            record_count=("record_id", "size"),
            combination_count=("combination_id", "nunique"),
            minimum_label_number=("label_number", "min"),
            maximum_label_number=("label_number", "max"),
            minimum_class_imbalance=("class_imbalance", "min"),
            maximum_class_imbalance=("class_imbalance", "max"),
            minimum_disagreement=("annotation_disagreement", "min"),
            maximum_disagreement=("annotation_disagreement", "max"),
            minimum_training_size=("training_sample_size", "min"),
            maximum_training_size=("training_sample_size", "max"),
            minimum_similarity=("semantic_similarity", "min"),
            maximum_similarity=("semantic_similarity", "max"),
        )
        .reset_index()
    )
    support["model_variant"] = support["model_variant"].astype(str)
    return data, support


def build_design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["model_variant", *NUMERIC_BASE_FEATURES]
    require_columns(frame, required, "prediction input")
    output = frame[NUMERIC_BASE_FEATURES].astype(float).copy()
    variants = frame["model_variant"].astype(str)
    for variant in MODEL_VARIANTS[1:]:
        dummy_name = f"model__{variant}"
        output[dummy_name] = variants.eq(variant).astype(float)
    interaction_features = {
        "DeBERTa": NUMERIC_BASE_FEATURES,
        "LLM_zero_shot": [
            "label_number_log2",
            "annotation_disagreement",
            "semantic_similarity",
        ],
        "LLM_few_shot": [
            "label_number_log2",
            "annotation_disagreement",
            "semantic_similarity",
        ],
        "LLM_codebook_guided": [
            "label_number_log2",
            "annotation_disagreement",
            "semantic_similarity",
        ],
    }
    for variant, features in interaction_features.items():
        dummy = variants.eq(variant).astype(float)
        for feature in features:
            output[f"interaction__{variant}__{feature}"] = dummy * output[feature]
    return output


def variant_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["model_variant"].astype(str).value_counts()
    weights = frame["model_variant"].astype(str).map(lambda value: 1.0 / counts[value]).to_numpy(float)
    return weights / weights.mean()


def variant_balanced_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variants: pd.Series,
) -> float:
    values = []
    variants = variants.astype(str).reset_index(drop=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    for variant in sorted(variants.unique()):
        mask = variants.eq(variant).to_numpy()
        values.append(float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))))
    return float(np.mean(values))


def algorithm_grids() -> dict[str, list[dict[str, Any]]]:
    return {
        "ridge": [{"alpha": value} for value in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]],
        "gradient_boosting": [
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
            }
            for n_estimators in [100, 250]
            for learning_rate in [0.03, 0.08]
            for max_depth in [1, 2]
            for min_samples_leaf in [5]
        ],
        "mlp": [
            {"hidden_layer_sizes": (16,), "alpha": 0.001},
            {"hidden_layer_sizes": (16,), "alpha": 0.01},
            {"hidden_layer_sizes": (32,), "alpha": 0.01},
            {"hidden_layer_sizes": (16, 8), "alpha": 0.01},
        ],
    }


def make_estimator(algorithm: str, parameters: dict[str, Any], seed: int) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if algorithm == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=float(parameters["alpha"]))),
            ]
        )
    if algorithm == "gradient_boosting":
        return GradientBoostingRegressor(
            random_state=seed,
            loss="squared_error",
            n_estimators=int(parameters["n_estimators"]),
            learning_rate=float(parameters["learning_rate"]),
            max_depth=int(parameters["max_depth"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=None,
        )
    if algorithm == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=tuple(parameters["hidden_layer_sizes"]),
                        alpha=float(parameters["alpha"]),
                        learning_rate_init=0.001,
                        activation="relu",
                        solver="adam",
                        max_iter=1000,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown prediction algorithm: {algorithm}")


def fit_estimator(
    estimator: Any,
    algorithm: str,
    x: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
) -> Any:
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if algorithm in {"ridge", "mlp"}:
            try:
                return estimator.fit(x, y, model__sample_weight=weights)
            except TypeError:
                return estimator.fit(x, y)
        return estimator.fit(x, y, sample_weight=weights)


def tune_algorithm(
    frame: pd.DataFrame,
    x: pd.DataFrame,
    y: np.ndarray,
    algorithm: str,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    from sklearn.model_selection import GroupKFold

    groups = frame["combination_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=min(INNER_FOLDS, len(np.unique(groups))))
    rows: list[dict[str, Any]] = []
    for parameter_index, parameters in enumerate(algorithm_grids()[algorithm]):
        fold_scores = []
        for inner_fold, (train_index, validation_index) in enumerate(splitter.split(x, y, groups), start=1):
            estimator = make_estimator(algorithm, parameters, seed + parameter_index * 101 + inner_fold)
            train_frame = frame.iloc[train_index]
            weights = variant_balanced_weights(train_frame)
            fit_estimator(estimator, algorithm, x.iloc[train_index], y[train_index], weights)
            prediction = estimator.predict(x.iloc[validation_index])
            fold_scores.append(
                variant_balanced_rmse(
                    y[validation_index],
                    prediction,
                    frame.iloc[validation_index]["model_variant"],
                )
            )
        rows.append(
            {
                "algorithm": algorithm,
                "parameters": json.dumps(json_safe(parameters), sort_keys=True),
                "mean_inner_variant_balanced_rmse": float(np.mean(fold_scores)),
                "standard_deviation_inner_score": float(np.std(fold_scores, ddof=1)),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["mean_inner_variant_balanced_rmse", "parameters"]
    ).reset_index(drop=True)
    best = json.loads(scores.iloc[0]["parameters"])
    if "hidden_layer_sizes" in best:
        best["hidden_layer_sizes"] = tuple(best["hidden_layer_sizes"])
    return best, scores


def candidate_mean_predict(
    training_frame: pd.DataFrame,
    training_y: np.ndarray,
    test_frame: pd.DataFrame,
) -> np.ndarray:
    means = (
        pd.DataFrame(
            {
                "model_variant": training_frame["model_variant"].astype(str).to_numpy(),
                "target": training_y,
            }
        )
        .groupby("model_variant")["target"]
        .mean()
    )
    global_mean = float(np.mean(training_y))
    return test_frame["model_variant"].astype(str).map(means).fillna(global_mean).to_numpy(float)


def nested_group_predictions(
    data: pd.DataFrame,
    x: pd.DataFrame,
    outcome: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    from sklearn.model_selection import GroupKFold

    y = data[outcome].to_numpy(float)
    groups = data["combination_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=OUTER_FOLDS)
    prediction_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    for outer_fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
        train_frame = data.iloc[train_index].reset_index(drop=True)
        test_frame = data.iloc[test_index].reset_index(drop=True)
        x_train = x.iloc[train_index].reset_index(drop=True)
        x_test = x.iloc[test_index].reset_index(drop=True)
        y_train = y[train_index]
        y_test = y[test_index]
        baseline_prediction = candidate_mean_predict(train_frame, y_train, test_frame)
        for row_index, prediction in enumerate(baseline_prediction):
            prediction_rows.append(
                {
                    "outcome": outcome,
                    "algorithm": "candidate_mean",
                    "outer_fold": outer_fold,
                    "record_id": test_frame.iloc[row_index]["record_id"],
                    "combination_id": test_frame.iloc[row_index]["combination_id"],
                    "model_variant": str(test_frame.iloc[row_index]["model_variant"]),
                    "is_common24_primary": bool(test_frame.iloc[row_index]["is_common24_primary"]),
                    "actual": y_test[row_index],
                    "prediction": prediction,
                }
            )
        for algorithm in ["ridge", "gradient_boosting", "mlp"]:
            best_parameters, tuning_scores = tune_algorithm(
                train_frame,
                x_train,
                y_train,
                algorithm,
                seed + outer_fold * 1000,
            )
            tuning_scores["outcome"] = outcome
            tuning_scores["outer_fold"] = outer_fold
            tuning_scores["selected"] = tuning_scores["parameters"].eq(
                json.dumps(json_safe(best_parameters), sort_keys=True)
            )
            tuning_rows.extend(tuning_scores.to_dict(orient="records"))
            estimator = make_estimator(algorithm, best_parameters, seed + outer_fold * 1000)
            weights = variant_balanced_weights(train_frame)
            fit_estimator(estimator, algorithm, x_train, y_train, weights)
            prediction = estimator.predict(x_test)
            for row_index, value in enumerate(prediction):
                prediction_rows.append(
                    {
                        "outcome": outcome,
                        "algorithm": algorithm,
                        "outer_fold": outer_fold,
                        "record_id": test_frame.iloc[row_index]["record_id"],
                        "combination_id": test_frame.iloc[row_index]["combination_id"],
                        "model_variant": str(test_frame.iloc[row_index]["model_variant"]),
                        "is_common24_primary": bool(test_frame.iloc[row_index]["is_common24_primary"]),
                        "actual": y_test[row_index],
                        "prediction": value,
                    }
                )
        print(f"[step6] outcome={outcome} outer_fold={outer_fold}/{OUTER_FOLDS} complete")

    final_parameters: dict[str, Any] = {}
    final_scores: dict[str, pd.DataFrame] = {}
    for algorithm in ["ridge", "gradient_boosting", "mlp"]:
        parameters, scores = tune_algorithm(data, x, y, algorithm, seed + 90000)
        final_parameters[algorithm] = parameters
        final_scores[algorithm] = scores
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(tuning_rows),
        final_parameters,
        final_scores,
    )


def basic_prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    from scipy.stats import spearmanr

    actual = frame["actual"].to_numpy(float)
    prediction = frame["prediction"].to_numpy(float)
    residual = actual - prediction
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else np.nan
    design = np.column_stack([np.ones(len(prediction)), prediction])
    calibration = np.linalg.lstsq(design, actual, rcond=None)[0]
    group_rmse = frame.assign(squared_error=residual**2).groupby("combination_id")[
        "squared_error"
    ].mean().pow(0.5)
    variant_rmse = frame.assign(squared_error=residual**2).groupby("model_variant")[
        "squared_error"
    ].mean().pow(0.5)
    correlation = spearmanr(actual, prediction)
    return {
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared,
        "spearman_rho": float(correlation.statistic),
        "calibration_intercept": float(calibration[0]),
        "calibration_slope": float(calibration[1]),
        "group_balanced_rmse": float(group_rmse.mean()),
        "variant_balanced_rmse": float(variant_rmse.mean()),
        "prediction_minimum": float(prediction.min()),
        "prediction_maximum": float(prediction.max()),
        "predictions_outside_minus1_plus1": int(np.sum((prediction < -1) | (prediction > 1))),
    }


def group_bootstrap_rmse(
    frame: pd.DataFrame,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = frame["combination_id"].astype(str).unique()
    group_errors = {
        group: (
            frame.loc[frame["combination_id"].astype(str).eq(group), "actual"].to_numpy(float)
            - frame.loc[frame["combination_id"].astype(str).eq(group), "prediction"].to_numpy(float)
        )
        ** 2
        for group in groups
    }
    draws = np.empty(BOOTSTRAP_REPETITIONS)
    for repetition in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        squared = np.concatenate([group_errors[group] for group in sampled])
        draws[repetition] = np.sqrt(np.mean(squared))
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def summarize_predictions(oof: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        for algorithm_index, algorithm in enumerate(ALGORITHMS):
            subset = oof[(oof["outcome"] == outcome) & (oof["algorithm"] == algorithm)].copy()
            metrics = basic_prediction_metrics(subset)
            ci_low, ci_high = group_bootstrap_rmse(
                subset, seed + outcome_index * 100 + algorithm_index
            )
            summary_rows.append(
                {
                    "outcome": outcome,
                    "algorithm": algorithm,
                    "algorithm_display": ALGORITHM_DISPLAY[algorithm],
                    "record_count": len(subset),
                    "combination_count": subset["combination_id"].nunique(),
                    **metrics,
                    "rmse_group_bootstrap_ci95_low": ci_low,
                    "rmse_group_bootstrap_ci95_high": ci_high,
                }
            )
            for variant in MODEL_VARIANTS:
                variant_subset = subset[subset["model_variant"] == variant]
                variant_metrics = basic_prediction_metrics(variant_subset)
                variant_rows.append(
                    {
                        "outcome": outcome,
                        "algorithm": algorithm,
                        "algorithm_display": ALGORITHM_DISPLAY[algorithm],
                        "model_variant": variant,
                        "model_variant_display": VARIANT_DISPLAY[variant],
                        "record_count": len(variant_subset),
                        "combination_count": variant_subset["combination_id"].nunique(),
                        **variant_metrics,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(variant_rows)


def summarize_predictions_by_family(oof: pd.DataFrame) -> pd.DataFrame:
    output = oof.copy()
    output["prediction_family"] = np.where(
        output["model_variant"].isin(["SVM", "DeBERTa"]),
        "supervised_models",
        "llm_prompt_variants",
    )
    rows: list[dict[str, Any]] = []
    for (outcome, algorithm, family), subset in output.groupby(
        ["outcome", "algorithm", "prediction_family"]
    ):
        rows.append(
            {
                "outcome": outcome,
                "algorithm": algorithm,
                "algorithm_display": ALGORITHM_DISPLAY[algorithm],
                "prediction_family": family,
                "record_count": len(subset),
                "combination_count": subset["combination_id"].nunique(),
                **basic_prediction_metrics(subset),
            }
        )
    return pd.DataFrame(rows)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return pd.Series(output, index=p_values.index)
    order = valid[np.argsort(values[valid])]
    adjusted = np.maximum.accumulate(
        np.array([(len(order) - rank) * values[index] for rank, index in enumerate(order)])
    )
    adjusted = np.minimum(adjusted, 1.0)
    for index, value in zip(order, adjusted):
        output[index] = value
    return pd.Series(output, index=p_values.index)


def compare_algorithm_errors(oof: pd.DataFrame, seed: int) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    contrasts = [
        ("ridge_minus_mean", "ridge", "candidate_mean"),
        ("gbt_minus_ridge", "gradient_boosting", "ridge"),
        ("mlp_minus_ridge", "mlp", "ridge"),
        ("gbt_minus_mlp", "gradient_boosting", "mlp"),
    ]
    rows: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        subset = oof[oof["outcome"] == outcome].copy()
        subset["squared_error"] = (subset["actual"] - subset["prediction"]) ** 2
        group_rmse = (
            subset.groupby(["combination_id", "algorithm"])["squared_error"]
            .mean()
            .pow(0.5)
            .unstack("algorithm")
        )
        for contrast_index, (contrast, algorithm_a, algorithm_b) in enumerate(contrasts):
            difference = (group_rmse[algorithm_a] - group_rmse[algorithm_b]).to_numpy(float)
            rng = np.random.default_rng(seed + outcome_index * 100 + contrast_index)
            indices = rng.integers(0, len(difference), size=(BOOTSTRAP_REPETITIONS, len(difference)))
            bootstrap = difference[indices].mean(axis=1)
            ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
            test = wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
            rows.append(
                {
                    "outcome": outcome,
                    "contrast": contrast,
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "interpretation": "Negative values favor algorithm_a",
                    "combination_count": len(difference),
                    "mean_group_rmse_difference": difference.mean(),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "wilcoxon_statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                }
            )
    output = pd.DataFrame(rows)
    output["p_value_holm_within_outcome"] = output.groupby("outcome", group_keys=False)[
        "p_value"
    ].apply(holm_adjust)
    return output


def residual_quantiles(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (outcome, algorithm, model_variant), subset in oof.groupby(
        ["outcome", "algorithm", "model_variant"]
    ):
        absolute_error = np.abs(subset["actual"] - subset["prediction"])
        rows.append(
            {
                "outcome": outcome,
                "algorithm": algorithm,
                "model_variant": model_variant,
                "record_count": len(subset),
                "absolute_error_q80": np.quantile(absolute_error, 0.80),
                "absolute_error_q90": np.quantile(absolute_error, 0.90),
                "absolute_error_q95": np.quantile(absolute_error, 0.95),
            }
        )
    return pd.DataFrame(rows)


def common24_routing(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor_variants = ["SVM", "DeBERTa", "LLM_codebook_guided"]
    kappa = oof[(oof["outcome"] == "kappa") & (oof["is_common24_primary"])].copy()
    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    actual_wide = (
        kappa[kappa["algorithm"] == "ridge"]
        .pivot(index="combination_id", columns="model_variant", values="actual")
        .loc[:, anchor_variants]
    )
    for algorithm in ALGORITHMS:
        prediction_wide = (
            kappa[kappa["algorithm"] == algorithm]
            .pivot(index="combination_id", columns="model_variant", values="prediction")
            .loc[actual_wide.index, anchor_variants]
        )
        for combination_id in actual_wide.index:
            actual = actual_wide.loc[combination_id]
            prediction = prediction_wide.loc[combination_id]
            actual_winner = actual.idxmax()
            predicted_winner = prediction.idxmax()
            selected_actual = float(actual[predicted_winner])
            oracle_actual = float(actual.max())
            details.append(
                {
                    "algorithm": algorithm,
                    "combination_id": combination_id,
                    "predicted_winner": predicted_winner,
                    "actual_winner": actual_winner,
                    "winner_correct": predicted_winner == actual_winner,
                    "selected_actual_kappa": selected_actual,
                    "oracle_actual_kappa": oracle_actual,
                    "regret": oracle_actual - selected_actual,
                    "within_0_02_of_oracle": oracle_actual - selected_actual <= 0.02,
                }
            )
    for combination_id in actual_wide.index:
        actual = actual_wide.loc[combination_id]
        selected_actual = float(actual["DeBERTa"])
        oracle_actual = float(actual.max())
        details.append(
            {
                "algorithm": "always_deberta",
                "combination_id": combination_id,
                "predicted_winner": "DeBERTa",
                "actual_winner": actual.idxmax(),
                "winner_correct": actual.idxmax() == "DeBERTa",
                "selected_actual_kappa": selected_actual,
                "oracle_actual_kappa": oracle_actual,
                "regret": oracle_actual - selected_actual,
                "within_0_02_of_oracle": oracle_actual - selected_actual <= 0.02,
            }
        )
    detail_frame = pd.DataFrame(details)
    for algorithm, subset in detail_frame.groupby("algorithm"):
        summaries.append(
            {
                "algorithm": algorithm,
                "algorithm_display": (
                    "Always DeBERTa"
                    if algorithm == "always_deberta"
                    else ALGORITHM_DISPLAY[algorithm]
                ),
                "combination_count": len(subset),
                "winner_accuracy": subset["winner_correct"].mean(),
                "within_0_02_of_oracle_share": subset["within_0_02_of_oracle"].mean(),
                "mean_selected_kappa": subset["selected_actual_kappa"].mean(),
                "mean_oracle_kappa": subset["oracle_actual_kappa"].mean(),
                "mean_regret": subset["regret"].mean(),
                "median_regret": subset["regret"].median(),
                "maximum_regret": subset["regret"].max(),
            }
        )
    return detail_frame, pd.DataFrame(summaries)


def compare_routing_regret(routing_detail: pd.DataFrame, seed: int) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    pivot = routing_detail.pivot(
        index="combination_id", columns="algorithm", values="regret"
    )
    contrasts = [
        ("mean_minus_always_deberta", "candidate_mean", "always_deberta"),
        ("ridge_minus_always_deberta", "ridge", "always_deberta"),
        ("gbt_minus_always_deberta", "gradient_boosting", "always_deberta"),
        ("mlp_minus_always_deberta", "mlp", "always_deberta"),
        ("gbt_minus_ridge", "gradient_boosting", "ridge"),
    ]
    rows: list[dict[str, Any]] = []
    for contrast_index, (contrast, algorithm_a, algorithm_b) in enumerate(contrasts):
        difference = (pivot[algorithm_a] - pivot[algorithm_b]).to_numpy(float)
        rng = np.random.default_rng(seed + contrast_index)
        indices = rng.integers(0, len(difference), size=(BOOTSTRAP_REPETITIONS, len(difference)))
        bootstrap = difference[indices].mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
        if np.all(np.abs(difference) <= 1e-12):
            statistic, p_value = 0.0, 1.0
        else:
            test = wilcoxon(difference, zero_method="zsplit", alternative="two-sided")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        rows.append(
            {
                "contrast": contrast,
                "algorithm_a": algorithm_a,
                "algorithm_b": algorithm_b,
                "interpretation": "Negative values favor algorithm_a",
                "combination_count": len(difference),
                "mean_regret_difference": difference.mean(),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    output = pd.DataFrame(rows)
    output["p_value_holm"] = holm_adjust(output["p_value"])
    return output


def final_model_artifacts(
    data: pd.DataFrame,
    x: pd.DataFrame,
    outcome: str,
    final_parameters: dict[str, Any],
    quantiles: pd.DataFrame,
    support: pd.DataFrame,
    models_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import joblib

    coefficient_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    y = data[outcome].to_numpy(float)
    for algorithm in ["ridge", "gradient_boosting", "mlp"]:
        estimator = make_estimator(algorithm, final_parameters[algorithm], seed + 99000)
        weights = variant_balanced_weights(data)
        fit_estimator(estimator, algorithm, x, y, weights)
        algorithm_quantiles = quantiles[
            (quantiles["outcome"] == outcome) & (quantiles["algorithm"] == algorithm)
        ]
        bundle = {
            "created_at_utc": utc_now(),
            "outcome": outcome,
            "algorithm": algorithm,
            "parameters": final_parameters[algorithm],
            "feature_columns": list(x.columns),
            "model_variants": MODEL_VARIANTS,
            "estimator": estimator,
            "absolute_error_quantiles": algorithm_quantiles.to_dict(orient="records"),
            "training_support": support.to_dict(orient="records"),
            "interpretation_boundary": (
                "Predictions apply to tasks within the observed complexity support and estimate association, not causation."
            ),
        }
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, models_dir / f"{outcome}__{algorithm}.joblib")
        if algorithm == "ridge":
            model = estimator.named_steps["model"]
            for feature, coefficient in zip(x.columns, model.coef_):
                coefficient_rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "standardized_coefficient": float(coefficient),
                        "absolute_coefficient": abs(float(coefficient)),
                        "selected_alpha": final_parameters[algorithm]["alpha"],
                    }
                )
            coefficient_rows.append(
                {
                    "outcome": outcome,
                    "feature": "Intercept",
                    "standardized_coefficient": float(model.intercept_),
                    "absolute_coefficient": abs(float(model.intercept_)),
                    "selected_alpha": final_parameters[algorithm]["alpha"],
                }
            )
        if algorithm == "gradient_boosting":
            for feature, importance in zip(x.columns, estimator.feature_importances_):
                importance_rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "feature_importance": float(importance),
                    }
                )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(importance_rows)


def make_prediction_template(output_dir: Path, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = {
        "label_number": int(data["label_number"].median()),
        "class_imbalance": float(data.loc[data["source"] == "supervised", "class_imbalance"].median()),
        "annotation_disagreement": float(data["annotation_disagreement"].median()),
        "training_sample_size": int(data.loc[data["source"] == "supervised", "training_sample_size"].median()),
        "semantic_similarity": float(data["semantic_similarity"].median()),
    }
    rows = []
    for variant in MODEL_VARIANTS:
        llm = variant.startswith("LLM_")
        rows.append(
            {
                "new_task_id": f"example_{variant}",
                "model_variant": variant,
                "label_number": medians["label_number"],
                "class_imbalance": np.nan if llm else medians["class_imbalance"],
                "annotation_disagreement": medians["annotation_disagreement"],
                "training_sample_size": np.nan if llm else medians["training_sample_size"],
                "semantic_similarity": medians["semantic_similarity"],
            }
        )
    template = pd.DataFrame(rows)
    schema = pd.DataFrame(
        [
            ["new_task_id", "string", "User-defined task/configuration identifier"],
            ["model_variant", "category", "One of: " + ", ".join(MODEL_VARIANTS)],
            ["label_number", "integer", "Number of candidate labels"],
            ["class_imbalance", "numeric", "Training-set imbalance index; blank for LLM"],
            ["annotation_disagreement", "numeric", "Task annotation-disagreement rate"],
            ["training_sample_size", "integer", "Supervised training examples; blank for LLM"],
            ["semantic_similarity", "numeric", "Average pairwise label semantic similarity"],
        ],
        columns=["column", "type", "definition"],
    )
    template.to_csv(output_dir / "prediction_input_template.csv", index=False, encoding="utf-8-sig")
    schema.to_csv(output_dir / "prediction_input_schema.csv", index=False, encoding="utf-8-sig")
    return template, schema


def configure_matplotlib() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
        }
    )
    return plt


def save_figure(figure: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        figure.savefig(output_dir / f"{stem}.{suffix}", facecolor="white")


def plot_oof_calibration(plt: Any, oof: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(12.2, 3.5), sharex=True, sharey=True)
    for ax, algorithm in zip(axes, ALGORITHMS):
        subset = oof[(oof["outcome"] == "kappa") & (oof["algorithm"] == algorithm)]
        ax.scatter(
            subset["actual"],
            subset["prediction"],
            s=11,
            alpha=0.35,
            color=ALGORITHM_COLORS[algorithm],
            edgecolor="none",
        )
        low = min(subset["actual"].min(), subset["prediction"].min()) - 0.03
        high = max(subset["actual"].max(), subset["prediction"].max()) + 0.03
        ax.plot([low, high], [low, high], color="#444444", linestyle="--", linewidth=0.8)
        row = summary[(summary["outcome"] == "kappa") & (summary["algorithm"] == algorithm)].iloc[0]
        ax.set_title(ALGORITHM_DISPLAY[algorithm])
        ax.text(
            0.04,
            0.96,
            f"RMSE={row['rmse']:.3f}\nR2={row['r_squared']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
        )
        ax.grid(color="#E2E2E2", linewidth=0.5)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("Observed kappa")
    axes[0].set_ylabel("Out-of-fold predicted kappa")
    figure.suptitle("Grouped out-of-fold model-performance calibration", y=1.02, fontsize=10)
    figure.tight_layout()
    save_figure(figure, output_dir, "figure17_performance_prediction_calibration")
    plt.close(figure)


def plot_algorithm_accuracy(plt: Any, summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ax = axes[0]
    x = np.arange(len(OUTCOMES))
    width = 0.19
    for index, algorithm in enumerate(ALGORITHMS):
        subset = summary[summary["algorithm"] == algorithm].set_index("outcome").reindex(OUTCOMES)
        offset = (index - 1.5) * width
        values = subset["rmse"].to_numpy(float)
        lower = values - subset["rmse_group_bootstrap_ci95_low"].to_numpy(float)
        upper = subset["rmse_group_bootstrap_ci95_high"].to_numpy(float) - values
        ax.bar(
            x + offset,
            values,
            width,
            color=ALGORITHM_COLORS[algorithm],
            label=ALGORITHM_DISPLAY[algorithm],
            yerr=np.vstack([lower, upper]),
            capsize=2,
            error_kw={"linewidth": 0.8},
        )
    ax.set_xticks(x, [OUTCOME_DISPLAY[outcome] for outcome in OUTCOMES])
    ax.set_ylabel("Grouped out-of-fold RMSE")
    ax.set_title("A. Prediction error by outcome")
    ax.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    kappa = summary[summary["outcome"] == "kappa"].copy()
    ax.barh(
        [ALGORITHM_DISPLAY[value] for value in kappa["algorithm"]],
        kappa["variant_balanced_rmse"],
        color=[ALGORITHM_COLORS[value] for value in kappa["algorithm"]],
    )
    for index, value in enumerate(kappa["variant_balanced_rmse"]):
        ax.text(value + 0.002, index, f"{value:.3f}", va="center", fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Variant-balanced Kappa RMSE")
    ax.set_title("B. Equal-weight error across five candidates")
    ax.grid(axis="x", color="#E2E2E2", linewidth=0.6)
    figure.tight_layout(w_pad=2.0)
    save_figure(figure, output_dir, "figure18_prediction_algorithm_comparison")
    plt.close(figure)


def plot_routing(plt: Any, routing: pd.DataFrame, output_dir: Path) -> None:
    order = ["candidate_mean", "ridge", "gradient_boosting", "mlp", "always_deberta"]
    display = {
        **ALGORITHM_DISPLAY,
        "always_deberta": "Always DeBERTa",
    }
    colors = {
        **ALGORITHM_COLORS,
        "always_deberta": "#4B4B4B",
    }
    data = routing.set_index("algorithm").reindex(order).reset_index()
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))
    metrics = [
        ("winner_accuracy", "Exact winner accuracy", "A. Correct Common24 model selection"),
        ("within_0_02_of_oracle_share", "Share within 0.02 of oracle", "B. Near-oracle selections"),
        ("mean_regret", "Mean Kappa regret", "C. Loss relative to oracle"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        bars = ax.bar(
            np.arange(len(data)),
            data[metric],
            color=[colors[value] for value in data["algorithm"]],
        )
        ax.set_xticks(
            np.arange(len(data)),
            [display[value] for value in data["algorithm"]],
            rotation=35,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.6)
        for bar, value in zip(bars, data[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.012 if metric != "mean_regret" else 0.001),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )
    axes[0].set_ylim(0, 1.08)
    axes[1].set_ylim(0, 1.08)
    figure.tight_layout(w_pad=1.8)
    save_figure(figure, output_dir, "figure19_common24_prediction_routing")
    plt.close(figure)


def plot_variant_errors(plt: Any, by_variant: pd.DataFrame, output_dir: Path) -> None:
    kappa = by_variant[by_variant["outcome"] == "kappa"].copy()
    pivot = kappa.pivot(index="model_variant", columns="algorithm", values="rmse").reindex(MODEL_VARIANTS)
    figure, ax = plt.subplots(figsize=(8.8, 4.3))
    x = np.arange(len(MODEL_VARIANTS))
    width = 0.19
    for index, algorithm in enumerate(ALGORITHMS):
        ax.bar(
            x + (index - 1.5) * width,
            pivot[algorithm].to_numpy(float),
            width,
            color=ALGORITHM_COLORS[algorithm],
            label=ALGORITHM_DISPLAY[algorithm],
        )
    ax.set_xticks(x, [VARIANT_DISPLAY[value] for value in MODEL_VARIANTS], rotation=18, ha="right")
    ax.set_ylabel("Grouped out-of-fold Kappa RMSE")
    ax.set_title("Prediction error differs across candidate model variants")
    ax.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    figure.tight_layout()
    save_figure(figure, output_dir, "figure20_prediction_error_by_model_variant")
    plt.close(figure)


def main() -> None:
    import joblib

    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    prediction_dir = output_dir / "predictions"
    diagnostic_dir = output_dir / "diagnostics"
    figure_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    for directory in [table_dir, prediction_dir, diagnostic_dir, figure_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data, support = prepare_unified_data(input_dir)
    x = build_design_matrix(data)
    all_oof: list[pd.DataFrame] = []
    all_tuning: list[pd.DataFrame] = []
    final_parameters_by_outcome: dict[str, dict[str, Any]] = {}
    final_score_rows: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        oof, tuning, final_parameters, final_scores = nested_group_predictions(
            data, x, outcome, args.seed + outcome_index * 100000
        )
        all_oof.append(oof)
        all_tuning.append(tuning)
        final_parameters_by_outcome[outcome] = final_parameters
        for algorithm, scores in final_scores.items():
            selected = scores.iloc[0].to_dict()
            final_score_rows.append(
                {
                    "outcome": outcome,
                    "algorithm": algorithm,
                    "selected_parameters": json.dumps(
                        json_safe(final_parameters[algorithm]), sort_keys=True
                    ),
                    "full_data_inner_variant_balanced_rmse": selected[
                        "mean_inner_variant_balanced_rmse"
                    ],
                    "full_data_inner_score_sd": selected["standard_deviation_inner_score"],
                }
            )
    oof = pd.concat(all_oof, ignore_index=True)
    oof["residual"] = oof["actual"] - oof["prediction"]
    oof["absolute_error"] = np.abs(oof["residual"])
    tuning = pd.concat(all_tuning, ignore_index=True)
    final_selection = pd.DataFrame(final_score_rows)

    summary, by_variant = summarize_predictions(oof, args.seed)
    by_family = summarize_predictions_by_family(oof)
    algorithm_comparison = compare_algorithm_errors(oof, args.seed + 500000)
    quantiles = residual_quantiles(oof)
    routing_detail, routing_summary = common24_routing(oof)
    routing_comparison = compare_routing_regret(routing_detail, args.seed + 600000)

    ridge_coefficients: list[pd.DataFrame] = []
    gbt_importance: list[pd.DataFrame] = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        coefficients, importance = final_model_artifacts(
            data,
            x,
            outcome,
            final_parameters_by_outcome[outcome],
            quantiles,
            support,
            models_dir,
            args.seed + outcome_index * 100000,
        )
        ridge_coefficients.append(coefficients)
        gbt_importance.append(importance)
    ridge_coefficients_frame = pd.concat(ridge_coefficients, ignore_index=True)
    gbt_importance_frame = pd.concat(gbt_importance, ignore_index=True)

    fold_map = oof[oof["algorithm"] == "candidate_mean"][
        ["outcome", "record_id", "combination_id", "outer_fold"]
    ].drop_duplicates()
    if fold_map.groupby("combination_id")["outer_fold"].nunique().max() != 1:
        raise ValueError("A label combination appears in multiple outer folds")
    if oof.groupby(["outcome", "algorithm"])["record_id"].nunique().ne(len(data)).any():
        raise ValueError("OOF predictions do not cover every record exactly once")

    support.to_csv(table_dir / "prediction_design_support.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "prediction_model_cv_summary.csv", index=False, encoding="utf-8-sig")
    by_variant.to_csv(table_dir / "prediction_model_cv_by_variant.csv", index=False, encoding="utf-8-sig")
    by_family.to_csv(table_dir / "prediction_model_cv_by_family.csv", index=False, encoding="utf-8-sig")
    algorithm_comparison.to_csv(
        table_dir / "prediction_algorithm_pairwise_comparison.csv", index=False, encoding="utf-8-sig"
    )
    final_selection.to_csv(table_dir / "prediction_final_hyperparameters.csv", index=False, encoding="utf-8-sig")
    routing_summary.to_csv(table_dir / "common24_prediction_routing_summary.csv", index=False, encoding="utf-8-sig")
    routing_comparison.to_csv(
        table_dir / "common24_prediction_routing_pairwise_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ridge_coefficients_frame.to_csv(table_dir / "prediction_ridge_coefficients.csv", index=False, encoding="utf-8-sig")
    gbt_importance_frame.to_csv(table_dir / "prediction_gbt_feature_importance.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(prediction_dir / "prediction_oof_records.csv", index=False, encoding="utf-8-sig")
    routing_detail.to_csv(prediction_dir / "common24_prediction_routing_detail.csv", index=False, encoding="utf-8-sig")
    quantiles.to_csv(prediction_dir / "prediction_absolute_error_quantiles.csv", index=False, encoding="utf-8-sig")
    tuning.to_csv(diagnostic_dir / "prediction_nested_tuning_log.csv", index=False, encoding="utf-8-sig")
    fold_map.to_csv(diagnostic_dir / "prediction_outer_fold_assignment.csv", index=False, encoding="utf-8-sig")
    data.to_csv(diagnostic_dir / "prediction_unified_analysis_data.csv", index=False, encoding="utf-8-sig")
    make_prediction_template(prediction_dir, data)

    kappa_summary = summary[summary["outcome"] == "kappa"].set_index("algorithm")
    ridge_rmse = float(kappa_summary.loc["ridge", "variant_balanced_rmse"])
    gbt_rmse = float(kappa_summary.loc["gradient_boosting", "variant_balanced_rmse"])
    comparison_row = algorithm_comparison[
        (algorithm_comparison["outcome"] == "kappa")
        & (algorithm_comparison["contrast"] == "gbt_minus_ridge")
    ].iloc[0]
    if (
        gbt_rmse <= ridge_rmse * 0.95
        and comparison_row["p_value_holm_within_outcome"] < 0.05
    ):
        recommended_algorithm = "gradient_boosting"
        recommendation_reason = (
            "Gradient boosting reduced variant-balanced Kappa RMSE by at least 5% and the grouped "
            "error difference remained significant after Holm correction."
        )
    else:
        recommended_algorithm = "ridge"
        recommendation_reason = (
            "Gradient boosting did not provide a Holm-robust improvement of at least 5%; ridge is retained "
            "for interpretability and transportability."
        )

    figure_status = "skipped"
    if not args.skip_figures:
        plt = configure_matplotlib()
        plot_oof_calibration(plt, oof, summary, figure_dir)
        plot_algorithm_accuracy(plt, summary, figure_dir)
        plot_routing(plt, routing_summary, figure_dir)
        plot_variant_errors(plt, by_variant, figure_dir)
        figure_status = "rendered"

    routing_index = routing_summary.set_index("algorithm")
    report = {
        "generated_at_utc": utc_now(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "prediction_target": (
            "Predict Kappa, Macro-F1, and MCC from task-complexity dimensions and candidate model configuration."
        ),
        "analysis_unit": "model-task configuration; LLM prompt cells are averaged across three calls",
        "validation_design": (
            f"Nested grouped cross-validation with {OUTER_FOLDS} outer folds and {INNER_FOLDS} inner folds; "
            "all records from a label combination remain in one fold."
        ),
        "record_count": len(data),
        "combination_count": data["combination_id"].nunique(),
        "model_variant_count": data["model_variant"].nunique(),
        "common24_routing_combination_count": routing_detail["combination_id"].nunique(),
        "kappa_cv_metrics": summary[summary["outcome"] == "kappa"][
            [
                "algorithm",
                "rmse",
                "mae",
                "r_squared",
                "group_balanced_rmse",
                "variant_balanced_rmse",
                "calibration_intercept",
                "calibration_slope",
            ]
        ].to_dict(orient="records"),
        "kappa_cv_metrics_by_family": by_family[by_family["outcome"] == "kappa"][
            [
                "algorithm",
                "prediction_family",
                "record_count",
                "rmse",
                "mae",
                "r_squared",
                "group_balanced_rmse",
            ]
        ].to_dict(orient="records"),
        "common24_routing_metrics": routing_summary.to_dict(orient="records"),
        "common24_routing_regret_comparisons": routing_comparison.to_dict(orient="records"),
        "recommended_primary_predictor": recommended_algorithm,
        "recommendation_reason": recommendation_reason,
        "mlp_role": (
            "Supplementary benchmark only. It tests whether a small neural network improves grouped out-of-fold "
            "prediction enough to justify its lower interpretability at the current effective sample size."
        ),
        "saved_model_bundles": sorted(path.name for path in models_dir.glob("*__*.joblib")),
        "figure_status": figure_status,
        "figure_count": 4 if figure_status == "rendered" else 0,
        "output_checks": {
            "expected_600_records": len(data) == 600,
            "expected_40_combinations": data["combination_id"].nunique() == 40,
            "complete_oof_predictions": bool(
                oof.groupby(["outcome", "algorithm"])["record_id"].nunique().eq(len(data)).all()
            ),
            "no_group_leakage_across_outer_folds": bool(
                fold_map.groupby(["outcome", "combination_id"])["outer_fold"].nunique().eq(1).all()
            ),
            "finite_cv_metrics": bool(
                np.isfinite(summary[["rmse", "mae", "r_squared", "group_balanced_rmse"]]).all().all()
            ),
            "finite_model_predictions": bool(np.isfinite(oof["prediction"]).all()),
            "common24_complete_candidate_sets": bool(
                routing_detail.groupby(["algorithm", "combination_id"]).size().eq(1).all()
            ),
            "model_bundle_count_is_nine": len(list(models_dir.glob("*__*.joblib"))) == 9,
        },
        "interpretation_boundary": (
            "Prediction accuracy estimates transport to unseen label combinations within the observed complexity "
            "support. The data contain 40 independent label combinations from one contract-coding domain; results "
            "do not establish universal performance for other engineering-text domains."
        ),
    }
    write_json(output_dir / "step6_performance_prediction_report.json", report)
    joblib.dump(
        {
            "recommended_algorithm": recommended_algorithm,
            "recommendation_reason": recommendation_reason,
            "report_path": str(output_dir / "step6_performance_prediction_report.json"),
        },
        models_dir / "recommended_predictor.joblib",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "records": len(data),
                "combinations": data["combination_id"].nunique(),
                "recommended_primary_predictor": recommended_algorithm,
                "kappa_variant_balanced_rmse": {
                    algorithm: float(kappa_summary.loc[algorithm, "variant_balanced_rmse"])
                    for algorithm in ALGORITHMS
                },
                "common24_winner_accuracy": {
                    algorithm: float(routing_index.loc[algorithm, "winner_accuracy"])
                    for algorithm in routing_index.index
                },
                "output_checks": report["output_checks"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
