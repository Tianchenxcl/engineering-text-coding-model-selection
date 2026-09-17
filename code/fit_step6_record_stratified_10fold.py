from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fit_step6_performance_prediction import (
    ALGORITHMS,
    ALGORITHM_DISPLAY,
    MODEL_VARIANTS,
    OUTCOMES,
    algorithm_grids,
    basic_prediction_metrics,
    build_design_matrix,
    candidate_mean_predict,
    fit_estimator,
    holm_adjust,
    make_estimator,
    prepare_unified_data,
    variant_balanced_rmse,
    variant_balanced_weights,
)


OUTER_FOLDS = 10
INNER_FOLDS = 10


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
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record-level stratified 10-fold sensitivity analysis for Step 6"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def tune_with_record_stratification(
    frame: pd.DataFrame,
    x: pd.DataFrame,
    y: np.ndarray,
    algorithm: str,
    seed: int,
    n_splits: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    from sklearn.model_selection import StratifiedKFold

    strata = frame["model_variant"].astype(str).to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    for parameter_index, parameters in enumerate(algorithm_grids()[algorithm]):
        fold_scores = []
        for inner_fold, (train_index, validation_index) in enumerate(
            splitter.split(x, strata), start=1
        ):
            estimator = make_estimator(
                algorithm,
                parameters,
                seed + parameter_index * 101 + inner_fold,
            )
            train_frame = frame.iloc[train_index]
            weights = variant_balanced_weights(train_frame)
            fit_estimator(
                estimator,
                algorithm,
                x.iloc[train_index],
                y[train_index],
                weights,
            )
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


def nested_record_level_predictions(
    data: pd.DataFrame,
    x: pd.DataFrame,
    outcome: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import StratifiedKFold

    y = data[outcome].to_numpy(float)
    strata = data["model_variant"].astype(str).to_numpy()
    splitter = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    prediction_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for outer_fold, (train_index, test_index) in enumerate(
        splitter.split(x, strata), start=1
    ):
        train_frame = data.iloc[train_index].reset_index(drop=True)
        test_frame = data.iloc[test_index].reset_index(drop=True)
        x_train = x.iloc[train_index].reset_index(drop=True)
        x_test = x.iloc[test_index].reset_index(drop=True)
        y_train = y[train_index]
        y_test = y[test_index]
        train_combinations = set(train_frame["combination_id"].astype(str))
        test_combinations = set(test_frame["combination_id"].astype(str))
        overlapping = train_combinations & test_combinations
        overlap_record_share = test_frame["combination_id"].astype(str).isin(overlapping).mean()
        fold_rows.append(
            {
                "outcome": outcome,
                "outer_fold": outer_fold,
                "training_records": len(train_index),
                "test_records": len(test_index),
                "training_combinations": len(train_combinations),
                "test_combinations": len(test_combinations),
                "overlapping_combinations": len(overlapping),
                "test_record_share_with_combination_seen_in_training": overlap_record_share,
            }
        )

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
                    "actual": y_test[row_index],
                    "prediction": prediction,
                }
            )

        for algorithm in ["ridge", "gradient_boosting", "mlp"]:
            best_parameters, tuning_scores = tune_with_record_stratification(
                train_frame,
                x_train,
                y_train,
                algorithm,
                seed + outer_fold * 1000,
                INNER_FOLDS,
            )
            tuning_scores["outcome"] = outcome
            tuning_scores["outer_fold"] = outer_fold
            tuning_scores["selected"] = tuning_scores["parameters"].eq(
                json.dumps(json_safe(best_parameters), sort_keys=True)
            )
            tuning_rows.extend(tuning_scores.to_dict(orient="records"))
            estimator = make_estimator(
                algorithm,
                best_parameters,
                seed + outer_fold * 1000,
            )
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
                        "actual": y_test[row_index],
                        "prediction": value,
                    }
                )
        print(
            f"[record-10fold] outcome={outcome} outer_fold={outer_fold}/{OUTER_FOLDS} "
            f"train={len(train_index)} test={len(test_index)} overlap_share={overlap_record_share:.3f}",
            flush=True,
        )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(fold_rows),
    )


def summarize_predictions(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        for algorithm in ALGORITHMS:
            subset = oof[
                (oof["outcome"] == outcome) & (oof["algorithm"] == algorithm)
            ].copy()
            rows.append(
                {
                    "outcome": outcome,
                    "algorithm": algorithm,
                    "algorithm_display": ALGORITHM_DISPLAY[algorithm],
                    "record_count": len(subset),
                    **basic_prediction_metrics(subset),
                }
            )
    return pd.DataFrame(rows)


def compare_algorithms_by_outer_fold(oof: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    contrasts = [
        ("ridge_minus_mean", "ridge", "candidate_mean"),
        ("gbt_minus_ridge", "gradient_boosting", "ridge"),
        ("mlp_minus_ridge", "mlp", "ridge"),
    ]
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        subset = oof[oof["outcome"] == outcome].copy()
        subset["squared_error"] = (subset["actual"] - subset["prediction"]) ** 2
        fold_rmse = (
            subset.groupby(["outer_fold", "algorithm"])["squared_error"]
            .mean()
            .pow(0.5)
            .unstack("algorithm")
        )
        for contrast, algorithm_a, algorithm_b in contrasts:
            difference = (fold_rmse[algorithm_a] - fold_rmse[algorithm_b]).to_numpy(float)
            test = wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
            rows.append(
                {
                    "outcome": outcome,
                    "contrast": contrast,
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "interpretation": "Negative values favor algorithm_a",
                    "outer_fold_count": len(difference),
                    "mean_fold_rmse_difference": difference.mean(),
                    "minimum_fold_difference": difference.min(),
                    "maximum_fold_difference": difference.max(),
                    "wilcoxon_statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                }
            )
    output = pd.DataFrame(rows)
    output["p_value_holm_within_outcome"] = output.groupby(
        "outcome", group_keys=False
    )["p_value"].apply(holm_adjust)
    return output


def select_and_fit_final_models(
    data: pd.DataFrame,
    x: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for outcome_index, outcome in enumerate(OUTCOMES):
        y = data[outcome].to_numpy(float)
        for algorithm_index, algorithm in enumerate(
            ["ridge", "gradient_boosting", "mlp"]
        ):
            parameters, scores = tune_with_record_stratification(
                data,
                x,
                y,
                algorithm,
                seed + outcome_index * 100000 + algorithm_index * 1000,
                OUTER_FOLDS,
            )
            estimator = make_estimator(
                algorithm,
                parameters,
                seed + outcome_index * 100000 + algorithm_index * 1000,
            )
            weights = variant_balanced_weights(data)
            fit_estimator(estimator, algorithm, x, y, weights)
            bundle = {
                "created_at_utc": utc_now(),
                "analysis": "record_level_stratified_10fold_sensitivity",
                "outcome": outcome,
                "algorithm": algorithm,
                "parameters": parameters,
                "feature_columns": list(x.columns),
                "model_variants": MODEL_VARIANTS,
                "estimator": estimator,
                "warning": (
                    "This model was selected with record-level stratified CV. Its validation estimates "
                    "measure interpolation within largely observed label combinations."
                ),
            }
            joblib.dump(bundle, models_dir / f"{outcome}__{algorithm}.joblib")
            rows.append(
                {
                    "outcome": outcome,
                    "algorithm": algorithm,
                    "selected_parameters": json.dumps(
                        json_safe(parameters), sort_keys=True
                    ),
                    "mean_10fold_variant_balanced_rmse": scores.iloc[0][
                        "mean_inner_variant_balanced_rmse"
                    ],
                    "standard_deviation_10fold_score": scores.iloc[0][
                        "standard_deviation_inner_score"
                    ],
                }
            )
    return pd.DataFrame(rows)


def grouped_comparison_if_available(
    output_dir: Path,
    record_summary: pd.DataFrame,
) -> pd.DataFrame:
    candidates = [
        output_dir.parent / "step6_performance_prediction" / "tables" / "prediction_model_cv_summary.csv",
        Path("/hy-tmp/hy_tmp_experiment_package/outputs/statistical_analysis/step6_performance_prediction/tables/prediction_model_cv_summary.csv"),
    ]
    grouped_path = next((path for path in candidates if path.exists()), None)
    if grouped_path is None:
        return pd.DataFrame()
    grouped = pd.read_csv(grouped_path)
    grouped = grouped[["outcome", "algorithm", "rmse", "r_squared"]].rename(
        columns={"rmse": "grouped_rmse", "r_squared": "grouped_r_squared"}
    )
    record = record_summary[["outcome", "algorithm", "rmse", "r_squared"]].rename(
        columns={"rmse": "record_stratified_rmse", "r_squared": "record_stratified_r_squared"}
    )
    comparison = record.merge(grouped, on=["outcome", "algorithm"], how="inner")
    comparison["rmse_reduction_from_record_split"] = (
        comparison["grouped_rmse"] - comparison["record_stratified_rmse"]
    )
    comparison["r_squared_increase_from_record_split"] = (
        comparison["record_stratified_r_squared"] - comparison["grouped_r_squared"]
    )
    return comparison


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    prediction_dir = output_dir / "predictions"
    diagnostic_dir = output_dir / "diagnostics"
    for directory in [table_dir, prediction_dir, diagnostic_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data, support = prepare_unified_data(input_dir)
    x = build_design_matrix(data)
    all_predictions: list[pd.DataFrame] = []
    all_tuning: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        predictions, tuning, folds = nested_record_level_predictions(
            data,
            x,
            outcome,
            args.seed + outcome_index * 100000,
        )
        all_predictions.append(predictions)
        all_tuning.append(tuning)
        all_folds.append(folds)
    oof = pd.concat(all_predictions, ignore_index=True)
    oof["residual"] = oof["actual"] - oof["prediction"]
    oof["absolute_error"] = np.abs(oof["residual"])
    tuning = pd.concat(all_tuning, ignore_index=True)
    folds = pd.concat(all_folds, ignore_index=True)
    summary = summarize_predictions(oof)
    comparisons = compare_algorithms_by_outer_fold(oof)
    final_models = select_and_fit_final_models(
        data,
        x,
        output_dir,
        args.seed + 900000,
    )
    grouped_comparison = grouped_comparison_if_available(output_dir, summary)

    support.to_csv(table_dir / "record_10fold_design_support.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "record_10fold_cv_summary.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(
        table_dir / "record_10fold_algorithm_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_models.to_csv(
        table_dir / "record_10fold_final_hyperparameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not grouped_comparison.empty:
        grouped_comparison.to_csv(
            table_dir / "grouped_vs_record_split_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )
    oof.to_csv(
        prediction_dir / "record_10fold_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tuning.to_csv(
        diagnostic_dir / "record_10fold_nested_tuning_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    folds.to_csv(
        diagnostic_dir / "record_10fold_fold_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    kappa = summary[summary["outcome"] == "kappa"].set_index("algorithm")
    report = {
        "generated_at_utc": utc_now(),
        "analysis_role": "record-level stratified 10-fold sensitivity analysis",
        "primary_analysis_replacement": False,
        "record_count": len(data),
        "label_combination_count": data["combination_id"].nunique(),
        "outer_split": (
            "StratifiedKFold(10), stratified by model_variant: 540 training records and 60 test records per fold."
        ),
        "inner_split": (
            "StratifiedKFold(10) inside each outer training set for hyperparameter selection."
        ),
        "refit_rule": (
            "After inner selection, refit on all 540 outer-training records and evaluate once on the 60 outer-test records. "
            "After evaluation, select parameters with 10-fold CV on all 600 records and fit deployment models on all records."
        ),
        "stratification_variable": "model_variant",
        "label_number_not_used_for_stratification_reason": (
            "Some LLM model-variant by 11-label strata contain only one record and cannot support 10 folds."
        ),
        "kappa_metrics": summary[summary["outcome"] == "kappa"][
            ["algorithm", "rmse", "mae", "r_squared", "calibration_slope"]
        ].to_dict(orient="records"),
        "mean_test_record_share_with_combination_seen_in_training": float(
            folds["test_record_share_with_combination_seen_in_training"].mean()
        ),
        "grouped_vs_record_split": grouped_comparison.to_dict(orient="records"),
        "output_checks": {
            "all_outer_folds_have_540_train_60_test": bool(
                folds["training_records"].eq(540).all()
                and folds["test_records"].eq(60).all()
            ),
            "all_records_have_one_oof_prediction_per_algorithm_outcome": bool(
                oof.groupby(["outcome", "algorithm"])["record_id"].nunique().eq(600).all()
            ),
            "all_predictions_finite": bool(np.isfinite(oof["prediction"]).all()),
            "nine_final_model_bundles": len(list((output_dir / "models").glob("*__*.joblib"))) == 9,
        },
        "interpretation_warning": (
            "Because records from the same label combination occur in both training and test folds, these estimates "
            "measure interpolation to new configurations of largely known label combinations. They should not be "
            "reported as generalization to a new coding task or new label system."
        ),
    }
    write_json(output_dir / "record_10fold_sensitivity_report.json", report)
    print(
        json.dumps(
            {
                "status": "ok",
                "analysis_role": report["analysis_role"],
                "kappa_rmse": {
                    algorithm: float(kappa.loc[algorithm, "rmse"])
                    for algorithm in ALGORITHMS
                },
                "mean_combination_overlap_share": report[
                    "mean_test_record_share_with_combination_seen_in_training"
                ],
                "output_checks": report["output_checks"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
