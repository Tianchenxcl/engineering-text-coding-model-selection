from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from build_step7_diagnostic_guidance import (
    LABEL_LEVELS,
    MODEL_DISPLAY,
    build_thresholds,
    predict_bundle,
    scenario_predictions,
)
from fit_step6_performance_prediction import (
    MODEL_VARIANTS,
    OUTCOMES,
    basic_prediction_metrics,
    prepare_unified_data,
)


ALGORITHMS = ["candidate_mean", "ridge", "gradient_boosting", "mlp"]
PREDICTIVE_ALGORITHMS = ["ridge", "gradient_boosting"]
THRESHOLD_PERCENTILES = [0.70, 0.75, 0.80]
NEAR_TIE_THRESHOLDS = [0.01, 0.02, 0.03, 0.05]
BOOTSTRAP_REPS = 2000
SEED = 20260717


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 8 robustness checks for task-complexity prediction and diagnosis"
    )
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--step6-dir", type=Path, required=True)
    parser.add_argument("--step7-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize_oof_subsets(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subsets = {
        "full": oof,
        "exclude_11_labels": oof[oof["label_number"] != 11],
    }
    for subset_name, frame in subsets.items():
        for outcome in OUTCOMES:
            for algorithm in ALGORITHMS:
                values = frame[
                    frame["outcome"].eq(outcome)
                    & frame["algorithm"].eq(algorithm)
                ].copy()
                metrics = basic_prediction_metrics(values)
                rows.append(
                    {
                        "evaluation_subset": subset_name,
                        "outcome": outcome,
                        "algorithm": algorithm,
                        "records": len(values),
                        "label_combinations": values["combination_id"].nunique(),
                        **metrics,
                    }
                )
    output = pd.DataFrame(rows)
    output["rmse_rank_within_subset_outcome"] = output.groupby(
        ["evaluation_subset", "outcome"]
    )["rmse"].rank(method="min")
    return output


def cluster_bootstrap_metrics(
    oof: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        for algorithm in ALGORITHMS:
            subset = oof[
                oof["outcome"].eq(outcome)
                & oof["algorithm"].eq(algorithm)
            ].copy()
            grouped = {
                key: value.reset_index(drop=True)
                for key, value in subset.groupby("combination_id")
            }
            groups = np.asarray(sorted(grouped), dtype=object)
            rmse_values = np.empty(reps, dtype=float)
            mae_values = np.empty(reps, dtype=float)
            r2_values = np.empty(reps, dtype=float)
            for replicate in range(reps):
                sampled = rng.choice(groups, size=len(groups), replace=True)
                frame = pd.concat([grouped[key] for key in sampled], ignore_index=True)
                metrics = basic_prediction_metrics(frame)
                rmse_values[replicate] = metrics["rmse"]
                mae_values[replicate] = metrics["mae"]
                r2_values[replicate] = metrics["r_squared"]
            rows.append(
                {
                    "outcome": outcome,
                    "algorithm": algorithm,
                    "bootstrap_unit": "label_combination",
                    "bootstrap_reps": reps,
                    "rmse_ci95_low": np.quantile(rmse_values, 0.025),
                    "rmse_ci95_high": np.quantile(rmse_values, 0.975),
                    "mae_ci95_low": np.quantile(mae_values, 0.025),
                    "mae_ci95_high": np.quantile(mae_values, 0.975),
                    "r_squared_ci95_low": np.quantile(r2_values, 0.025),
                    "r_squared_ci95_high": np.quantile(r2_values, 0.975),
                    "share_positive_r_squared": float((r2_values > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def validation_design_robustness(step6_dir: Path) -> pd.DataFrame:
    comparison = read_csv(
        step6_dir / "tables" / "grouped_vs_record_split_comparison.csv"
    )
    comparison["record_to_grouped_rmse_ratio"] = (
        comparison["record_stratified_rmse"] / comparison["grouped_rmse"]
    )
    comparison["positive_r_squared_under_both_splits"] = (
        comparison["record_stratified_r_squared"].gt(0)
        & comparison["grouped_r_squared"].gt(0)
    )
    comparison["same_basic_predictive_conclusion"] = comparison[
        "positive_r_squared_under_both_splits"
    ]
    return comparison


def scenario_metric_algorithm_robustness(
    data: pd.DataFrame,
    step6_dir: Path,
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    for outcome in OUTCOMES:
        for algorithm in PREDICTIVE_ALGORITHMS:
            bundle = joblib.load(
                step6_dir / "models" / f"{outcome}__{algorithm}.joblib"
            )
            _, summary = scenario_predictions(data, bundle, thresholds)
            summary = summary.rename(
                columns={
                    "mean_predicted_kappa_difference": "mean_predicted_metric_difference",
                    "median_predicted_kappa_difference": "median_predicted_metric_difference",
                    "q10_predicted_kappa_difference": "q10_predicted_metric_difference",
                    "q90_predicted_kappa_difference": "q90_predicted_metric_difference",
                }
            )
            summary["outcome"] = outcome
            summary["prediction_algorithm"] = algorithm
            summaries.append(summary)
    all_summary = pd.concat(summaries, ignore_index=True)

    metric_rows = []
    for algorithm in PREDICTIVE_ALGORITHMS:
        pivot = all_summary[
            all_summary["prediction_algorithm"].eq(algorithm)
        ].pivot_table(
            index=["scenario", "model_variant"],
            columns="outcome",
            values="mean_predicted_metric_difference",
        )
        for index, row in pivot.dropna().iterrows():
            signs = np.sign(row[OUTCOMES].to_numpy(float))
            metric_rows.append(
                {
                    "prediction_algorithm": algorithm,
                    "scenario": index[0],
                    "model_variant": index[1],
                    "kappa_difference": row["kappa"],
                    "macro_f1_difference": row["macro_f1"],
                    "mcc_difference": row["mcc"],
                    "same_direction_all_three_metrics": bool(
                        np.all(signs > 0) or np.all(signs < 0) or np.all(signs == 0)
                    ),
                }
            )
    metric_consistency = pd.DataFrame(metric_rows)

    algorithm_rows = []
    pivot = all_summary.pivot_table(
        index=["outcome", "scenario", "model_variant"],
        columns="prediction_algorithm",
        values="mean_predicted_metric_difference",
    ).dropna()
    for index, row in pivot.iterrows():
        algorithm_rows.append(
            {
                "outcome": index[0],
                "scenario": index[1],
                "model_variant": index[2],
                "ridge_difference": row["ridge"],
                "gradient_boosting_difference": row["gradient_boosting"],
                "same_direction_across_algorithms": bool(
                    np.sign(row["ridge"]) == np.sign(row["gradient_boosting"])
                ),
                "absolute_difference_between_algorithms": abs(
                    row["ridge"] - row["gradient_boosting"]
                ),
            }
        )
    algorithm_consistency = pd.DataFrame(algorithm_rows)
    return all_summary, metric_consistency, algorithm_consistency


def threshold_values(
    data: pd.DataFrame,
    percentile: float,
) -> dict[str, float]:
    combinations = data.drop_duplicates("combination_id")
    supervised = data[data["source"].eq("supervised")]
    return {
        "percentile": percentile,
        "high_label_number": 8.0,
        "high_class_imbalance": float(
            supervised["class_imbalance"].quantile(percentile)
        ),
        "low_training_sample_size": float(
            supervised["training_sample_size"].quantile(1.0 - percentile)
        ),
        "high_annotation_disagreement": float(
            combinations["annotation_disagreement"].quantile(percentile)
        ),
        "high_semantic_similarity": float(
            combinations["semantic_similarity"].quantile(percentile)
        ),
    }


def risk_flags(frame: pd.DataFrame, values: dict[str, float]) -> pd.DataFrame:
    output = frame[["profile_id"]].copy()
    output["label_space_flag"] = frame["label_number"].ge(
        values["high_label_number"]
    )
    output["imbalance_flag"] = frame["class_imbalance"].ge(
        values["high_class_imbalance"]
    )
    output["small_sample_flag"] = frame["training_sample_size"].le(
        values["low_training_sample_size"]
    )
    output["disagreement_flag"] = frame["annotation_disagreement"].ge(
        values["high_annotation_disagreement"]
    )
    output["similarity_flag"] = frame["semantic_similarity"].ge(
        values["high_semantic_similarity"]
    )
    flag_columns = [column for column in output if column.endswith("_flag")]
    output["risk_flag_count"] = output[flag_columns].sum(axis=1)
    output["any_risk_flag"] = output["risk_flag_count"].gt(0)
    output["flag_signature"] = output[flag_columns].astype(int).astype(str).agg("".join, axis=1)
    return output


def threshold_robustness(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = (
        data[data["source"].eq("supervised")]
        .drop_duplicates(
            [
                "combination_id",
                "label_number",
                "class_imbalance",
                "training_sample_size",
                "annotation_disagreement",
                "semantic_similarity",
            ]
        )
        .reset_index(drop=True)
    )
    profiles["profile_id"] = "profile_" + profiles.index.astype(str)
    threshold_rows = []
    flag_sets: dict[float, pd.DataFrame] = {}
    for percentile in THRESHOLD_PERCENTILES:
        values = threshold_values(data, percentile)
        flags = risk_flags(profiles, values)
        flag_sets[percentile] = flags
        flag_columns = [column for column in flags if column.endswith("_flag")]
        row = values.copy()
        row["profiles"] = len(flags)
        row["profiles_with_any_risk"] = int(flags["any_risk_flag"].sum())
        row["share_with_any_risk"] = float(flags["any_risk_flag"].mean())
        for column in flag_columns:
            row[f"{column}_count"] = int(flags[column].sum())
        threshold_rows.append(row)

    baseline = flag_sets[0.75].set_index("profile_id")
    agreement_rows = []
    for percentile, flags in flag_sets.items():
        current = flags.set_index("profile_id")
        agreement_rows.append(
            {
                "percentile": percentile,
                "reference_percentile": 0.75,
                "exact_flag_signature_agreement": float(
                    current["flag_signature"].eq(baseline["flag_signature"]).mean()
                ),
                "any_risk_decision_agreement": float(
                    current["any_risk_flag"].eq(baseline["any_risk_flag"]).mean()
                ),
                "mean_absolute_flag_count_difference": float(
                    (current["risk_flag_count"] - baseline["risk_flag_count"])
                    .abs()
                    .mean()
                ),
            }
        )
    return pd.DataFrame(threshold_rows), pd.DataFrame(agreement_rows)


def candidate_predictions_for_profiles(
    data: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    profiles = (
        data[data["source"].eq("supervised")]
        .drop_duplicates(
            [
                "combination_id",
                "label_number",
                "class_imbalance",
                "training_sample_size",
                "annotation_disagreement",
                "semantic_similarity",
            ]
        )
        .reset_index(drop=True)
    )
    rows = []
    for profile_index, profile in profiles.iterrows():
        candidates = []
        for variant in MODEL_VARIANTS:
            candidates.append(
                {
                    "profile_id": f"profile_{profile_index}",
                    "combination_id": profile["combination_id"],
                    "model_variant": variant,
                    "label_number": profile["label_number"],
                    "class_imbalance": (
                        profile["class_imbalance"]
                        if variant in ["SVM", "DeBERTa"]
                        else np.nan
                    ),
                    "annotation_disagreement": profile[
                        "annotation_disagreement"
                    ],
                    "training_sample_size": (
                        profile["training_sample_size"]
                        if variant in ["SVM", "DeBERTa"]
                        else np.nan
                    ),
                    "semantic_similarity": profile["semantic_similarity"],
                }
            )
        candidate_frame = pd.DataFrame(candidates)
        candidate_frame["prediction"] = predict_bundle(bundle, candidate_frame)
        candidate_frame = candidate_frame.sort_values("prediction", ascending=False)
        candidate_frame["rank"] = np.arange(1, len(candidate_frame) + 1)
        rows.append(candidate_frame)
    return pd.concat(rows, ignore_index=True)


def near_tie_robustness(
    data: pd.DataFrame,
    step6_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    for outcome in OUTCOMES:
        bundle = joblib.load(
            step6_dir / "models" / f"{outcome}__gradient_boosting.joblib"
        )
        frame = candidate_predictions_for_profiles(data, bundle)
        frame["outcome"] = outcome
        predictions.append(frame)
    predictions = pd.concat(predictions, ignore_index=True)

    margins = []
    for (outcome, profile_id), frame in predictions.groupby(
        ["outcome", "profile_id"]
    ):
        ordered = frame.sort_values("rank")
        margins.append(
            {
                "outcome": outcome,
                "profile_id": profile_id,
                "combination_id": ordered.iloc[0]["combination_id"],
                "winner": ordered.iloc[0]["model_variant"],
                "runner_up": ordered.iloc[1]["model_variant"],
                "winner_prediction": ordered.iloc[0]["prediction"],
                "runner_up_prediction": ordered.iloc[1]["prediction"],
                "predicted_margin": ordered.iloc[0]["prediction"]
                - ordered.iloc[1]["prediction"],
            }
        )
    margins = pd.DataFrame(margins)
    rows = []
    for outcome in OUTCOMES:
        subset = margins[margins["outcome"].eq(outcome)]
        for threshold in NEAR_TIE_THRESHOLDS:
            near_tie = subset["predicted_margin"].lt(threshold)
            rows.append(
                {
                    "outcome": outcome,
                    "near_tie_threshold": threshold,
                    "profiles": len(subset),
                    "near_tie_profiles": int(near_tie.sum()),
                    "near_tie_share": float(near_tie.mean()),
                    "eligible_for_bounded_routing": int((~near_tie).sum()),
                    "routing_share": float((~near_tie).mean()),
                }
            )
    return pd.DataFrame(rows), margins


def model_winner_metric_consistency(margins: pd.DataFrame) -> pd.DataFrame:
    pivot = margins.pivot(
        index="profile_id", columns="outcome", values="winner"
    ).reset_index()
    pivot["same_predicted_winner_all_metrics"] = (
        pivot[OUTCOMES].nunique(axis=1).eq(1)
    )
    return pivot


def common24_metric_consistency(analysis_root: Path) -> pd.DataFrame:
    winners = read_csv(
        analysis_root
        / "step5_common24_cross_model"
        / "predictions"
        / "common24_task_winners.csv"
    )
    pivot = winners.pivot(
        index="combination_id", columns="outcome", values="winner_short"
    ).reset_index()
    pivot["same_observed_winner_all_metrics"] = pivot[OUTCOMES].nunique(axis=1).eq(1)
    return pivot


def contribution_robustness_summary(
    oof_summary: pd.DataFrame,
    validation: pd.DataFrame,
    metric_consistency: pd.DataFrame,
    algorithm_consistency: pd.DataFrame,
    threshold_agreement: pd.DataFrame,
) -> pd.DataFrame:
    full_gbt = oof_summary[
        oof_summary["evaluation_subset"].eq("full")
        & oof_summary["outcome"].eq("kappa")
        & oof_summary["algorithm"].eq("gradient_boosting")
    ].iloc[0]
    no11_gbt = oof_summary[
        oof_summary["evaluation_subset"].eq("exclude_11_labels")
        & oof_summary["outcome"].eq("kappa")
        & oof_summary["algorithm"].eq("gradient_boosting")
    ].iloc[0]
    grouped_gbt = validation[
        validation["outcome"].eq("kappa")
        & validation["algorithm"].eq("gradient_boosting")
    ].iloc[0]
    grouped_ridge = validation[
        validation["outcome"].eq("kappa")
        & validation["algorithm"].eq("ridge")
    ].iloc[0]
    threshold_min = threshold_agreement[
        threshold_agreement["percentile"].isin([0.70, 0.80])
    ]["any_risk_decision_agreement"].min()
    metric_gbt = metric_consistency[
        metric_consistency["prediction_algorithm"].eq("gradient_boosting")
    ]["same_direction_all_three_metrics"].mean()
    metric_ridge = metric_consistency[
        metric_consistency["prediction_algorithm"].eq("ridge")
    ]["same_direction_all_three_metrics"].mean()
    algorithm_min = algorithm_consistency.groupby("outcome")[
        "same_direction_across_algorithms"
    ].mean().min()
    return pd.DataFrame(
        [
            {
                "核心贡献": "贡献一：任务复杂度框架",
                "稳健性证据": (
                    f"排除11标签后Kappa预测R²由{full_gbt['r_squared']:.3f}变为"
                    f"{no11_gbt['r_squared']:.3f}；70%/80%阈值下总体风险判断最低一致率为"
                    f"{threshold_min:.1%}"
                ),
                "稳健性判断": "总体复杂度风险识别稳健，但具体瓶颈标签对阈值中度敏感",
                "论文主张边界": "五维框架可用于分维度诊断；经验阈值不是跨领域固定标准",
            },
            {
                "核心贡献": "贡献二：模型—任务适配预判",
                "稳健性证据": (
                    f"记录层GBT Kappa R²={full_gbt['r_squared']:.3f}；组合层GBT R²="
                    f"{grouped_gbt['grouped_r_squared']:.3f}；组合层岭回归RMSE="
                    f"{grouped_ridge['grouped_rmse']:.3f}"
                ),
                "稳健性判断": "复杂度具有稳定预测信息，但最优预测算法随泛化目标变化",
                "论文主张边界": "GBT用于已知设计空间插值；岭回归和分组验证用于未见组合的保守校验",
            },
            {
                "核心贡献": "贡献三：数据与模型改进诊断",
                "稳健性证据": (
                    f"诊断方向跨三指标一致率：GBT {metric_gbt:.1%}、岭回归{metric_ridge:.1%}；"
                    f"跨算法最低一致率{algorithm_min:.1%}"
                ),
                "稳健性判断": "主要建议具有较高方向一致性，边缘建议需要降级",
                "论文主张边界": "仅将多指标或多算法一致的建议列为强证据；情景差值不作因果解释",
            },
        ]
    )




def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    step6_dir = args.step6_dir.resolve()
    step7_dir = args.step7_dir.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    diagnostic_dir = output_dir / "diagnostics"
    table_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    data, _ = prepare_unified_data(analysis_root / "step1_data_quality")
    oof = read_csv(
        step6_dir / "predictions" / "record_10fold_oof_predictions.csv"
    )
    oof = oof.merge(
        data[["record_id", "label_number"]],
        on="record_id",
        how="left",
        validate="many_to_one",
    )
    if oof["label_number"].isna().any():
        raise ValueError("OOF predictions could not be mapped to label numbers")

    _, thresholds = build_thresholds(data)
    oof_summary = summarize_oof_subsets(oof)
    bootstrap = cluster_bootstrap_metrics(
        oof,
        reps=args.bootstrap_reps,
        seed=args.seed,
    )
    validation = validation_design_robustness(step6_dir)
    scenario_summary, metric_consistency, algorithm_consistency = (
        scenario_metric_algorithm_robustness(data, step6_dir, thresholds)
    )
    threshold_summary, threshold_agreement = threshold_robustness(data)
    near_tie, margins = near_tie_robustness(data, step6_dir)
    predicted_winners = model_winner_metric_consistency(margins)
    observed_winners = common24_metric_consistency(analysis_root)
    contribution_summary = contribution_robustness_summary(
        oof_summary,
        validation,
        metric_consistency,
        algorithm_consistency,
        threshold_agreement,
    )

    oof_summary.to_csv(
        table_dir / "metric_and_11label_robustness.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap.to_csv(
        table_dir / "combination_cluster_bootstrap_intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation.to_csv(
        table_dir / "validation_split_robustness.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_consistency.to_csv(
        table_dir / "scenario_metric_direction_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    algorithm_consistency.to_csv(
        table_dir / "scenario_algorithm_direction_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    threshold_summary.to_csv(
        table_dir / "diagnostic_threshold_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    threshold_agreement.to_csv(
        table_dir / "diagnostic_threshold_agreement.csv",
        index=False,
        encoding="utf-8-sig",
    )
    near_tie.to_csv(
        table_dir / "near_tie_threshold_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predicted_winners.to_csv(
        table_dir / "predicted_winner_metric_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    observed_winners.to_csv(
        table_dir / "common24_winner_metric_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contribution_summary.to_csv(
        table_dir / "core_contribution_robustness_summary_zh.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenario_summary.to_csv(
        diagnostic_dir / "all_metric_algorithm_scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    margins.to_csv(
        diagnostic_dir / "all_profile_candidate_margins.csv",
        index=False,
        encoding="utf-8-sig",
    )


    full_gbt = oof_summary[
        oof_summary["evaluation_subset"].eq("full")
        & oof_summary["algorithm"].eq("gradient_boosting")
    ]
    report = {
        "analysis_role": "robustness checks for the three core contributions",
        "bootstrap_reps": int(args.bootstrap_reps),
        "robustness_dimensions": [
            "outcome metric",
            "validation split",
            "prediction algorithm",
            "11-label evaluation condition",
            "diagnostic threshold",
            "near-tie routing threshold",
        ],
        "full_gbt_metrics": full_gbt[
            ["outcome", "rmse", "mae", "r_squared"]
        ].to_dict(orient="records"),
        "output_checks": {
            "all_three_outcomes_present": bool(
                set(oof_summary["outcome"]) == set(OUTCOMES)
            ),
            "all_four_prediction_algorithms_present": bool(
                set(oof_summary["algorithm"]) == set(ALGORITHMS)
            ),
            "bootstrap_intervals_ordered": bool(
                bootstrap["rmse_ci95_low"].le(bootstrap["rmse_ci95_high"]).all()
                and bootstrap["r_squared_ci95_low"].le(
                    bootstrap["r_squared_ci95_high"]
                ).all()
            ),
            "three_threshold_percentiles_checked": bool(
                len(threshold_summary) == len(THRESHOLD_PERCENTILES)
            ),
            "four_near_tie_thresholds_per_outcome": bool(
                near_tie.groupby("outcome").size().eq(len(NEAR_TIE_THRESHOLDS)).all()
            ),
            "scenario_predictions_finite": bool(
                np.isfinite(
                    scenario_summary["mean_predicted_metric_difference"]
                ).all()
            ),
            "three_core_contributions_summarized": bool(
                len(contribution_summary) == 3
            ),
        },
        "interpretation_boundary": (
            "Robustness checks use existing experiment outputs and fitted meta-predictors. "
            "They assess analytical stability, not causal intervention effects or unrestricted cross-domain generalization."
        ),
    }
    write_json(output_dir / "step8_robustness_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
