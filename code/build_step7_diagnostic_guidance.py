from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fit_step6_performance_prediction import (
    MODEL_VARIANTS,
    build_design_matrix,
    prepare_unified_data,
)


MODEL_DISPLAY = {
    "SVM": "SVM",
    "DeBERTa": "DeBERTa-v3",
    "LLM_zero_shot": "LLM zero-shot",
    "LLM_few_shot": "LLM few-shot",
    "LLM_codebook_guided": "LLM codebook-guided",
}

LABEL_LEVELS = [3, 5, 8, 11]
PRACTICAL_TIE_MARGIN = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build evidence-constrained diagnostic guidance for engineering text coding"
    )
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--step6-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def prepare_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["label_number"] = pd.to_numeric(output["label_number"], errors="coerce")
    output["class_imbalance"] = pd.to_numeric(
        output["class_imbalance"], errors="coerce"
    )
    output["annotation_disagreement"] = pd.to_numeric(
        output["annotation_disagreement"], errors="coerce"
    )
    output["training_sample_size"] = pd.to_numeric(
        output["training_sample_size"], errors="coerce"
    )
    output["semantic_similarity"] = pd.to_numeric(
        output["semantic_similarity"], errors="coerce"
    )
    output["label_number_log2"] = np.log2(output["label_number"].astype(float))
    output["class_imbalance_filled"] = output["class_imbalance"].fillna(0.0)
    output["training_size_log2_filled"] = np.where(
        output["training_sample_size"].notna(),
        np.log2(output["training_sample_size"].fillna(1.0)),
        0.0,
    )
    output["model_variant"] = pd.Categorical(
        output["model_variant"], categories=MODEL_VARIANTS, ordered=True
    )
    return output


def predict_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    prepared = prepare_prediction_frame(frame)
    matrix = build_design_matrix(prepared)
    matrix = matrix[bundle["feature_columns"]]
    prediction = np.asarray(bundle["estimator"].predict(matrix), dtype=float)
    return np.clip(prediction, -1.0, 1.0)


def build_thresholds(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    combinations = data.drop_duplicates("combination_id")
    supervised = data[data["source"].eq("supervised")].copy()
    values = {
        "high_label_number": 8.0,
        "high_class_imbalance": float(supervised["class_imbalance"].quantile(0.75)),
        "low_training_sample_size": float(
            supervised["training_sample_size"].quantile(0.25)
        ),
        "high_annotation_disagreement": float(
            combinations["annotation_disagreement"].quantile(0.75)
        ),
        "high_semantic_similarity": float(
            combinations["semantic_similarity"].quantile(0.75)
        ),
        "target_annotation_disagreement": float(
            combinations["annotation_disagreement"].quantile(0.25)
        ),
        "target_semantic_similarity": float(
            combinations["semantic_similarity"].quantile(0.25)
        ),
        "balanced_class_imbalance": 0.0,
        "practical_near_tie_margin": PRACTICAL_TIE_MARGIN,
    }
    rows = [
        {
            "indicator": "label_number",
            "trigger": ">=",
            "threshold": values["high_label_number"],
            "derivation": "Pre-specified high-complexity level in the 3/5/8/11 design",
            "interpretation": "Large label space",
        },
        {
            "indicator": "class_imbalance",
            "trigger": ">=",
            "threshold": values["high_class_imbalance"],
            "derivation": "75th percentile of supervised model-task records",
            "interpretation": "High class imbalance",
        },
        {
            "indicator": "training_sample_size",
            "trigger": "<=",
            "threshold": values["low_training_sample_size"],
            "derivation": "25th percentile of supervised model-task records",
            "interpretation": "Limited labeled training data",
        },
        {
            "indicator": "annotation_disagreement",
            "trigger": ">=",
            "threshold": values["high_annotation_disagreement"],
            "derivation": "75th percentile across the 40 label combinations",
            "interpretation": "High coding disagreement",
        },
        {
            "indicator": "semantic_similarity",
            "trigger": ">=",
            "threshold": values["high_semantic_similarity"],
            "derivation": "75th percentile across the 40 label combinations",
            "interpretation": "High semantic overlap between labels",
        },
        {
            "indicator": "predicted_model_margin",
            "trigger": "<",
            "threshold": PRACTICAL_TIE_MARGIN,
            "derivation": "Pre-specified practical near-tie threshold used in Common24",
            "interpretation": "Candidate models are practically indistinguishable",
        },
    ]
    return pd.DataFrame(rows), values


def scenario_predictions(
    data: pd.DataFrame,
    bundle: dict[str, Any],
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = data.copy()
    base["baseline_prediction"] = predict_bundle(bundle, base)
    rows: list[pd.DataFrame] = []

    def add_scenario(
        subset: pd.DataFrame,
        scenario_name: str,
        action: str,
        evidence_grade: str,
        changed: pd.DataFrame,
        interpretation_boundary: str,
    ) -> None:
        if subset.empty:
            return
        output = subset[
            ["record_id", "combination_id", "source", "model_variant", "baseline_prediction"]
        ].copy()
        output["scenario"] = scenario_name
        output["recommended_action"] = action
        output["evidence_grade"] = evidence_grade
        output["scenario_prediction"] = predict_bundle(bundle, changed)
        output["predicted_kappa_difference"] = (
            output["scenario_prediction"] - output["baseline_prediction"]
        )
        output["interpretation_boundary"] = interpretation_boundary
        rows.append(output)

    supervised_mask = base["source"].eq("supervised")
    supervised = base[supervised_mask].copy()

    eligible = supervised[supervised["training_sample_size"] < supervised["training_sample_size"].max()].copy()
    if not eligible.empty:
        levels = sorted(supervised["training_sample_size"].dropna().unique())
        next_level = {
            level: next((candidate for candidate in levels if candidate > level), level)
            for level in levels
        }
        changed = eligible.copy()
        changed["training_sample_size"] = changed["training_sample_size"].map(next_level)
        add_scenario(
            eligible,
            "increase_training_one_level",
            "Add labeled examples to the next observed training-size level",
            "strong",
            changed,
            "Predictive contrast within the observed training-size design; not a causal estimate.",
        )

    eligible = supervised[supervised["class_imbalance"] > 1e-12].copy()
    changed = eligible.copy()
    changed["class_imbalance"] = thresholds["balanced_class_imbalance"]
    add_scenario(
        eligible,
        "rebalance_classes",
        "Prioritize minority-class collection, weighting, or resampling",
        "strong",
        changed,
        "Predictive contrast to the balanced reference within the observed design.",
    )

    eligible = base[base["label_number"] > min(LABEL_LEVELS)].copy()
    changed = eligible.copy()
    previous_level = {
        level: max(candidate for candidate in LABEL_LEVELS if candidate < level)
        for level in LABEL_LEVELS[1:]
    }
    changed["label_number"] = changed["label_number"].map(previous_level)
    add_scenario(
        eligible,
        "reduce_label_space_one_level",
        "Use hierarchical coding, merge unsupported labels, or decompose the task",
        "strong_supervised_exploratory_llm",
        changed,
        "Labels are held otherwise constant; this is a model-based design scenario, not a literal intervention effect.",
    )

    eligible = base[
        base["semantic_similarity"] > thresholds["target_semantic_similarity"]
    ].copy()
    changed = eligible.copy()
    changed["semantic_similarity"] = thresholds["target_semantic_similarity"]
    add_scenario(
        eligible,
        "reduce_semantic_overlap_to_q25",
        "Clarify label boundaries, add contrastive examples, or introduce hierarchical routing",
        "moderate",
        changed,
        "The numeric contrast represents a redesigned label system and should not be read causally.",
    )

    eligible = base[
        base["annotation_disagreement"] > thresholds["target_annotation_disagreement"]
    ].copy()
    changed = eligible.copy()
    changed["annotation_disagreement"] = thresholds[
        "target_annotation_disagreement"
    ]
    add_scenario(
        eligible,
        "reduce_disagreement_to_q25",
        "Revise the codebook, adjudicate disputed examples, and retain human review",
        "conditional",
        changed,
        "Disagreement had no stable independent supervised effect but moderated LLM codebook gains.",
    )

    eligible = base[
        base["model_variant"].astype(str).isin(["LLM_zero_shot", "LLM_few_shot"])
    ].copy()
    changed = eligible.copy()
    changed["model_variant"] = "LLM_codebook_guided"
    add_scenario(
        eligible,
        "upgrade_to_codebook_prompt",
        "Provide complete label definitions, coding rules, and discriminative examples",
        "strong_average_conditional_gain",
        changed,
        "Average gain is supported, but it shrinks as disagreement and semantic overlap increase.",
    )

    detail = pd.concat(rows, ignore_index=True)
    detail["model_variant"] = detail["model_variant"].astype(str)
    summary = (
        detail.groupby(
            [
                "scenario",
                "recommended_action",
                "evidence_grade",
                "model_variant",
                "interpretation_boundary",
            ],
            as_index=False,
        )
        .agg(
            eligible_records=("record_id", "size"),
            mean_predicted_kappa_difference=("predicted_kappa_difference", "mean"),
            median_predicted_kappa_difference=("predicted_kappa_difference", "median"),
            q10_predicted_kappa_difference=(
                "predicted_kappa_difference",
                lambda values: values.quantile(0.10),
            ),
            q90_predicted_kappa_difference=(
                "predicted_kappa_difference",
                lambda values: values.quantile(0.90),
            ),
            share_positive_prediction=(
                "predicted_kappa_difference",
                lambda values: float((values > 0).mean()),
            ),
        )
    )
    summary["model_display"] = summary["model_variant"].map(MODEL_DISPLAY)
    return detail, summary


def prediction_error_by_variant(step6_dir: Path) -> pd.DataFrame:
    oof = read_csv(
        step6_dir / "predictions" / "record_10fold_oof_predictions.csv"
    )
    subset = oof[
        oof["outcome"].eq("kappa") & oof["algorithm"].eq("gradient_boosting")
    ].copy()
    subset["absolute_error"] = np.abs(subset["actual"] - subset["prediction"])
    output = (
        subset.groupby("model_variant", as_index=False)
        .agg(
            oof_records=("record_id", "size"),
            rmse=("residual", lambda values: float(np.sqrt(np.mean(values**2)))),
            mae=("absolute_error", "mean"),
            absolute_error_q80=("absolute_error", lambda values: values.quantile(0.80)),
            absolute_error_q90=("absolute_error", lambda values: values.quantile(0.90)),
        )
    )
    output["model_display"] = output["model_variant"].map(MODEL_DISPLAY)
    return output


def burden_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    components = {
        "label_burden": output["label_number"].rank(pct=True, method="average"),
        "imbalance_burden": output["class_imbalance"].rank(pct=True, method="average"),
        "disagreement_burden": output["annotation_disagreement"].rank(
            pct=True, method="average"
        ),
        "similarity_burden": output["semantic_similarity"].rank(
            pct=True, method="average"
        ),
        "small_sample_burden": 1.0
        - output["training_sample_size"].rank(pct=True, method="average"),
    }
    for name, values in components.items():
        output[name] = values
    output["illustrative_complexity_burden"] = output[list(components)].mean(axis=1)
    return output


def illustrative_profiles(
    data: pd.DataFrame,
    bundle: dict[str, Any],
    errors: pd.DataFrame,
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    supervised = (
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
    scored = burden_percentiles(supervised)
    targets = {
        "low_complexity_observed_profile": scored["illustrative_complexity_burden"].min(),
        "median_complexity_observed_profile": scored["illustrative_complexity_burden"].median(),
        "high_complexity_observed_profile": scored["illustrative_complexity_burden"].max(),
    }
    chosen_rows = []
    used: set[int] = set()
    for profile_name, target in targets.items():
        order = (scored["illustrative_complexity_burden"] - target).abs().sort_values().index
        index = next(int(value) for value in order if int(value) not in used)
        used.add(index)
        row = scored.loc[index].copy()
        row["profile_name"] = profile_name
        chosen_rows.append(row)
    chosen = pd.DataFrame(chosen_rows)

    prediction_rows = []
    diagnostic_rows = []
    error_map = errors.set_index("model_variant")["absolute_error_q90"].to_dict()
    for _, profile in chosen.iterrows():
        candidates = []
        for variant in MODEL_VARIANTS:
            candidate = {
                "profile_name": profile["profile_name"],
                "combination_id": profile["combination_id"],
                "model_variant": variant,
                "label_number": profile["label_number"],
                "class_imbalance": profile["class_imbalance"] if variant in ["SVM", "DeBERTa"] else np.nan,
                "annotation_disagreement": profile["annotation_disagreement"],
                "training_sample_size": profile["training_sample_size"] if variant in ["SVM", "DeBERTa"] else np.nan,
                "semantic_similarity": profile["semantic_similarity"],
            }
            candidates.append(candidate)
        candidate_frame = pd.DataFrame(candidates)
        candidate_frame["predicted_kappa"] = predict_bundle(bundle, candidate_frame)
        candidate_frame["approximate_q90_error"] = candidate_frame["model_variant"].map(error_map)
        candidate_frame["approximate_lower_bound"] = np.maximum(
            -1.0,
            candidate_frame["predicted_kappa"] - candidate_frame["approximate_q90_error"],
        )
        candidate_frame["approximate_upper_bound"] = np.minimum(
            1.0,
            candidate_frame["predicted_kappa"] + candidate_frame["approximate_q90_error"],
        )
        candidate_frame["model_display"] = candidate_frame["model_variant"].map(MODEL_DISPLAY)
        candidate_frame = candidate_frame.sort_values("predicted_kappa", ascending=False)
        prediction_rows.append(candidate_frame)

        top = candidate_frame.iloc[0]
        runner_up = candidate_frame.iloc[1]
        margin = float(top["predicted_kappa"] - runner_up["predicted_kappa"])
        flags = []
        actions = []
        if profile["label_number"] >= thresholds["high_label_number"]:
            flags.append("large_label_space")
            actions.append("hierarchical coding or task decomposition")
        if profile["class_imbalance"] >= thresholds["high_class_imbalance"]:
            flags.append("high_class_imbalance")
            actions.append("minority-class collection, weighting, or resampling")
        if profile["training_sample_size"] <= thresholds["low_training_sample_size"]:
            flags.append("limited_training_data")
            actions.append("increase labeled samples before increasing model complexity")
        if profile["annotation_disagreement"] >= thresholds["high_annotation_disagreement"]:
            flags.append("high_annotation_disagreement")
            actions.append("adjudication, codebook revision, and human review")
        if profile["semantic_similarity"] >= thresholds["high_semantic_similarity"]:
            flags.append("high_semantic_similarity")
            actions.append("contrastive label definitions and boundary examples")
        if margin < PRACTICAL_TIE_MARGIN:
            flags.append("predicted_model_near_tie")
            actions.append("run a small matched benchmark instead of automatic routing")
        complexity_flags = [
            flag for flag in flags if flag != "predicted_model_near_tie"
        ]
        diagnostic_rows.append(
            {
                "profile_name": profile["profile_name"],
                "combination_id": profile["combination_id"],
                "illustrative_complexity_burden": profile[
                    "illustrative_complexity_burden"
                ],
                "label_number": profile["label_number"],
                "class_imbalance": profile["class_imbalance"],
                "training_sample_size": profile["training_sample_size"],
                "annotation_disagreement": profile["annotation_disagreement"],
                "semantic_similarity": profile["semantic_similarity"],
                "predicted_best_model": top["model_display"],
                "predicted_best_kappa": top["predicted_kappa"],
                "predicted_runner_up": runner_up["model_display"],
                "predicted_margin": margin,
                "contribution_1_complexity_diagnosis": (
                    "; ".join(complexity_flags)
                    if complexity_flags
                    else "no high-complexity trigger"
                ),
                "contribution_2_model_task_fit": (
                    f"{top['model_display']} ranked first; predicted margin={margin:.3f}; "
                    + ("pilot comparison required" if margin < PRACTICAL_TIE_MARGIN else "eligible for bounded pre-screening")
                ),
                "contribution_3_improvement_guidance": (
                    "; ".join(actions) if actions else "retain current design"
                ),
                "diagnostic_flags": "; ".join(flags) if flags else "none",
                "recommended_actions": "; ".join(actions) if actions else "retain current design",
                "example_status": "Illustrative observed profile; not external validation",
            }
        )
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(diagnostic_rows)


def build_contribution_alignment() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contribution": "C1 Task complexity framework",
                "diagnostic_stage": "Complexity profiling",
                "input": "Label number, class imbalance, annotation disagreement, training size, semantic similarity",
                "output": "Dimension-specific risk flags and an observed-support check",
                "decision_supported": "Identify why the coding task is difficult and which complexity source dominates",
                "evidence_used": "Steps 3 and 4 complexity effects plus Step 5 matched comparison",
            },
            {
                "contribution": "C2 Model-task fit prediction",
                "diagnostic_stage": "Candidate performance forecasting",
                "input": "Complexity profile and candidate model family or prompt strategy",
                "output": "Predicted Kappa, approximate error range, model ranking, and near-tie flag",
                "decision_supported": "Pre-screen SVM, DeBERTa, and LLM configurations before exhaustive training",
                "evidence_used": "Step 6 record-stratified prediction with grouped-validation boundary",
            },
            {
                "contribution": "C3 Diagnostic guidance for improvement",
                "diagnostic_stage": "Evidence-constrained scenario diagnosis",
                "input": "Detected bottlenecks and fitted performance predictor",
                "output": "Ranked data, label-system, human-review, and prompt interventions",
                "decision_supported": "Prioritize the most relevant improvement instead of indiscriminate model search",
                "evidence_used": "Predicted scenario contrasts constrained by Steps 3-5 statistical evidence",
            },
        ]
    )


def build_rulebook(thresholds: dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "rule_id": "R1_label_space",
            "diagnostic_condition": f"Label number >= {thresholds['high_label_number']:.0f}",
            "interpreted_bottleneck": "Large and fine-grained label space",
            "recommended_action": "Use hierarchical coding, decompose the task, or merge theoretically unsupported labels",
            "model_guidance": "Do not assume a more complex model offsets label-space complexity",
            "evidence_grade": "strong for supervised; exploratory for LLM",
            "evidence_basis": "Step 3 adjusted marginal changes and model-complexity interactions",
        },
        {
            "rule_id": "R2_class_imbalance",
            "diagnostic_condition": f"Class imbalance >= {thresholds['high_class_imbalance']:.3f}",
            "interpreted_bottleneck": "Insufficient minority-class information",
            "recommended_action": "Collect minority-class examples first; then consider class weighting or resampling",
            "model_guidance": "Evaluate Macro-F1 and MCC together with Kappa",
            "evidence_grade": "strong within supervised tasks",
            "evidence_basis": "Step 3 task-mean and model-task-fit analyses",
        },
        {
            "rule_id": "R3_training_size",
            "diagnostic_condition": f"Training size <= {thresholds['low_training_sample_size']:.0f}",
            "interpreted_bottleneck": "Limited supervised information",
            "recommended_action": "Increase labeled examples to the next feasible level and prioritize error-driven sampling",
            "model_guidance": "DeBERTa's relative advantage increased with training size; SVM remains a resource-efficient baseline",
            "evidence_grade": "strong within supervised tasks",
            "evidence_basis": "Step 3 marginal changes and DeBERTa-SVM interaction",
        },
        {
            "rule_id": "R4_semantic_overlap",
            "diagnostic_condition": f"Semantic similarity >= {thresholds['high_semantic_similarity']:.3f}",
            "interpreted_bottleneck": "Overlapping conceptual boundaries",
            "recommended_action": "Rewrite contrastive definitions, add boundary cases, or introduce hierarchical routing",
            "model_guidance": "Prompt expansion alone is unlikely to remove this bottleneck",
            "evidence_grade": "moderate and directionally consistent",
            "evidence_basis": "Step 3 supervised effect and Step 4 prompt-gain moderation",
        },
        {
            "rule_id": "R5_annotation_disagreement",
            "diagnostic_condition": f"Annotation disagreement >= {thresholds['high_annotation_disagreement']:.3f}",
            "interpreted_bottleneck": "Uncertain or contested coding rules",
            "recommended_action": "Adjudicate disputed examples, revise the codebook, and retain targeted human review",
            "model_guidance": "High disagreement reduced the extra gain from codebook-guided prompting",
            "evidence_grade": "conditional",
            "evidence_basis": "No stable supervised main effect; significant LLM prompt-gain moderation",
        },
        {
            "rule_id": "R6_codebook_prompt",
            "diagnostic_condition": "LLM zero-shot or few-shot selected and label boundaries are clear",
            "interpreted_bottleneck": "Under-specified task instructions",
            "recommended_action": "Use a codebook-guided prompt with complete definitions and discriminative examples",
            "model_guidance": "Codebook-guided prompting had the highest adjusted mean performance",
            "evidence_grade": "strong average effect with conditional moderation",
            "evidence_basis": "Step 4 adjusted prompt effects",
        },
        {
            "rule_id": "R7_model_near_tie",
            "diagnostic_condition": f"Predicted top-two model margin < {PRACTICAL_TIE_MARGIN:.2f}",
            "interpreted_bottleneck": "Model choice is uncertain at practical resolution",
            "recommended_action": "Run a small matched benchmark and select using reliability, cost, and deployment constraints",
            "model_guidance": "Do not automatically route on a near-tie prediction",
            "evidence_grade": "decision guardrail",
            "evidence_basis": "Common24 near-tie definition and Step 6 prediction error",
        },
        {
            "rule_id": "R8_out_of_support",
            "diagnostic_condition": "Any input lies outside the observed support",
            "interpreted_bottleneck": "Extrapolation beyond the validated design space",
            "recommended_action": "Treat the forecast as unavailable and obtain a pilot benchmark",
            "model_guidance": "The current predictor supports interpolation, not unrestricted new-task generalization",
            "evidence_grade": "mandatory boundary",
            "evidence_basis": "100% label-combination overlap in record-level validation and grouped-CV sensitivity",
        },
    ]
    return pd.DataFrame(rows)


def build_rulebook_zh(thresholds: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "规则编号": "R1",
                "诊断条件": f"标签数量不少于{thresholds['high_label_number']:.0f}类",
                "主要瓶颈": "标签空间大且类别粒度较细",
                "建议措施": "采用层级编码或任务分解；合并缺乏理论支持的细分类别",
                "模型选择含义": "不应假定提高模型复杂度能够抵消标签空间复杂度",
                "证据等级": "监督学习强证据；LLM探索性证据",
            },
            {
                "规则编号": "R2",
                "诊断条件": f"类别不平衡指标不低于{thresholds['high_class_imbalance']:.3f}",
                "主要瓶颈": "少数类监督信息不足",
                "建议措施": "优先补充少数类样本，再考虑类别加权或重采样",
                "模型选择含义": "模型评估应同时报告Macro-F1、MCC和Kappa",
                "证据等级": "监督学习强证据",
            },
            {
                "规则编号": "R3",
                "诊断条件": f"训练样本量不超过{thresholds['low_training_sample_size']:.0f}条",
                "主要瓶颈": "可用监督信息有限",
                "建议措施": "将标注样本增加到下一可行规模，并优先标注高错误率样本",
                "模型选择含义": "DeBERTa的相对优势随样本量增加；小样本下保留SVM基线",
                "证据等级": "监督学习强证据",
            },
            {
                "规则编号": "R4",
                "诊断条件": f"标签语义相似度不低于{thresholds['high_semantic_similarity']:.3f}",
                "主要瓶颈": "标签概念边界重叠",
                "建议措施": "重写对比式定义，增加边界样例，或采用层级路由",
                "模型选择含义": "仅扩展prompt难以消除语义边界瓶颈",
                "证据等级": "方向一致的中等证据",
            },
            {
                "规则编号": "R5",
                "诊断条件": f"人工编码争议率不低于{thresholds['high_annotation_disagreement']:.3f}",
                "主要瓶颈": "编码规则存在不确定性或争议",
                "建议措施": "复核争议样本、修订codebook并保留针对性人工复核",
                "模型选择含义": "高争议会削弱codebook-guided prompt的额外收益",
                "证据等级": "条件性证据",
            },
            {
                "规则编号": "R6",
                "诊断条件": "拟采用LLM且标签边界相对清晰",
                "主要瓶颈": "任务说明不足",
                "建议措施": "提供完整标签定义、编码规则和有区分度的示例",
                "模型选择含义": "优先采用codebook-guided prompt，而非zero-shot",
                "证据等级": "平均效应强，但存在复杂度调节",
            },
            {
                "规则编号": "R7",
                "诊断条件": f"预测最优与次优模型的Kappa差小于{PRACTICAL_TIE_MARGIN:.2f}",
                "主要瓶颈": "模型选择不确定",
                "建议措施": "开展小规模同任务匹配测试，并结合成本和部署约束选择",
                "模型选择含义": "近似并列时不执行自动模型路由",
                "证据等级": "决策保护规则",
            },
            {
                "规则编号": "R8",
                "诊断条件": "任一输入超出本研究观测范围",
                "主要瓶颈": "超出已验证设计空间",
                "建议措施": "停止自动预判并开展先导基准测试",
                "模型选择含义": "当前预测器支持范围内插值，不支持无约束外推",
                "证据等级": "强制边界规则",
            },
        ]
    )


def build_evidence_registry(
    analysis_root: Path,
    cv_summary: pd.DataFrame,
    grouped_comparison: pd.DataFrame,
) -> pd.DataFrame:
    marginal = read_csv(
        analysis_root
        / "step3_supervised_complexity"
        / "predictions"
        / "adjusted_marginal_changes.csv"
    )
    prompt = read_csv(
        analysis_root
        / "step4_llm_prompt_analysis"
        / "predictions"
        / "llm_adjusted_overall_prompt_effects.csv"
    )
    winners = read_csv(
        analysis_root
        / "step5_common24_cross_model"
        / "tables"
        / "common24_winner_summary.csv"
    )

    rows: list[dict[str, Any]] = []
    dimension_rule = {
        "label_number": "R1_label_space",
        "class_imbalance_z": "R2_class_imbalance",
        "log2_training_size_z": "R3_training_size",
        "semantic_similarity_z": "R4_semantic_overlap",
        "annotation_disagreement_z": "R5_annotation_disagreement",
    }
    for _, row in marginal.iterrows():
        dimension = str(row["dimension"])
        change = float(row["predicted_change_high_minus_low"])
        rows.append(
            {
                "evidence_id": f"S3_{dimension}_{row['model']}",
                "source_step": "Step 3 supervised complexity analysis",
                "metric": "Cohen's kappa",
                "comparison": f"{row['dimension_display']}: {float(row['low_x']):.3f} to {float(row['high_x']):.3f}",
                "estimate": change,
                "finding": f"Predicted high-minus-low change for {row['model']}",
                "evidence_status": (
                    "conditional/weak"
                    if dimension == "annotation_disagreement_z"
                    else "primary adjusted marginal evidence"
                ),
                "supports_rule_id": dimension_rule[dimension],
            }
        )

    prompt_rule = {
        "codebook_minus_zero": "R6_codebook_prompt",
        "few_minus_zero": "R6_codebook_prompt",
        "codebook_minus_few": "R6_codebook_prompt",
    }
    for strategy, rule_id in prompt_rule.items():
        row = prompt[prompt["prompt_strategy"].eq(strategy)].iloc[0]
        rows.append(
            {
                "evidence_id": f"S4_{strategy}",
                "source_step": "Step 4 LLM prompt analysis",
                "metric": "Cohen's kappa",
                "comparison": row["prompt_display"],
                "estimate": float(row["adjusted_mean"]),
                "finding": "Adjusted average prompt-strategy difference",
                "evidence_status": "primary adjusted prompt contrast",
                "supports_rule_id": rule_id,
            }
        )

    kappa_winners = winners[winners["outcome"].eq("kappa")]
    for _, row in kappa_winners.iterrows():
        rows.append(
            {
                "evidence_id": f"S5_wins_{row['winner_short']}",
                "source_step": "Step 5 Common24 matched comparison",
                "metric": "Cohen's kappa",
                "comparison": "Task wins out of 24 matched label combinations",
                "estimate": float(row["task_wins"]),
                "finding": f"{row['winner_short']} task wins",
                "evidence_status": "matched cross-model evidence",
                "supports_rule_id": "R7_model_near_tie",
            }
        )

    record_gbt = cv_summary[
        cv_summary["outcome"].eq("kappa")
        & cv_summary["algorithm"].eq("gradient_boosting")
    ].iloc[0]
    grouped_gbt = grouped_comparison[
        grouped_comparison["outcome"].eq("kappa")
        & grouped_comparison["algorithm"].eq("gradient_boosting")
    ].iloc[0]
    rows.extend(
        [
            {
                "evidence_id": "S6_record_gbt_r2",
                "source_step": "Step 6 record-stratified nested CV",
                "metric": "R-squared",
                "comparison": "GBT prediction of Kappa across 600 records",
                "estimate": float(record_gbt["r_squared"]),
                "finding": "Interpolation-oriented prediction performance",
                "evidence_status": "primary operational prediction evidence",
                "supports_rule_id": "R7_model_near_tie",
            },
            {
                "evidence_id": "S6_grouped_gbt_r2",
                "source_step": "Step 6 grouped-by-label-combination CV",
                "metric": "R-squared",
                "comparison": "GBT prediction with unseen label combinations",
                "estimate": float(grouped_gbt["grouped_r_squared"]),
                "finding": "Conservative new-combination prediction performance",
                "evidence_status": "scope-boundary robustness evidence",
                "supports_rule_id": "R8_out_of_support",
            },
        ]
    )
    output = pd.DataFrame(rows)
    if output["evidence_id"].duplicated().any():
        raise ValueError("Diagnostic evidence IDs are not unique")
    return output




def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    step6_dir = args.step6_dir.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    prediction_dir = output_dir / "predictions"
    table_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    step1_dir = analysis_root / "step1_data_quality"
    data, support = prepare_unified_data(step1_dir)
    thresholds_table, thresholds = build_thresholds(data)

    bundle_path = step6_dir / "models" / "kappa__gradient_boosting.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    bundle = joblib.load(bundle_path)
    if bundle.get("analysis") != "record_level_stratified_10fold_sensitivity":
        raise ValueError("Unexpected Step 6 model bundle")

    errors = prediction_error_by_variant(step6_dir)
    scenario_detail, scenario_summary = scenario_predictions(data, bundle, thresholds)
    profile_predictions, profile_diagnoses = illustrative_profiles(
        data, bundle, errors, thresholds
    )
    rulebook = build_rulebook(thresholds)
    rulebook_zh = build_rulebook_zh(thresholds)
    contribution_alignment = build_contribution_alignment()

    cv_summary = read_csv(step6_dir / "tables" / "record_10fold_cv_summary.csv")
    algorithm_comparison = read_csv(
        step6_dir / "tables" / "record_10fold_algorithm_comparison.csv"
    )
    grouped_comparison = read_csv(
        step6_dir / "tables" / "grouped_vs_record_split_comparison.csv"
    )
    evidence_registry = build_evidence_registry(
        analysis_root,
        cv_summary,
        grouped_comparison,
    )

    thresholds_table.to_csv(
        table_dir / "diagnostic_thresholds.csv", index=False, encoding="utf-8-sig"
    )
    rulebook.to_csv(
        table_dir / "diagnostic_rulebook.csv", index=False, encoding="utf-8-sig"
    )
    rulebook_zh.to_csv(
        table_dir / "diagnostic_rulebook_zh.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evidence_registry.to_csv(
        table_dir / "diagnostic_evidence_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contribution_alignment.to_csv(
        table_dir / "diagnostic_contribution_alignment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    errors.to_csv(
        table_dir / "prediction_error_by_model_variant.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenario_summary.to_csv(
        table_dir / "diagnostic_scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    profile_diagnoses.to_csv(
        table_dir / "illustrative_task_diagnoses.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenario_detail.to_csv(
        prediction_dir / "diagnostic_scenario_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    profile_predictions.to_csv(
        prediction_dir / "illustrative_task_model_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )


    report = {
        "analysis_role": "evidence-constrained diagnostic guidance",
        "input_records": int(len(data)),
        "label_combinations": int(data["combination_id"].nunique()),
        "prediction_model": "record-stratified gradient boosting for Kappa",
        "prediction_scope": "interpolation within the observed contract-coding design space",
        "thresholds": thresholds,
        "scenario_rows": int(len(scenario_detail)),
        "illustrative_profiles": int(len(profile_diagnoses)),
        "rule_count": int(len(rulebook)),
        "evidence_entries": int(len(evidence_registry)),
        "contribution_stages": int(len(contribution_alignment)),
        "output_checks": {
            "all_scenario_predictions_finite": bool(
                np.isfinite(scenario_detail["scenario_prediction"]).all()
            ),
            "all_scenario_predictions_in_metric_range": bool(
                scenario_detail["scenario_prediction"].between(-1, 1).all()
            ),
            "three_distinct_illustrative_profiles": bool(
                profile_diagnoses["combination_id"].nunique() == 3
            ),
            "five_candidate_predictions_per_profile": bool(
                profile_predictions.groupby("profile_name").size().eq(5).all()
            ),
            "rulebook_has_boundary_rule": bool(
                rulebook["evidence_grade"].eq("mandatory boundary").any()
            ),
            "all_rules_have_evidence_basis": bool(
                rulebook["evidence_basis"].notna().all()
            ),
            "chinese_rulebook_has_eight_rules": bool(len(rulebook_zh) == 8),
            "evidence_ids_are_unique": bool(
                evidence_registry["evidence_id"].is_unique
            ),
            "three_core_contributions_are_mapped": bool(
                len(contribution_alignment) == 3
            ),
        },
        "interpretation_boundary": (
            "Scenario differences are model-predicted contrasts and are not causal intervention effects. "
            "The record-level predictor does not establish unrestricted generalization to new coding tasks."
        ),
    }
    write_json(output_dir / "step7_diagnostic_guidance_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
