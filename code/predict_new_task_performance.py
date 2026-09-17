from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


OUTCOMES = ["kappa", "macro_f1", "mcc"]
MODEL_VARIANTS = [
    "SVM",
    "DeBERTa",
    "LLM_zero_shot",
    "LLM_few_shot",
    "LLM_codebook_guided",
]
NUMERIC_BASE_FEATURES = [
    "label_number_log2",
    "class_imbalance_filled",
    "annotation_disagreement",
    "training_size_log2_filled",
    "semantic_similarity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict coding-model performance for new tasks")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["ridge", "gradient_boosting", "mlp"])
    return parser.parse_args()


def prepare_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "new_task_id",
        "model_variant",
        "label_number",
        "class_imbalance",
        "annotation_disagreement",
        "training_sample_size",
        "semantic_similarity",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction input is missing columns: {missing}")
    output = frame.copy()
    for column in [
        "label_number",
        "class_imbalance",
        "annotation_disagreement",
        "training_sample_size",
        "semantic_similarity",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if output["new_task_id"].isna().any() or output["new_task_id"].duplicated().any():
        raise ValueError("new_task_id values must be present and unique")
    invalid_variants = sorted(set(output["model_variant"].astype(str)) - set(MODEL_VARIANTS))
    if invalid_variants:
        raise ValueError(f"Unsupported model variants: {invalid_variants}")
    universal = ["label_number", "annotation_disagreement", "semantic_similarity"]
    if output[universal].isna().any().any():
        raise ValueError("Label number, annotation disagreement, and semantic similarity are required")
    supervised = output["model_variant"].isin(["SVM", "DeBERTa"])
    if output.loc[supervised, ["class_imbalance", "training_sample_size"]].isna().any().any():
        raise ValueError("SVM and DeBERTa rows require class imbalance and training sample size")
    output["label_number_log2"] = np.log2(output["label_number"].astype(float))
    output["class_imbalance_filled"] = output["class_imbalance"].fillna(0.0)
    output["training_size_log2_filled"] = np.where(
        output["training_sample_size"].notna(),
        np.log2(output["training_sample_size"].fillna(1.0)),
        0.0,
    )
    return output


def build_design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[NUMERIC_BASE_FEATURES].astype(float).copy()
    variants = frame["model_variant"].astype(str)
    for variant in MODEL_VARIANTS[1:]:
        output[f"model__{variant}"] = variants.eq(variant).astype(float)
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


def support_warning(row: pd.Series, support: pd.DataFrame) -> str:
    variant_support = support[support["model_variant"].astype(str).eq(str(row["model_variant"]))]
    if variant_support.empty:
        return "model variant absent from training support"
    profile = variant_support.iloc[0]
    warnings = []
    checks = [
        ("label_number", "minimum_label_number", "maximum_label_number"),
        ("annotation_disagreement", "minimum_disagreement", "maximum_disagreement"),
        ("semantic_similarity", "minimum_similarity", "maximum_similarity"),
    ]
    if row["model_variant"] in ["SVM", "DeBERTa"]:
        checks.extend(
            [
                ("class_imbalance", "minimum_class_imbalance", "maximum_class_imbalance"),
                ("training_sample_size", "minimum_training_size", "maximum_training_size"),
            ]
        )
    for input_column, minimum_column, maximum_column in checks:
        value = float(row[input_column])
        minimum = float(profile[minimum_column])
        maximum = float(profile[maximum_column])
        if value < minimum or value > maximum:
            warnings.append(f"{input_column} outside [{minimum:.4g}, {maximum:.4g}]")
    return "; ".join(warnings) if warnings else "within observed support"


def main() -> None:
    args = parse_args()
    frame = prepare_input(pd.read_csv(args.input))
    recommendation_path = args.models_dir / "recommended_predictor.joblib"
    recommendation = joblib.load(recommendation_path) if recommendation_path.exists() else {}
    algorithm = args.algorithm or recommendation.get("recommended_algorithm", "ridge")
    design = build_design_matrix(frame)
    output = frame[
        [
            "new_task_id",
            "model_variant",
            "label_number",
            "class_imbalance",
            "annotation_disagreement",
            "training_sample_size",
            "semantic_similarity",
        ]
    ].copy()
    support_reference = None
    for outcome in OUTCOMES:
        bundle_path = args.models_dir / f"{outcome}__{algorithm}.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing model bundle: {bundle_path}")
        bundle = joblib.load(bundle_path)
        expected_columns = list(bundle["feature_columns"])
        missing_features = [column for column in expected_columns if column not in design.columns]
        if missing_features:
            raise ValueError(f"Derived design is missing model features: {missing_features}")
        prediction = bundle["estimator"].predict(design[expected_columns])
        output[f"predicted_{outcome}"] = prediction
        quantiles = pd.DataFrame(bundle["absolute_error_quantiles"])
        quantile_map = quantiles.set_index("model_variant")["absolute_error_q95"]
        width = frame["model_variant"].astype(str).map(quantile_map).to_numpy(float)
        output[f"predicted_{outcome}_lower_approx95"] = np.maximum(-1.0, prediction - width)
        output[f"predicted_{outcome}_upper_approx95"] = np.minimum(1.0, prediction + width)
        support_reference = pd.DataFrame(bundle["training_support"])
    output["prediction_algorithm"] = algorithm
    output["support_warning"] = frame.apply(
        lambda row: support_warning(row, support_reference), axis=1
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(output)} predictions using {algorithm}: {args.output.resolve()}")


if __name__ == "__main__":
    main()
