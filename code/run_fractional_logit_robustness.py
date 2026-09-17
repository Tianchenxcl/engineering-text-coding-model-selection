from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/step1_data_quality/supervised_analysis_ready.csv"
LINEAR_EFFECTS = ROOT / "results/step3_supervised_complexity/tables/mixed_model_fixed_effects.csv"
OUTPUT_DIR = ROOT / "reproduced/bounded_kappa_robustness"


MODEL_TERM = "C(model_code, Treatment(reference='SVM'))"
LABEL_TERM = "C(label_number_factor, Treatment(reference='3'))"
FORMULA = (
    "kappa_fraction ~ "
    f"{MODEL_TERM} * ("
    f"{LABEL_TERM} + class_imbalance_z + annotation_disagreement_z + "
    "log2_training_size_z + semantic_similarity_z)"
)


def prepare_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA, low_memory=False)
    required = [
        "kappa",
        "model_variant",
        "label_number_factor",
        "class_imbalance_z",
        "annotation_disagreement_z",
        "log2_training_size_z",
        "semantic_similarity_z",
        "combination_id",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame = frame.copy()
    frame["model_code"] = frame["model_variant"].map(
        {
            "TF-IDF + LinearSVC": "SVM",
            "DeBERTa-v3-base": "DeBERTa",
        }
    )
    if frame["model_code"].isna().any():
        raise ValueError("Unexpected supervised model variant")
    frame["label_number_factor"] = frame["label_number_factor"].astype(str)
    frame["kappa_fraction"] = ((frame["kappa"].astype(float) + 1.0) / 2.0).clip(
        1e-6, 1.0 - 1e-6
    )
    if len(frame) != 480 or frame["combination_id"].nunique() != 24:
        raise ValueError("Expected 480 records and 24 label combinations")
    return frame


def fit_fractional_logit(frame: pd.DataFrame, analysis: str) -> pd.DataFrame:
    result = smf.glm(
        formula=FORMULA,
        data=frame,
        family=sm.families.Binomial(link=sm.families.links.Logit()),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["combination_id"], "use_correction": True},
    )
    confidence = result.conf_int()
    output = pd.DataFrame(
        {
            "analysis": analysis,
            "term": result.params.index,
            "estimate_log_odds": result.params.to_numpy(float),
            "cluster_robust_standard_error": result.bse.to_numpy(float),
            "z_value": result.tvalues.to_numpy(float),
            "p_value": result.pvalues.to_numpy(float),
            "ci95_low": confidence.iloc[:, 0].to_numpy(float),
            "ci95_high": confidence.iloc[:, 1].to_numpy(float),
        }
    )
    output["predicted_minimum"] = float(result.predict(frame).min())
    output["predicted_maximum"] = float(result.predict(frame).max())
    output["record_count"] = len(frame)
    output["combination_count"] = frame["combination_id"].nunique()
    return output


def add_linear_direction_check(coefficients: pd.DataFrame) -> pd.DataFrame:
    linear = pd.read_csv(LINEAR_EFFECTS)
    linear = linear[
        linear["analysis"].eq("primary_all_labels") & linear["outcome"].eq("kappa")
    ][["term", "estimate"]].rename(columns={"estimate": "linear_mixed_estimate"})
    output = coefficients.merge(linear, on="term", how="left")
    output["same_direction_as_linear_mixed"] = np.where(
        output["linear_mixed_estimate"].notna(),
        np.sign(output["estimate_log_odds"])
        == np.sign(output["linear_mixed_estimate"]),
        np.nan,
    )
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = prepare_data()
    all_labels = fit_fractional_logit(frame, "all_labels")
    without_11 = fit_fractional_logit(
        frame[frame["label_number_factor"].ne("11")].copy(),
        "exclude_11_labels",
    )
    coefficients = pd.concat([all_labels, without_11], ignore_index=True)
    coefficients = add_linear_direction_check(coefficients)
    coefficients.to_csv(
        OUTPUT_DIR / "bounded_kappa_fractional_logit_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparable = coefficients[
        coefficients["analysis"].eq("all_labels")
        & coefficients["linear_mixed_estimate"].notna()
        & coefficients["term"].ne("Intercept")
    ]
    report = {
        "method": "Fractional-logit GLM with label-combination cluster-robust standard errors",
        "response": "(Kappa + 1) / 2, clipped to [1e-6, 1-1e-6]",
        "formula": FORMULA,
        "record_count": int(len(frame)),
        "combination_count": int(frame["combination_id"].nunique()),
        "coefficient_direction_matches": int(
            comparable["same_direction_as_linear_mixed"].fillna(False).sum()
        ),
        "coefficient_direction_total": int(len(comparable)),
        "direction_agreement_rate": float(
            comparable["same_direction_as_linear_mixed"].mean()
        ),
        "bounded_prediction_minimum": float(all_labels["predicted_minimum"].iloc[0]),
        "bounded_prediction_maximum": float(all_labels["predicted_maximum"].iloc[0]),
        "notes": [
            "This is an outcome-distribution robustness check based on existing model outputs.",
            "Cluster-robust inference uses 24 label combinations and should be interpreted cautiously.",
            "Coefficient magnitudes are on the fractional-logit scale and are not directly comparable with linear mixed-model coefficients.",
        ],
    }
    (OUTPUT_DIR / "bounded_kappa_fractional_logit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
