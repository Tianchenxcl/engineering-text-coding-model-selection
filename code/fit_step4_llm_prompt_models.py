from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTCOMES = ["kappa", "macro_f1", "mcc"]
PROMPT_ORDER = ["zero_shot", "few_shot", "codebook_guided"]
PROMPT_DISPLAY = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "codebook_guided": "Codebook-guided",
}
PROMPT_COLORS = {
    "zero_shot": "#8B8B8B",
    "few_shot": "#D28B3C",
    "codebook_guided": "#B55245",
}
CONTRASTS = [
    ("codebook_minus_zero", "codebook_guided", "zero_shot"),
    ("few_minus_zero", "few_shot", "zero_shot"),
    ("codebook_minus_few", "codebook_guided", "few_shot"),
]
CONTRAST_DISPLAY = {
    "codebook_minus_zero": "Codebook minus zero-shot",
    "few_minus_zero": "Few-shot minus zero-shot",
    "codebook_minus_few": "Codebook minus few-shot",
}
CONTRAST_COLORS = {
    "codebook_minus_zero": "#B55245",
    "few_minus_zero": "#D28B3C",
    "codebook_minus_few": "#5B4A42",
}
LABEL_LEVELS = ["3", "5", "8", "11"]
CONTINUOUS_TERMS = ["annotation_disagreement_z", "semantic_similarity_z"]
CONTINUOUS_SOURCE = {
    "annotation_disagreement_z": "annotation_disagreement_complexity",
    "semantic_similarity_z": "semantic_similarity_complexity",
}
DIMENSION_DISPLAY = {
    "label_number": "Label number",
    "annotation_disagreement_z": "Annotation disagreement",
    "semantic_similarity_z": "Label semantic similarity",
}
OUTCOME_DISPLAY = {
    "kappa": "Cohen's kappa",
    "macro_f1": "Macro-F1",
    "mcc": "MCC",
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
    parser = argparse.ArgumentParser(description="Step 4: LLM prompt-strategy models")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], dataset: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = input_dir / "llm_analysis_ready.csv"
    parameter_path = input_dir / "standardization_parameters.csv"
    if not data_path.exists() or not parameter_path.exists():
        raise FileNotFoundError(
            "Step 4 requires llm_analysis_ready.csv and standardization_parameters.csv"
        )
    return (
        pd.read_csv(data_path, low_memory=False),
        pd.read_csv(parameter_path, low_memory=False),
    )


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "combination_id",
        "prompt_strategy",
        "repeat_id",
        "label_number_factor",
        *CONTINUOUS_TERMS,
        *[CONTINUOUS_SOURCE[column] for column in CONTINUOUS_TERMS],
        *OUTCOMES,
    ]
    require_columns(data, required, "llm_analysis_ready")
    output = data.copy()
    output["prompt_strategy"] = output["prompt_strategy"].astype(str)
    output["label_number_factor"] = (
        pd.to_numeric(output["label_number_factor"], errors="raise").astype(int).astype(str)
    )
    output["repeat_id"] = pd.to_numeric(output["repeat_id"], errors="raise").astype(int)
    output["prompt_combination_id"] = (
        output["combination_id"].astype(str) + "::" + output["prompt_strategy"]
    )
    for column in [*CONTINUOUS_TERMS, *OUTCOMES]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if output[["prompt_strategy", "label_number_factor", *CONTINUOUS_TERMS, *OUTCOMES]].isna().any().any():
        raise ValueError("LLM analysis data contain missing model variables")
    if set(output["prompt_strategy"].unique()) != set(PROMPT_ORDER):
        raise ValueError(f"Unexpected prompt strategies: {output['prompt_strategy'].unique().tolist()}")
    cell_counts = output.groupby(["combination_id", "prompt_strategy"]).size()
    if not cell_counts.eq(3).all():
        raise ValueError("Every label-combination and prompt cell must contain three repeats")
    strategy_counts = output.groupby("combination_id")["prompt_strategy"].nunique()
    if not strategy_counts.eq(3).all():
        raise ValueError("Every label combination must contain all three prompt strategies")
    output = output.sort_values(
        ["combination_id", "prompt_strategy", "repeat_id"]
    ).reset_index(drop=True)

    reference = output.drop_duplicates("combination_id")
    support = (
        reference.groupby("label_number_factor", sort=False)
        .agg(
            combination_count=("combination_id", "nunique"),
            minimum_disagreement=("annotation_disagreement_complexity", "min"),
            maximum_disagreement=("annotation_disagreement_complexity", "max"),
            minimum_similarity=("semantic_similarity_complexity", "min"),
            maximum_similarity=("semantic_similarity_complexity", "max"),
        )
        .reset_index()
    )
    support["label_number"] = pd.to_numeric(support["label_number_factor"])
    support = support.sort_values("label_number").drop(columns="label_number_factor")
    support["inference_status"] = np.where(
        support["combination_count"] < 2,
        "exploratory_single_combination",
        "replicated_across_combinations",
    )
    return output, support


def complexity_expression() -> str:
    return (
        "C(label_number_factor, Treatment(reference='3'))"
        " + annotation_disagreement_z"
        " + semantic_similarity_z"
    )


def model_formulas(outcome: str) -> tuple[str, str]:
    prompt = "C(prompt_strategy, Treatment(reference='zero_shot'))"
    complexity = complexity_expression()
    return (
        f"{outcome} ~ {prompt} + {complexity}",
        f"{outcome} ~ {prompt} * ({complexity})",
    )


def structure_specification(name: str) -> dict[str, Any]:
    if name == "combination_plus_prompt_cell":
        return {
            "groups": "combination_id",
            "vc_formula": {"prompt_cell": "0 + C(prompt_combination_id)"},
        }
    if name == "combination_only":
        return {"groups": "combination_id", "vc_formula": None}
    raise ValueError(f"Unknown random-effects structure: {name}")


def fit_mixed_once(
    data: pd.DataFrame,
    formula: str,
    structure: str,
    reml: bool,
    outcome: str,
    role: str,
    log_rows: list[dict[str, Any]],
) -> Any:
    import statsmodels.formula.api as smf

    spec = structure_specification(structure)
    candidates = []
    for optimizer in ["lbfgs", "powell", "cg"]:
        result = None
        error = ""
        warning_text = ""
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    data,
                    groups=data[spec["groups"]],
                    re_formula="1",
                    vc_formula=spec["vc_formula"],
                    missing="raise",
                )
                result = model.fit(
                    reml=reml,
                    method=optimizer,
                    maxiter=3000,
                    disp=False,
                    full_output=True,
                )
                warning_text = " | ".join(str(item.message) for item in caught)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        converged = bool(getattr(result, "converged", False)) if result is not None else False
        log_rows.append(
            {
                "outcome": outcome,
                "fit_role": role,
                "structure": structure,
                "reml": reml,
                "optimizer": optimizer,
                "converged": converged,
                "log_likelihood": float(getattr(result, "llf", np.nan)) if result is not None else np.nan,
                "warning": warning_text,
                "error": error,
            }
        )
        if result is not None:
            candidates.append(result)
        if converged:
            return result
    if candidates:
        return max(candidates, key=lambda item: float(getattr(item, "llf", -np.inf)))
    raise RuntimeError(f"All mixed-model optimizers failed for {outcome}, {role}")


def fit_ml_pair(
    data: pd.DataFrame,
    additive_formula: str,
    interaction_formula: str,
    outcome: str,
    log_rows: list[dict[str, Any]],
) -> tuple[Any, Any, str]:
    errors = []
    for structure in ["combination_plus_prompt_cell", "combination_only"]:
        try:
            additive = fit_mixed_once(
                data, additive_formula, structure, False, outcome, "additive_ml", log_rows
            )
            interaction = fit_mixed_once(
                data, interaction_formula, structure, False, outcome, "interaction_ml", log_rows
            )
            if bool(additive.converged) and bool(interaction.converged):
                return additive, interaction, structure
            errors.append(f"{structure}: non-convergence")
        except Exception as exc:
            errors.append(f"{structure}: {exc}")
    raise RuntimeError(f"No common converged structure for {outcome}: {'; '.join(errors)}")


def fit_estimation_model(
    data: pd.DataFrame,
    formula: str,
    structure: str,
    outcome: str,
    role: str,
    log_rows: list[dict[str, Any]],
) -> tuple[Any, str]:
    result = fit_mixed_once(data, formula, structure, True, outcome, role, log_rows)
    if bool(result.converged):
        return result, "REML"
    return (
        fit_mixed_once(data, formula, structure, False, outcome, f"{role}_ml_fallback", log_rows),
        "ML_fallback",
    )


def prompt_from_term(term: str) -> str:
    if "codebook_guided" in term:
        return "codebook_guided"
    if "few_shot" in term:
        return "few_shot"
    return "zero_shot"


def classify_term(term: str) -> tuple[str, str, str, str]:
    interaction = ":" in term
    prompt = prompt_from_term(term)
    if term == "Intercept":
        return "intercept", "Intercept", "Reference expected performance", prompt
    if "prompt_strategy" in term and not interaction:
        return (
            "prompt_main",
            "Prompt strategy",
            f"{PROMPT_DISPLAY[prompt]} minus zero-shot at reference conditions",
            prompt,
        )
    if "label_number_factor" in term:
        levels = re.findall(r"\[T\.([^\]]+)\]", term)
        label = levels[-1] if levels else ""
        if interaction:
            return (
                "prompt_interaction",
                "Label number",
                f"{PROMPT_DISPLAY[prompt]}-zero-shot difference: {label} versus 3 labels",
                prompt,
            )
        return (
            "complexity_main",
            "Label number",
            f"Zero-shot difference: {label} versus 3 labels",
            prompt,
        )
    for variable in CONTINUOUS_TERMS:
        if variable in term:
            display = DIMENSION_DISPLAY[variable]
            if interaction:
                return (
                    "prompt_interaction",
                    display,
                    f"Difference in {display.lower()} slope: {PROMPT_DISPLAY[prompt]} minus zero-shot",
                    prompt,
                )
            return (
                "complexity_main",
                display,
                f"Zero-shot slope for {display.lower()} (per 1 SD)",
                prompt,
            )
    return "other", "Other", term, prompt


def fixed_effect_table(
    result: Any,
    outcome: str,
    structure: str,
    estimation: str,
    analysis: str,
) -> pd.DataFrame:
    params = result.fe_params
    bse = result.bse_fe
    ci = result.conf_int().loc[params.index]
    rows = []
    for term in params.index:
        role, dimension, description, prompt = classify_term(term)
        rows.append(
            {
                "analysis": analysis,
                "outcome": outcome,
                "random_effects_structure": structure,
                "estimation": estimation,
                "term": term,
                "term_role": role,
                "dimension": dimension,
                "prompt_contrast_numerator": prompt,
                "description": description,
                "estimate": float(params[term]),
                "standard_error": float(bse[term]),
                "z_value": float(params[term] / bse[term]),
                "p_value": float(result.pvalues[term]),
                "ci95_low": float(ci.loc[term, 0]),
                "ci95_high": float(ci.loc[term, 1]),
            }
        )
    return pd.DataFrame(rows)


def matching_terms(names: list[str], test_name: str) -> list[str]:
    interactions = [name for name in names if ":" in name and "prompt_strategy" in name]
    if test_name == "all_prompt_complexity_interactions":
        return interactions
    if test_name == "prompt_main":
        return [name for name in names if "prompt_strategy" in name and ":" not in name]
    if test_name == "label_number_main":
        return [name for name in names if "label_number_factor" in name and ":" not in name]
    if test_name == "prompt_x_label_number":
        return [name for name in interactions if "label_number_factor" in name]
    if test_name.endswith("_main"):
        variable = test_name.removesuffix("_main")
        return [name for name in names if name == variable]
    if test_name.startswith("prompt_x_"):
        variable = test_name.removeprefix("prompt_x_")
        return [name for name in interactions if variable in name]
    return []


def wald_test_rows(result: Any, outcome: str, analysis: str) -> pd.DataFrame:
    names = list(result.params.index)
    tests = [
        "prompt_main",
        "label_number_main",
        "annotation_disagreement_z_main",
        "semantic_similarity_z_main",
        "all_prompt_complexity_interactions",
        "prompt_x_label_number",
        "prompt_x_annotation_disagreement_z",
        "prompt_x_semantic_similarity_z",
    ]
    rows = []
    for test_name in tests:
        terms = matching_terms(names, test_name)
        if not terms:
            continue
        restriction = np.zeros((len(terms), len(names)))
        for index, term in enumerate(terms):
            restriction[index, names.index(term)] = 1.0
        test = result.wald_test(restriction, scalar=True)
        rows.append(
            {
                "analysis": analysis,
                "outcome": outcome,
                "test": test_name,
                "tested_terms": " | ".join(terms),
                "df": len(terms),
                "chi_square": float(np.asarray(test.statistic).squeeze()),
                "p_value": float(np.asarray(test.pvalue).squeeze()),
            }
        )
    return pd.DataFrame(rows)


def variance_and_r2(result: Any, outcome: str, structure: str, estimation: str) -> dict[str, Any]:
    fixed_prediction = np.asarray(result.model.exog @ result.fe_params, dtype=float)
    fixed_variance = float(np.var(fixed_prediction, ddof=0))
    combination_variance = float(result.cov_re.iloc[0, 0]) if result.cov_re.size else 0.0
    cell_variance = float(np.sum(result.vcomp)) if np.size(result.vcomp) else 0.0
    residual_variance = float(result.scale)
    total = fixed_variance + combination_variance + cell_variance + residual_variance
    return {
        "outcome": outcome,
        "random_effects_structure": structure,
        "estimation": estimation,
        "fixed_prediction_variance": fixed_variance,
        "combination_random_intercept_variance": combination_variance,
        "prompt_cell_random_intercept_variance": cell_variance,
        "residual_variance": residual_variance,
        "marginal_r2": fixed_variance / total,
        "conditional_r2": (fixed_variance + combination_variance + cell_variance) / total,
        "combination_icc": combination_variance / total,
        "prompt_cell_icc": cell_variance / total,
    }


def residual_diagnostics(
    result: Any,
    data: pd.DataFrame,
    outcome: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    from scipy import stats

    fixed = np.asarray(result.model.exog @ result.fe_params, dtype=float)
    try:
        conditional = np.asarray(result.fittedvalues, dtype=float)
        fitted_type = "conditional"
    except Exception:
        conditional = fixed
        fitted_type = "fixed_only_fallback"
    observed = data[outcome].to_numpy(dtype=float)
    residual = observed - conditional
    shapiro_w, shapiro_p = stats.shapiro(residual)
    summary = {
        "outcome": outcome,
        "fitted_value_type": fitted_type,
        "observation_count": len(residual),
        "residual_mean": float(np.mean(residual)),
        "residual_sd": float(np.std(residual, ddof=1)),
        "residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "residual_skewness": float(stats.skew(residual, bias=False)),
        "residual_excess_kurtosis": float(stats.kurtosis(residual, bias=False)),
        "shapiro_w": float(shapiro_w),
        "shapiro_p": float(shapiro_p),
        "fixed_predictions_below_zero": int((fixed < 0).sum()),
        "fixed_predictions_above_one": int((fixed > 1).sum()),
    }
    observations = data[
        ["combination_id", "prompt_strategy", "repeat_id", "label_number_factor"]
    ].copy()
    observations["outcome"] = outcome
    observations["observed"] = observed
    observations["fixed_prediction"] = fixed
    observations["conditional_prediction"] = conditional
    observations["conditional_residual"] = residual
    return summary, observations


def standardization_lookup(parameters: pd.DataFrame) -> dict[str, dict[str, float]]:
    subset = parameters[parameters["dataset"] == "llm_combination_reference"]
    return {
        str(row.standardized_variable): {
            "mean": float(row.mean),
            "std": float(row.std_ddof0),
            "minimum": float(row.minimum),
            "maximum": float(row.maximum),
        }
        for row in subset.itertuples(index=False)
    }


def marginal_prediction_grid(
    result: Any,
    data: pd.DataFrame,
    parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from patsy import build_design_matrices

    lookup = standardization_lookup(parameters)
    reference = data.drop_duplicates("combination_id")
    empirical_profiles = reference[["label_number_factor", *CONTINUOUS_TERMS]].copy()
    names = list(result.fe_params.index)
    beta = result.fe_params.reindex(names).to_numpy(dtype=float)
    covariance = result.cov_params().loc[names, names].to_numpy(dtype=float)

    def design_mean(frame: pd.DataFrame) -> np.ndarray:
        matrix = build_design_matrices(
            [result.model.data.design_info], frame, return_type="dataframe"
        )[0].reindex(columns=names, fill_value=0.0)
        return matrix.to_numpy(dtype=float).mean(axis=0)

    curve_rows = []
    design_cache: dict[tuple[str, float, str], np.ndarray] = {}
    for dimension in ["label_number", *CONTINUOUS_TERMS]:
        if dimension == "label_number":
            points = [(float(level), float(level), level) for level in LABEL_LEVELS]
        else:
            source = CONTINUOUS_SOURCE[dimension]
            low, high = reference[source].quantile([0.05, 0.95]).tolist()
            originals = np.linspace(float(low), float(high), 21)
            points = [
                (
                    float(value),
                    (float(value) - lookup[dimension]["mean"]) / lookup[dimension]["std"],
                    None,
                )
                for value in originals
            ]
        for prompt in PROMPT_ORDER:
            for original, standardized, label_level in points:
                profiles = empirical_profiles.copy()
                profiles["prompt_strategy"] = prompt
                if dimension == "label_number":
                    profiles["label_number_factor"] = label_level
                else:
                    profiles[dimension] = standardized
                xbar = design_mean(profiles)
                prediction = float(xbar @ beta)
                se = float(np.sqrt(max(xbar @ covariance @ xbar, 0.0)))
                design_cache[(dimension, original, prompt)] = xbar
                curve_rows.append(
                    {
                        "prediction_method": "marginal_standardization_over_40_observed_label_combinations",
                        "dimension": dimension,
                        "dimension_display": DIMENSION_DISPLAY[dimension],
                        "x_original": original,
                        "x_standardized": standardized,
                        "prompt_strategy": prompt,
                        "prompt_display": PROMPT_DISPLAY[prompt],
                        "predicted_performance": prediction,
                        "ci95_low": prediction - 1.96 * se,
                        "ci95_high": prediction + 1.96 * se,
                    }
                )
    curves = pd.DataFrame(curve_rows)

    contrast_rows = []
    for (dimension, x_original), _ in curves.groupby(["dimension", "x_original"], sort=False):
        for contrast, numerator, denominator in CONTRASTS:
            difference_design = (
                design_cache[(dimension, x_original, numerator)]
                - design_cache[(dimension, x_original, denominator)]
            )
            estimate = float(difference_design @ beta)
            se = float(np.sqrt(max(difference_design @ covariance @ difference_design, 0.0)))
            contrast_rows.append(
                {
                    "dimension": dimension,
                    "dimension_display": DIMENSION_DISPLAY[dimension],
                    "x_original": x_original,
                    "contrast": contrast,
                    "contrast_display": CONTRAST_DISPLAY[contrast],
                    "estimated_difference": estimate,
                    "standard_error": se,
                    "ci95_low": estimate - 1.96 * se,
                    "ci95_high": estimate + 1.96 * se,
                }
            )
    contrasts = pd.DataFrame(contrast_rows)

    overall_rows = []
    overall_designs = {}
    for prompt in PROMPT_ORDER:
        profiles = empirical_profiles.copy()
        profiles["prompt_strategy"] = prompt
        xbar = design_mean(profiles)
        overall_designs[prompt] = xbar
        estimate = float(xbar @ beta)
        se = float(np.sqrt(max(xbar @ covariance @ xbar, 0.0)))
        overall_rows.append(
            {
                "prompt_strategy": prompt,
                "prompt_display": PROMPT_DISPLAY[prompt],
                "adjusted_mean": estimate,
                "standard_error": se,
                "ci95_low": estimate - 1.96 * se,
                "ci95_high": estimate + 1.96 * se,
            }
        )
    for contrast, numerator, denominator in CONTRASTS:
        xdiff = overall_designs[numerator] - overall_designs[denominator]
        estimate = float(xdiff @ beta)
        se = float(np.sqrt(max(xdiff @ covariance @ xdiff, 0.0)))
        overall_rows.append(
            {
                "prompt_strategy": contrast,
                "prompt_display": CONTRAST_DISPLAY[contrast],
                "adjusted_mean": estimate,
                "standard_error": se,
                "ci95_low": estimate - 1.96 * se,
                "ci95_high": estimate + 1.96 * se,
            }
        )
    return curves, contrasts, pd.DataFrame(overall_rows)


def aggregate_combination_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = data.drop_duplicates("combination_id")[
        ["combination_id", "label_number_factor", *CONTINUOUS_TERMS]
    ]
    prompt_means = (
        data.groupby(["combination_id", "prompt_strategy"], as_index=False)[OUTCOMES]
        .mean()
    )
    task_mean = data.groupby("combination_id", as_index=False)[OUTCOMES].mean()
    task_mean = metadata.merge(task_mean, on="combination_id", validate="one_to_one")
    pivot = prompt_means.pivot(index="combination_id", columns="prompt_strategy", values=OUTCOMES)
    pivot.columns = [f"{outcome}__{prompt}" for outcome, prompt in pivot.columns]
    pivot = pivot.reset_index()
    deltas = metadata.merge(pivot, on="combination_id", validate="one_to_one")
    for outcome in OUTCOMES:
        for contrast, numerator, denominator in CONTRASTS:
            deltas[f"{contrast}__{outcome}"] = (
                deltas[f"{outcome}__{numerator}"] - deltas[f"{outcome}__{denominator}"]
            )
    return task_mean, deltas


def hc3_complexity_models(
    frame: pd.DataFrame,
    targets: list[tuple[str, str, str]],
    analysis: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.formula.api as smf

    coefficient_rows = []
    test_rows = []
    for target, outcome, model_role in targets:
        formula = f"{target} ~ {complexity_expression()}"
        result = smf.ols(formula, frame).fit(cov_type="HC3", use_t=True)
        ci = result.conf_int()
        for term in result.params.index:
            if term == "Intercept":
                dimension = "Intercept"
                description = f"{model_role} at 3 labels and mean continuous complexity"
            elif "label_number_factor" in term:
                level = re.findall(r"\[T\.([^\]]+)\]", term)[-1]
                dimension = "Label number"
                description = f"{model_role}: {level} versus 3 labels"
            else:
                dimension = DIMENSION_DISPLAY.get(term, term)
                description = f"{model_role} slope for {dimension.lower()} (per 1 SD)"
            coefficient_rows.append(
                {
                    "analysis": analysis,
                    "outcome": outcome,
                    "model_role": model_role,
                    "term": term,
                    "dimension": dimension,
                    "description": description,
                    "estimate": float(result.params[term]),
                    "hc3_standard_error": float(result.bse[term]),
                    "t_value": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                    "ci95_low": float(ci.loc[term, 0]),
                    "ci95_high": float(ci.loc[term, 1]),
                    "combination_count": len(frame),
                    "r_squared": float(result.rsquared),
                    "adjusted_r_squared": float(result.rsquared_adj),
                }
            )
        names = list(result.params.index)
        groups = {
            "all_complexity_terms": [name for name in names if name != "Intercept"],
            "label_number": [name for name in names if "label_number_factor" in name],
            "annotation_disagreement_z": [name for name in names if name == "annotation_disagreement_z"],
            "semantic_similarity_z": [name for name in names if name == "semantic_similarity_z"],
        }
        for test_name, terms in groups.items():
            if not terms:
                continue
            restriction = np.zeros((len(terms), len(names)))
            for index, term in enumerate(terms):
                restriction[index, names.index(term)] = 1.0
            test = result.wald_test(restriction, use_f=True, scalar=True)
            test_rows.append(
                {
                    "analysis": analysis,
                    "outcome": outcome,
                    "model_role": model_role,
                    "test": test_name,
                    "tested_terms": " | ".join(terms),
                    "numerator_df": len(terms),
                    "denominator_df": float(np.asarray(test.df_denom).squeeze()),
                    "f_statistic": float(np.asarray(test.statistic).squeeze()),
                    "p_value": float(np.asarray(test.pvalue).squeeze()),
                }
            )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(test_rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def repeat_stability_analysis(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from scipy import stats

    cell_sd = (
        data.groupby(["combination_id", "prompt_strategy"], as_index=False)[OUTCOMES]
        .std(ddof=1)
        .rename(columns={outcome: f"{outcome}_repeat_sd" for outcome in OUTCOMES})
    )
    friedman_rows = []
    pairwise_rows = []
    for outcome in OUTCOMES:
        value = f"{outcome}_repeat_sd"
        pivot = cell_sd.pivot(index="combination_id", columns="prompt_strategy", values=value)
        statistic, p_value = stats.friedmanchisquare(
            *[pivot[prompt].to_numpy(dtype=float) for prompt in PROMPT_ORDER]
        )
        friedman_rows.append(
            {
                "outcome": outcome,
                "combination_count": len(pivot),
                "friedman_chi_square": float(statistic),
                "df": 2,
                "p_value": float(p_value),
            }
        )
        metric_rows = []
        for contrast, numerator, denominator in CONTRASTS:
            difference = pivot[numerator] - pivot[denominator]
            try:
                test = stats.wilcoxon(
                    pivot[numerator], pivot[denominator], zero_method="wilcox", method="auto"
                )
                raw_p = float(test.pvalue)
                test_stat = float(test.statistic)
            except ValueError:
                raw_p = 1.0
                test_stat = 0.0
            metric_rows.append(
                {
                    "outcome": outcome,
                    "contrast": contrast,
                    "contrast_display": CONTRAST_DISPLAY[contrast],
                    "combination_count": len(pivot),
                    "mean_sd_difference": float(difference.mean()),
                    "median_sd_difference": float(difference.median()),
                    "positive_difference_share": float((difference > 0).mean()),
                    "wilcoxon_statistic": test_stat,
                    "raw_p_value": raw_p,
                }
            )
        adjusted = holm_adjust([row["raw_p_value"] for row in metric_rows])
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["holm_adjusted_p_value"] = adjusted_p
            pairwise_rows.append(row)
    return cell_sd, pd.DataFrame(friedman_rows), pd.DataFrame(pairwise_rows)


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


def plot_interaction_forest(plt: Any, coefficients: pd.DataFrame, figure_dir: Path) -> None:
    frame = coefficients[
        (coefficients["analysis"] == "primary_all_labels")
        & (coefficients["outcome"] == "kappa")
        & (coefficients["term_role"] == "prompt_interaction")
    ].copy()
    frame["plot_label"] = frame["description"].str.replace(" difference: ", ": ", regex=False)
    frame["plot_label"] = frame["plot_label"].str.replace(
        "Difference in ", "", regex=False
    ).str.replace(" slope: ", " slope: ", regex=False)
    frame["plot_label"] = frame["plot_label"].str.replace(
        "11 versus 3 labels", "11 versus 3 labels (exploratory)", regex=False
    )
    frame = frame.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for index, row in frame.iterrows():
        prompt = row["prompt_contrast_numerator"]
        color = "#B08A3E" if "exploratory" in row["plot_label"] else PROMPT_COLORS[prompt]
        ax.plot([row["ci95_low"], row["ci95_high"]], [index, index], color=color, linewidth=1.4)
        ax.scatter(row["estimate"], index, color=color, s=25, zorder=3)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels(frame["plot_label"])
    ax.set_xlabel("Prompt-complexity interaction coefficient")
    ax.set_title("Task complexity moderates prompt-strategy gains")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, figure_dir, "figure9_llm_prompt_interaction_forest")
    plt.close(fig)


def plot_adjusted_performance(plt: Any, curves: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    specs = [
        ("label_number", "Number of labels", "A"),
        ("annotation_disagreement_z", "Annotation disagreement rate", "B"),
        ("semantic_similarity_z", "Mean semantic similarity", "C"),
    ]
    for ax, (dimension, xlabel, panel) in zip(axes, specs):
        subset = curves[curves["dimension"] == dimension]
        for prompt in PROMPT_ORDER:
            group = subset[subset["prompt_strategy"] == prompt].sort_values("x_original")
            x = group["x_original"].to_numpy(dtype=float)
            mean = group["predicted_performance"].to_numpy(dtype=float)
            low = group["ci95_low"].to_numpy(dtype=float)
            high = group["ci95_high"].to_numpy(dtype=float)
            ax.plot(
                x,
                mean,
                marker="o" if len(x) <= 10 else None,
                linewidth=1.3,
                color=PROMPT_COLORS[prompt],
                label=PROMPT_DISPLAY[prompt],
            )
            ax.fill_between(x, low, high, color=PROMPT_COLORS[prompt], alpha=0.10, linewidth=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Adjusted predicted kappa")
        ax.set_ylim(0, 0.85)
        ax.set_title(DIMENSION_DISPLAY[dimension])
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        panel_label(ax, panel)
        if dimension == "label_number":
            ax.set_xticks([3, 5, 8, 11])
            ax.set_xticklabels(["3", "5", "8", "11*"])
            ax.text(0.98, 0.02, "* exploratory", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#6F6F6F")
    axes[0].legend(fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    save_figure(fig, figure_dir, "figure10_llm_adjusted_prompt_performance")
    plt.close(fig)


def plot_prompt_gains(plt: Any, contrasts: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    specs = [
        ("label_number", "Number of labels", "A"),
        ("annotation_disagreement_z", "Annotation disagreement rate", "B"),
        ("semantic_similarity_z", "Mean semantic similarity", "C"),
    ]
    for ax, (dimension, xlabel, panel) in zip(axes, specs):
        subset = contrasts[contrasts["dimension"] == dimension]
        for contrast, _, _ in CONTRASTS:
            group = subset[subset["contrast"] == contrast].sort_values("x_original")
            x_values = group["x_original"].to_numpy(dtype=float)
            ax.plot(
                x_values,
                group["estimated_difference"],
                marker="o" if len(group) <= 10 else None,
                linewidth=1.3,
                color=CONTRAST_COLORS[contrast],
                label=CONTRAST_DISPLAY[contrast],
            )
            ax.fill_between(
                x_values,
                group["ci95_low"].to_numpy(dtype=float),
                group["ci95_high"].to_numpy(dtype=float),
                color=CONTRAST_COLORS[contrast],
                alpha=0.08,
                linewidth=0,
            )
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Adjusted performance difference")
        ax.set_title(DIMENSION_DISPLAY[dimension])
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        panel_label(ax, panel)
        if dimension == "label_number":
            ax.set_xticks([3, 5, 8, 11])
            ax.set_xticklabels(["3", "5", "8", "11*"])
            ax.text(0.98, 0.02, "* exploratory", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#6F6F6F")
    axes[0].legend(fontsize=6.2)
    fig.tight_layout(w_pad=1.0)
    save_figure(fig, figure_dir, "figure11_llm_adjusted_prompt_gains")
    plt.close(fig)


def plot_reliability_diagnostics(
    plt: Any,
    overall: pd.DataFrame,
    cell_sd: pd.DataFrame,
    residuals: pd.DataFrame,
    figure_dir: Path,
    seed: int,
) -> None:
    from scipy import stats

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    means = overall[overall["prompt_strategy"].isin(PROMPT_ORDER)].set_index("prompt_strategy").reindex(PROMPT_ORDER)
    x = np.arange(3)
    for index, prompt in enumerate(PROMPT_ORDER):
        row = means.loc[prompt]
        axes[0].errorbar(
            index,
            row["adjusted_mean"],
            yerr=np.asarray([[row["adjusted_mean"] - row["ci95_low"]], [row["ci95_high"] - row["adjusted_mean"]]]),
            fmt="o",
            markersize=6,
            capsize=3,
            color=PROMPT_COLORS[prompt],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([PROMPT_DISPLAY[prompt] for prompt in PROMPT_ORDER], rotation=18, ha="right")
    axes[0].set_ylabel("Adjusted mean kappa")
    axes[0].set_ylim(0.25, 0.65)
    axes[0].set_title("Adjusted prompt performance")
    axes[0].grid(axis="y", color="#E6E6E6", linewidth=0.6)
    panel_label(axes[0], "A")

    rng = np.random.default_rng(seed)
    box_data = [
        cell_sd.loc[cell_sd["prompt_strategy"] == prompt, "kappa_repeat_sd"].to_numpy(dtype=float)
        for prompt in PROMPT_ORDER
    ]
    box = axes[1].boxplot(box_data, positions=x, widths=0.55, patch_artist=True, showfliers=False)
    for index, (prompt, values) in enumerate(zip(PROMPT_ORDER, box_data)):
        box["boxes"][index].set_facecolor(PROMPT_COLORS[prompt])
        box["boxes"][index].set_alpha(0.55)
        axes[1].scatter(
            np.full(len(values), index) + rng.normal(0, 0.05, len(values)),
            values,
            s=9,
            alpha=0.45,
            color=PROMPT_COLORS[prompt],
            edgecolors="none",
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([PROMPT_DISPLAY[prompt] for prompt in PROMPT_ORDER], rotation=18, ha="right")
    axes[1].set_ylabel("Kappa SD across three calls")
    axes[1].set_title("Repeat-call variability")
    axes[1].grid(axis="y", color="#E6E6E6", linewidth=0.6)
    panel_label(axes[1], "B")

    frame = residuals[residuals["outcome"] == "kappa"]
    theoretical, ordered = stats.probplot(frame["conditional_residual"], dist="norm", fit=False)
    axes[2].scatter(theoretical, ordered, s=10, alpha=0.50, color="#6F7A80")
    slope, intercept, _ = stats.probplot(frame["conditional_residual"], dist="norm", fit=True)[1]
    line_x = np.asarray([min(theoretical), max(theoretical)])
    axes[2].plot(line_x, intercept + slope * line_x, color="#555555", linewidth=0.8)
    axes[2].set_xlabel("Normal theoretical quantile")
    axes[2].set_ylabel("Residual quantile")
    axes[2].set_title("Mixed-model residual Q-Q plot")
    axes[2].grid(color="#E6E6E6", linewidth=0.6)
    panel_label(axes[2], "C")
    fig.tight_layout(w_pad=1.0)
    save_figure(fig, figure_dir, "figure12_llm_reliability_and_diagnostics")
    plt.close(fig)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    from scipy.stats import chi2

    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    prediction_dir = output_dir / "predictions"
    diagnostic_dir = output_dir / "diagnostics"
    figure_dir = output_dir / "figures"
    for directory in [table_dir, prediction_dir, diagnostic_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    raw_data, parameters = load_inputs(input_dir)
    data, support = prepare_data(raw_data)
    convergence_rows: list[dict[str, Any]] = []
    fixed_tables = []
    wald_tables = []
    fit_rows = []
    variance_rows = []
    residual_rows = []
    residual_observation_tables = []
    estimation_results: dict[str, Any] = {}

    for outcome in OUTCOMES:
        additive_formula, interaction_formula = model_formulas(outcome)
        additive_ml, interaction_ml, structure = fit_ml_pair(
            data,
            additive_formula,
            interaction_formula,
            outcome,
            convergence_rows,
        )
        estimation, estimation_method = fit_estimation_model(
            data,
            interaction_formula,
            structure,
            outcome,
            "interaction_reml",
            convergence_rows,
        )
        estimation_results[outcome] = estimation
        df_difference = len(interaction_ml.fe_params) - len(additive_ml.fe_params)
        lr_statistic = max(0.0, 2 * (interaction_ml.llf - additive_ml.llf))
        fit_rows.extend(
            [
                {
                    "outcome": outcome,
                    "model": "additive",
                    "random_effects_structure": structure,
                    "fixed_effect_parameters": len(additive_ml.fe_params),
                    "log_likelihood": additive_ml.llf,
                    "aic": additive_ml.aic,
                    "bic": additive_ml.bic,
                    "converged": additive_ml.converged,
                    "interaction_lrt_chi_square": np.nan,
                    "interaction_lrt_df": np.nan,
                    "interaction_lrt_p": np.nan,
                },
                {
                    "outcome": outcome,
                    "model": "prompt_by_complexity_interactions",
                    "random_effects_structure": structure,
                    "fixed_effect_parameters": len(interaction_ml.fe_params),
                    "log_likelihood": interaction_ml.llf,
                    "aic": interaction_ml.aic,
                    "bic": interaction_ml.bic,
                    "converged": interaction_ml.converged,
                    "interaction_lrt_chi_square": lr_statistic,
                    "interaction_lrt_df": df_difference,
                    "interaction_lrt_p": float(chi2.sf(lr_statistic, df_difference)),
                },
            ]
        )
        fixed_tables.append(
            fixed_effect_table(
                estimation,
                outcome,
                structure,
                estimation_method,
                "primary_all_labels",
            )
        )
        wald_tables.append(wald_test_rows(estimation, outcome, "primary_all_labels"))
        variance_rows.append(variance_and_r2(estimation, outcome, structure, estimation_method))
        residual_summary, residual_observations = residual_diagnostics(estimation, data, outcome)
        residual_rows.append(residual_summary)
        residual_observation_tables.append(residual_observations)

    sensitivity_data = data[data["label_number_factor"] != "11"].copy()
    _, sensitivity_formula = model_formulas("kappa")
    sensitivity_result, sensitivity_method = fit_estimation_model(
        sensitivity_data,
        sensitivity_formula,
        "combination_plus_prompt_cell",
        "kappa",
        "exclude_k11_reml",
        convergence_rows,
    )
    fixed_tables.append(
        fixed_effect_table(
            sensitivity_result,
            "kappa",
            "combination_plus_prompt_cell",
            sensitivity_method,
            "sensitivity_excluding_11_labels",
        )
    )
    wald_tables.append(
        wald_test_rows(
            sensitivity_result, "kappa", "sensitivity_excluding_11_labels"
        )
    )

    fixed_effects = pd.concat(fixed_tables, ignore_index=True)
    wald_tests = pd.concat(wald_tables, ignore_index=True)
    fit_comparison = pd.DataFrame(fit_rows)
    variance_r2 = pd.DataFrame(variance_rows)
    residual_summary = pd.DataFrame(residual_rows)
    residual_observations = pd.concat(residual_observation_tables, ignore_index=True)
    convergence = pd.DataFrame(convergence_rows)

    task_mean, prompt_deltas = aggregate_combination_data(data)
    task_targets = [(outcome, outcome, "Average performance across prompt strategies") for outcome in OUTCOMES]
    task_mean_coefficients, task_mean_tests = hc3_complexity_models(
        task_mean, task_targets, "primary_all_labels"
    )
    task_mean_no11_coefficients, task_mean_no11_tests = hc3_complexity_models(
        task_mean[task_mean["label_number_factor"] != "11"].copy(),
        task_targets,
        "sensitivity_excluding_11_labels",
    )
    task_mean_coefficients = pd.concat(
        [task_mean_coefficients, task_mean_no11_coefficients], ignore_index=True
    )
    task_mean_tests = pd.concat([task_mean_tests, task_mean_no11_tests], ignore_index=True)

    delta_targets = []
    for outcome in OUTCOMES:
        for contrast, _, _ in CONTRASTS:
            delta_targets.append(
                (f"{contrast}__{outcome}", outcome, CONTRAST_DISPLAY[contrast])
            )
    delta_coefficients, delta_tests = hc3_complexity_models(
        prompt_deltas, delta_targets, "primary_all_labels"
    )
    delta_no11_coefficients, delta_no11_tests = hc3_complexity_models(
        prompt_deltas[prompt_deltas["label_number_factor"] != "11"].copy(),
        delta_targets,
        "sensitivity_excluding_11_labels",
    )
    delta_coefficients = pd.concat(
        [delta_coefficients, delta_no11_coefficients], ignore_index=True
    )
    delta_tests = pd.concat([delta_tests, delta_no11_tests], ignore_index=True)

    cell_sd, stability_friedman, stability_pairwise = repeat_stability_analysis(data)
    curves, contrast_curves, overall_prompt = marginal_prediction_grid(
        estimation_results["kappa"], data, parameters
    )

    primary_interactions = fixed_effects[
        (fixed_effects["analysis"] == "primary_all_labels")
        & (fixed_effects["term_role"] == "prompt_interaction")
    ].copy()
    robustness = primary_interactions.pivot_table(
        index=["dimension", "description"],
        columns="outcome",
        values=["estimate", "p_value"],
        aggfunc="first",
    )
    robustness.columns = [f"{metric}_{outcome}" for metric, outcome in robustness.columns]
    robustness = robustness.reset_index()
    robustness["same_sign_all_metrics"] = (
        np.sign(robustness["estimate_kappa"]) == np.sign(robustness["estimate_macro_f1"])
    ) & (
        np.sign(robustness["estimate_kappa"]) == np.sign(robustness["estimate_mcc"])
    )
    significant_interactions = primary_interactions[
        primary_interactions["p_value"] < 0.05
    ][["outcome", "dimension", "description", "estimate", "p_value", "ci95_low", "ci95_high"]]

    save_csv(support, table_dir / "llm_design_support.csv")
    save_csv(fixed_effects, table_dir / "llm_mixed_model_fixed_effects.csv")
    save_csv(wald_tests, table_dir / "llm_mixed_model_wald_tests.csv")
    save_csv(fit_comparison, table_dir / "llm_mixed_model_fit_comparison.csv")
    save_csv(variance_r2, table_dir / "llm_mixed_model_variance_and_r2.csv")
    save_csv(task_mean_coefficients, table_dir / "llm_task_mean_hc3_coefficients.csv")
    save_csv(task_mean_tests, table_dir / "llm_task_mean_hc3_tests.csv")
    save_csv(delta_coefficients, table_dir / "llm_prompt_gain_hc3_coefficients.csv")
    save_csv(delta_tests, table_dir / "llm_prompt_gain_hc3_tests.csv")
    save_csv(stability_friedman, table_dir / "llm_repeat_stability_friedman.csv")
    save_csv(stability_pairwise, table_dir / "llm_repeat_stability_pairwise_wilcoxon.csv")
    save_csv(robustness, table_dir / "llm_interaction_metric_robustness.csv")
    save_csv(significant_interactions, table_dir / "significant_prompt_complexity_interactions.csv")
    save_csv(curves, prediction_dir / "llm_adjusted_performance_grid.csv")
    save_csv(contrast_curves, prediction_dir / "llm_adjusted_prompt_gain_grid.csv")
    save_csv(overall_prompt, prediction_dir / "llm_adjusted_overall_prompt_effects.csv")
    save_csv(cell_sd, diagnostic_dir / "llm_prompt_cell_repeat_sd.csv")
    save_csv(convergence, diagnostic_dir / "llm_model_convergence_log.csv")
    save_csv(residual_summary, diagnostic_dir / "llm_residual_diagnostics.csv")
    save_csv(residual_observations, diagnostic_dir / "llm_residual_observations.csv")

    figure_status = "skipped"
    if not args.skip_figures:
        plt = configure_matplotlib()
        plot_interaction_forest(plt, fixed_effects, figure_dir)
        plot_adjusted_performance(plt, curves, figure_dir)
        plot_prompt_gains(plt, contrast_curves, figure_dir)
        plot_reliability_diagnostics(
            plt,
            overall_prompt,
            cell_sd,
            residual_observations,
            figure_dir,
            args.seed,
        )
        figure_status = "rendered"

    lrt = (
        fit_comparison[fit_comparison["model"] == "prompt_by_complexity_interactions"]
        .set_index("outcome")["interaction_lrt_p"]
        .to_dict()
    )
    overall_lookup = overall_prompt.set_index("prompt_strategy")["adjusted_mean"].to_dict()
    report = {
        "generated_at_utc": utc_now(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "primary_outcome": "kappa",
        "robustness_outcomes": ["macro_f1", "mcc"],
        "fixed_effects": (
            "prompt strategy * (categorical label number + standardized annotation disagreement + "
            "standardized semantic similarity)"
        ),
        "preferred_random_effects": (
            "label-combination intercept plus nested prompt-by-combination cell intercept"
        ),
        "adjusted_prediction_method": (
            "Marginal standardization over the 40 observed label-combination profiles."
        ),
        "analysis_rows": len(data),
        "combination_count": data["combination_id"].nunique(),
        "prompt_cell_count": data["prompt_combination_id"].nunique(),
        "repeat_count_per_prompt_cell": 3,
        "interaction_lrt_p_values": lrt,
        "adjusted_kappa_means_and_contrasts": overall_lookup,
        "significant_interaction_count": len(significant_interactions),
        "significant_interactions": significant_interactions.to_dict(orient="records"),
        "figure_status": figure_status,
        "figure_count": 4 if figure_status == "rendered" else 0,
        "output_checks": {
            "all_primary_models_converged": bool(fit_comparison["converged"].fillna(False).all()),
            "missing_primary_fixed_effect_cells": int(
                fixed_effects[fixed_effects["analysis"] == "primary_all_labels"]
                [["estimate", "standard_error", "p_value"]]
                .isna()
                .sum()
                .sum()
            ),
            "fixed_predictions_outside_unit_interval": int(
                residual_summary["fixed_predictions_below_zero"].sum()
                + residual_summary["fixed_predictions_above_one"].sum()
            ),
            "all_prompt_gain_hc3_coefficients_finite": bool(
                np.isfinite(delta_coefficients["estimate"]).all()
                and np.isfinite(delta_coefficients["hc3_standard_error"]).all()
            ),
        },
        "support_warning": (
            "The 11-label condition contains one label combination. Its main and prompt-interaction "
            "coefficients are exploratory; use the excluding-11 sensitivity results for replicated claims."
        ),
        "interpretation_boundary": (
            "Prompt-complexity interactions estimate how prompt gains change across the constructed "
            "task space. They are associations, not causal effects. Repeat-SD comparisons are based "
            "on three calls per prompt cell and should be treated as secondary reliability evidence."
        ),
    }
    write_json(output_dir / "step4_llm_prompt_report.json", report)
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    print(f"\nStep 4 outputs: {output_dir}")


if __name__ == "__main__":
    main()
