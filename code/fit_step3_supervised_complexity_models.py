from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTCOMES = ["kappa", "macro_f1", "mcc"]
MODEL_ORDER = ["SVM", "DeBERTa"]
LABEL_LEVELS = ["3", "5", "8", "11"]
CONTINUOUS_TERMS = [
    "class_imbalance_z",
    "annotation_disagreement_z",
    "log2_training_size_z",
    "semantic_similarity_z",
]
CONTINUOUS_SOURCE = {
    "class_imbalance_z": "class_imbalance_complexity",
    "annotation_disagreement_z": "annotation_disagreement_complexity",
    "log2_training_size_z": "log2_training_size",
    "semantic_similarity_z": "semantic_similarity_complexity",
}
DIMENSION_DISPLAY = {
    "label_number": "Label number",
    "class_imbalance_z": "Class imbalance",
    "annotation_disagreement_z": "Annotation disagreement",
    "log2_training_size_z": "Training sample size",
    "semantic_similarity_z": "Label semantic similarity",
}
OUTCOME_DISPLAY = {
    "kappa": "Cohen's kappa",
    "macro_f1": "Macro-F1",
    "mcc": "MCC",
}
MODEL_COLORS = {"SVM": "#4F6B5B", "DeBERTa": "#2F6B9A"}


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
    parser = argparse.ArgumentParser(
        description="Step 3: supervised-learning complexity-effect models"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], dataset: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "supervised": input_dir / "supervised_analysis_ready.csv",
        "pairs": input_dir / "supervised_task_pairs.csv",
        "parameters": input_dir / "standardization_parameters.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Step 1 files: {missing}")
    supervised = pd.read_csv(paths["supervised"], low_memory=False)
    pairs = pd.read_csv(paths["pairs"], low_memory=False)
    parameters = pd.read_csv(paths["parameters"], low_memory=False)
    return supervised, pairs, parameters


def prepare_data(
    supervised: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        "original_task_id",
        "combination_id",
        "model_name",
        "label_number_factor",
        *CONTINUOUS_TERMS,
        *OUTCOMES,
    ]
    require_columns(supervised, required, "supervised_analysis_ready")
    data = supervised.copy()
    model_map = {"TF-IDF + LinearSVC": "SVM", "DeBERTa-v3-base": "DeBERTa"}
    data["model_code"] = data["model_name"].map(model_map)
    data["label_number_factor"] = (
        pd.to_numeric(data["label_number_factor"], errors="raise").astype(int).astype(str)
    )
    numeric_columns = [*CONTINUOUS_TERMS, *OUTCOMES]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    invalid = data[["model_code", "label_number_factor", *numeric_columns]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"Supervised model data contain {int(invalid.sum())} incomplete rows")
    if set(data["model_code"].unique()) != set(MODEL_ORDER):
        raise ValueError(f"Unexpected supervised model names: {data['model_name'].unique().tolist()}")
    model_counts = data.groupby("original_task_id")["model_code"].nunique()
    if not model_counts.eq(2).all():
        raise ValueError("Every supervised task must contain both SVM and DeBERTa")
    data = data.sort_values(["original_task_id", "model_code"]).reset_index(drop=True)

    task_reference = data.drop_duplicates("original_task_id").copy()
    support = (
        task_reference.groupby("label_number_factor", sort=False)
        .agg(
            task_count=("original_task_id", "nunique"),
            combination_count=("combination_id", "nunique"),
            minimum_training_size=("training_sample_size", "min"),
            maximum_training_size=("training_sample_size", "max"),
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

    pair_required = [
        "original_task_id",
        "combination_id",
        "kappa__DeBERTa-v3-base",
        "kappa__TF-IDF + LinearSVC",
        "macro_f1__DeBERTa-v3-base",
        "macro_f1__TF-IDF + LinearSVC",
        "mcc__DeBERTa-v3-base",
        "mcc__TF-IDF + LinearSVC",
    ]
    require_columns(pairs, pair_required, "supervised_task_pairs")
    pair_data = pairs.copy()
    metadata = task_reference[
        [
            "original_task_id",
            "label_number_factor",
            *CONTINUOUS_TERMS,
        ]
    ]
    pair_data = pair_data.drop(
        columns=[column for column in ["label_number_factor", *CONTINUOUS_TERMS] if column in pair_data],
        errors="ignore",
    ).merge(metadata, on="original_task_id", how="left", validate="one_to_one")
    for outcome in OUTCOMES:
        pair_data[f"delta_{outcome}"] = (
            pd.to_numeric(pair_data[f"{outcome}__DeBERTa-v3-base"], errors="coerce")
            - pd.to_numeric(pair_data[f"{outcome}__TF-IDF + LinearSVC"], errors="coerce")
        )
    if pair_data[["label_number_factor", *CONTINUOUS_TERMS, *[f"delta_{x}" for x in OUTCOMES]]].isna().any().any():
        raise ValueError("Paired-difference data contain missing analysis values")
    return data, pair_data, support


def complexity_expression(label_levels: list[str]) -> str:
    del label_levels
    return (
        "C(label_number_factor, Treatment(reference='3'))"
        " + class_imbalance_z"
        " + annotation_disagreement_z"
        " + log2_training_size_z"
        " + semantic_similarity_z"
    )


def model_formulas(outcome: str, label_levels: list[str]) -> tuple[str, str]:
    complexity = complexity_expression(label_levels)
    model = "C(model_code, Treatment(reference='SVM'))"
    additive = f"{outcome} ~ {model} + {complexity}"
    interaction = f"{outcome} ~ {model} * ({complexity})"
    return additive, interaction


def structure_specification(name: str) -> dict[str, Any]:
    if name == "combination_plus_task":
        return {
            "groups": "combination_id",
            "re_formula": "1",
            "vc_formula": {"task": "0 + C(original_task_id)"},
        }
    if name == "task_only":
        return {
            "groups": "original_task_id",
            "re_formula": "1",
            "vc_formula": None,
        }
    raise ValueError(f"Unknown random-effects structure: {name}")


def fit_mixed_once(
    data: pd.DataFrame,
    formula: str,
    structure: str,
    reml: bool,
    outcome: str,
    fit_role: str,
    log_rows: list[dict[str, Any]],
) -> Any:
    import statsmodels.formula.api as smf

    spec = structure_specification(structure)
    candidates = []
    for optimizer in ["lbfgs", "powell", "cg"]:
        error = ""
        warning_text = ""
        result = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    data,
                    groups=data[spec["groups"]],
                    re_formula=spec["re_formula"],
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
        llf = float(getattr(result, "llf", np.nan)) if result is not None else np.nan
        log_rows.append(
            {
                "outcome": outcome,
                "fit_role": fit_role,
                "structure": structure,
                "reml": reml,
                "optimizer": optimizer,
                "converged": converged,
                "log_likelihood": llf,
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
    raise RuntimeError(
        f"All optimizers failed for {outcome}, {fit_role}, structure={structure}, reml={reml}"
    )


def fit_ml_model_pair(
    data: pd.DataFrame,
    additive_formula: str,
    interaction_formula: str,
    outcome: str,
    log_rows: list[dict[str, Any]],
) -> tuple[Any, Any, str]:
    errors = []
    for structure in ["combination_plus_task", "task_only"]:
        try:
            additive = fit_mixed_once(
                data, additive_formula, structure, False, outcome, "additive_ml", log_rows
            )
            interaction = fit_mixed_once(
                data, interaction_formula, structure, False, outcome, "interaction_ml", log_rows
            )
            if bool(additive.converged) and bool(interaction.converged):
                return additive, interaction, structure
            errors.append(f"{structure}: at least one model did not converge")
        except Exception as exc:
            errors.append(f"{structure}: {exc}")
    raise RuntimeError(f"No common converged structure for {outcome}: {'; '.join(errors)}")


def fit_estimation_model(
    data: pd.DataFrame,
    formula: str,
    structure: str,
    outcome: str,
    fit_role: str,
    log_rows: list[dict[str, Any]],
) -> tuple[Any, str]:
    result = fit_mixed_once(data, formula, structure, True, outcome, fit_role, log_rows)
    if bool(result.converged):
        return result, "REML"
    result = fit_mixed_once(data, formula, structure, False, outcome, f"{fit_role}_ml_fallback", log_rows)
    return result, "ML_fallback"


def classify_term(term: str) -> tuple[str, str, str]:
    is_interaction = ":" in term
    if term == "Intercept":
        return "intercept", "Intercept", "Reference-task expected performance"
    if "model_code" in term and not is_interaction:
        return "model_main", "Model family", "DeBERTa minus SVM at reference conditions"
    if "label_number_factor" in term:
        levels = re.findall(r"\[T\.([^\]]+)\]", term)
        label = levels[-1] if levels else ""
        if is_interaction:
            return "model_interaction", "Label number", f"DeBERTa-SVM difference: {label} versus 3 labels"
        return "complexity_main", "Label number", f"SVM difference: {label} versus 3 labels"
    for variable, display in DIMENSION_DISPLAY.items():
        if variable != "label_number" and variable in term:
            if is_interaction:
                return "model_interaction", display, f"Difference in {display.lower()} slope: DeBERTa minus SVM"
            return "complexity_main", display, f"SVM slope for {display.lower()} (per 1 SD)"
    return "other", "Other", term


def fixed_effect_table(
    result: Any,
    outcome: str,
    structure: str,
    estimation: str,
    analysis: str = "primary_all_labels",
) -> pd.DataFrame:
    params = result.fe_params
    bse = result.bse_fe
    ci = result.conf_int().loc[params.index]
    rows = []
    for term in params.index:
        role, dimension, description = classify_term(term)
        rows.append(
            {
                "analysis": analysis,
                "outcome": outcome,
                "random_effects_structure": structure,
                "estimation": estimation,
                "term": term,
                "term_role": role,
                "dimension": dimension,
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
    interactions = [name for name in names if ":" in name and "model_code" in name]
    if test_name == "all_model_complexity_interactions":
        return interactions
    if test_name == "model_main":
        return [name for name in names if "model_code" in name and ":" not in name]
    if test_name == "label_number_main":
        return [name for name in names if "label_number_factor" in name and ":" not in name]
    if test_name == "model_x_label_number":
        return [name for name in interactions if "label_number_factor" in name]
    if test_name.endswith("_main"):
        variable = test_name.removesuffix("_main")
        return [name for name in names if name == variable]
    if test_name.startswith("model_x_"):
        variable = test_name.removeprefix("model_x_")
        return [name for name in interactions if variable in name]
    return []


def wald_test_rows(result: Any, outcome: str, analysis: str) -> pd.DataFrame:
    names = list(result.params.index)
    tests = [
        "model_main",
        "label_number_main",
        "class_imbalance_z_main",
        "annotation_disagreement_z_main",
        "log2_training_size_z_main",
        "semantic_similarity_z_main",
        "all_model_complexity_interactions",
        "model_x_label_number",
        "model_x_class_imbalance_z",
        "model_x_annotation_disagreement_z",
        "model_x_log2_training_size_z",
        "model_x_semantic_similarity_z",
    ]
    rows = []
    for test_name in tests:
        terms = matching_terms(names, test_name)
        if not terms:
            continue
        restriction = np.zeros((len(terms), len(names)))
        for row_index, term in enumerate(terms):
            restriction[row_index, names.index(term)] = 1.0
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
    group_variance = float(result.cov_re.iloc[0, 0]) if result.cov_re.size else 0.0
    component_variance = float(np.sum(result.vcomp)) if np.size(result.vcomp) else 0.0
    residual_variance = float(result.scale)
    total = fixed_variance + group_variance + component_variance + residual_variance
    if structure == "combination_plus_task":
        combination_variance = group_variance
        task_variance = component_variance
    else:
        combination_variance = np.nan
        task_variance = group_variance
    return {
        "outcome": outcome,
        "random_effects_structure": structure,
        "estimation": estimation,
        "fixed_prediction_variance": fixed_variance,
        "combination_random_intercept_variance": combination_variance,
        "task_random_intercept_variance": task_variance,
        "residual_variance": residual_variance,
        "marginal_r2": fixed_variance / total,
        "conditional_r2": (fixed_variance + group_variance + component_variance) / total,
        "combination_icc": combination_variance / total if np.isfinite(combination_variance) else np.nan,
        "task_icc": task_variance / total,
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
        fit_type = "conditional"
    except Exception:
        conditional = fixed
        fit_type = "fixed_only_fallback"
    observed = data[outcome].to_numpy(dtype=float)
    residual = observed - conditional
    shapiro_stat, shapiro_p = stats.shapiro(residual)
    diagnostics = {
        "outcome": outcome,
        "fitted_value_type": fit_type,
        "observation_count": len(residual),
        "residual_mean": float(np.mean(residual)),
        "residual_sd": float(np.std(residual, ddof=1)),
        "residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "residual_skewness": float(stats.skew(residual, bias=False)),
        "residual_excess_kurtosis": float(stats.kurtosis(residual, bias=False)),
        "shapiro_w": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "fixed_predictions_below_zero": int((fixed < 0).sum()),
        "fixed_predictions_above_one": int((fixed > 1).sum()),
    }
    observations = data[
        ["original_task_id", "combination_id", "model_code", "label_number_factor"]
    ].copy()
    observations["outcome"] = outcome
    observations["observed"] = observed
    observations["fixed_prediction"] = fixed
    observations["conditional_prediction"] = conditional
    observations["conditional_residual"] = residual
    return diagnostics, observations


def paired_cluster_models(
    pair_data: pd.DataFrame,
    analysis: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.formula.api as smf

    coefficient_rows = []
    test_rows = []
    for outcome in OUTCOMES:
        target = f"delta_{outcome}"
        formula = f"{target} ~ {complexity_expression(sorted(pair_data['label_number_factor'].unique()))}"
        result = smf.ols(formula, pair_data).fit(
            cov_type="cluster",
            cov_kwds={"groups": pair_data["combination_id"], "use_correction": True},
            use_t=True,
        )
        ci = result.conf_int()
        for term in result.params.index:
            role, dimension, description = classify_term(
                f"C(model_code, Treatment(reference='SVM'))[T.DeBERTa]:{term}"
                if term != "Intercept"
                else "C(model_code, Treatment(reference='SVM'))[T.DeBERTa]"
            )
            if term == "Intercept":
                description = "DeBERTa minus SVM at 3 labels and mean continuous complexity"
                dimension = "Model family"
            coefficient_rows.append(
                {
                    "analysis": analysis,
                    "outcome": outcome,
                    "term": term,
                    "dimension": dimension,
                    "description": description,
                    "estimate": float(result.params[term]),
                    "cluster_robust_standard_error": float(result.bse[term]),
                    "t_value": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                    "ci95_low": float(ci.loc[term, 0]),
                    "ci95_high": float(ci.loc[term, 1]),
                    "task_count": len(pair_data),
                    "cluster_count": pair_data["combination_id"].nunique(),
                    "r_squared": float(result.rsquared),
                    "adjusted_r_squared": float(result.rsquared_adj),
                }
            )
        names = list(result.params.index)
        groups = {
            "all_complexity_terms": [name for name in names if name != "Intercept"],
            "label_number": [name for name in names if "label_number_factor" in name],
        }
        for variable in CONTINUOUS_TERMS:
            groups[variable] = [name for name in names if name == variable]
        for test_name, terms in groups.items():
            if not terms:
                continue
            restriction = np.zeros((len(terms), len(names)))
            for row_index, term in enumerate(terms):
                restriction[row_index, names.index(term)] = 1.0
            test = result.wald_test(restriction, use_f=True, scalar=True)
            test_rows.append(
                {
                    "analysis": analysis,
                    "outcome": outcome,
                    "test": test_name,
                    "tested_terms": " | ".join(terms),
                    "numerator_df": len(terms),
                    "denominator_df": float(np.asarray(test.df_denom).squeeze()),
                    "f_statistic": float(np.asarray(test.statistic).squeeze()),
                    "p_value": float(np.asarray(test.pvalue).squeeze()),
                }
            )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(test_rows)


def task_mean_cluster_models(
    data: pd.DataFrame,
    analysis: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.formula.api as smf

    metadata_columns = [
        "original_task_id",
        "combination_id",
        "label_number_factor",
        *CONTINUOUS_TERMS,
    ]
    metadata = data.drop_duplicates("original_task_id")[metadata_columns]
    means = data.groupby("original_task_id", as_index=False)[OUTCOMES].mean()
    task_data = metadata.merge(means, on="original_task_id", how="left", validate="one_to_one")
    coefficient_rows = []
    test_rows = []
    for outcome in OUTCOMES:
        target = f"task_mean_{outcome}"
        task_data[target] = task_data[outcome]
        formula = f"{target} ~ {complexity_expression(sorted(task_data['label_number_factor'].unique()))}"
        result = smf.ols(formula, task_data).fit(
            cov_type="cluster",
            cov_kwds={"groups": task_data["combination_id"], "use_correction": True},
            use_t=True,
        )
        ci = result.conf_int()
        for term in result.params.index:
            if term == "Intercept":
                dimension = "Intercept"
                description = "Average SVM-DeBERTa performance at 3 labels and mean continuous complexity"
            elif "label_number_factor" in term:
                level = re.findall(r"\[T\.([^\]]+)\]", term)[-1]
                dimension = "Label number"
                description = f"Average performance difference: {level} versus 3 labels"
            else:
                dimension = DIMENSION_DISPLAY.get(term, term)
                description = f"Average performance slope for {dimension.lower()} (per 1 SD)"
            coefficient_rows.append(
                {
                    "analysis": analysis,
                    "outcome": outcome,
                    "term": term,
                    "dimension": dimension,
                    "description": description,
                    "estimate": float(result.params[term]),
                    "cluster_robust_standard_error": float(result.bse[term]),
                    "t_value": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                    "ci95_low": float(ci.loc[term, 0]),
                    "ci95_high": float(ci.loc[term, 1]),
                    "task_count": len(task_data),
                    "cluster_count": task_data["combination_id"].nunique(),
                    "r_squared": float(result.rsquared),
                    "adjusted_r_squared": float(result.rsquared_adj),
                }
            )
        names = list(result.params.index)
        groups = {
            "all_complexity_terms": [name for name in names if name != "Intercept"],
            "label_number": [name for name in names if "label_number_factor" in name],
        }
        for variable in CONTINUOUS_TERMS:
            groups[variable] = [name for name in names if name == variable]
        for test_name, terms in groups.items():
            if not terms:
                continue
            restriction = np.zeros((len(terms), len(names)))
            for row_index, term in enumerate(terms):
                restriction[row_index, names.index(term)] = 1.0
            test = result.wald_test(restriction, use_f=True, scalar=True)
            test_rows.append(
                {
                    "analysis": analysis,
                    "outcome": outcome,
                    "test": test_name,
                    "tested_terms": " | ".join(terms),
                    "numerator_df": len(terms),
                    "denominator_df": float(np.asarray(test.df_denom).squeeze()),
                    "f_statistic": float(np.asarray(test.statistic).squeeze()),
                    "p_value": float(np.asarray(test.pvalue).squeeze()),
                }
            )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(test_rows)


def standardization_lookup(parameters: pd.DataFrame) -> dict[str, dict[str, float]]:
    subset = parameters[parameters["dataset"] == "supervised_task_reference"]
    lookup = {}
    for row in subset.itertuples(index=False):
        lookup[str(row.standardized_variable)] = {
            "mean": float(row.mean),
            "std": float(row.std_ddof0),
            "minimum": float(row.minimum),
            "maximum": float(row.maximum),
        }
    return lookup


def predict_fixed_with_ci(result: Any, new_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from patsy import build_design_matrices

    matrix = build_design_matrices(
        [result.model.data.design_info], new_data, return_type="dataframe"
    )[0]
    names = list(result.fe_params.index)
    matrix = matrix.reindex(columns=names, fill_value=0.0)
    x = matrix.to_numpy(dtype=float)
    beta = result.fe_params.reindex(names).to_numpy(dtype=float)
    covariance = result.cov_params().loc[names, names].to_numpy(dtype=float)
    prediction = x @ beta
    variance = np.einsum("ij,jk,ik->i", x, covariance, x)
    se = np.sqrt(np.maximum(variance, 0))
    return prediction, prediction - 1.96 * se, prediction + 1.96 * se


def prediction_grid(
    result: Any,
    data: pd.DataFrame,
    parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from patsy import build_design_matrices

    lookup = standardization_lookup(parameters)
    task_reference = data.drop_duplicates("original_task_id")
    profile_columns = ["label_number_factor", *CONTINUOUS_TERMS]
    empirical_profiles = task_reference[profile_columns].copy().reset_index(drop=True)
    fixed_names = list(result.fe_params.index)
    beta = result.fe_params.reindex(fixed_names).to_numpy(dtype=float)
    covariance = result.cov_params().loc[fixed_names, fixed_names].to_numpy(dtype=float)
    rows = []
    design_means: dict[tuple[str, float, str], np.ndarray] = {}
    dimensions = ["label_number", *CONTINUOUS_TERMS]
    for dimension in dimensions:
        if dimension == "label_number":
            points = [(float(level), float(level), level) for level in LABEL_LEVELS]
        elif dimension == "log2_training_size_z":
            sizes = sorted(task_reference["training_sample_size"].astype(int).unique())
            points = [
                (
                    float(size),
                    (float(np.log2(size)) - lookup[dimension]["mean"]) / lookup[dimension]["std"],
                    None,
                )
                for size in sizes
            ]
        else:
            source = CONTINUOUS_SOURCE[dimension]
            values = pd.to_numeric(task_reference[source], errors="coerce")
            low, high = values.quantile([0.05, 0.95]).tolist()
            originals = np.linspace(float(low), float(high), 21)
            points = [
                (
                    float(value),
                    (float(value) - lookup[dimension]["mean"]) / lookup[dimension]["std"],
                    None,
                )
                for value in originals
            ]
        for model in MODEL_ORDER:
            for original, standardized, label_level in points:
                profiles = empirical_profiles.copy()
                profiles["model_code"] = model
                if dimension == "label_number":
                    profiles["label_number_factor"] = label_level
                else:
                    profiles[dimension] = standardized
                matrix = build_design_matrices(
                    [result.model.data.design_info], profiles, return_type="dataframe"
                )[0].reindex(columns=fixed_names, fill_value=0.0)
                design_mean = matrix.to_numpy(dtype=float).mean(axis=0)
                prediction = float(design_mean @ beta)
                prediction_se = float(
                    np.sqrt(max(design_mean @ covariance @ design_mean, 0.0))
                )
                design_means[(dimension, original, model)] = design_mean
                rows.append(
                    {
                        "prediction_method": "marginal_standardization_over_240_observed_task_profiles",
                        "model_code": model,
                        "dimension": dimension,
                        "dimension_display": DIMENSION_DISPLAY[dimension],
                        "x_original": original,
                        "x_standardized": standardized,
                        "predicted_kappa": prediction,
                        "ci95_low": prediction - 1.96 * prediction_se,
                        "ci95_high": prediction + 1.96 * prediction_se,
                    }
                )
    grid = pd.DataFrame(rows)

    contrast_rows = []
    for (dimension, x_original), group in grid.groupby(["dimension", "x_original"], sort=False):
        if set(group["model_code"]) != set(MODEL_ORDER):
            continue
        svm_design = design_means[(dimension, x_original, "SVM")]
        deberta_design = design_means[(dimension, x_original, "DeBERTa")]
        difference_design = deberta_design - svm_design
        difference = float(difference_design @ beta)
        difference_se = float(
            np.sqrt(max(difference_design @ covariance @ difference_design, 0.0))
        )
        contrast_rows.append(
            {
                "dimension": dimension,
                "dimension_display": DIMENSION_DISPLAY[dimension],
                "x_original": x_original,
                "predicted_difference_deberta_minus_svm": difference,
                "ci95_low": difference - 1.96 * difference_se,
                "ci95_high": difference + 1.96 * difference_se,
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    return grid, contrasts


def marginal_change_summary(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dimension, model), group in grid.groupby(["dimension", "model_code"], sort=False):
        ordered = group.sort_values("x_original")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        rows.append(
            {
                "dimension": dimension,
                "dimension_display": DIMENSION_DISPLAY[dimension],
                "model": model,
                "low_x": first["x_original"],
                "high_x": last["x_original"],
                "predicted_kappa_low": first["predicted_kappa"],
                "predicted_kappa_high": last["predicted_kappa"],
                "predicted_change_high_minus_low": last["predicted_kappa"] - first["predicted_kappa"],
            }
        )
    return pd.DataFrame(rows)


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
        & (coefficients["term_role"] == "model_interaction")
    ].copy()
    frame["plot_label"] = frame["description"].str.replace("DeBERTa-SVM difference: ", "", regex=False)
    frame["plot_label"] = frame["plot_label"].str.replace("Difference in ", "", regex=False)
    frame["plot_label"] = frame["plot_label"].str.replace(" slope: DeBERTa minus SVM", " slope", regex=False)
    frame["plot_label"] = frame["plot_label"].str.replace(
        "11 versus 3 labels", "11 versus 3 labels (exploratory)", regex=False
    )
    frame = frame.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(frame))
    significant = frame["p_value"] < 0.05
    exploratory = frame["plot_label"].str.contains("exploratory")
    colors = np.where(exploratory, "#B08A3E", np.where(significant, "#B55245", "#6F7A80"))
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    for index, row in frame.iterrows():
        ax.plot([row["ci95_low"], row["ci95_high"]], [index, index], color=colors[index], linewidth=1.5)
        ax.scatter(row["estimate"], index, color=colors[index], s=28, zorder=3)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(frame["plot_label"])
    ax.set_xlabel("Change in DeBERTa-SVM kappa difference")
    ax.set_title("Model-task fit: interaction coefficients with 95% Wald intervals")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, figure_dir, "figure5_supervised_interaction_forest")
    plt.close(fig)


def plot_prediction_landscape(
    plt: Any,
    grid: pd.DataFrame,
    contrasts: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.15))
    specs = [
        ("label_number", "Number of labels", "A"),
        ("class_imbalance_z", "Class imbalance index", "B"),
        ("annotation_disagreement_z", "Annotation disagreement rate", "C"),
        ("log2_training_size_z", "Training samples", "D"),
        ("semantic_similarity_z", "Mean semantic similarity", "E"),
    ]
    for ax, (dimension, xlabel, panel) in zip(axes.flat[:5], specs):
        subset = grid[grid["dimension"] == dimension]
        for model in MODEL_ORDER:
            group = subset[subset["model_code"] == model].sort_values("x_original")
            x = group["x_original"].to_numpy(dtype=float)
            mean = group["predicted_kappa"].to_numpy(dtype=float)
            low = group["ci95_low"].to_numpy(dtype=float)
            high = group["ci95_high"].to_numpy(dtype=float)
            ax.plot(x, mean, marker="o" if len(x) <= 10 else None, linewidth=1.4, color=MODEL_COLORS[model], label=model)
            ax.fill_between(x, low, high, color=MODEL_COLORS[model], alpha=0.12, linewidth=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Predicted kappa")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        ax.set_title(DIMENSION_DISPLAY[dimension])
        panel_label(ax, panel)
        if dimension == "log2_training_size_z":
            ax.set_xticks([300, 3000, 6000, 9000, 12000, 15000])
            ax.tick_params(axis="x", labelrotation=45, labelsize=6)
            for label in ax.get_xticklabels():
                label.set_ha("right")
    axes[0, 0].legend(fontsize=7)

    ax = axes[1, 2]
    subset = contrasts[contrasts["dimension"] == "log2_training_size_z"].sort_values("x_original")
    ax.plot(
        subset["x_original"],
        subset["predicted_difference_deberta_minus_svm"],
        marker="o",
        linewidth=1.4,
        color="#5B4A42",
    )
    ax.fill_between(
        subset["x_original"].to_numpy(dtype=float),
        subset["ci95_low"].to_numpy(dtype=float),
        subset["ci95_high"].to_numpy(dtype=float),
        color="#5B4A42",
        alpha=0.12,
        linewidth=0,
    )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Predicted kappa difference")
    ax.set_title("DeBERTa minus SVM")
    ax.set_xticks([300, 3000, 6000, 9000, 12000, 15000])
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    panel_label(ax, "F")
    fig.tight_layout(w_pad=1.1, h_pad=1.5)
    save_figure(fig, figure_dir, "figure6_supervised_adjusted_predictions")
    plt.close(fig)


def plot_robustness(plt: Any, coefficients: pd.DataFrame, figure_dir: Path) -> None:
    frame = coefficients[
        (coefficients["analysis"] == "primary_all_labels")
        & (coefficients["term_role"] == "model_interaction")
    ].copy()
    frame["short_label"] = frame["description"].str.replace("DeBERTa-SVM difference: ", "", regex=False)
    frame["short_label"] = frame["short_label"].str.replace("Difference in ", "", regex=False)
    frame["short_label"] = frame["short_label"].str.replace(" slope: DeBERTa minus SVM", " slope", regex=False)
    frame["short_label"] = frame["short_label"].str.replace(
        "11 versus 3 labels", "11 versus 3 labels (exploratory)", regex=False
    )
    labels = frame[frame["outcome"] == "kappa"]["short_label"].tolist()[::-1]
    positions = np.arange(len(labels))
    offsets = {"kappa": -0.18, "macro_f1": 0.0, "mcc": 0.18}
    colors = {"kappa": "#2F6B9A", "macro_f1": "#B55245", "mcc": "#4F6B5B"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for outcome in OUTCOMES:
        group = frame[frame["outcome"] == outcome].set_index("short_label").reindex(labels)
        y = positions + offsets[outcome]
        ax.errorbar(
            group["estimate"],
            y,
            xerr=np.vstack([group["estimate"] - group["ci95_low"], group["ci95_high"] - group["estimate"]]),
            fmt="o",
            markersize=4,
            capsize=2,
            linewidth=1,
            color=colors[outcome],
            label=OUTCOME_DISPLAY[outcome],
        )
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Model-complexity interaction coefficient")
    ax.set_title("Robustness of model-task fit patterns across performance metrics")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, figure_dir, "figure7_supervised_metric_robustness")
    plt.close(fig)


def plot_diagnostics(plt: Any, observations: pd.DataFrame, figure_dir: Path) -> None:
    from scipy import stats

    frame = observations[observations["outcome"] == "kappa"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    for model in MODEL_ORDER:
        group = frame[frame["model_code"] == model]
        axes[0].scatter(group["observed"], group["conditional_prediction"], s=10, alpha=0.45, color=MODEL_COLORS[model], label=model)
        axes[1].scatter(group["conditional_prediction"], group["conditional_residual"], s=10, alpha=0.45, color=MODEL_COLORS[model])
    axes[0].plot([0, 1], [0, 1], color="#555555", linewidth=0.8)
    axes[0].set_xlabel("Observed kappa")
    axes[0].set_ylabel("Conditional fitted kappa")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7)
    axes[0].set_title("Observed versus fitted")
    panel_label(axes[0], "A")

    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xlabel("Conditional fitted kappa")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual pattern")
    panel_label(axes[1], "B")

    theoretical, ordered = stats.probplot(frame["conditional_residual"], dist="norm", fit=False)
    axes[2].scatter(theoretical, ordered, s=10, alpha=0.55, color="#6F7A80")
    slope, intercept, _ = stats.probplot(frame["conditional_residual"], dist="norm", fit=True)[1]
    xline = np.asarray([min(theoretical), max(theoretical)])
    axes[2].plot(xline, intercept + slope * xline, color="#555555", linewidth=0.8)
    axes[2].set_xlabel("Normal theoretical quantile")
    axes[2].set_ylabel("Residual quantile")
    axes[2].set_title("Residual Q-Q plot")
    panel_label(axes[2], "C")
    for ax in axes:
        ax.grid(color="#EAEAEA", linewidth=0.5)
    fig.tight_layout(w_pad=1.1)
    save_figure(fig, figure_dir, "figure8_supervised_model_diagnostics")
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

    supervised, pairs, parameters = load_inputs(input_dir)
    data, pair_data, support = prepare_data(supervised, pairs)
    convergence_rows: list[dict[str, Any]] = []
    fixed_tables = []
    wald_tables = []
    fit_rows = []
    variance_rows = []
    diagnostic_rows = []
    diagnostic_observations = []
    estimation_results: dict[str, Any] = {}

    for outcome in OUTCOMES:
        additive_formula, interaction_formula = model_formulas(outcome, LABEL_LEVELS)
        additive_ml, interaction_ml, structure = fit_ml_model_pair(
            data,
            additive_formula,
            interaction_formula,
            outcome,
            convergence_rows,
        )
        estimation_result, estimation_method = fit_estimation_model(
            data,
            interaction_formula,
            structure,
            outcome,
            "interaction_reml",
            convergence_rows,
        )
        estimation_results[outcome] = estimation_result
        degrees = len(interaction_ml.fe_params) - len(additive_ml.fe_params)
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
                    "model": "model_by_complexity_interactions",
                    "random_effects_structure": structure,
                    "fixed_effect_parameters": len(interaction_ml.fe_params),
                    "log_likelihood": interaction_ml.llf,
                    "aic": interaction_ml.aic,
                    "bic": interaction_ml.bic,
                    "converged": interaction_ml.converged,
                    "interaction_lrt_chi_square": lr_statistic,
                    "interaction_lrt_df": degrees,
                    "interaction_lrt_p": float(chi2.sf(lr_statistic, degrees)),
                },
            ]
        )
        fixed_tables.append(
            fixed_effect_table(estimation_result, outcome, structure, estimation_method)
        )
        wald_tables.append(wald_test_rows(estimation_result, outcome, "primary_all_labels"))
        variance_rows.append(variance_and_r2(estimation_result, outcome, structure, estimation_method))
        diagnostics, observations = residual_diagnostics(estimation_result, data, outcome)
        diagnostic_rows.append(diagnostics)
        diagnostic_observations.append(observations)

    # Sensitivity analysis: remove the structurally unsupported 11-label level.
    sensitivity_data = data[data["label_number_factor"] != "11"].copy()
    _, sensitivity_formula = model_formulas("kappa", ["3", "5", "8"])
    sensitivity_result, sensitivity_method = fit_estimation_model(
        sensitivity_data,
        sensitivity_formula,
        "combination_plus_task",
        "kappa",
        "exclude_k11_reml",
        convergence_rows,
    )
    sensitivity_fixed = fixed_effect_table(
        sensitivity_result,
        "kappa",
        "combination_plus_task",
        sensitivity_method,
        analysis="sensitivity_excluding_11_labels",
    )
    fixed_tables.append(sensitivity_fixed)
    wald_tables.append(
        wald_test_rows(sensitivity_result, "kappa", "sensitivity_excluding_11_labels")
    )

    paired_coefficients, paired_tests = paired_cluster_models(pair_data, "primary_all_labels")
    paired_no11_coefficients, paired_no11_tests = paired_cluster_models(
        pair_data[pair_data["label_number_factor"] != "11"].copy(),
        "sensitivity_excluding_11_labels",
    )
    paired_coefficients = pd.concat(
        [paired_coefficients, paired_no11_coefficients], ignore_index=True
    )
    paired_tests = pd.concat([paired_tests, paired_no11_tests], ignore_index=True)

    task_mean_coefficients, task_mean_tests = task_mean_cluster_models(
        data, "primary_all_labels"
    )
    task_mean_no11_coefficients, task_mean_no11_tests = task_mean_cluster_models(
        sensitivity_data, "sensitivity_excluding_11_labels"
    )
    task_mean_coefficients = pd.concat(
        [task_mean_coefficients, task_mean_no11_coefficients], ignore_index=True
    )
    task_mean_tests = pd.concat([task_mean_tests, task_mean_no11_tests], ignore_index=True)

    fixed_effects = pd.concat(fixed_tables, ignore_index=True)
    wald_tests = pd.concat(wald_tables, ignore_index=True)
    fit_comparison = pd.DataFrame(fit_rows)
    variance_r2 = pd.DataFrame(variance_rows)
    convergence = pd.DataFrame(convergence_rows)
    residual_summary = pd.DataFrame(diagnostic_rows)
    residual_observations = pd.concat(diagnostic_observations, ignore_index=True)

    primary_kappa = estimation_results["kappa"]
    predictions, contrasts = prediction_grid(primary_kappa, data, parameters)
    marginal_changes = marginal_change_summary(predictions)

    primary_interactions = fixed_effects[
        (fixed_effects["analysis"] == "primary_all_labels")
        & (fixed_effects["term_role"] == "model_interaction")
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
        np.sign(robustness["estimate_kappa"])
        == np.sign(robustness["estimate_macro_f1"])
    ) & (
        np.sign(robustness["estimate_kappa"])
        == np.sign(robustness["estimate_mcc"])
    )

    primary_coefficients = fixed_effects[fixed_effects["analysis"] == "primary_all_labels"]
    significant_interactions = primary_coefficients[
        (primary_coefficients["term_role"] == "model_interaction")
        & (primary_coefficients["p_value"] < 0.05)
    ][["outcome", "dimension", "description", "estimate", "p_value", "ci95_low", "ci95_high"]]

    save_csv(support, table_dir / "supervised_design_support.csv")
    save_csv(fixed_effects, table_dir / "mixed_model_fixed_effects.csv")
    save_csv(wald_tests, table_dir / "mixed_model_wald_tests.csv")
    save_csv(fit_comparison, table_dir / "mixed_model_fit_comparison.csv")
    save_csv(variance_r2, table_dir / "mixed_model_variance_and_r2.csv")
    save_csv(paired_coefficients, table_dir / "paired_difference_cluster_robust_coefficients.csv")
    save_csv(paired_tests, table_dir / "paired_difference_cluster_robust_tests.csv")
    save_csv(task_mean_coefficients, table_dir / "task_mean_cluster_robust_coefficients.csv")
    save_csv(task_mean_tests, table_dir / "task_mean_cluster_robust_tests.csv")
    save_csv(robustness, table_dir / "interaction_metric_robustness.csv")
    save_csv(significant_interactions, table_dir / "significant_model_complexity_interactions.csv")
    save_csv(convergence, diagnostic_dir / "model_convergence_log.csv")
    save_csv(residual_summary, diagnostic_dir / "residual_diagnostics.csv")
    save_csv(residual_observations, diagnostic_dir / "residual_observations.csv")
    save_csv(predictions, prediction_dir / "adjusted_kappa_prediction_grid.csv")
    save_csv(contrasts, prediction_dir / "adjusted_model_difference_grid.csv")
    save_csv(marginal_changes, prediction_dir / "adjusted_marginal_changes.csv")

    figure_status = "skipped"
    if not args.skip_figures:
        plt = configure_matplotlib()
        plot_interaction_forest(plt, fixed_effects, figure_dir)
        plot_prediction_landscape(plt, predictions, contrasts, figure_dir)
        plot_robustness(plt, fixed_effects, figure_dir)
        plot_diagnostics(plt, residual_observations, figure_dir)
        figure_status = "rendered"

    lrt_summary = (
        fit_comparison[fit_comparison["model"] == "model_by_complexity_interactions"]
        .set_index("outcome")["interaction_lrt_p"]
        .to_dict()
    )
    report = {
        "generated_at_utc": utc_now(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "primary_outcome": "kappa",
        "robustness_outcomes": ["macro_f1", "mcc"],
        "fixed_effects": (
            "model family * (categorical label number + standardized class imbalance + "
            "standardized annotation disagreement + standardized log2 training size + "
            "standardized semantic similarity)"
        ),
        "preferred_random_effects": "label-combination intercept plus nested task intercept",
        "adjusted_prediction_method": (
            "Marginal standardization over the 240 observed task profiles; the focal complexity "
            "dimension and model are varied while the empirical distribution of other dimensions is retained."
        ),
        "selected_random_effects_by_outcome": {
            row["outcome"]: row["random_effects_structure"]
            for row in fit_rows
            if row["model"] == "model_by_complexity_interactions"
        },
        "analysis_rows": len(data),
        "task_count": data["original_task_id"].nunique(),
        "combination_count": data["combination_id"].nunique(),
        "interaction_lrt_p_values": lrt_summary,
        "significant_interaction_count": len(significant_interactions),
        "significant_interactions": significant_interactions.to_dict(orient="records"),
        "figure_status": figure_status,
        "figure_count": 4 if figure_status == "rendered" else 0,
        "output_checks": {
            "all_primary_models_converged": bool(
                fit_comparison["converged"].fillna(False).all()
            ),
            "missing_primary_fixed_effect_cells": int(
                primary_coefficients[["estimate", "standard_error", "p_value"]].isna().sum().sum()
            ),
            "fixed_predictions_outside_unit_interval": int(
                residual_summary["fixed_predictions_below_zero"].sum()
                + residual_summary["fixed_predictions_above_one"].sum()
            ),
            "paired_difference_cluster_count": int(pair_data["combination_id"].nunique()),
            "all_paired_difference_coefficients_finite": bool(
                np.isfinite(paired_coefficients["estimate"]).all()
                and np.isfinite(paired_coefficients["cluster_robust_standard_error"]).all()
            ),
            "all_task_mean_coefficients_finite": bool(
                np.isfinite(task_mean_coefficients["estimate"]).all()
                and np.isfinite(task_mean_coefficients["cluster_robust_standard_error"]).all()
            ),
        },
        "support_warning": (
            "The 11-label condition contains one label combination. Its categorical main and "
            "interaction coefficients are exploratory; use the excluding-11 sensitivity model "
            "when assessing replicated model-task fit patterns."
        ),
        "interpretation_boundary": (
            "Coefficients describe associations across deliberately constructed tasks. Continuous "
            "coefficients are kappa/metric changes per one task-level standard deviation. Model-by-"
            "complexity terms estimate how DeBERTa sensitivity differs from SVM; they are not causal effects."
        ),
        "distributional_robustness": (
            "Mixed-model residual normality is assessed explicitly. Cluster-robust task-mean and "
            "paired-difference regressions are provided so common complexity effects and model-fit "
            "effects do not rely solely on Gaussian mixed-model standard errors."
        ),
    }
    write_json(output_dir / "step3_supervised_complexity_report.json", report)
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    print(f"\nStep 3 outputs: {output_dir}")


if __name__ == "__main__":
    main()
