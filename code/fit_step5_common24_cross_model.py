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
CANDIDATE_ORDER = [
    "TF-IDF + LinearSVC",
    "DeBERTa-v3-base",
    "DeepSeek V4 Flash (codebook_guided)",
]
CANDIDATE_SHORT = {
    "TF-IDF + LinearSVC": "SVM",
    "DeBERTa-v3-base": "DeBERTa",
    "DeepSeek V4 Flash (codebook_guided)": "LLM codebook",
}
CANDIDATE_COLORS = {
    "TF-IDF + LinearSVC": "#6F7378",
    "DeBERTa-v3-base": "#247B7B",
    "DeepSeek V4 Flash (codebook_guided)": "#B55245",
}
CONTRASTS = [
    (
        "deberta_minus_svm",
        "DeBERTa-v3-base",
        "TF-IDF + LinearSVC",
        "DeBERTa minus SVM",
    ),
    (
        "llm_minus_svm",
        "DeepSeek V4 Flash (codebook_guided)",
        "TF-IDF + LinearSVC",
        "LLM codebook minus SVM",
    ),
    (
        "deberta_minus_llm",
        "DeBERTa-v3-base",
        "DeepSeek V4 Flash (codebook_guided)",
        "DeBERTa minus LLM codebook",
    ),
]
CONTRAST_COLORS = {
    "deberta_minus_svm": "#247B7B",
    "llm_minus_svm": "#B55245",
    "deberta_minus_llm": "#4B4B4B",
}
BOOTSTRAP_REPETITIONS = 50000


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
    parser = argparse.ArgumentParser(description="Step 5: Common24 cross-model comparison")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], dataset: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")


def load_and_validate(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = input_dir / "common24_primary_analysis_ready.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step 1 Common24 table: {path}")
    data = pd.read_csv(path, low_memory=False)
    required = [
        "combination_id",
        "candidate_model",
        "candidate_family",
        "prompt_strategy",
        "repeat_count",
        "training_sample_size",
        "training_class_imbalance",
        "common_label_number",
        "common_label_number_factor",
        "common_annotation_disagreement",
        "common_semantic_similarity",
        "common_annotation_disagreement_z",
        "common_semantic_similarity_z",
        "common_complexity_quadrant",
        *OUTCOMES,
    ]
    require_columns(data, required, "common24_primary_analysis_ready")
    data = data.copy()
    for column in [
        "repeat_count",
        "training_sample_size",
        "common_label_number",
        "common_annotation_disagreement",
        "common_semantic_similarity",
        "common_annotation_disagreement_z",
        "common_semantic_similarity_z",
        *OUTCOMES,
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    universally_required = [
        column
        for column in required
        if column not in {"training_sample_size", "training_class_imbalance"}
    ]
    if data[universally_required].isna().any().any():
        missing = data[universally_required].isna().sum()
        raise ValueError(f"Common24 contains missing required values: {missing[missing > 0].to_dict()}")
    if not np.isfinite(data[[*OUTCOMES, "common_annotation_disagreement_z", "common_semantic_similarity_z"]]).all().all():
        raise ValueError("Common24 contains non-finite analysis values")
    if not data[OUTCOMES].apply(lambda column: column.between(-1, 1).all()).all():
        raise ValueError("At least one performance metric falls outside [-1, 1]")
    if data.duplicated(["combination_id", "candidate_model"]).any():
        raise ValueError("Common24 must contain one row per combination and candidate")
    if set(data["candidate_model"].unique()) != set(CANDIDATE_ORDER):
        raise ValueError(f"Unexpected candidates: {sorted(data['candidate_model'].unique())}")
    candidate_counts = data.groupby("combination_id")["candidate_model"].nunique()
    if len(candidate_counts) != 24 or not candidate_counts.eq(3).all():
        raise ValueError("Common24 must contain exactly 24 complete three-candidate blocks")
    metadata_columns = [
        "common_label_number",
        "common_label_number_factor",
        "common_annotation_disagreement",
        "common_semantic_similarity",
        "common_annotation_disagreement_z",
        "common_semantic_similarity_z",
        "common_complexity_quadrant",
    ]
    for column in metadata_columns:
        if not data.groupby("combination_id")[column].nunique(dropna=False).eq(1).all():
            raise ValueError(f"Complexity metadata differ across candidates: {column}")
    supervised = data[data["candidate_model"] != CANDIDATE_ORDER[2]]
    if supervised[["training_sample_size", "training_class_imbalance"]].isna().any().any():
        raise ValueError("Supervised Common24 anchors contain missing training-design values")
    if not supervised["training_sample_size"].eq(3000).all():
        raise ValueError("Common24 supervised anchors must all use 3,000 training examples")
    supervised["training_class_imbalance"] = pd.to_numeric(
        supervised["training_class_imbalance"], errors="coerce"
    )
    if not np.isfinite(supervised["training_class_imbalance"]).all():
        raise ValueError("Common24 supervised anchors contain invalid imbalance values")
    if not supervised.groupby("combination_id")["training_class_imbalance"].nunique().eq(1).all():
        raise ValueError("SVM and DeBERTa imbalance values differ within Common24 tasks")
    llm_repeats = data.loc[data["candidate_model"] == CANDIDATE_ORDER[2], "repeat_count"]
    supervised_repeats = supervised["repeat_count"]
    if not llm_repeats.eq(3).all() or not supervised_repeats.eq(1).all():
        raise ValueError("Expected LLM means over 3 calls and one supervised result per task")

    data["candidate_model"] = pd.Categorical(
        data["candidate_model"], categories=CANDIDATE_ORDER, ordered=True
    )
    data["candidate_short"] = data["candidate_model"].map(CANDIDATE_SHORT).astype(str)
    data["label_number_factor"] = data["common_label_number"].astype(int).astype(str)
    data = data.sort_values(["combination_id", "candidate_model"]).reset_index(drop=True)

    metadata = data.drop_duplicates("combination_id").copy()
    support = (
        metadata.groupby("common_label_number", sort=True)
        .agg(
            combination_count=("combination_id", "nunique"),
            minimum_disagreement=("common_annotation_disagreement", "min"),
            maximum_disagreement=("common_annotation_disagreement", "max"),
            minimum_similarity=("common_semantic_similarity", "min"),
            maximum_similarity=("common_semantic_similarity", "max"),
        )
        .reset_index()
    )
    support["inference_status"] = np.where(
        support["combination_count"].ge(2),
        "replicated_across_combinations",
        "exploratory_single_combination",
    )
    return data, support


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(values, [0.025, 0.975]).astype(float))


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPETITIONS, len(values)))
    return values[indices].mean(axis=1)


def model_summaries(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        for candidate_index, candidate in enumerate(CANDIDATE_ORDER):
            values = (
                data.loc[data["candidate_model"] == candidate]
                .sort_values("combination_id")[outcome]
                .to_numpy(float)
            )
            rng = np.random.default_rng(seed + outcome_index * 100 + candidate_index)
            boot = bootstrap_mean(values, rng)
            ci_low, ci_high = percentile_interval(boot)
            rows.append(
                {
                    "outcome": outcome,
                    "candidate_model": candidate,
                    "candidate_short": CANDIDATE_SHORT[candidate],
                    "combination_count": len(values),
                    "mean": values.mean(),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "standard_deviation": values.std(ddof=1),
                    "minimum": values.min(),
                    "quartile_25": np.quantile(values, 0.25),
                    "median": np.median(values),
                    "quartile_75": np.quantile(values, 0.75),
                    "maximum": values.max(),
                    "share_kappa_at_least_0_60": (
                        np.mean(values >= 0.60) if outcome == "kappa" else np.nan
                    ),
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
    adjusted_sorted = np.maximum.accumulate(
        np.array([(len(order) - rank) * values[index] for rank, index in enumerate(order)])
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    for index, adjusted in zip(order, adjusted_sorted):
        output[index] = adjusted
    return pd.Series(output, index=p_values.index)


def rank_biserial(diff: np.ndarray) -> float:
    from scipy.stats import rankdata

    diff = np.asarray(diff, dtype=float)
    diff = diff[np.abs(diff) > 1e-12]
    if len(diff) == 0:
        return 0.0
    ranks = rankdata(np.abs(diff))
    total = ranks.sum()
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / total)


def paired_inference(data: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import friedmanchisquare, wilcoxon

    omnibus_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        wide = data.pivot(index="combination_id", columns="candidate_model", values=outcome)
        wide = wide[CANDIDATE_ORDER].sort_index()
        friedman = friedmanchisquare(*[wide[column].to_numpy(float) for column in CANDIDATE_ORDER])
        omnibus_rows.append(
            {
                "outcome": outcome,
                "combination_count": len(wide),
                "friedman_chi_square": float(friedman.statistic),
                "degrees_of_freedom": len(CANDIDATE_ORDER) - 1,
                "p_value": float(friedman.pvalue),
                "kendalls_w": float(friedman.statistic / (len(wide) * (len(CANDIDATE_ORDER) - 1))),
            }
        )
        for contrast_index, (contrast, candidate_a, candidate_b, display) in enumerate(CONTRASTS):
            diff = (wide[candidate_a] - wide[candidate_b]).to_numpy(float)
            rng = np.random.default_rng(seed + 1000 + outcome_index * 100 + contrast_index)
            boot = bootstrap_mean(diff, rng)
            ci_low, ci_high = percentile_interval(boot)
            test = wilcoxon(diff, zero_method="wilcox", correction=False, alternative="two-sided")
            pair_rows.append(
                {
                    "outcome": outcome,
                    "contrast": contrast,
                    "contrast_display": display,
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                    "combination_count": len(diff),
                    "mean_difference": diff.mean(),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "median_difference": np.median(diff),
                    "standard_deviation_difference": diff.std(ddof=1),
                    "candidate_a_wins": int(np.sum(diff > 1e-12)),
                    "candidate_b_wins": int(np.sum(diff < -1e-12)),
                    "exact_ties": int(np.sum(np.abs(diff) <= 1e-12)),
                    "wilcoxon_statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "rank_biserial_correlation": rank_biserial(diff),
                }
            )
    pairwise = pd.DataFrame(pair_rows)
    pairwise["p_value_holm_within_outcome"] = pairwise.groupby("outcome", group_keys=False)[
        "p_value"
    ].apply(holm_adjust)
    return pd.DataFrame(omnibus_rows), pairwise


def task_winners(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_columns = [
        "combination_id",
        "common_label_number",
        "common_annotation_disagreement",
        "common_semantic_similarity",
        "common_complexity_quadrant",
    ]
    metadata = data.drop_duplicates("combination_id")[metadata_columns].set_index("combination_id")
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        wide = data.pivot(index="combination_id", columns="candidate_model", values=outcome)
        wide = wide[CANDIDATE_ORDER]
        for combination_id, values in wide.iterrows():
            ordered = values.sort_values(ascending=False)
            rows.append(
                {
                    "outcome": outcome,
                    "combination_id": combination_id,
                    "winner_model": ordered.index[0],
                    "winner_short": CANDIDATE_SHORT[ordered.index[0]],
                    "winner_value": ordered.iloc[0],
                    "runner_up_model": ordered.index[1],
                    "runner_up_short": CANDIDATE_SHORT[ordered.index[1]],
                    "runner_up_value": ordered.iloc[1],
                    "winning_margin": ordered.iloc[0] - ordered.iloc[1],
                    "near_tie_within_0_02": bool(ordered.iloc[0] - ordered.iloc[1] <= 0.02),
                    **metadata.loc[combination_id].to_dict(),
                }
            )
    winners = pd.DataFrame(rows)
    summary = (
        winners.groupby(["outcome", "winner_model", "winner_short"], observed=True)
        .agg(
            task_wins=("combination_id", "size"),
            mean_winning_margin=("winning_margin", "mean"),
            median_winning_margin=("winning_margin", "median"),
        )
        .reset_index()
    )
    all_rows = pd.MultiIndex.from_product(
        [OUTCOMES, CANDIDATE_ORDER], names=["outcome", "winner_model"]
    ).to_frame(index=False)
    all_rows["winner_model"] = pd.Categorical(
        all_rows["winner_model"], categories=CANDIDATE_ORDER, ordered=True
    )
    all_rows["winner_short"] = all_rows["winner_model"].map(CANDIDATE_SHORT).astype(str)
    summary = all_rows.merge(summary, how="left", on=["outcome", "winner_model", "winner_short"])
    summary["task_wins"] = summary["task_wins"].fillna(0).astype(int)
    summary["win_share"] = summary["task_wins"] / winners["combination_id"].nunique()
    return winners, summary


def make_gain_data(data: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "combination_id",
        "common_label_number",
        "common_annotation_disagreement",
        "common_semantic_similarity",
        "common_annotation_disagreement_z",
        "common_semantic_similarity_z",
        "common_complexity_quadrant",
    ]
    metadata = data.drop_duplicates("combination_id")[metadata_columns]
    output = metadata.copy()
    output["label_number_factor"] = output["common_label_number"].astype(int).astype(str)
    output["label_number_log2"] = np.log2(output["common_label_number"].astype(float))
    output["label_number_log2_z"] = (
        output["label_number_log2"] - output["label_number_log2"].mean()
    ) / output["label_number_log2"].std(ddof=0)
    for outcome in OUTCOMES:
        wide = data.pivot(index="combination_id", columns="candidate_model", values=outcome)
        for contrast, candidate_a, candidate_b, _ in CONTRASTS:
            mapping = (wide[candidate_a] - wide[candidate_b]).rename(
                f"{outcome}__{contrast}"
            )
            output = output.merge(mapping, left_on="combination_id", right_index=True, how="left")
    return output.sort_values("combination_id").reset_index(drop=True)


def term_description(term: str) -> tuple[str, str]:
    if term == "Intercept":
        return "Intercept", "Expected gain at 3 labels and mean continuous complexity"
    if "label_number_factor" in term:
        level = term.split("[T.", 1)[1].split("]", 1)[0]
        return "Label number", f"Gain difference: {level} versus 3 labels"
    if term == "common_annotation_disagreement_z":
        return "Annotation disagreement", "Gain change per 1 SD annotation disagreement"
    if term == "common_semantic_similarity_z":
        return "Label semantic similarity", "Gain change per 1 SD semantic similarity"
    if term == "label_number_log2_z":
        return "Label number", "Gain change per 1 SD log2 label number"
    return term, term


def fit_gain_model(
    data: pd.DataFrame,
    outcome_column: str,
    formula_terms: str,
) -> Any:
    import statsmodels.formula.api as smf

    return smf.ols(f"{outcome_column} ~ {formula_terms}", data=data).fit(cov_type="HC3")


def wald_test_rows(
    result: Any,
    outcome: str,
    contrast: str,
    contrast_display: str,
    analysis: str,
    groups: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = list(result.params.index)
    for test_name, terms in groups.items():
        present = [term for term in terms if term in names]
        if not present:
            continue
        restriction = np.zeros((len(present), len(names)))
        for row_index, term in enumerate(present):
            restriction[row_index, names.index(term)] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = result.wald_test(restriction, use_f=True, scalar=True)
        rows.append(
            {
                "analysis": analysis,
                "outcome": outcome,
                "contrast": contrast,
                "contrast_display": contrast_display,
                "test": test_name,
                "degrees_of_freedom": len(present),
                "f_statistic": float(test.statistic),
                "denominator_df": float(test.df_denom),
                "p_value": float(test.pvalue),
            }
        )
    return rows


def gain_regressions(gain_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    primary = gain_data[gain_data["common_label_number"] != 11].copy()
    analyses = [
        (
            "primary_categorical_excluding_11",
            primary,
            "C(label_number_factor, Treatment(reference='3'))"
            " + common_annotation_disagreement_z"
            " + common_semantic_similarity_z",
        ),
        (
            "sensitivity_log2_all_labels",
            gain_data.copy(),
            "label_number_log2_z"
            " + common_annotation_disagreement_z"
            " + common_semantic_similarity_z",
        ),
    ]
    for analysis, analysis_data, formula_terms in analyses:
        for outcome in OUTCOMES:
            for contrast, _, _, contrast_display in CONTRASTS:
                outcome_column = f"{outcome}__{contrast}"
                result = fit_gain_model(analysis_data, outcome_column, formula_terms)
                confidence = result.conf_int(alpha=0.05)
                fit_rows.append(
                    {
                        "analysis": analysis,
                        "outcome": outcome,
                        "contrast": contrast,
                        "contrast_display": contrast_display,
                        "combination_count": int(result.nobs),
                        "parameters": int(len(result.params)),
                        "r_squared": float(result.rsquared),
                        "adjusted_r_squared": float(result.rsquared_adj),
                        "root_mean_squared_error": float(np.sqrt(np.mean(result.resid**2))),
                        "condition_number": float(result.condition_number),
                    }
                )
                for term in result.params.index:
                    dimension, description = term_description(term)
                    coefficient_rows.append(
                        {
                            "analysis": analysis,
                            "outcome": outcome,
                            "contrast": contrast,
                            "contrast_display": contrast_display,
                            "term": term,
                            "dimension": dimension,
                            "description": description,
                            "estimate": float(result.params[term]),
                            "hc3_standard_error": float(result.bse[term]),
                            "t_value": float(result.tvalues[term]),
                            "p_value": float(result.pvalues[term]),
                            "ci95_low": float(confidence.loc[term, 0]),
                            "ci95_high": float(confidence.loc[term, 1]),
                        }
                    )
                if analysis == "primary_categorical_excluding_11":
                    names = list(result.params.index)
                    label_terms = [name for name in names if "label_number_factor" in name]
                    groups = {
                        "all_complexity_terms": [name for name in names if name != "Intercept"],
                        "label_number": label_terms,
                        "annotation_disagreement": ["common_annotation_disagreement_z"],
                        "semantic_similarity": ["common_semantic_similarity_z"],
                    }
                else:
                    groups = {
                        "all_complexity_terms": [
                            "label_number_log2_z",
                            "common_annotation_disagreement_z",
                            "common_semantic_similarity_z",
                        ],
                        "label_number": ["label_number_log2_z"],
                        "annotation_disagreement": ["common_annotation_disagreement_z"],
                        "semantic_similarity": ["common_semantic_similarity_z"],
                    }
                test_rows.extend(
                    wald_test_rows(
                        result,
                        outcome,
                        contrast,
                        contrast_display,
                        analysis,
                        groups,
                    )
                )

    coefficients = pd.DataFrame(coefficient_rows)
    coefficients["p_value_holm_within_analysis_outcome"] = np.nan
    for (_, _), indices in coefficients[coefficients["term"] != "Intercept"].groupby(
        ["analysis", "outcome"]
    ).groups.items():
        coefficients.loc[indices, "p_value_holm_within_analysis_outcome"] = holm_adjust(
            coefficients.loc[indices, "p_value"]
        ).to_numpy()
    tests = pd.DataFrame(test_rows)
    tests["adjustment_family"] = np.where(
        tests["test"] == "all_complexity_terms",
        "overall_complexity_omnibus",
        "dimension_specific",
    )
    tests["p_value_holm_within_analysis_outcome"] = tests.groupby(
        ["analysis", "outcome", "adjustment_family"], group_keys=False
    )["p_value"].apply(holm_adjust)
    return coefficients, tests, pd.DataFrame(fit_rows)


def leave_one_combination_out(
    data: pd.DataFrame,
    gain_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_detail_rows: list[dict[str, Any]] = []
    pair_summary_rows: list[dict[str, Any]] = []
    gain_detail_rows: list[dict[str, Any]] = []
    gain_summary_rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        wide = data.pivot(index="combination_id", columns="candidate_model", values=outcome)
        for contrast, candidate_a, candidate_b, contrast_display in CONTRASTS:
            differences = (wide[candidate_a] - wide[candidate_b]).sort_index()
            full_estimate = float(differences.mean())
            estimates = []
            for omitted_combination in differences.index:
                estimate = float(differences.drop(index=omitted_combination).mean())
                estimates.append(estimate)
                pair_detail_rows.append(
                    {
                        "outcome": outcome,
                        "contrast": contrast,
                        "contrast_display": contrast_display,
                        "omitted_combination": omitted_combination,
                        "leave_one_out_mean_difference": estimate,
                    }
                )
            estimates_array = np.asarray(estimates, dtype=float)
            pair_summary_rows.append(
                {
                    "outcome": outcome,
                    "contrast": contrast,
                    "contrast_display": contrast_display,
                    "full_mean_difference": full_estimate,
                    "leave_one_out_minimum": estimates_array.min(),
                    "leave_one_out_maximum": estimates_array.max(),
                    "same_sign_share": float(np.mean(np.sign(estimates_array) == np.sign(full_estimate))),
                }
            )

    primary = gain_data[gain_data["common_label_number"] != 11].copy()
    formula_terms = (
        "C(label_number_factor, Treatment(reference='3'))"
        " + common_annotation_disagreement_z"
        " + common_semantic_similarity_z"
    )
    for outcome in OUTCOMES:
        for contrast, _, _, contrast_display in CONTRASTS:
            outcome_column = f"{outcome}__{contrast}"
            full_result = fit_gain_model(primary, outcome_column, formula_terms)
            for omitted_combination in primary["combination_id"]:
                subset = primary[primary["combination_id"] != omitted_combination]
                result = fit_gain_model(subset, outcome_column, formula_terms)
                for term, estimate in result.params.items():
                    if term == "Intercept":
                        continue
                    gain_detail_rows.append(
                        {
                            "outcome": outcome,
                            "contrast": contrast,
                            "contrast_display": contrast_display,
                            "term": term,
                            "omitted_combination": omitted_combination,
                            "leave_one_out_estimate": float(estimate),
                        }
                    )
            detail = pd.DataFrame(gain_detail_rows)
            contrast_detail = detail[
                (detail["outcome"] == outcome) & (detail["contrast"] == contrast)
            ]
            for term, term_data in contrast_detail.groupby("term"):
                full_estimate = float(full_result.params[term])
                estimates = term_data["leave_one_out_estimate"].to_numpy(float)
                dimension, description = term_description(term)
                gain_summary_rows.append(
                    {
                        "outcome": outcome,
                        "contrast": contrast,
                        "contrast_display": contrast_display,
                        "term": term,
                        "dimension": dimension,
                        "description": description,
                        "full_estimate": full_estimate,
                        "leave_one_out_minimum": estimates.min(),
                        "leave_one_out_maximum": estimates.max(),
                        "same_sign_share": float(
                            np.mean(np.sign(estimates) == np.sign(full_estimate))
                        ),
                    }
                )
    return (
        pd.DataFrame(pair_detail_rows),
        pd.DataFrame(pair_summary_rows),
        pd.DataFrame(gain_detail_rows),
        pd.DataFrame(gain_summary_rows),
    )


def stratified_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for stratifier in ["common_label_number", "common_complexity_quadrant"]:
        grouped = (
            data.groupby([stratifier, "candidate_model"], observed=True)[OUTCOMES]
            .agg(["size", "mean", "std"])
            .reset_index()
        )
        grouped.columns = [
            str(column[0]) if column[1] == "" else f"{column[0]}_{column[1]}"
            for column in grouped.columns
        ]
        long = grouped.melt(
            id_vars=[stratifier, "candidate_model"],
            var_name="metric_statistic",
            value_name="value",
        )
        split = long["metric_statistic"].str.rsplit("_", n=1, expand=True)
        long["outcome"] = split[0]
        long["statistic"] = split[1]
        long["stratifier"] = stratifier
        long["stratum"] = long[stratifier].astype(str)
        rows.append(long[["stratifier", "stratum", "candidate_model", "outcome", "statistic", "value"]])
    return pd.concat(rows, ignore_index=True)


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
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
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


def plot_overall_comparison(
    plt: Any,
    data: pd.DataFrame,
    summaries: pd.DataFrame,
    pairwise: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), gridspec_kw={"width_ratios": [1.05, 1.35]})
    ax = axes[0]
    rng = np.random.default_rng(714)
    for candidate_index, candidate in enumerate(CANDIDATE_ORDER):
        values = data.loc[data["candidate_model"] == candidate, "kappa"].to_numpy(float)
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full(len(values), candidate_index) + jitter,
            values,
            s=18,
            alpha=0.46,
            color=CANDIDATE_COLORS[candidate],
            edgecolor="none",
        )
        row = summaries[(summaries["outcome"] == "kappa") & (summaries["candidate_model"] == candidate)].iloc[0]
        ax.errorbar(
            candidate_index,
            row["mean"],
            yerr=[[row["mean"] - row["bootstrap_ci95_low"]], [row["bootstrap_ci95_high"] - row["mean"]]],
            fmt="o",
            color="black",
            markerfacecolor=CANDIDATE_COLORS[candidate],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7,
            capsize=3,
            linewidth=1.2,
            zorder=5,
        )
    ax.set_xticks(range(len(CANDIDATE_ORDER)), [CANDIDATE_SHORT[c] for c in CANDIDATE_ORDER])
    ax.set_ylabel("Cohen's kappa")
    ax.set_ylim(0.15, 0.90)
    ax.set_title("A. Common24 task distributions and means")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.text(0.01, 0.02, "Dots: tasks; markers: mean and 95% task-bootstrap CI", transform=ax.transAxes, fontsize=7.2)

    ax = axes[1]
    plot_rows = pairwise.copy()
    y_positions = []
    labels = []
    position = 0
    for outcome in OUTCOMES:
        subset = plot_rows[plot_rows["outcome"] == outcome]
        for _, row in subset.iterrows():
            y_positions.append(position)
            labels.append(f"{OUTCOME_DISPLAY[outcome]} | {row['contrast_display']}")
            color = CONTRAST_COLORS[row["contrast"]]
            ax.errorbar(
                row["mean_difference"],
                position,
                xerr=[[row["mean_difference"] - row["bootstrap_ci95_low"]], [row["bootstrap_ci95_high"] - row["mean_difference"]]],
                fmt="o",
                color=color,
                capsize=2.5,
                markersize=5,
                linewidth=1.1,
            )
            position += 1
        position += 0.45
    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired performance difference")
    ax.set_title("B. Pairwise gaps with 95% task-bootstrap CIs")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    figure.tight_layout(w_pad=2.2)
    save_figure(figure, output_dir, "figure13_common24_overall_comparison")
    plt.close(figure)


def plot_task_heatmap(plt: Any, data: pd.DataFrame, output_dir: Path) -> None:
    metadata = data.drop_duplicates("combination_id").set_index("combination_id")
    order = (
        metadata.sort_values(
            ["common_label_number", "common_annotation_disagreement", "common_semantic_similarity"]
        )
        .index.tolist()
    )
    wide = data.pivot(index="combination_id", columns="candidate_model", values="kappa").loc[order, CANDIDATE_ORDER]
    figure, ax = plt.subplots(figsize=(7.5, 7.2))
    image = ax.imshow(wide.to_numpy(float), aspect="auto", cmap="cividis", vmin=0.25, vmax=0.85)
    for row_index, (_, values) in enumerate(wide.iterrows()):
        winner_index = int(np.argmax(values.to_numpy(float)))
        ax.text(winner_index, row_index, "*", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    labels = [
        f"{combination} | K={int(metadata.loc[combination, 'common_label_number'])}"
        + ("*" if int(metadata.loc[combination, "common_label_number"]) == 11 else "")
        for combination in order
    ]
    ax.set_yticks(range(len(order)), labels)
    ax.set_xticks(range(len(CANDIDATE_ORDER)), [CANDIDATE_SHORT[c] for c in CANDIDATE_ORDER])
    ax.set_title("Common24 task-level kappa (* marks task winner)")
    colorbar = figure.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
    colorbar.set_label("Cohen's kappa")
    ax.text(0.0, -0.055, "K=11* is exploratory because it contains one label combination.", transform=ax.transAxes, fontsize=7.5)
    figure.tight_layout()
    save_figure(figure, output_dir, "figure14_common24_task_heatmap")
    plt.close(figure)


def plot_gain_moderation(plt: Any, coefficients: pd.DataFrame, output_dir: Path) -> None:
    subset = coefficients[
        (coefficients["analysis"] == "primary_categorical_excluding_11")
        & (coefficients["outcome"] == "kappa")
        & (coefficients["term"] != "Intercept")
    ].copy()
    desired = [
        "Gain difference: 5 versus 3 labels",
        "Gain difference: 8 versus 3 labels",
        "Gain change per 1 SD annotation disagreement",
        "Gain change per 1 SD semantic similarity",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(11.6, 4.0), sharex=True, sharey=True)
    short_labels = ["5 vs 3 labels", "8 vs 3 labels", "Disagreement (1 SD)", "Semantic similarity (1 SD)"]
    for ax, (contrast, _, _, contrast_display) in zip(axes, CONTRASTS):
        contrast_data = subset[subset["contrast"] == contrast].set_index("description").reindex(desired)
        y = np.arange(len(desired))
        for index, (_, row) in enumerate(contrast_data.iterrows()):
            significant = row["p_value_holm_within_analysis_outcome"] < 0.05
            ax.errorbar(
                row["estimate"],
                index,
                xerr=[[row["estimate"] - row["ci95_low"]], [row["ci95_high"] - row["estimate"]]],
                fmt="o",
                color=CONTRAST_COLORS[contrast],
                markeredgecolor="black" if significant else "none",
                markeredgewidth=0.7,
                markersize=6,
                capsize=2.5,
                linewidth=1.1,
            )
        ax.axvline(0, color="#444444", linestyle="--", linewidth=0.8)
        ax.set_title(contrast_display)
        ax.set_yticks(y, short_labels)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.set_xlabel("Change in paired kappa gap")
    axes[0].set_ylabel("Task-complexity moderator")
    figure.suptitle("Complexity moderation of cross-model kappa gaps (HC3, K=3/5/8)", y=1.02, fontsize=10)
    figure.text(0.5, -0.01, "Black marker outline indicates Holm-adjusted p < 0.05 within the kappa family.", ha="center", fontsize=7.5)
    figure.tight_layout()
    save_figure(figure, output_dir, "figure15_common24_gain_moderation")
    plt.close(figure)


def plot_winner_map(
    plt: Any,
    winners: pd.DataFrame,
    winner_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    kappa = winners[winners["outcome"] == "kappa"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.35, 0.8]})
    ax = axes[0]
    markers = {3: "o", 5: "s", 8: "^", 11: "*"}
    for label_number, marker in markers.items():
        label_data = kappa[kappa["common_label_number"] == label_number]
        for candidate in CANDIDATE_ORDER:
            subset = label_data[label_data["winner_model"] == candidate]
            if subset.empty:
                continue
            ax.scatter(
                subset["common_semantic_similarity"],
                subset["common_annotation_disagreement"],
                s=95 if label_number == 11 else 55,
                marker=marker,
                color=CANDIDATE_COLORS[candidate],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.9,
            )
    for candidate in CANDIDATE_ORDER:
        ax.scatter([], [], color=CANDIDATE_COLORS[candidate], label=CANDIDATE_SHORT[candidate])
    ax.set_xlabel("Label semantic similarity")
    ax.set_ylabel("Annotation disagreement rate")
    ax.set_title("A. Kappa winner across Common24 task space")
    ax.grid(color="#E2E2E2", linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    label_legend = [
        plt.Line2D([0], [0], marker=marker, color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=7, label=f"{label} labels" + ("*" if label == 11 else ""))
        for label, marker in markers.items()
    ]
    first_legend = ax.get_legend()
    ax.add_artist(first_legend)
    ax.legend(handles=label_legend, frameon=False, loc="lower left", fontsize=7.3)

    ax = axes[1]
    summary = winner_summary.copy()
    x = np.arange(len(OUTCOMES))
    width = 0.23
    for candidate_index, candidate in enumerate(CANDIDATE_ORDER):
        values = (
            summary[summary["winner_model"] == candidate]
            .set_index("outcome")
            .reindex(OUTCOMES)["win_share"]
            .to_numpy(float)
        )
        bars = ax.bar(
            x + (candidate_index - 1) * width,
            values,
            width=width,
            color=CANDIDATE_COLORS[candidate],
            label=CANDIDATE_SHORT[candidate],
        )
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{int(round(value * 24))}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x, [OUTCOME_DISPLAY[outcome] for outcome in OUTCOMES], rotation=15, ha="right")
    ax.set_ylabel("Share of 24 tasks won")
    ax.set_ylim(0, 1.08)
    ax.set_title("B. Task wins by outcome")
    ax.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    ax.text(0.0, -0.20, "Numbers above bars are task counts; 11-label point is exploratory.", transform=ax.transAxes, fontsize=7.3)
    figure.tight_layout(w_pad=2.0)
    save_figure(figure, output_dir, "figure16_common24_model_task_fit")
    plt.close(figure)


def main() -> None:
    from scipy.stats import shapiro

    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    prediction_dir = output_dir / "predictions"
    diagnostic_dir = output_dir / "diagnostics"
    figure_dir = output_dir / "figures"
    for directory in [table_dir, prediction_dir, diagnostic_dir, figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data, support = load_and_validate(input_dir)
    summaries = model_summaries(data, args.seed)
    omnibus, pairwise = paired_inference(data, args.seed)
    winners, winner_summary = task_winners(data)
    gain_data = make_gain_data(data)
    gain_coefficients, gain_tests, gain_fit = gain_regressions(gain_data)
    (
        leave_one_pair_detail,
        leave_one_pair_summary,
        leave_one_gain_detail,
        leave_one_gain_summary,
    ) = leave_one_combination_out(data, gain_data)
    stratified = stratified_summary(data)

    residual_rows: list[dict[str, Any]] = []
    primary_coefficients = gain_coefficients[
        gain_coefficients["analysis"] == "primary_categorical_excluding_11"
    ]
    for outcome in OUTCOMES:
        for contrast, _, _, _ in CONTRASTS:
            fit_data = gain_data[gain_data["common_label_number"] != 11].copy()
            outcome_column = f"{outcome}__{contrast}"
            result = fit_gain_model(
                fit_data,
                outcome_column,
                "C(label_number_factor, Treatment(reference='3'))"
                " + common_annotation_disagreement_z"
                " + common_semantic_similarity_z",
            )
            residual = np.asarray(result.resid, dtype=float)
            shapiro_result = shapiro(residual)
            residual_rows.append(
                {
                    "outcome": outcome,
                    "contrast": contrast,
                    "residual_mean": residual.mean(),
                    "residual_standard_deviation": residual.std(ddof=1),
                    "residual_minimum": residual.min(),
                    "residual_maximum": residual.max(),
                    "shapiro_w": float(shapiro_result.statistic),
                    "shapiro_p": float(shapiro_result.pvalue),
                    "maximum_absolute_studentized_residual": float(
                        np.max(np.abs(result.get_influence().resid_studentized_external))
                    ),
                    "maximum_cooks_distance": float(
                        np.max(result.get_influence().cooks_distance[0])
                    ),
                }
            )
    diagnostics = pd.DataFrame(residual_rows)

    support.to_csv(table_dir / "common24_design_support.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(table_dir / "common24_model_summary.csv", index=False, encoding="utf-8-sig")
    omnibus.to_csv(table_dir / "common24_friedman_omnibus.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(table_dir / "common24_pairwise_comparisons.csv", index=False, encoding="utf-8-sig")
    winner_summary.to_csv(table_dir / "common24_winner_summary.csv", index=False, encoding="utf-8-sig")
    gain_coefficients.to_csv(table_dir / "common24_gain_hc3_coefficients.csv", index=False, encoding="utf-8-sig")
    gain_tests.to_csv(table_dir / "common24_gain_hc3_tests.csv", index=False, encoding="utf-8-sig")
    gain_fit.to_csv(table_dir / "common24_gain_model_fit.csv", index=False, encoding="utf-8-sig")
    stratified.to_csv(table_dir / "common24_stratified_summary.csv", index=False, encoding="utf-8-sig")
    gain_data.to_csv(prediction_dir / "common24_task_pairwise_gains.csv", index=False, encoding="utf-8-sig")
    winners.to_csv(prediction_dir / "common24_task_winners.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(diagnostic_dir / "common24_gain_model_diagnostics.csv", index=False, encoding="utf-8-sig")
    leave_one_pair_detail.to_csv(
        diagnostic_dir / "common24_leave_one_out_pairwise_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_one_pair_summary.to_csv(
        diagnostic_dir / "common24_leave_one_out_pairwise_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_one_gain_detail.to_csv(
        diagnostic_dir / "common24_leave_one_out_gain_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_one_gain_summary.to_csv(
        diagnostic_dir / "common24_leave_one_out_gain_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    data.to_csv(diagnostic_dir / "common24_validated_analysis_records.csv", index=False, encoding="utf-8-sig")

    figure_status = "skipped"
    if not args.skip_figures:
        plt = configure_matplotlib()
        plot_overall_comparison(plt, data, summaries, pairwise, figure_dir)
        plot_task_heatmap(plt, data, figure_dir)
        plot_gain_moderation(plt, gain_coefficients, figure_dir)
        plot_winner_map(plt, winners, winner_summary, figure_dir)
        figure_status = "rendered"

    kappa_summary = summaries[summaries["outcome"] == "kappa"].set_index("candidate_short")
    kappa_pairs = pairwise[pairwise["outcome"] == "kappa"].set_index("contrast")
    kappa_wins = winner_summary[winner_summary["outcome"] == "kappa"].set_index("winner_short")
    primary_tests = gain_tests[
        (gain_tests["analysis"] == "primary_categorical_excluding_11")
        & (gain_tests["outcome"] == "kappa")
    ]
    significant_primary = primary_coefficients[
        (primary_coefficients["outcome"] == "kappa")
        &
        (primary_coefficients["term"] != "Intercept")
        & (primary_coefficients["p_value_holm_within_analysis_outcome"] < 0.05)
    ]
    report = {
        "generated_at_utc": utc_now(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "analysis_unit": "24 matched label combinations with one result per candidate; LLM result is the mean of three calls",
        "primary_outcome": "kappa",
        "robustness_outcomes": ["macro_f1", "mcc"],
        "candidate_models": CANDIDATE_ORDER,
        "comparison_scope": "Supervised anchors use 3,000 natural-imbalance training examples. LLM uses codebook-guided prompting without task-specific training.",
        "complexity_moderators": [
            "categorical label number",
            "standardized annotation disagreement",
            "standardized label semantic similarity",
        ],
        "analysis_rows": len(data),
        "combination_count": data["combination_id"].nunique(),
        "kappa_model_means": {
            candidate: float(kappa_summary.loc[candidate, "mean"])
            for candidate in kappa_summary.index
        },
        "kappa_pairwise_differences": {
            contrast: {
                "mean_difference": float(kappa_pairs.loc[contrast, "mean_difference"]),
                "ci95_low": float(kappa_pairs.loc[contrast, "bootstrap_ci95_low"]),
                "ci95_high": float(kappa_pairs.loc[contrast, "bootstrap_ci95_high"]),
                "wilcoxon_p_holm": float(kappa_pairs.loc[contrast, "p_value_holm_within_outcome"]),
                "wins": int(kappa_pairs.loc[contrast, "candidate_a_wins"]),
                "losses": int(kappa_pairs.loc[contrast, "candidate_b_wins"]),
            }
            for contrast in kappa_pairs.index
        },
        "kappa_friedman": omnibus[omnibus["outcome"] == "kappa"].iloc[0].to_dict(),
        "kappa_task_wins": {
            candidate: int(kappa_wins.loc[candidate, "task_wins"])
            for candidate in kappa_wins.index
        },
        "kappa_leave_one_out_pairwise_sign_stability": leave_one_pair_summary[
            leave_one_pair_summary["outcome"] == "kappa"
        ][
            [
                "contrast",
                "full_mean_difference",
                "leave_one_out_minimum",
                "leave_one_out_maximum",
                "same_sign_share",
            ]
        ].to_dict(orient="records"),
        "primary_kappa_complexity_tests": primary_tests.to_dict(orient="records"),
        "holm_significant_primary_kappa_moderators": significant_primary[
            ["contrast", "description", "estimate", "p_value_holm_within_analysis_outcome"]
        ].to_dict(orient="records"),
        "figure_status": figure_status,
        "figure_count": 4 if figure_status == "rendered" else 0,
        "output_checks": {
            "complete_three_candidate_blocks": bool(
                data.groupby("combination_id")["candidate_model"].nunique().eq(3).all()
            ),
            "metric_values_within_minus1_plus1": bool(
                data[OUTCOMES].apply(lambda column: column.between(-1, 1).all()).all()
            ),
            "all_hc3_coefficients_finite": bool(
                np.isfinite(gain_coefficients[["estimate", "hc3_standard_error", "p_value"]]).all().all()
            ),
            "all_bootstrap_intervals_finite": bool(
                np.isfinite(
                    pairwise[["bootstrap_ci95_low", "bootstrap_ci95_high"]]
                ).all().all()
            ),
            "gain_models_have_adequate_residual_df": bool(
                gain_fit["combination_count"].sub(gain_fit["parameters"]).ge(10).all()
            ),
        },
        "support_warning": (
            "The 11-label condition contains one combination. It is retained in overall paired comparisons "
            "but excluded from the primary categorical complexity-moderation regressions."
        ),
        "interpretation_boundary": (
            "Common24 identifies matched cross-model differences at a fixed supervised training design. "
            "It does not identify training-size or class-imbalance moderation, and complexity coefficients "
            "are associative rather than causal."
        ),
    }
    write_json(output_dir / "step5_common24_cross_model_report.json", report)

    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(data),
                "combinations": data["combination_id"].nunique(),
                "output_dir": str(output_dir),
                "kappa_means": report["kappa_model_means"],
                "output_checks": report["output_checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
