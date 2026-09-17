from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "processed"
INPUT = PROCESSED / "all_label_combinations.csv"

FEATURES = [
    "avg_pairwise_semantic_similarity",
    "unweighted_label_disagreement_rate",
    "natural_class_imbalance_index",
]
COMMON_QUOTAS = {3: 8, 5: 8, 8: 7, 11: 1}
LLM_QUOTAS = {3: 12, 5: 15, 8: 12, 11: 1}


def label_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in str(value).split(","))


def standardize(frame: pd.DataFrame) -> np.ndarray:
    values = frame[FEATURES].to_numpy(float)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return (values - values.mean(axis=0)) / scale


def select_maximin(
    candidates: pd.DataFrame,
    quota: int,
    initial_ids: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    candidates = candidates.reset_index(drop=True).copy()
    z = standardize(candidates)
    labels = [set(label_tuple(x)) for x in candidates["selected_label_ids"]]
    id_to_idx = {value: idx for idx, value in enumerate(candidates["combination_id"])}
    selected = [id_to_idx[x] for x in (initial_ids or []) if x in id_to_idx]
    reasons = {candidates.loc[idx, "combination_id"]: "included_from_common_subset" for idx in selected}

    # Seed the design with one point from each semantic-disagreement quadrant.
    quadrant_targets = {
        "low_semantic_low_disagreement": np.array([-1.0, -1.0]),
        "low_semantic_high_disagreement": np.array([-1.0, 1.0]),
        "high_semantic_low_disagreement": np.array([1.0, -1.0]),
        "high_semantic_high_disagreement": np.array([1.0, 1.0]),
    }
    for quadrant, target in quadrant_targets.items():
        if len(selected) >= quota:
            break
        pool = candidates.index[candidates["complexity_quadrant"] == quadrant].tolist()
        pool = [idx for idx in pool if idx not in selected]
        if not pool:
            continue
        distances = [np.linalg.norm(z[idx, :2] - target) for idx in pool]
        chosen = pool[int(np.argmin(distances))]
        selected.append(chosen)
        reasons[candidates.loc[chosen, "combination_id"]] = f"quadrant_anchor:{quadrant}"

    while len(selected) < quota:
        best_idx = None
        best_score = -np.inf
        for idx in candidates.index:
            if idx in selected:
                continue
            if selected:
                feature_distance = min(np.linalg.norm(z[idx] - z[j]) for j in selected)
                max_jaccard = max(len(labels[idx] & labels[j]) / len(labels[idx] | labels[j]) for j in selected)
                label_diversity = 1.0 - max_jaccard
            else:
                feature_distance = float(np.linalg.norm(z[idx]))
                label_diversity = 1.0
            score = feature_distance + 0.35 * label_diversity
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        reasons[candidates.loc[best_idx, "combination_id"]] = "maximin_space_filling"

    return [candidates.loc[idx, "combination_id"] for idx in selected], reasons


def build_selection(combos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_ids: list[str] = []
    common_reasons: dict[str, str] = {}
    for k, quota in COMMON_QUOTAS.items():
        subset = combos[combos["label_number"] == k].copy()
        if k != 11:
            subset = subset[(subset["natural_max_feasible_n"] >= 3000) & (subset["balanced_max_feasible_n"] >= 500)]
        selected, reasons = select_maximin(subset, quota)
        common_ids.extend(selected)
        common_reasons.update(reasons)

    common = combos[combos["combination_id"].isin(common_ids)].copy()
    common["selection_scope"] = "common_cross_model_24"
    common["selection_reason"] = common["combination_id"].map(common_reasons)
    common["selection_order"] = common["combination_id"].map({value: i + 1 for i, value in enumerate(common_ids)})
    common = common.sort_values("selection_order")

    llm_ids: list[str] = []
    llm_reasons: dict[str, str] = {}
    for k, quota in LLM_QUOTAS.items():
        subset = combos[combos["label_number"] == k].copy()
        initial = [x for x in common_ids if x in set(subset["combination_id"])]
        selected, reasons = select_maximin(subset, quota, initial_ids=initial)
        llm_ids.extend(selected)
        llm_reasons.update(reasons)

    llm = combos[combos["combination_id"].isin(llm_ids)].copy()
    llm["selection_scope"] = "llm_extended_40"
    llm["selection_reason"] = llm["combination_id"].map(llm_reasons)
    llm["selection_order"] = llm["combination_id"].map({value: i + 1 for i, value in enumerate(llm_ids)})
    llm = llm.sort_values("selection_order")

    summary_rows = []
    for name, frame in [("all_793", combos), ("common_24", common), ("llm_40", llm)]:
        corr = frame["avg_pairwise_semantic_similarity"].corr(frame["unweighted_label_disagreement_rate"])
        summary_rows.append(
            {
                "selection": name,
                "combination_count": len(frame),
                "semantic_disagreement_correlation": corr,
                "k3_count": int((frame["label_number"] == 3).sum()),
                "k5_count": int((frame["label_number"] == 5).sum()),
                "k8_count": int((frame["label_number"] == 8).sum()),
                "k11_count": int((frame["label_number"] == 11).sum()),
                "low_low_count": int((frame["complexity_quadrant"] == "low_semantic_low_disagreement").sum()),
                "low_high_count": int((frame["complexity_quadrant"] == "low_semantic_high_disagreement").sum()),
                "high_low_count": int((frame["complexity_quadrant"] == "high_semantic_low_disagreement").sum()),
                "high_high_count": int((frame["complexity_quadrant"] == "high_semantic_high_disagreement").sum()),
            }
        )
    return common, llm, pd.DataFrame(summary_rows)


def main() -> None:
    combos = pd.read_csv(INPUT)
    common, llm, summary = build_selection(combos)
    common.to_csv(PROCESSED / "recommended_common_24_label_sets.csv", index=False, encoding="utf-8-sig")
    llm.to_csv(PROCESSED / "recommended_llm_40_label_sets.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(PROCESSED / "selection_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print("\nCommon subset by K/quadrant:\n", pd.crosstab(common["label_number"], common["complexity_quadrant"]).to_string())
    print("\nLLM subset by K/quadrant:\n", pd.crosstab(llm["label_number"], llm["complexity_quadrant"]).to_string())


if __name__ == "__main__":
    main()
