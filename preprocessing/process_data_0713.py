from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "processed"
OUT.mkdir(parents=True, exist_ok=True)

RAW_PATH = ROOT / "人工编码一致率.xlsx"
OLD_TEST_PATH = ROOT / "data" / "test.xlsx"
SEMANTIC_MATRIX_PATH = ROOT / "metadata" / "label_semantic_similarity_matrix.xlsx"
RANDOM_SEED = 20260713
TRAINING_SIZES = [100, 300, 500, 1000, 3000, 6000, 9000, 12000, 15000]
LABEL_COUNTS = [3, 5, 8, 11]
LONG_TAIL_EXPONENT = 1.5
MIN_CLASS_SAMPLES = 5

LABEL_NAMES = {
    0: "Interpretations",
    1: "Task descriptions",
    2: "Communication",
    3: "Positional power",
    4: "Rights entitlement",
    5: "Obligations",
    6: "Penalties",
    7: "Behavior restrictions",
    8: "Adjudication",
    9: "Environmental changes",
    10: "Task changes",
}

TEST_QUOTAS = {0: 100, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100, 7: 100, 8: 33, 9: 100, 10: 58}

# Manual adjudication of the 101 exact-text groups with tied final-label votes.
# Keys are the deterministic tie sequence produced by the source workbook order.
TIE_ADJUDICATION_PATH = ROOT / "private_inputs" / "tie_adjudications.json"
if not TIE_ADJUDICATION_PATH.exists():
    raise FileNotFoundError("Private manual adjudications are required: private_inputs/tie_adjudications.json")
TIE_ADJUDICATIONS = {int(k): tuple(v) for k, v in json.loads(TIE_ADJUDICATION_PATH.read_text(encoding="utf-8")).items()}


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def normalized_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def imbalance_index(counts: list[int] | np.ndarray) -> float:
    arr = np.asarray(counts, dtype=float)
    arr = arr[arr > 0]
    if len(arr) <= 1:
        return 0.0
    p = arr / arr.sum()
    return float(1.0 + np.sum(p * np.log(p)) / np.log(len(arr)))


def allocation_largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    raw = weights * total
    allocation = np.floor(raw).astype(int)
    remainder = total - int(allocation.sum())
    if remainder:
        order = np.argsort(-(raw - allocation), kind="stable")
        allocation[order[:remainder]] += 1
    return allocation


def allocation_for_mode(available: np.ndarray, total: int, mode: str) -> np.ndarray:
    k = len(available)
    reserved = k * MIN_CLASS_SAMPLES
    if total < reserved:
        raise ValueError(
            f"Training size {total} cannot provide at least {MIN_CLASS_SAMPLES} samples "
            f"for each of {k} labels"
        )
    if mode == "balanced":
        weights = np.ones(k, dtype=float)
    elif mode == "natural_imbalance":
        weights = available.astype(float)
    elif mode == "enhanced_long_tail":
        weights = available.astype(float) ** LONG_TAIL_EXPONENT
    else:
        raise ValueError(mode)
    return np.full(k, MIN_CLASS_SAMPLES, dtype=int) + allocation_largest_remainder(weights, total - reserved)


def counts_string(labels: tuple[int, ...], counts: np.ndarray) -> str:
    return "; ".join(f"{label}:{int(count)}" for label, count in zip(labels, counts))


def read_semantic_pairs() -> dict[tuple[int, int], float]:
    matrix = pd.read_excel(SEMANTIC_MATRIX_PATH)
    row_name_col = matrix.columns[0]
    id_by_name = {normalized_name(name): label for label, name in LABEL_NAMES.items()}
    pairs: dict[tuple[int, int], float] = {}
    for _, row in matrix.iterrows():
        row_id = id_by_name[normalized_name(row[row_name_col])]
        for column in matrix.columns[1:]:
            col_id = id_by_name[normalized_name(column)]
            if row_id == col_id:
                continue
            key = tuple(sorted((row_id, col_id)))
            pairs[key] = float(row[column])
    if len(pairs) != 55:
        raise ValueError(f"Expected 55 semantic pairs, found {len(pairs)}")
    return pairs


def build_cleaned_data(raw: pd.DataFrame, old_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = raw.rename(columns={"text": "text", "labels": "original_label", "是否一致": "annotation_disagreement"}).copy()
    raw["source_row_id"] = np.arange(1, len(raw) + 1)
    raw["normalized_text"] = raw["text"].map(normalize_text)
    raw["original_label"] = raw["original_label"].astype(int)
    raw["annotation_disagreement"] = raw["annotation_disagreement"].astype(float)

    old_test_norms = Counter(old_test.iloc[:, 0].map(normalize_text).tolist())
    records = []
    audit_rows = []
    tie_rows = []
    tie_index = 0

    for normalized_text, group in raw.groupby("normalized_text", sort=False):
        votes = group["original_label"].value_counts().sort_index()
        max_votes = int(votes.max())
        winners = sorted(votes[votes == max_votes].index.astype(int).tolist())
        vote_string = "; ".join(f"{int(label)}:{int(count)}" for label, count in votes.items())
        common = {
            "normalized_text": normalized_text,
            "representative_text": str(group.iloc[0]["text"]),
            "source_row_ids": ",".join(map(str, group["source_row_id"].astype(int).tolist())),
            "duplicate_count": int(len(group)),
            "distinct_label_count": int(len(votes)),
            "label_vote_counts": vote_string,
            "top_vote_count": max_votes,
            "top_vote_share": max_votes / len(group),
            "mean_disagreement_all_occurrences": float(group["annotation_disagreement"].mean()),
            "original_test_occurrence_count": int(old_test_norms.get(normalized_text, 0)),
        }
        if len(winners) > 1:
            if tie_index not in TIE_ADJUDICATIONS:
                raise ValueError(f"Missing manual adjudication for tie index {tie_index}")
            majority_label, rationale = TIE_ADJUDICATIONS[tie_index]
            if majority_label not in LABEL_NAMES:
                raise ValueError(f"Invalid adjudicated label {majority_label} for tie index {tie_index}")
            tie_rows.append(
                {
                    **common,
                    "tie_adjudication_id": tie_index,
                    "tied_labels": ",".join(map(str, winners)),
                    "adjudicated_label": majority_label,
                    "adjudicated_label_name": LABEL_NAMES[majority_label],
                    "adjudication_rationale": rationale,
                    "decision": "tie_adjudicated_by_contract_function",
                }
            )
            tie_index += 1
        else:
            majority_label = winners[0]
            rationale = ""

        majority_rows = group[group["original_label"] == majority_label]
        representative = majority_rows.iloc[0] if not majority_rows.empty else group.iloc[0]
        record = {
            "record_id": len(records) + 1,
            "text": str(representative["text"]),
            "normalized_text": normalized_text,
            "label": majority_label,
            "label_name": LABEL_NAMES[majority_label],
            "annotation_disagreement_rate": float(group["annotation_disagreement"].mean()),
            "majority_label_disagreement_rate": float(majority_rows["annotation_disagreement"].mean()) if not majority_rows.empty else float(group["annotation_disagreement"].mean()),
            "duplicate_count": int(len(group)),
            "distinct_label_count": int(len(votes)),
            "label_vote_counts": vote_string,
            "majority_vote_share": max_votes / len(group),
            "source_row_ids": common["source_row_ids"],
            "original_test_occurrence_count": common["original_test_occurrence_count"],
            "tie_adjudicated": int(len(winners) > 1),
            "adjudication_rationale": rationale,
        }
        records.append(record)
        if len(winners) > 1:
            decision = "tie_adjudicated_by_contract_function"
        else:
            decision = "kept_unique" if len(group) == 1 else ("kept_duplicate_same_label" if len(votes) == 1 else "kept_majority_label")
        audit_rows.append({**common, "majority_label": majority_label, "majority_label_name": LABEL_NAMES[majority_label], "decision": decision})

    if tie_index != len(TIE_ADJUDICATIONS):
        raise ValueError(f"Expected {len(TIE_ADJUDICATIONS)} tied groups, found {tie_index}")

    cleaned = pd.DataFrame(records)
    audit = pd.DataFrame(audit_rows)
    ties = pd.DataFrame(tie_rows)
    return cleaned, audit, ties


def create_fixed_test(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_parts = []
    rng = np.random.default_rng(RANDOM_SEED)
    for label, quota in TEST_QUOTAS.items():
        candidates = cleaned[cleaned["label"] == label].copy()
        preserved = candidates[candidates["original_test_occurrence_count"] > 0].copy()
        if len(preserved) > quota:
            keep_index = rng.choice(preserved.index.to_numpy(), size=quota, replace=False)
            preserved = preserved.loc[sorted(keep_index)]
        preserved["test_selection_source"] = "preserved_original_test_unique_text"
        needed = quota - len(preserved)
        if needed > 0:
            remaining = candidates.drop(index=preserved.index)
            if len(remaining) < needed:
                raise ValueError(f"Label {label} has only {len(candidates)} unique eligible texts for quota {quota}")
            top_up_index = rng.choice(remaining.index.to_numpy(), size=needed, replace=False)
            top_up = remaining.loc[sorted(top_up_index)].copy()
            top_up["test_selection_source"] = "random_top_up_after_deduplication"
            selected = pd.concat([preserved, top_up], ignore_index=False)
        else:
            selected = preserved
        selected_parts.append(selected)

    test = pd.concat(selected_parts, ignore_index=False).sort_values(["label", "record_id"]).copy()
    test["test_row_id"] = np.arange(1, len(test) + 1)
    train = cleaned.drop(index=test.index).sort_values(["label", "record_id"]).copy()
    train["train_pool_row_id"] = np.arange(1, len(train) + 1)

    if len(test) != sum(TEST_QUOTAS.values()):
        raise AssertionError(f"Expected 991 test rows, found {len(test)}")
    if test["normalized_text"].duplicated().any() or train["normalized_text"].duplicated().any():
        raise AssertionError("Duplicate normalized texts remain after cleaning")
    if set(test["normalized_text"]) & set(train["normalized_text"]):
        raise AssertionError("Train-test normalized text leakage remains")
    if test["label"].value_counts().sort_index().to_dict() != TEST_QUOTAS:
        raise AssertionError("Test label quotas do not match the fixed design")
    return test.reset_index(drop=True), train.reset_index(drop=True)


def build_label_stats(raw: pd.DataFrame, cleaned: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, name in LABEL_NAMES.items():
        raw_label = raw[raw["labels"] == label]
        clean_label = cleaned[cleaned["label"] == label]
        test_label = test[test["label"] == label]
        train_label = train[train["label"] == label]
        rows.append(
            {
                "label_id": label,
                "label_name": name,
                "raw_occurrence_n": len(raw_label),
                "cleaned_unique_n": len(clean_label),
                "fixed_test_n": len(test_label),
                "available_train_pool_n": len(train_label),
                "raw_disagreement_rate": float(raw_label["是否一致"].mean()),
                "cleaned_unique_disagreement_rate": float(clean_label["annotation_disagreement_rate"].mean()),
                "train_pool_disagreement_rate": float(train_label["annotation_disagreement_rate"].mean()),
                "train_pool_duplicate_source_occurrences": int(train_label["duplicate_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_combinations_and_tasks(label_stats: pd.DataFrame, semantic_pairs: dict[tuple[int, int], float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = label_stats.set_index("label_id")["available_train_pool_n"].to_dict()
    disagreement = label_stats.set_index("label_id")["train_pool_disagreement_rate"].to_dict()
    combination_rows = []
    task_rows = []

    for k in LABEL_COUNTS:
        for combo_index, labels in enumerate(combinations(LABEL_NAMES.keys(), k), start=1):
            labels = tuple(int(x) for x in labels)
            counts = np.array([available[x] for x in labels], dtype=int)
            rates = np.array([disagreement[x] for x in labels], dtype=float)
            pair_values = np.array([semantic_pairs[tuple(sorted(pair))] for pair in combinations(labels, 2)], dtype=float)
            combo_id = f"K{k:02d}_C{combo_index:03d}"
            natural_imbalance = imbalance_index(counts)
            weighted_disagreement = float(np.average(rates, weights=counts))
            unweighted_disagreement = float(rates.mean())

            mode_feasible_sizes: dict[str, list[int]] = {}
            for mode in ["balanced", "natural_imbalance", "enhanced_long_tail"]:
                feasible_sizes = []
                for n in TRAINING_SIZES:
                    allocation = allocation_for_mode(counts, n, mode)
                    feasible = bool(np.all(allocation <= counts))
                    if feasible:
                        feasible_sizes.append(n)
                    task_rows.append(
                        {
                            "candidate_task_id": f"{combo_id}_{mode}_N{n}",
                            "combination_id": combo_id,
                            "label_number": k,
                            "selected_label_ids": ",".join(map(str, labels)),
                            "selected_labels": "; ".join(f"{x}={LABEL_NAMES[x]}" for x in labels),
                            "class_balance_design": mode,
                            "requested_training_sample_size": n,
                            "feasible": feasible,
                            "requested_samples_per_label": counts_string(labels, allocation),
                            "available_samples_per_label": counts_string(labels, counts),
                            "realized_class_imbalance_index": imbalance_index(allocation),
                            "expected_annotation_disagreement_rate": float(np.average(rates, weights=allocation)),
                            "avg_pairwise_semantic_similarity": float(pair_values.mean()),
                            "max_pairwise_semantic_similarity": float(pair_values.max()),
                            "infeasibility_reason": "" if feasible else "requested per-label allocation exceeds the deduplicated training pool",
                        }
                    )
                mode_feasible_sizes[mode] = feasible_sizes

            combination_rows.append(
                {
                    "combination_id": combo_id,
                    "label_number": k,
                    "selected_label_ids": ",".join(map(str, labels)),
                    "selected_labels": "; ".join(f"{x}={LABEL_NAMES[x]}" for x in labels),
                    "available_samples_per_label": counts_string(labels, counts),
                    "available_total": int(counts.sum()),
                    "minimum_label_available_n": int(counts.min()),
                    "maximum_label_available_n": int(counts.max()),
                    "natural_class_imbalance_index": natural_imbalance,
                    "weighted_annotation_disagreement_rate": weighted_disagreement,
                    "unweighted_label_disagreement_rate": unweighted_disagreement,
                    "minimum_label_disagreement_rate": float(rates.min()),
                    "maximum_label_disagreement_rate": float(rates.max()),
                    "pair_count": len(pair_values),
                    "avg_pairwise_semantic_similarity": float(pair_values.mean()),
                    "max_pairwise_semantic_similarity": float(pair_values.max()),
                    "min_pairwise_semantic_similarity": float(pair_values.min()),
                    "feasible_balanced_sizes": ",".join(map(str, mode_feasible_sizes["balanced"])),
                    "feasible_natural_sizes": ",".join(map(str, mode_feasible_sizes["natural_imbalance"])),
                    "feasible_long_tail_sizes": ",".join(map(str, mode_feasible_sizes["enhanced_long_tail"])),
                    "balanced_max_feasible_n": max(mode_feasible_sizes["balanced"], default=0),
                    "natural_max_feasible_n": max(mode_feasible_sizes["natural_imbalance"], default=0),
                    "long_tail_max_feasible_n": max(mode_feasible_sizes["enhanced_long_tail"], default=0),
                }
            )

    combinations_df = pd.DataFrame(combination_rows)
    tasks_df = pd.DataFrame(task_rows)
    for k in LABEL_COUNTS:
        mask = combinations_df["label_number"] == k
        for source, target in [
            ("avg_pairwise_semantic_similarity", "semantic_percentile_within_k"),
            ("unweighted_label_disagreement_rate", "disagreement_percentile_within_k"),
        ]:
            combinations_df.loc[mask, target] = combinations_df.loc[mask, source].rank(pct=True, method="average")
    combinations_df["complexity_quadrant"] = np.select(
        [
            (combinations_df["semantic_percentile_within_k"] <= 0.5) & (combinations_df["disagreement_percentile_within_k"] <= 0.5),
            (combinations_df["semantic_percentile_within_k"] <= 0.5) & (combinations_df["disagreement_percentile_within_k"] > 0.5),
            (combinations_df["semantic_percentile_within_k"] > 0.5) & (combinations_df["disagreement_percentile_within_k"] <= 0.5),
            (combinations_df["semantic_percentile_within_k"] > 0.5) & (combinations_df["disagreement_percentile_within_k"] > 0.5),
        ],
        ["low_semantic_low_disagreement", "low_semantic_high_disagreement", "high_semantic_low_disagreement", "high_semantic_high_disagreement"],
        default="unclassified",
    )
    return combinations_df, tasks_df


def main() -> None:
    stale_output = OUT / "tied_label_conflicts_excluded.csv"
    if stale_output.exists():
        stale_output.unlink()
    raw = pd.read_excel(RAW_PATH)
    old_test = pd.read_excel(OLD_TEST_PATH)
    semantic_pairs = read_semantic_pairs()

    cleaned, duplicate_audit, tied_conflicts = build_cleaned_data(raw, old_test)
    fixed_test, train_pool = create_fixed_test(cleaned)
    label_stats = build_label_stats(raw, cleaned, fixed_test, train_pool)
    label_combinations, task_candidates = build_combinations_and_tasks(label_stats, semantic_pairs)
    feasible_tasks = task_candidates[task_candidates["feasible"]].copy()

    outputs = {
        "cleaned_unique_data.csv": cleaned,
        "duplicate_decision_audit.csv": duplicate_audit,
        "tied_label_adjudications.csv": tied_conflicts,
        "fixed_test_991.csv": fixed_test,
        "deduplicated_training_pool.csv": train_pool,
        "label_statistics.csv": label_stats,
        "all_label_combinations.csv": label_combinations,
        "all_task_candidates.csv": task_candidates,
        "feasible_task_candidates.csv": feasible_tasks,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    summary = {
        "source_file": str(RAW_PATH.name),
        "random_seed": RANDOM_SEED,
        "normalization": "Unicode NFKC, collapse all whitespace, strip leading/trailing whitespace, case preserved",
        "raw_rows": int(len(raw)),
        "normalized_unique_text_groups": int(len(duplicate_audit)),
        "duplicate_groups": int((duplicate_audit["duplicate_count"] > 1).sum()),
        "rows_removed_by_grouping_before_ties": int(len(raw) - len(duplicate_audit)),
        "cross_label_duplicate_groups": int((duplicate_audit["distinct_label_count"] > 1).sum()),
        "tied_majority_groups_adjudicated": int(len(tied_conflicts)),
        "cleaned_unique_rows": int(len(cleaned)),
        "fixed_test_rows": int(len(fixed_test)),
        "training_pool_rows": int(len(train_pool)),
        "train_test_normalized_text_overlap": int(len(set(fixed_test["normalized_text"]) & set(train_pool["normalized_text"]))),
        "label_combination_rows": int(len(label_combinations)),
        "task_candidate_rows": int(len(task_candidates)),
        "feasible_task_candidate_rows": int(len(feasible_tasks)),
        "training_sizes": TRAINING_SIZES,
        "label_counts": LABEL_COUNTS,
        "long_tail_exponent": LONG_TAIL_EXPONENT,
        "minimum_samples_per_class": MIN_CLASS_SAMPLES,
        "fixed_test_quotas": TEST_QUOTAS,
    }
    (OUT / "processing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nLabel statistics:\n", label_stats.to_string(index=False))
    print("\nCombination counts:\n", label_combinations.groupby("label_number").size().to_string())
    print("\nFeasible task counts:\n", feasible_tasks.groupby(["label_number", "class_balance_design"]).size().to_string())


if __name__ == "__main__":
    main()
