from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROCESSED = HERE / "processed"
OUTPUT = HERE / "generated_datasets"
INTERMEDIATE = OUTPUT / "intermediate_json"
RANDOM_SEED = 20260713

TRAIN_POOL_PATH = PROCESSED / "deduplicated_training_pool.csv"
FIXED_TEST_PATH = PROCESSED / "fixed_test_991.csv"
COMMON_PATH = PROCESSED / "recommended_common_24_label_sets.csv"
LLM_PATH = PROCESSED / "recommended_llm_40_label_sets.csv"
FEASIBLE_PATH = PROCESSED / "feasible_task_candidates.csv"
LEGACY_EXAMPLES_PATH = ROOT / "data" / "deepseek_v4_11_label_examples.xlsx"

LABELS = {
    0: {
        "name": "Interpretations",
        "function": "Coordination",
        "description": "Provisions that clarify the meanings of terms to support a shared understanding between the parties.",
    },
    1: {
        "name": "Task descriptions",
        "function": "Coordination",
        "description": "Provisions that describe each party's tasks for completing the work.",
    },
    2: {
        "name": "Communication",
        "function": "Coordination",
        "description": "Provisions concerning information exchange and sharing between the parties.",
    },
    3: {
        "name": "Positional power",
        "function": "Coordination",
        "description": "Provisions that authorize positional power for a party during the work process.",
    },
    4: {
        "name": "Rights entitlement",
        "function": "Control",
        "description": "Provisions that endow each party with rights for protecting its assets or interests.",
    },
    5: {
        "name": "Obligations",
        "function": "Control",
        "description": "Provisions that oblige each party to perform responsibilities and comply with allocated risks, rules, and laws.",
    },
    6: {
        "name": "Penalties",
        "function": "Control",
        "description": "Provisions concerning compensation, damages, or additional rights arising from another party's failure to perform.",
    },
    7: {
        "name": "Behavior restrictions",
        "function": "Control",
        "description": "Provisions that constrain a party from performing specified actions.",
    },
    8: {
        "name": "Adjudication",
        "function": "Control",
        "description": "Provisions concerning the arbitration or adjudication of disputes between the parties.",
    },
    9: {
        "name": "Environmental changes",
        "function": "Adaptation",
        "description": "Provisions concerning responses to future natural or institutional changes.",
    },
    10: {
        "name": "Task changes",
        "function": "Adaptation",
        "description": "Provisions concerning future changes to work scope, standards, or sequence.",
    },
}

CORE_NATURAL_SIZES = [300, 500, 1000, 3000]
HIGH_N_QUOTAS = {6000: 12, 9000: 6, 12000: 4, 15000: 2}


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, (RANDOM_SEED, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def parse_label_ids(value: object) -> tuple[int, ...]:
    return tuple(int(item) for item in str(value).split(","))


def parse_counts(value: object) -> dict[int, int]:
    result: dict[int, int] = {}
    for item in str(value).split(";"):
        label, count = item.strip().split(":")
        result[int(label)] = int(count)
    return result


def normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def imbalance_index(counts: list[int] | np.ndarray) -> float:
    values = np.asarray(counts, dtype=float)
    proportions = values / values.sum()
    return float(1.0 + np.sum(proportions * np.log(proportions)) / np.log(len(values)))


def task_lookup(frame: pd.DataFrame, combo_id: str, mode: str, size: int) -> pd.Series:
    rows = frame[
        (frame["combination_id"] == combo_id)
        & (frame["class_balance_design"] == mode)
        & (frame["requested_training_sample_size"] == size)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one feasible task for {combo_id}, {mode}, N={size}; found {len(rows)}")
    return rows.iloc[0]


def feature_matrix(frame: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    continuous = pd.DataFrame(
        {
            "log_n": np.log(frame["requested_training_sample_size"].astype(float)),
            "imbalance": frame["realized_class_imbalance_index"].astype(float),
            "disagreement": frame["expected_annotation_disagreement_rate"].astype(float),
            "semantic": frame["avg_pairwise_semantic_similarity"].astype(float),
        },
        index=frame.index,
    )
    reference_continuous = pd.DataFrame(
        {
            "log_n": np.log(reference["requested_training_sample_size"].astype(float)),
            "imbalance": reference["realized_class_imbalance_index"].astype(float),
            "disagreement": reference["expected_annotation_disagreement_rate"].astype(float),
            "semantic": reference["avg_pairwise_semantic_similarity"].astype(float),
        },
        index=reference.index,
    )
    means = reference_continuous.mean()
    scales = reference_continuous.std(ddof=0).replace(0, 1.0)
    z = (continuous - means) / scales
    k = frame["label_number"].astype(int)
    mode = frame["class_balance_design"].astype(str)
    matrix = np.column_stack(
        [
            np.ones(len(frame)),
            (k == 5).astype(float),
            (k == 8).astype(float),
            (k == 11).astype(float),
            z["log_n"],
            z["imbalance"],
            z["disagreement"],
            z["semantic"],
            (mode == "balanced").astype(float),
            (mode == "enhanced_long_tail").astype(float),
            z["log_n"] * z["imbalance"],
            z["semantic"] * z["disagreement"],
        ]
    )
    return matrix.astype(float)


def select_d_optimal_high_n(
    feasible_common: pd.DataFrame,
    already_selected: pd.DataFrame,
) -> pd.DataFrame:
    reference = feasible_common.copy()
    selected = already_selected.copy()
    selected_x = feature_matrix(selected, reference)
    information = selected_x.T @ selected_x + np.eye(selected_x.shape[1]) * 1e-6
    inverse = np.linalg.inv(information)
    selected_rows: list[pd.Series] = []
    high_count_by_combo: dict[str, int] = {}
    high_count_by_k: dict[int, int] = {}

    for size, quota in sorted(HIGH_N_QUOTAS.items(), reverse=True):
        pool = feasible_common[
            (feasible_common["requested_training_sample_size"] == size)
            & (~feasible_common["candidate_task_id"].isin(already_selected["candidate_task_id"]))
        ].copy()
        if len(pool) < quota:
            raise ValueError(f"Only {len(pool)} candidates available for N={size}, need {quota}")
        pool_x = feature_matrix(pool, reference)
        pool_indices = list(pool.index)
        x_by_index = {idx: pool_x[pos] for pos, idx in enumerate(pool_indices)}

        for _ in range(quota):
            best_index = None
            best_score = -math.inf
            for idx, row in pool.iterrows():
                combo_id = str(row["combination_id"])
                if high_count_by_combo.get(combo_id, 0) >= 3:
                    continue
                x = x_by_index[idx]
                leverage = float(x @ inverse @ x)
                combo_bonus = 1.0 + 0.20 / (1 + high_count_by_combo.get(combo_id, 0))
                k_bonus = 1.0 + 0.10 / (1 + high_count_by_k.get(int(row["label_number"]), 0))
                score = leverage * combo_bonus * k_bonus
                if score > best_score:
                    best_score = score
                    best_index = idx
            if best_index is None:
                raise ValueError(f"Could not complete D-optimal selection for N={size}")
            chosen = pool.loc[best_index]
            x = x_by_index[best_index]
            denominator = 1.0 + float(x @ inverse @ x)
            inverse = inverse - np.outer(inverse @ x, x @ inverse) / denominator
            selected_rows.append(chosen)
            combo_id = str(chosen["combination_id"])
            k = int(chosen["label_number"])
            high_count_by_combo[combo_id] = high_count_by_combo.get(combo_id, 0) + 1
            high_count_by_k[k] = high_count_by_k.get(k, 0) + 1
            pool = pool.drop(index=best_index)

    result = pd.DataFrame(selected_rows).reset_index(drop=True)
    if len(result) != sum(HIGH_N_QUOTAS.values()):
        raise AssertionError("High-N D-optimal selection did not produce 24 tasks")
    return result


def select_supervised_tasks(feasible: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    common_ids = common.sort_values("selection_order")["combination_id"].tolist()
    feasible_common = feasible[feasible["combination_id"].isin(common_ids)].copy()
    selected_parts: list[pd.DataFrame] = []

    core_rows = []
    for combo_id in common_ids:
        for size in CORE_NATURAL_SIZES:
            row = task_lookup(feasible_common, combo_id, "natural_imbalance", size).copy()
            row["design_tier"] = "core_learning_curve"
            core_rows.append(row)
        for mode, size in [
            ("balanced", 300),
            ("balanced", 500),
            ("enhanced_long_tail", 500),
        ]:
            row = task_lookup(feasible_common, combo_id, mode, size).copy()
            row["design_tier"] = f"core_imbalance_anchor_N{size}"
            core_rows.append(row)
    core = pd.DataFrame(core_rows)
    selected_parts.append(core)

    contrast_rows = []
    for combo_id in common_ids:
        row = task_lookup(feasible_common, combo_id, "enhanced_long_tail", 3000).copy()
        row["design_tier"] = "moderate_long_tail_N3000"
        contrast_rows.append(row)
        balanced_1000 = feasible_common[
            (feasible_common["combination_id"] == combo_id)
            & (feasible_common["class_balance_design"] == "balanced")
            & (feasible_common["requested_training_sample_size"] == 1000)
        ]
        if len(balanced_1000) == 1:
            row = balanced_1000.iloc[0].copy()
            row["design_tier"] = "moderate_balanced_N1000"
        else:
            row = task_lookup(feasible_common, combo_id, "enhanced_long_tail", 1000).copy()
            row["design_tier"] = "moderate_long_tail_N1000_fallback"
        contrast_rows.append(row)
    contrasts = pd.DataFrame(contrast_rows)
    selected_parts.append(contrasts)

    initial = pd.concat(selected_parts, ignore_index=True)
    high = select_d_optimal_high_n(feasible_common, initial)
    high["design_tier"] = "d_optimal_high_n"
    selected_parts.append(high)

    tasks = pd.concat(selected_parts, ignore_index=True)
    if len(tasks) != 240 or tasks["candidate_task_id"].duplicated().any():
        raise AssertionError("The supervised design must contain 240 unique tasks")
    common_order = common.set_index("combination_id")["selection_order"].to_dict()
    tasks["common24_order"] = tasks["combination_id"].map(common_order).astype(int)
    tier_order = {
        "core_learning_curve": 1,
        "core_imbalance_anchor_N300": 2,
        "core_imbalance_anchor_N500": 2,
        "moderate_long_tail_N3000": 3,
        "moderate_balanced_N1000": 4,
        "moderate_long_tail_N1000_fallback": 4,
        "d_optimal_high_n": 5,
    }
    tasks["tier_order"] = tasks["design_tier"].map(tier_order)
    tasks = tasks.sort_values(
        ["common24_order", "tier_order", "requested_training_sample_size", "class_balance_design", "candidate_task_id"]
    ).reset_index(drop=True)
    tasks.insert(0, "task_no", np.arange(1, len(tasks) + 1))
    tasks.insert(
        1,
        "task_id",
        [f"SUP{number:03d}_{candidate}" for number, candidate in zip(tasks["task_no"], tasks["candidate_task_id"])],
    )
    tasks["comparison_anchor"] = (
        (tasks["class_balance_design"] == "natural_imbalance")
        & (tasks["requested_training_sample_size"] == 3000)
    ).astype(int)
    return tasks


def make_label_codebook() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "label_id": label,
                "label_name": data["name"],
                "function": data["function"],
                "description": data["description"],
            }
            for label, data in LABELS.items()
        ]
    )


def ordered_training_pool(train_pool: pd.DataFrame, combo_id: str, label: int) -> pd.DataFrame:
    subset = train_pool[train_pool["label"] == label].copy()
    rng = np.random.default_rng(stable_seed("sample_order", combo_id, label))
    subset["sampling_key"] = rng.random(len(subset))
    subset = subset.sort_values(["sampling_key", "train_pool_row_id"]).reset_index(drop=True)
    subset["within_label_rank"] = np.arange(1, len(subset) + 1)
    return subset.drop(columns=["sampling_key"])


def prepare_supervised_payloads(
    tasks: pd.DataFrame,
    train_pool: pd.DataFrame,
    fixed_test: pd.DataFrame,
    codebook: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    payloads: dict[str, dict[str, object]] = {}
    order_cache: dict[tuple[str, int], pd.DataFrame] = {}
    manifest_rows = []

    for _, task in tasks.iterrows():
        task_id = str(task["task_id"])
        combo_id = str(task["combination_id"])
        labels = parse_label_ids(task["selected_label_ids"])
        quotas = parse_counts(task["requested_samples_per_label"])
        mapping = {label: index for index, label in enumerate(labels)}
        development_parts = []
        for label in labels:
            cache_key = (combo_id, label)
            if cache_key not in order_cache:
                order_cache[cache_key] = ordered_training_pool(train_pool, combo_id, label)
            selected = order_cache[cache_key].head(quotas[label]).copy()
            if len(selected) != quotas[label]:
                raise AssertionError(f"Insufficient samples for {task_id}, label {label}")
            selected["continuous_label"] = mapping[label]
            selected["split"] = np.where(
                (selected["within_label_rank"] - 1) % 10 == 0,
                "validation",
                "fit_train",
            )
            development_parts.append(selected)
        development = pd.concat(development_parts, ignore_index=True)
        development.insert(0, "task_id", task_id)
        development.insert(1, "sample_id", [f"{task_id}_D{i:05d}" for i in range(1, len(development) + 1)])

        test = fixed_test[fixed_test["label"].isin(labels)].copy()
        test["continuous_label"] = test["label"].map(mapping).astype(int)
        test.insert(0, "task_id", task_id)
        test.insert(1, "sample_id", [f"{task_id}_T{i:04d}" for i in range(1, len(test) + 1)])

        sampled_counts = development["label"].value_counts().reindex(labels).astype(int)
        sampled_disagreement = float(development["annotation_disagreement_rate"].mean())
        metadata = {
            "task_id": task_id,
            "candidate_task_id": task["candidate_task_id"],
            "combination_id": combo_id,
            "design_tier": task["design_tier"],
            "comparison_anchor": int(task["comparison_anchor"]),
            "label_number": int(task["label_number"]),
            "selected_label_ids": task["selected_label_ids"],
            "selected_labels": task["selected_labels"],
            "class_balance_design": task["class_balance_design"],
            "development_pool_n": int(len(development)),
            "fit_train_n": int((development["split"] == "fit_train").sum()),
            "validation_n": int((development["split"] == "validation").sum()),
            "fixed_test_n": int(len(test)),
            "requested_samples_per_label": task["requested_samples_per_label"],
            "sampled_samples_per_label": "; ".join(f"{label}:{sampled_counts[label]}" for label in labels),
            "realized_class_imbalance_index": imbalance_index(sampled_counts.to_numpy()),
            "expected_annotation_disagreement_rate": float(task["expected_annotation_disagreement_rate"]),
            "sampled_annotation_disagreement_rate": sampled_disagreement,
            "avg_pairwise_semantic_similarity": float(task["avg_pairwise_semantic_similarity"]),
            "max_pairwise_semantic_similarity": float(task["max_pairwise_semantic_similarity"]),
            "sampling_seed": RANDOM_SEED,
            "validation_rule": "Within each label, ranks 1, 11, 21, ... are validation; all other ranks are fit_train",
            "final_training_rule": "After hyperparameter or epoch selection, retrain on the full DevelopmentPool",
        }
        mapping_frame = codebook[codebook["label_id"].isin(labels)].copy()
        mapping_frame.insert(2, "continuous_label", mapping_frame["label_id"].map(mapping).astype(int))

        development_columns = [
            "task_id",
            "sample_id",
            "train_pool_row_id",
            "record_id",
            "text",
            "normalized_text",
            "label",
            "label_name",
            "continuous_label",
            "split",
            "within_label_rank",
            "annotation_disagreement_rate",
            "majority_vote_share",
            "tie_adjudicated",
            "source_row_ids",
        ]
        test_columns = [
            "task_id",
            "sample_id",
            "test_row_id",
            "record_id",
            "text",
            "normalized_text",
            "label",
            "label_name",
            "continuous_label",
            "annotation_disagreement_rate",
            "majority_vote_share",
            "tie_adjudicated",
            "test_selection_source",
            "source_row_ids",
        ]
        payloads[task_id] = {
            "metadata": metadata,
            "development_pool": development[development_columns].replace({np.nan: None}).to_dict(orient="records"),
            "fixed_test": test[test_columns].replace({np.nan: None}).to_dict(orient="records"),
            "label_mapping": mapping_frame.replace({np.nan: None}).to_dict(orient="records"),
        }
        manifest_rows.append(metadata)
    return pd.DataFrame(manifest_rows), payloads


def build_demo_bank(train_pool: pd.DataFrame) -> pd.DataFrame:
    legacy = pd.read_excel(LEGACY_EXAMPLES_PATH)
    legacy_by_label = {
        name: normalized_text(text)
        for text, name in zip(legacy["合同语句"], legacy["标签"])
    }
    rows = []
    for label, label_data in LABELS.items():
        candidates = train_pool[
            (train_pool["label"] == label)
            & (train_pool["annotation_disagreement_rate"] == 0)
            & (train_pool["tie_adjudicated"] == 0)
            & (train_pool["majority_vote_share"] == 1)
        ].copy()
        candidates["word_count"] = candidates["normalized_text"].str.split().str.len()
        moderate = candidates[candidates["word_count"].between(12, 80)].copy()
        if len(moderate) >= 4:
            candidates = moderate
        median_length = float(candidates["word_count"].median())
        candidates["length_distance"] = (candidates["word_count"] - median_length).abs()
        legacy_text = legacy_by_label.get(label_data["name"])
        candidates["legacy_priority"] = (candidates["normalized_text"] != legacy_text).astype(int)
        candidates["stable_key"] = candidates["train_pool_row_id"].map(
            lambda row_id: stable_seed("demo", label, row_id)
        )
        chosen = candidates.sort_values(
            ["legacy_priority", "length_distance", "stable_key", "train_pool_row_id"]
        ).head(4)
        if len(chosen) != 4:
            raise AssertionError(f"Could not select four demonstrations for label {label}")
        for rank, (_, sample) in enumerate(chosen.iterrows(), start=1):
            rows.append(
                {
                    "label_id": label,
                    "label_name": label_data["name"],
                    "demo_rank": rank,
                    "train_pool_row_id": int(sample["train_pool_row_id"]),
                    "record_id": int(sample["record_id"]),
                    "text": sample["text"],
                    "normalized_text": sample["normalized_text"],
                    "word_count": int(sample["word_count"]),
                    "selection_reason": "legacy_example_matched" if sample["normalized_text"] == legacy_text else "high_consensus_median_length",
                }
            )
    return pd.DataFrame(rows)


def prepare_llm_payloads(
    llm_combos: pd.DataFrame,
    common: pd.DataFrame,
    fixed_test: pd.DataFrame,
    codebook: pd.DataFrame,
    demo_bank: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    common_ids = set(common["combination_id"])
    payloads: dict[str, dict[str, object]] = {}
    combo_manifest_rows = []
    run_rows = []

    for _, combo in llm_combos.sort_values("selection_order").iterrows():
        combo_id = str(combo["combination_id"])
        llm_task_id = f"LLM_{combo_id}"
        labels = parse_label_ids(combo["selected_label_ids"])
        k = len(labels)
        base = 11 // k
        extra = 11 % k
        offset = (int(combo["selection_order"]) - 1) % k
        extra_labels = {labels[(offset + index) % k] for index in range(extra)}
        demo_counts = {label: base + int(label in extra_labels) for label in labels}
        demonstrations = pd.concat(
            [
                demo_bank[demo_bank["label_id"] == label].head(demo_counts[label])
                for label in labels
            ],
            ignore_index=True,
        )
        demonstrations.insert(0, "llm_task_id", llm_task_id)
        demonstrations.insert(1, "demonstration_order", np.arange(1, len(demonstrations) + 1))
        if len(demonstrations) != 11:
            raise AssertionError(f"{llm_task_id} does not contain exactly 11 demonstrations")

        test = fixed_test[fixed_test["label"].isin(labels)].copy()
        test.insert(0, "llm_task_id", llm_task_id)
        test.insert(1, "sample_id", [f"{llm_task_id}_T{i:04d}" for i in range(1, len(test) + 1)])
        selected_codebook = codebook[codebook["label_id"].isin(labels)].copy()

        is_common = int(combo_id in common_ids)
        metadata = {
            "llm_task_id": llm_task_id,
            "combination_id": combo_id,
            "selection_order": int(combo["selection_order"]),
            "is_common24": is_common,
            "analysis_scope": "cross_model_and_llm" if is_common else "llm_extension_only",
            "label_number": int(combo["label_number"]),
            "selected_label_ids": combo["selected_label_ids"],
            "selected_labels": combo["selected_labels"],
            "fixed_test_n": int(len(test)),
            "few_shot_total_examples": 11,
            "few_shot_examples_per_label": "; ".join(f"{label}:{demo_counts[label]}" for label in labels),
            "unweighted_label_disagreement_rate": float(combo["unweighted_label_disagreement_rate"]),
            "avg_pairwise_semantic_similarity": float(combo["avg_pairwise_semantic_similarity"]),
            "complexity_quadrant": combo["complexity_quadrant"],
            "demonstration_selection_seed": RANDOM_SEED,
        }
        prompt_templates = [
            {
                "prompt_strategy": "zero_shot",
                "prompt_components": "task instruction + allowed label IDs and names",
                "template": "Assign exactly one allowed contractual-function label to each clause. Return only the label ID.",
            },
            {
                "prompt_strategy": "few_shot",
                "prompt_components": "zero-shot components + 11 fixed demonstrations",
                "template": "Use the fixed demonstrations to assign exactly one allowed label. Return only the label ID.",
            },
            {
                "prompt_strategy": "codebook_guided",
                "prompt_components": "label definitions + 11 fixed demonstrations + output constraint",
                "template": "Apply the contractual-function codebook and fixed demonstrations. Assign exactly one allowed label and return only the label ID.",
            },
        ]
        runs = []
        for strategy in ["zero_shot", "few_shot", "codebook_guided"]:
            for repeat in range(1, 4):
                run_id = f"{llm_task_id}_{strategy}_R{repeat}"
                run = {
                    "llm_run_id": run_id,
                    "llm_task_id": llm_task_id,
                    "combination_id": combo_id,
                    "prompt_strategy": strategy,
                    "repeat_id": repeat,
                    "model_name": "DeepSeek V4",
                    "model_version": "TO_BE_RECORDED_AT_CALL_TIME",
                    "temperature": "TO_BE_FIXED",
                    "batch_size": "TO_BE_FIXED",
                    "call_date": "TO_BE_RECORDED",
                    "status": "planned",
                }
                runs.append(run)
                run_rows.append(run)

        test_columns = [
            "llm_task_id",
            "sample_id",
            "test_row_id",
            "record_id",
            "text",
            "normalized_text",
            "label",
            "label_name",
            "annotation_disagreement_rate",
            "majority_vote_share",
            "tie_adjudicated",
            "test_selection_source",
            "source_row_ids",
        ]
        payloads[llm_task_id] = {
            "metadata": metadata,
            "fixed_test": test[test_columns].replace({np.nan: None}).to_dict(orient="records"),
            "demonstrations": demonstrations.replace({np.nan: None}).to_dict(orient="records"),
            "codebook": selected_codebook.replace({np.nan: None}).to_dict(orient="records"),
            "prompt_templates": prompt_templates,
            "run_plan": runs,
        }
        combo_manifest_rows.append(metadata)
    return pd.DataFrame(combo_manifest_rows), pd.DataFrame(run_rows), payloads


def prepare_common_payloads(
    common: pd.DataFrame,
    fixed_test: pd.DataFrame,
    supervised_manifest: pd.DataFrame,
    llm_manifest: pd.DataFrame,
    codebook: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    payloads: dict[str, dict[str, object]] = {}
    rows = []
    for _, combo in common.sort_values("selection_order").iterrows():
        combo_id = str(combo["combination_id"])
        labels = parse_label_ids(combo["selected_label_ids"])
        test = fixed_test[fixed_test["label"].isin(labels)].copy()
        anchor = supervised_manifest[
            (supervised_manifest["combination_id"] == combo_id)
            & (supervised_manifest["class_balance_design"] == "natural_imbalance")
            & (supervised_manifest["development_pool_n"] == 3000)
        ]
        if len(anchor) != 1:
            raise AssertionError(f"Expected one N=3000 natural anchor for {combo_id}")
        llm_row = llm_manifest[llm_manifest["combination_id"] == combo_id]
        if len(llm_row) != 1:
            raise AssertionError(f"Expected one LLM task for {combo_id}")
        metadata = {
            "common_task_id": f"COMMON_{combo_id}",
            "combination_id": combo_id,
            "selection_order": int(combo["selection_order"]),
            "label_number": int(combo["label_number"]),
            "selected_label_ids": combo["selected_label_ids"],
            "selected_labels": combo["selected_labels"],
            "fixed_test_n": int(len(test)),
            "svm_deberta_anchor_task_id": anchor.iloc[0]["task_id"],
            "svm_deberta_anchor_condition": "natural_imbalance_N3000",
            "llm_task_id": llm_row.iloc[0]["llm_task_id"],
            "primary_llm_configuration": "codebook_guided",
            "unweighted_label_disagreement_rate": float(combo["unweighted_label_disagreement_rate"]),
            "avg_pairwise_semantic_similarity": float(combo["avg_pairwise_semantic_similarity"]),
            "complexity_quadrant": combo["complexity_quadrant"],
        }
        selected_codebook = codebook[codebook["label_id"].isin(labels)].copy()
        payloads[metadata["common_task_id"]] = {
            "metadata": metadata,
            "fixed_test": test.replace({np.nan: None}).to_dict(orient="records"),
            "codebook": selected_codebook.replace({np.nan: None}).to_dict(orient="records"),
        }
        rows.append(metadata)
    return pd.DataFrame(rows), payloads


def write_payloads(payloads: dict[str, dict[str, object]], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for task_id, payload in payloads.items():
        (folder / f"{task_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "manifests_csv").mkdir(parents=True)
    INTERMEDIATE.mkdir(parents=True)

    train_pool = pd.read_csv(TRAIN_POOL_PATH)
    fixed_test = pd.read_csv(FIXED_TEST_PATH)
    common = pd.read_csv(COMMON_PATH)
    llm_combos = pd.read_csv(LLM_PATH)
    feasible = pd.read_csv(FEASIBLE_PATH)
    codebook = make_label_codebook()

    tasks = select_supervised_tasks(feasible, common)
    supervised_manifest, supervised_payloads = prepare_supervised_payloads(
        tasks, train_pool, fixed_test, codebook
    )
    demo_bank = build_demo_bank(train_pool)
    llm_manifest, llm_runs, llm_payloads = prepare_llm_payloads(
        llm_combos, common, fixed_test, codebook, demo_bank
    )
    common_manifest, common_payloads = prepare_common_payloads(
        common, fixed_test, supervised_manifest, llm_manifest, codebook
    )

    manifests = OUTPUT / "manifests_csv"
    tasks.to_csv(manifests / "selected_supervised_task_design_240.csv", index=False, encoding="utf-8-sig")
    supervised_manifest.to_csv(manifests / "supervised_task_manifest_240.csv", index=False, encoding="utf-8-sig")
    common_manifest.to_csv(manifests / "common24_manifest.csv", index=False, encoding="utf-8-sig")
    llm_manifest.to_csv(manifests / "llm_combination_manifest_40.csv", index=False, encoding="utf-8-sig")
    llm_runs.to_csv(manifests / "llm_run_plan_360.csv", index=False, encoding="utf-8-sig")
    codebook.to_csv(manifests / "label_codebook.csv", index=False, encoding="utf-8-sig")
    demo_bank.to_csv(manifests / "llm_demonstration_bank_44.csv", index=False, encoding="utf-8-sig")

    write_payloads(supervised_payloads, INTERMEDIATE / "supervised")
    write_payloads(common_payloads, INTERMEDIATE / "common24")
    write_payloads(llm_payloads, INTERMEDIATE / "llm")

    summary = {
        "random_seed": RANDOM_SEED,
        "supervised_task_count": int(len(supervised_manifest)),
        "common24_count": int(len(common_manifest)),
        "llm_combination_count": int(len(llm_manifest)),
        "llm_run_count": int(len(llm_runs)),
        "supervised_design_tiers": tasks["design_tier"].value_counts().to_dict(),
        "supervised_training_sizes": tasks["requested_training_sample_size"].value_counts().sort_index().to_dict(),
        "supervised_balance_designs": tasks["class_balance_design"].value_counts().to_dict(),
        "supervised_label_numbers": tasks["label_number"].value_counts().sort_index().to_dict(),
        "high_n_quotas": HIGH_N_QUOTAS,
        "minimum_development_samples_per_label": 5,
        "supervised_train_test_overlap": 0,
        "llm_test_demo_overlap": 0,
    }
    (OUTPUT / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
