from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE / "generated_datasets"
INTERMEDIATE = ROOT / "intermediate_json"
MANIFESTS = ROOT / "manifests_csv"


def load_payloads(folder: Path) -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(folder.glob("*.json"))
    }


def count_rows(sheet) -> int:
    return sum(1 for _ in sheet.iter_rows(values_only=True))


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []

    supervised = load_payloads(INTERMEDIATE / "supervised")
    common = load_payloads(INTERMEDIATE / "common24")
    llm = load_payloads(INTERMEDIATE / "llm")
    supervised_manifest = pd.read_csv(MANIFESTS / "supervised_task_manifest_240.csv")
    llm_runs = pd.read_csv(MANIFESTS / "llm_run_plan_360.csv")

    if len(supervised) != 240:
        errors.append(f"Expected 240 supervised payloads, found {len(supervised)}")
    if len(common) != 24:
        errors.append(f"Expected 24 Common payloads, found {len(common)}")
    if len(llm) != 40:
        errors.append(f"Expected 40 LLM payloads, found {len(llm)}")
    if len(llm_runs) != 360:
        errors.append(f"Expected 360 LLM runs, found {len(llm_runs)}")
    if (supervised_manifest["development_pool_n"] == 100).any():
        errors.append("N=100 remains in the supervised task manifest")

    test_ids_by_combo: dict[str, set[int]] = {}
    nested_sets: dict[tuple[str, str, int], dict[int, set[int]]] = defaultdict(dict)
    nested_splits: dict[tuple[str, str, int], dict[int, dict[int, str]]] = defaultdict(dict)

    for task_id, payload in supervised.items():
        metadata = payload["metadata"]
        development = pd.DataFrame(payload["development_pool"])
        test = pd.DataFrame(payload["fixed_test"])
        mapping = pd.DataFrame(payload["label_mapping"])
        combo_id = metadata["combination_id"]
        mode = metadata["class_balance_design"]
        n = int(metadata["development_pool_n"])
        labels = tuple(int(value) for value in str(metadata["selected_label_ids"]).split(","))

        if len(development) != n:
            errors.append(f"{task_id}: development rows {len(development)} != {n}")
        if len(test) != int(metadata["fixed_test_n"]):
            errors.append(f"{task_id}: fixed test count mismatch")
        if set(development["label"].astype(int)) != set(labels):
            errors.append(f"{task_id}: development labels do not match selected labels")
        if set(test["label"].astype(int)) != set(labels):
            errors.append(f"{task_id}: test labels do not match selected labels")
        counts = development.groupby("label").size().reindex(labels, fill_value=0)
        if int(counts.min()) < 5:
            errors.append(f"{task_id}: a development label has fewer than five samples")
        split_counts = development.groupby(["label", "split"]).size().unstack(fill_value=0)
        for label in labels:
            if split_counts.loc[label].get("validation", 0) < 1:
                errors.append(f"{task_id}: label {label} absent from validation")
            if split_counts.loc[label].get("fit_train", 0) < 1:
                errors.append(f"{task_id}: label {label} absent from fit_train")
        overlap = set(development["normalized_text"]) & set(test["normalized_text"])
        if overlap:
            errors.append(f"{task_id}: {len(overlap)} train-test normalized text overlaps")
        expected_mapping = dict(zip(mapping["label_id"].astype(int), mapping["continuous_label"].astype(int)))
        if set(expected_mapping.values()) != set(range(len(labels))):
            errors.append(f"{task_id}: continuous label mapping is not 0..K-1")
        if not all(
            development.apply(lambda row: int(row["continuous_label"]) == expected_mapping[int(row["label"])], axis=1)
        ):
            errors.append(f"{task_id}: development continuous labels are inconsistent")

        test_ids = set(test["test_row_id"].astype(int))
        if combo_id in test_ids_by_combo and test_ids != test_ids_by_combo[combo_id]:
            errors.append(f"{task_id}: fixed test IDs differ within {combo_id}")
        test_ids_by_combo[combo_id] = test_ids

        for label in labels:
            label_rows = development[development["label"].astype(int) == label]
            key = (combo_id, mode, label)
            nested_sets[key][n] = set(label_rows["train_pool_row_id"].astype(int))
            nested_splits[key][n] = dict(
                zip(label_rows["train_pool_row_id"].astype(int), label_rows["split"].astype(str))
            )

    nested_comparisons = 0
    for key, sets_by_n in nested_sets.items():
        sizes = sorted(sets_by_n)
        for smaller, larger in zip(sizes, sizes[1:]):
            if not sets_by_n[smaller].issubset(sets_by_n[larger]):
                errors.append(f"Nested sampling failed for {key}: N={smaller} is not a subset of N={larger}")
            shared_ids = sets_by_n[smaller]
            small_splits = nested_splits[key][smaller]
            large_splits = nested_splits[key][larger]
            if any(small_splits[row_id] != large_splits[row_id] for row_id in shared_ids):
                errors.append(f"Split assignment changed across nested sizes for {key}")
            nested_comparisons += 1

    for common_id, payload in common.items():
        metadata = payload["metadata"]
        combo_id = metadata["combination_id"]
        test_ids = {int(row["test_row_id"]) for row in payload["fixed_test"]}
        if combo_id not in test_ids_by_combo or test_ids != test_ids_by_combo[combo_id]:
            errors.append(f"{common_id}: Common test IDs do not match supervised tasks")

    common_combo_ids = {payload["metadata"]["combination_id"] for payload in common.values()}
    for llm_id, payload in llm.items():
        metadata = payload["metadata"]
        combo_id = metadata["combination_id"]
        demonstrations = pd.DataFrame(payload["demonstrations"])
        test = pd.DataFrame(payload["fixed_test"])
        runs = pd.DataFrame(payload["run_plan"])
        if len(demonstrations) != 11:
            errors.append(f"{llm_id}: expected 11 demonstrations")
        if len(runs) != 9:
            errors.append(f"{llm_id}: expected 9 planned runs")
        if set(runs["prompt_strategy"]) != {"zero_shot", "few_shot", "codebook_guided"}:
            errors.append(f"{llm_id}: prompt strategy set is incomplete")
        if set(demonstrations["normalized_text"]) & set(test["normalized_text"]):
            errors.append(f"{llm_id}: demonstration-test overlap detected")
        labels = set(int(value) for value in str(metadata["selected_label_ids"]).split(","))
        if set(demonstrations["label_id"].astype(int)) != labels:
            errors.append(f"{llm_id}: demonstrations do not cover every selected label")
        test_ids = set(test["test_row_id"].astype(int))
        if combo_id in common_combo_ids and test_ids != test_ids_by_combo[combo_id]:
            errors.append(f"{llm_id}: Common24 LLM test IDs do not match supervised tasks")

    expected_workbooks = {
        "supervised": (ROOT / "03_supervised_tasks", 240, {"Metadata", "DevelopmentPool", "FixedTest", "LabelMapping"}),
        "common24": (ROOT / "02_common24", 24, {"Metadata", "FixedTest", "Codebook"}),
        "llm": (ROOT / "04_llm_tasks", 40, {"Metadata", "FixedTest", "Demonstrations", "Codebook", "PromptTemplates", "RunPlan"}),
    }
    workbook_counts = {}
    for kind, (folder, expected_count, expected_sheets) in expected_workbooks.items():
        files = sorted(folder.glob("*.xlsx"))
        workbook_counts[kind] = len(files)
        if len(files) != expected_count:
            errors.append(f"Expected {expected_count} {kind} workbooks, found {len(files)}")
        for file_path in files:
            workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=False)
            if set(workbook.sheetnames) != expected_sheets:
                errors.append(f"{file_path.name}: unexpected sheets {workbook.sheetnames}")
            workbook.close()

    representative_files = [
        ROOT / "03_supervised_tasks" / supervised_manifest.iloc[0]["task_id"].replace("/", "_") / "unused",
    ]
    del representative_files

    inspect_logs = list(ROOT.rglob("*.inspect.ndjson"))
    if inspect_logs:
        warnings.append(f"{len(inspect_logs)} generated inspect logs remain")

    task_size_counts = Counter(supervised_manifest["development_pool_n"].astype(int))
    design_counts = Counter(supervised_manifest["class_balance_design"].astype(str))
    label_number_counts = Counter(supervised_manifest["label_number"].astype(int))
    summary = {
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "supervised_payloads": len(supervised),
        "common24_payloads": len(common),
        "llm_payloads": len(llm),
        "llm_planned_runs": len(llm_runs),
        "supervised_workbooks": workbook_counts.get("supervised", 0),
        "common24_workbooks": workbook_counts.get("common24", 0),
        "llm_workbooks": workbook_counts.get("llm", 0),
        "nested_comparisons_checked": nested_comparisons,
        "training_size_counts": dict(sorted(task_size_counts.items())),
        "class_balance_counts": dict(sorted(design_counts.items())),
        "label_number_counts": dict(sorted(label_number_counts.items())),
        "minimum_training_size": int(supervised_manifest["development_pool_n"].min()),
        "maximum_training_size": int(supervised_manifest["development_pool_n"].max()),
        "inspect_logs_remaining": len(inspect_logs),
        "errors": errors,
        "warnings": warnings,
    }
    (ROOT / "validation_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
