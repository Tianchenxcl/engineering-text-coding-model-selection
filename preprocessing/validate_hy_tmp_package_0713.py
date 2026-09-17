from __future__ import annotations

import json
import math
import py_compile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "supervised_data"
MAPPING = ROOT / "manifests" / "task_file_mapping.xlsx"
LLM_DATA = ROOT / "llm_data"
REPORT = ROOT / "_qa" / "package_validation_report.json"


REQUIRED_COLUMNS = {
    "sample_id",
    "task_id",
    "text",
    "label",
    "labels",
    "original_label",
    "label_name",
    "agreement_disagreement",
    "annotation_disagreement_rate",
    "majority_vote_share",
    "tie_adjudicated",
    "record_id",
    "source_row_ids",
}


def normalized_text(values: pd.Series) -> set[str]:
    return set(values.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip())


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    mapping = pd.read_excel(MAPPING, sheet_name="TaskMapping")
    if len(mapping) != 240:
        errors.append(f"Mapping row count is {len(mapping)}, expected 240")
    if mapping["task_no"].tolist() != list(range(1, 241)):
        errors.append("Mapping task_no is not exactly 1..240")

    train_files = sorted(DATA.glob("train*.xlsx"), key=lambda path: int(path.stem.removeprefix("train")))
    test_files = sorted(DATA.glob("test*.xlsx"), key=lambda path: int(path.stem.removeprefix("test")))
    if len(train_files) != 240 or len(test_files) != 240:
        errors.append(f"File counts: train={len(train_files)}, test={len(test_files)}, expected 240 each")

    checked_rows = 0
    for task_no in range(1, 241):
        train_path = DATA / f"train{task_no}.xlsx"
        test_path = DATA / f"test{task_no}.xlsx"
        if not train_path.exists() or not test_path.exists():
            errors.append(f"Task {task_no}: missing train or test file")
            continue
        train = pd.read_excel(train_path, sheet_name="Data")
        test = pd.read_excel(test_path, sheet_name="Data")
        checked_rows += len(train) + len(test)
        for name, frame in [("train", train), ("test", test)]:
            missing = REQUIRED_COLUMNS - set(frame.columns)
            if missing:
                errors.append(f"Task {task_no} {name}: missing columns {sorted(missing)}")
            if frame["sample_id"].astype(str).duplicated().any():
                errors.append(f"Task {task_no} {name}: duplicated sample_id")
            if frame["text"].fillna("").astype(str).str.strip().eq("").any():
                errors.append(f"Task {task_no} {name}: empty text")
            if not (frame["label"].astype(int) == frame["labels"].astype(int)).all():
                errors.append(f"Task {task_no} {name}: label and labels differ")
        train_labels = sorted(train["label"].astype(int).unique().tolist())
        test_labels = sorted(test["label"].astype(int).unique().tolist())
        if train_labels != list(range(len(train_labels))):
            errors.append(f"Task {task_no}: train labels are not continuous 0..K-1")
        if train_labels != test_labels:
            errors.append(f"Task {task_no}: train/test label sets differ")
        if set(train["sample_id"].astype(str)) & set(test["sample_id"].astype(str)):
            errors.append(f"Task {task_no}: train/test sample_id overlap")
        if normalized_text(train["text"]) & normalized_text(test["text"]):
            errors.append(f"Task {task_no}: train/test normalized text overlap")
        counts = train["label"].value_counts()
        if counts.min() < 2:
            errors.append(f"Task {task_no}: class with fewer than two training rows")
        if math.ceil(len(train) * 0.10) < len(train_labels):
            errors.append(f"Task {task_no}: 10% validation set is smaller than the number of classes")
        map_row = mapping.loc[mapping["task_no"] == task_no].iloc[0]
        if len(train) != int(map_row["training_sample_size"]):
            errors.append(f"Task {task_no}: train row count differs from mapping")
        if len(test) != int(map_row["fixed_test_sample_size"]):
            errors.append(f"Task {task_no}: test row count differs from mapping")
        if str(train["task_id"].iloc[0]) != str(map_row["original_task_id"]):
            errors.append(f"Task {task_no}: task_id differs from mapping")

    llm_files = sorted(LLM_DATA.glob("LLM*.json"))
    if len(llm_files) != 40:
        errors.append(f"LLM task count is {len(llm_files)}, expected 40")
    llm_planned_runs = 0
    for path in llm_files:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        llm_planned_runs += len(payload["run_plan"])
        test_ids = {row["record_id"] for row in payload["fixed_test"]}
        demo_ids = {row["record_id"] for row in payload["demonstrations"]}
        if test_ids & demo_ids:
            errors.append(f"{path.name}: LLM test/demonstration record overlap")
        if len(payload["demonstrations"]) != 11:
            errors.append(f"{path.name}: expected 11 demonstrations")
    if llm_planned_runs != 360:
        errors.append(f"LLM planned run count is {llm_planned_runs}, expected 360")

    scripts = [
        ROOT / "code" / "experiment_common.py",
        ROOT / "code" / "train_svm.py",
        ROOT / "code" / "train_deberta_v3.py",
        ROOT / "code" / "run_deepseek_llm.py",
        ROOT / "code" / "assemble_analysis_tables.py",
    ]
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception as exc:
            errors.append(f"Python compile failed for {script.name}: {exc}")

    notebook_files = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if len(notebook_files) != 4:
        errors.append(f"Notebook count is {len(notebook_files)}, expected 4")
    notebook_code_cells = 0
    for notebook_path in notebook_files:
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            cell_ids = [cell.get("id") for cell in notebook["cells"]]
            if None in cell_ids or len(cell_ids) != len(set(cell_ids)):
                errors.append(f"{notebook_path.name}: missing or duplicated cell IDs")
            for cell_no, cell in enumerate(notebook["cells"], start=1):
                if cell["cell_type"] != "code":
                    continue
                notebook_code_cells += 1
                compile("".join(cell["source"]), f"{notebook_path.name}:cell{cell_no}", "exec")
        except Exception as exc:
            errors.append(f"Notebook validation failed for {notebook_path.name}: {exc}")

    for convenience_file in [ROOT / "task_file_mapping.xlsx", ROOT / "common24_manifest.csv"]:
        if not convenience_file.exists():
            errors.append(f"Missing /hy-tmp convenience file: {convenience_file.name}")

    inspect_logs = list(ROOT.rglob("*.inspect.ndjson"))
    if inspect_logs:
        warnings.append(f"{len(inspect_logs)} temporary inspect logs remain")

    report = {
        "status": "passed" if not errors else "failed",
        "supervised_pairs": min(len(train_files), len(test_files)),
        "mapping_rows": len(mapping),
        "llm_tasks": len(llm_files),
        "llm_planned_runs": llm_planned_runs,
        "total_supervised_rows_checked": checked_rows,
        "python_scripts_compiled": len(scripts),
        "notebooks_validated": len(notebook_files),
        "notebook_code_cells_compiled": notebook_code_cells,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
