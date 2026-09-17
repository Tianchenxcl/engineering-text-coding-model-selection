from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE / "generated_datasets"
SOURCE_SUPERVISED = SOURCE_ROOT / "intermediate_json" / "supervised"
SOURCE_LLM = SOURCE_ROOT / "intermediate_json" / "llm"
SOURCE_MANIFESTS = SOURCE_ROOT / "manifests_csv"
PACKAGE_ROOT = HERE.parent
INTERMEDIATE_ROOT = PACKAGE_ROOT / "_build_intermediate"
PAIR_JSON_ROOT = INTERMEDIATE_ROOT / "supervised_pairs"
MAPPING_CSV = INTERMEDIATE_ROOT / "task_file_mapping.csv"


DATA_COLUMNS = [
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
]


def model_record(record: dict) -> dict:
    continuous_label = int(record["continuous_label"])
    disagreement = float(record.get("annotation_disagreement_rate", 0.0))
    return {
        "sample_id": str(record["sample_id"]),
        "task_id": str(record["task_id"]),
        "text": str(record.get("normalized_text") or record["text"]).strip(),
        "label": continuous_label,
        "labels": continuous_label,
        "original_label": int(record["label"]),
        "label_name": str(record["label_name"]),
        "agreement_disagreement": disagreement,
        "annotation_disagreement_rate": disagreement,
        "majority_vote_share": float(record.get("majority_vote_share", 1.0)),
        "tie_adjudicated": int(record.get("tie_adjudicated", 0)),
        "record_id": int(record["record_id"]),
        "source_row_ids": str(record.get("source_row_ids", "")),
    }


def prepare_supervised_pairs() -> list[dict]:
    PAIR_JSON_ROOT.mkdir(parents=True, exist_ok=True)
    source_files = sorted(SOURCE_SUPERVISED.glob("SUP*.json"))
    if len(source_files) != 240:
        raise RuntimeError(f"Expected 240 supervised task payloads, found {len(source_files)}")

    mapping_rows: list[dict] = []
    for task_no, source_path in enumerate(source_files, start=1):
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload["metadata"]
        expected_prefix = f"SUP{task_no:03d}_"
        if not str(metadata["task_id"]).startswith(expected_prefix):
            raise RuntimeError(
                f"Task ordering mismatch at {task_no}: {metadata['task_id']} does not start with {expected_prefix}"
            )

        train_records = [model_record(row) for row in payload["development_pool"]]
        test_records = [model_record(row) for row in payload["fixed_test"]]
        pair_payload = {
            "task_no": task_no,
            "source_workbook": f"{metadata['task_id']}.xlsx",
            "train_file": f"train{task_no}.xlsx",
            "test_file": f"test{task_no}.xlsx",
            "metadata": metadata,
            "label_mapping": payload["label_mapping"],
            "columns": DATA_COLUMNS,
            "train": train_records,
            "test": test_records,
        }
        pair_path = PAIR_JSON_ROOT / f"pair{task_no:03d}.json"
        with pair_path.open("w", encoding="utf-8") as handle:
            json.dump(pair_payload, handle, ensure_ascii=False, separators=(",", ":"))

        mapping_rows.append(
            {
                "task_no": task_no,
                "original_task_id": metadata["task_id"],
                "original_workbook": f"{metadata['task_id']}.xlsx",
                "new_train_file": f"train{task_no}.xlsx",
                "new_test_file": f"test{task_no}.xlsx",
                "candidate_task_id": metadata["candidate_task_id"],
                "combination_id": metadata["combination_id"],
                "design_tier": metadata["design_tier"],
                "comparison_anchor": metadata["comparison_anchor"],
                "label_number": metadata["label_number"],
                "selected_label_ids": metadata["selected_label_ids"],
                "selected_labels": metadata["selected_labels"],
                "class_balance_design": metadata["class_balance_design"],
                "training_sample_size": len(train_records),
                "fixed_test_sample_size": len(test_records),
                "sampled_samples_per_label": metadata["sampled_samples_per_label"],
                "realized_class_imbalance_index": metadata["realized_class_imbalance_index"],
                "sampled_annotation_disagreement_rate": metadata["sampled_annotation_disagreement_rate"],
                "avg_pairwise_semantic_similarity": metadata["avg_pairwise_semantic_similarity"],
                "max_pairwise_semantic_similarity": metadata["max_pairwise_semantic_similarity"],
                "sampling_seed": metadata["sampling_seed"],
                "selection_split_rule": "Stratified 90/10 split inside trainN.xlsx using seed 20260713",
                "final_training_rule": "Retrain from scratch on all rows in trainN.xlsx using selected settings",
            }
        )

    MAPPING_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MAPPING_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)
    return mapping_rows


def copy_llm_assets() -> None:
    target_data = PACKAGE_ROOT / "llm_data"
    target_manifests = PACKAGE_ROOT / "manifests" / "source_manifests"
    target_data.mkdir(parents=True, exist_ok=True)
    target_manifests.mkdir(parents=True, exist_ok=True)

    for path in sorted(SOURCE_LLM.glob("LLM*.json")):
        shutil.copy2(path, target_data / path.name)

    for name in [
        "common24_manifest.csv",
        "llm_combination_manifest_40.csv",
        "llm_run_plan_360.csv",
        "label_codebook.csv",
        "supervised_task_manifest_240.csv",
    ]:
        shutil.copy2(SOURCE_MANIFESTS / name, target_manifests / name)


def main() -> None:
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    mapping_rows = prepare_supervised_pairs()
    copy_llm_assets()
    print(
        json.dumps(
            {
                "package_root": str(PACKAGE_ROOT),
                "supervised_pairs": len(mapping_rows),
                "llm_tasks": len(list((PACKAGE_ROOT / "llm_data").glob("LLM*.json"))),
                "mapping_csv": str(MAPPING_CSV),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
