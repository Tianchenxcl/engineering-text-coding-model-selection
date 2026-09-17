from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiment_common import package_root, utc_now, write_json


def parse_args() -> argparse.Namespace:
    root = package_root()
    parser = argparse.ArgumentParser(description="Assemble model outputs into analysis-ready long tables")
    parser.add_argument("--package-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "analysis_ready")
    return parser.parse_args()


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def standardize_supervised(frame: pd.DataFrame, model_family: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    output["model_family"] = model_family
    output["performance_unit_id"] = output["run_key"]
    output["label_number_complexity"] = output["label_number"]
    output["class_imbalance_complexity"] = output["realized_class_imbalance_index"]
    output["annotation_disagreement_complexity"] = output["sampled_annotation_disagreement_rate"]
    output["training_size_complexity"] = output["final_training_n"]
    output["semantic_similarity_complexity"] = output["avg_pairwise_semantic_similarity"]
    output["prompt_strategy"] = "not_applicable"
    output["repeat_id"] = output.get("seed")
    return output


def standardize_llm(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    output["model_family"] = "large_language_model"
    output["performance_unit_id"] = output["run_id"].astype(str) + "_" + output["config_hash"].astype(str)
    output["label_number_complexity"] = output["label_number"]
    output["class_imbalance_complexity"] = pd.NA
    output["annotation_disagreement_complexity"] = output["annotation_disagreement_rate"]
    output["training_size_complexity"] = pd.NA
    output["semantic_similarity_complexity"] = output["avg_pairwise_semantic_similarity"]
    return output


def metric_checks(frame: pd.DataFrame, name: str) -> list[str]:
    warnings = []
    for metric in ["kappa", "macro_f1", "mcc", "accuracy", "weighted_f1"]:
        if metric not in frame.columns:
            warnings.append(f"{name}: missing metric column {metric}")
            continue
        values = pd.to_numeric(frame[metric], errors="coerce")
        if values.isna().any():
            warnings.append(f"{name}: {int(values.isna().sum())} missing/non-numeric {metric} values")
        if ((values < -1) | (values > 1)).any():
            warnings.append(f"{name}: {metric} contains values outside [-1, 1]")
    return warnings


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.package_root
    svm = read_optional(root / "outputs" / "svm" / "task_results.csv")
    traditional_robustness = read_optional(
        root / "outputs" / "traditional_ml_robustness" / "task_results.csv"
    )
    deberta = read_optional(root / "outputs" / "deberta_v3" / "task_results.csv")
    llm = read_optional(root / "outputs" / "deepseek_llm" / "run_results.csv")
    if svm.empty and traditional_robustness.empty and deberta.empty and llm.empty:
        raise RuntimeError("No completed model result files were found")

    supervised_frames = []
    svm_long = pd.DataFrame()
    traditional_robustness_long = pd.DataFrame()
    if not svm.empty:
        require_columns(svm, ["run_key", "original_task_id", "combination_id", "kappa"], "SVM results")
        svm_long = standardize_supervised(svm, "traditional_ml")
        supervised_frames.append(svm_long)
    if not traditional_robustness.empty:
        require_columns(
            traditional_robustness,
            ["algorithm", "run_key", "original_task_id", "combination_id", "kappa"],
            "Additional traditional ML results",
        )
        traditional_robustness_long = standardize_supervised(
            traditional_robustness, "traditional_ml"
        )
    if not deberta.empty:
        require_columns(deberta, ["run_key", "original_task_id", "combination_id", "kappa"], "DeBERTa results")
        supervised_frames.append(standardize_supervised(deberta, "pretrained_language_model"))
    supervised = pd.concat(supervised_frames, ignore_index=True, sort=False) if supervised_frames else pd.DataFrame()
    llm_long = standardize_llm(llm)
    all_runs = pd.concat([frame for frame in [supervised, llm_long] if not frame.empty], ignore_index=True, sort=False)

    if not supervised.empty:
        supervised.to_csv(args.output_dir / "supervised_performance_long.csv", index=False, encoding="utf-8-sig")
    if not llm_long.empty:
        llm_long.to_csv(args.output_dir / "llm_performance_long.csv", index=False, encoding="utf-8-sig")
    all_runs.to_csv(args.output_dir / "all_performance_long.csv", index=False, encoding="utf-8-sig")

    traditional_algorithm_frames = [
        frame for frame in [svm_long, traditional_robustness_long] if not frame.empty
    ]
    traditional_algorithms = (
        pd.concat(traditional_algorithm_frames, ignore_index=True, sort=False)
        if traditional_algorithm_frames
        else pd.DataFrame()
    )
    if not traditional_algorithms.empty:
        traditional_algorithms.to_csv(
            args.output_dir / "traditional_ml_algorithm_long.csv",
            index=False,
            encoding="utf-8-sig",
        )

    common_candidates = [
        root / "common24_manifest.csv",
        root / "manifests" / "source_manifests" / "common24_manifest.csv",
    ]
    common_path = next((path for path in common_candidates if path.exists()), common_candidates[0])
    common_cross_model = pd.DataFrame()
    common_traditional_algorithms = pd.DataFrame()
    if common_path.exists():
        common = pd.read_csv(common_path)
        anchor_ids = set(common["svm_deberta_anchor_task_id"].astype(str))
        pieces = []
        if not supervised.empty:
            supervised_anchor = supervised[supervised["original_task_id"].astype(str).isin(anchor_ids)].copy()
            supervised_anchor = supervised_anchor.merge(
                common[["combination_id", "svm_deberta_anchor_task_id", "complexity_quadrant"]],
                on="combination_id",
                how="inner",
                suffixes=("", "_common"),
            )
            pieces.append(supervised_anchor)
        if not llm_long.empty:
            llm_common = llm_long.merge(
                common[["combination_id", "llm_task_id", "primary_llm_configuration", "complexity_quadrant"]],
                on="combination_id",
                how="inner",
                suffixes=("", "_common"),
            )
            pieces.append(llm_common)
        if pieces:
            common_cross_model = pd.concat(pieces, ignore_index=True, sort=False)
            common_cross_model.to_csv(
                args.output_dir / "common24_cross_model_long.csv", index=False, encoding="utf-8-sig"
            )
        if not traditional_algorithms.empty:
            common_traditional_algorithms = traditional_algorithms[
                traditional_algorithms["original_task_id"].astype(str).isin(anchor_ids)
            ].copy()
            common_traditional_algorithms = common_traditional_algorithms.merge(
                common[["combination_id", "svm_deberta_anchor_task_id", "complexity_quadrant"]],
                on="combination_id",
                how="inner",
                suffixes=("", "_common"),
            )
            common_traditional_algorithms.to_csv(
                args.output_dir / "common24_traditional_ml_algorithm_long.csv",
                index=False,
                encoding="utf-8-sig",
            )

    warnings = metric_checks(all_runs, "all_performance_long")
    duplicate_count = int(all_runs["performance_unit_id"].duplicated().sum())
    if duplicate_count:
        warnings.append(f"all_performance_long: {duplicate_count} duplicated performance_unit_id values")
    report = {
        "generated_at_utc": utc_now(),
        "svm_rows": len(svm),
        "additional_traditional_ml_rows": len(traditional_robustness),
        "traditional_ml_algorithm_rows": len(traditional_algorithms),
        "common24_traditional_ml_algorithm_rows": len(common_traditional_algorithms),
        "deberta_rows": len(deberta),
        "llm_rows": len(llm),
        "all_run_rows": len(all_runs),
        "common24_cross_model_rows": len(common_cross_model),
        "duplicate_performance_unit_ids": duplicate_count,
        "warnings": warnings,
        "analysis_note": (
            "Use supervised_performance_long for five-dimension model-family interaction models. "
            "Use llm_performance_long for prompt-strategy and repeat analyses. "
            "Use common24_cross_model_long only for the predeclared shared-task model comparison. "
            "Use traditional_ml_algorithm_long and common24_traditional_ml_algorithm_long only "
            "for the predeclared traditional-ML robustness analysis; do not append those rows to "
            "the primary family model without accounting for algorithm nesting."
        ),
    }
    write_json(args.output_dir / "assembly_validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
