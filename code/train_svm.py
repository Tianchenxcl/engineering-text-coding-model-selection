from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from experiment_common import (
    DEFAULT_SEED,
    append_jsonl,
    append_or_replace_csv,
    bootstrap_metric_intervals,
    completed_keys,
    environment_metadata,
    load_mapping,
    mapping_fields,
    metric_bundle,
    package_root,
    read_task_data,
    save_evaluation_artifacts,
    set_global_seed,
    sha256_file,
    stratified_development_split,
    task_numbers,
    utc_now,
    write_json,
)


PARAMETER_GRID = [
    {"ngram_range": (1, 1), "min_df": 1, "max_df": 1.00, "max_features": None, "C": 0.25, "class_weight": None},
    {"ngram_range": (1, 1), "min_df": 1, "max_df": 0.95, "max_features": None, "C": 0.50, "class_weight": None},
    {"ngram_range": (1, 1), "min_df": 1, "max_df": 0.95, "max_features": None, "C": 1.00, "class_weight": None},
    {"ngram_range": (1, 2), "min_df": 1, "max_df": 1.00, "max_features": 50000, "C": 0.50, "class_weight": None},
    {"ngram_range": (1, 2), "min_df": 1, "max_df": 0.95, "max_features": 50000, "C": 1.00, "class_weight": None},
    {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "max_features": 50000, "C": 2.00, "class_weight": None},
    {"ngram_range": (1, 1), "min_df": 1, "max_df": 0.95, "max_features": None, "C": 0.50, "class_weight": "balanced"},
    {"ngram_range": (1, 1), "min_df": 1, "max_df": 0.95, "max_features": None, "C": 1.00, "class_weight": "balanced"},
    {"ngram_range": (1, 2), "min_df": 1, "max_df": 0.95, "max_features": 50000, "C": 0.50, "class_weight": "balanced"},
    {"ngram_range": (1, 2), "min_df": 1, "max_df": 0.95, "max_features": 50000, "C": 1.00, "class_weight": "balanced"},
    {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "max_features": 50000, "C": 2.00, "class_weight": "balanced"},
    {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.90, "max_features": 75000, "C": 4.00, "class_weight": "balanced"},
]


def build_model(parameters: dict, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    norm="l2",
                    ngram_range=parameters["ngram_range"],
                    min_df=parameters["min_df"],
                    max_df=parameters["max_df"],
                    max_features=parameters["max_features"],
                ),
            ),
            (
                "svm",
                LinearSVC(
                    C=parameters["C"],
                    class_weight=parameters["class_weight"],
                    random_state=seed,
                    dual="auto",
                    max_iter=10000,
                ),
            ),
        ]
    )


def validation_key(row: dict) -> tuple[float, float, float, float]:
    return (
        row["validation_kappa"],
        row["validation_macro_f1"],
        row["validation_mcc"],
        -row["param_set_no"],
    )


def parse_args() -> argparse.Namespace:
    root = package_root()
    parser = argparse.ArgumentParser(description="Leakage-free TF-IDF + LinearSVC task experiments")
    parser.add_argument("--data-dir", type=Path, default=root / "supervised_data")
    parser.add_argument("--mapping", type=Path, default=root / "manifests" / "task_file_mapping.xlsx")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "svm")
    parser.add_argument("--start-task", type=int)
    parser.add_argument("--end-task", type=int)
    parser.add_argument("--tasks", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_task(
    task_no: int,
    seed: int,
    args: argparse.Namespace,
    mapping: pd.DataFrame,
) -> dict:
    started_at = utc_now()
    started = time.perf_counter()
    run_key = f"svm_task{task_no:03d}_seed{seed}"
    train_path = args.data_dir / f"train{task_no}.xlsx"
    test_path = args.data_dir / f"test{task_no}.xlsx"
    train = read_task_data(train_path)
    test = read_task_data(test_path)
    labels = sorted(train["label"].unique().astype(int).tolist())
    if sorted(test["label"].unique().astype(int).tolist()) != labels:
        raise ValueError(f"Task {task_no}: train/test label sets differ")
    if set(train["sample_id"]) & set(test["sample_id"]):
        raise ValueError(f"Task {task_no}: train/test sample overlap")

    set_global_seed(seed)
    fit, validation = stratified_development_split(train, seed, args.validation_fraction)
    split_frame = pd.concat(
        [
            fit[["sample_id", "label"]].assign(split="fit"),
            validation[["sample_id", "label"]].assign(split="validation"),
        ],
        ignore_index=True,
    )
    (args.output_dir / "splits").mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(args.output_dir / "splits" / f"{run_key}.csv", index=False, encoding="utf-8-sig")

    trial_rows = []
    for param_set_no, parameters in enumerate(PARAMETER_GRID, start=1):
        model = build_model(parameters, seed)
        trial_started = time.perf_counter()
        model.fit(fit["text"], fit["label"])
        prediction = model.predict(validation["text"])
        metrics = metric_bundle(validation["label"], prediction, labels)
        row = {
            "run_key": run_key,
            "task_no": task_no,
            "seed": seed,
            "param_set_no": param_set_no,
            "parameters": parameters,
            "fit_n": len(fit),
            "validation_n": len(validation),
            "validation_kappa": metrics["kappa"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_mcc": metrics["mcc"],
            "validation_accuracy": metrics["accuracy"],
            "validation_weighted_f1": metrics["weighted_f1"],
            "fit_seconds": time.perf_counter() - trial_started,
        }
        trial_rows.append(row)
        append_or_replace_csv(
            args.output_dir / "hyperparameter_trials.csv",
            row,
            ["run_key", "param_set_no"],
        )
    best = max(trial_rows, key=validation_key)

    final_model = build_model(best["parameters"], seed)
    final_started = time.perf_counter()
    final_model.fit(train["text"], train["label"])
    test_prediction = final_model.predict(test["text"]).astype(int)
    test_metrics = metric_bundle(test["label"], test_prediction, labels)
    intervals = bootstrap_metric_intervals(
        test["label"], test_prediction, labels, args.bootstrap_reps, seed + 100000 + task_no
    )
    scores = final_model.decision_function(test["text"])
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    save_evaluation_artifacts(
        args.output_dir,
        run_key,
        test,
        test_prediction,
        labels,
        {"decision_scores_json": [json.dumps(row.tolist()) for row in scores]},
    )

    row = {
        **mapping_fields(mapping, task_no),
        "run_key": run_key,
        "model_family": "traditional_ml",
        "model_name": "TF-IDF + LinearSVC",
        "seed": seed,
        "fit_n": len(fit),
        "validation_n": len(validation),
        "final_training_n": len(train),
        "test_n": len(test),
        "best_param_set_no": best["param_set_no"],
        "best_parameters": best["parameters"],
        "validation_kappa": best["validation_kappa"],
        "validation_macro_f1": best["validation_macro_f1"],
        "validation_mcc": best["validation_mcc"],
        **test_metrics,
        **intervals,
        "selection_metric": "validation_kappa; ties: macro_f1, mcc, lower param_set_no",
        "validation_fraction": args.validation_fraction,
        "bootstrap_repetitions": args.bootstrap_reps,
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "final_fit_seconds": time.perf_counter() - final_started,
        "total_seconds": time.perf_counter() - started,
        "status": "completed",
    }
    write_json(
        args.output_dir / "run_metadata" / f"{run_key}.json",
        {"summary": row, "environment": environment_metadata(), "validation_trials": trial_rows},
    )
    return row


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping(args.mapping)
    numbers = task_numbers(args.data_dir, args.start_task, args.end_task, args.tasks)
    if not numbers:
        raise RuntimeError("No supervised task files matched the requested range")
    summary_path = args.output_dir / "task_results.csv"
    completed = completed_keys(summary_path, ["task_no", "seed"]) if not args.overwrite else set()
    print(f"Tasks={len(numbers)} seeds={args.seeds} output={args.output_dir}")
    for task_no in numbers:
        for seed in args.seeds:
            if (str(task_no), str(seed)) in completed:
                print(f"[skip] task={task_no} seed={seed}")
                continue
            try:
                print(f"[start] task={task_no} seed={seed}")
                row = run_task(task_no, seed, args, mapping)
                append_or_replace_csv(summary_path, row, ["task_no", "seed"])
                print(f"[done] task={task_no} seed={seed} kappa={row['kappa']:.4f}")
            except Exception as exc:
                error = {
                    "task_no": task_no,
                    "seed": seed,
                    "time_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                append_jsonl(args.output_dir / "errors.jsonl", error)
                print(f"[error] task={task_no} seed={seed}: {exc}")


if __name__ == "__main__":
    main()
