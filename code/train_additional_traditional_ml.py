from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

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


ALGORITHMS = ["logistic_regression", "complement_nb", "random_forest"]

MODEL_NAMES = {
    "logistic_regression": "TF-IDF + Logistic Regression",
    "complement_nb": "TF-IDF + Complement Naive Bayes",
    "random_forest": "TF-IDF + Random Forest",
}


PARAMETER_GRIDS: dict[str, list[dict[str, Any]]] = {
    "logistic_regression": [
        {"ngram_range": (1, 1), "min_df": 1, "max_df": 1.00, "max_features": None, "C": 0.25, "class_weight": None},
        {"ngram_range": (1, 1), "min_df": 1, "max_df": 0.95, "max_features": None, "C": 1.00, "class_weight": None},
        {"ngram_range": (1, 1), "min_df": 2, "max_df": 0.95, "max_features": 75000, "C": 4.00, "class_weight": None},
        {"ngram_range": (1, 2), "min_df": 1, "max_df": 0.95, "max_features": 50000, "C": 1.00, "class_weight": None},
        {"ngram_range": (1, 1), "min_df": 1, "max_df": 0.95, "max_features": None, "C": 0.25, "class_weight": "balanced"},
        {"ngram_range": (1, 1), "min_df": 1, "max_df": 0.95, "max_features": None, "C": 1.00, "class_weight": "balanced"},
        {"ngram_range": (1, 1), "min_df": 2, "max_df": 0.95, "max_features": 75000, "C": 4.00, "class_weight": "balanced"},
        {"ngram_range": (1, 2), "min_df": 1, "max_df": 0.95, "max_features": 50000, "C": 1.00, "class_weight": "balanced"},
    ],
    "complement_nb": [
        {"ngram_range": (1, 1), "min_df": 1, "max_df": 1.00, "max_features": None, "alpha": 0.10, "norm": False},
        {"ngram_range": (1, 1), "min_df": 1, "max_df": 0.95, "max_features": None, "alpha": 0.50, "norm": False},
        {"ngram_range": (1, 1), "min_df": 1, "max_df": 0.95, "max_features": None, "alpha": 1.00, "norm": False},
        {"ngram_range": (1, 1), "min_df": 2, "max_df": 0.95, "max_features": 75000, "alpha": 2.00, "norm": False},
        {"ngram_range": (1, 2), "min_df": 1, "max_df": 0.95, "max_features": 50000, "alpha": 0.50, "norm": True},
        {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "max_features": 75000, "alpha": 1.00, "norm": True},
    ],
    "random_forest": [
        {
            "ngram_range": (1, 1), "min_df": 1, "max_df": 0.95, "max_features": 30000,
            "n_estimators": 200, "max_depth": None, "min_samples_leaf": 1,
            "tree_max_features": "sqrt", "class_weight": None,
        },
        {
            "ngram_range": (1, 1), "min_df": 1, "max_df": 0.95, "max_features": 30000,
            "n_estimators": 200, "max_depth": 60, "min_samples_leaf": 1,
            "tree_max_features": "sqrt", "class_weight": None,
        },
        {
            "ngram_range": (1, 1), "min_df": 2, "max_df": 0.95, "max_features": 30000,
            "n_estimators": 200, "max_depth": None, "min_samples_leaf": 2,
            "tree_max_features": "sqrt", "class_weight": "balanced_subsample",
        },
        {
            "ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "max_features": 50000,
            "n_estimators": 200, "max_depth": 100, "min_samples_leaf": 2,
            "tree_max_features": "sqrt", "class_weight": "balanced_subsample",
        },
    ],
}


def vectorizer(parameters: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        norm="l2",
        ngram_range=parameters["ngram_range"],
        min_df=parameters["min_df"],
        max_df=parameters["max_df"],
        max_features=parameters["max_features"],
        dtype=np.float32,
    )


def build_model(algorithm: str, parameters: dict[str, Any], seed: int, n_jobs: int) -> Pipeline:
    if algorithm == "logistic_regression":
        classifier = LogisticRegression(
            C=parameters["C"],
            class_weight=parameters["class_weight"],
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        )
    elif algorithm == "complement_nb":
        classifier = ComplementNB(
            alpha=parameters["alpha"],
            norm=parameters["norm"],
            force_alpha=True,
        )
    elif algorithm == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=parameters["n_estimators"],
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            max_features=parameters["tree_max_features"],
            class_weight=parameters["class_weight"],
            random_state=seed,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return Pipeline([("tfidf", vectorizer(parameters)), ("classifier", classifier)])


def validation_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        row["validation_kappa"],
        row["validation_macro_f1"],
        row["validation_mcc"],
        -row["param_set_no"],
    )


def score_columns(model: Pipeline, text: pd.Series) -> tuple[str, list[str]]:
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(text))
        score_type = "decision_function"
    elif hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(text))
        score_type = "predict_proba"
    else:
        return "not_available", ["[]"] * len(text)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return score_type, [json.dumps(row.tolist()) for row in scores]


def parse_args() -> argparse.Namespace:
    root = package_root()
    mapping_default = root / "task_file_mapping.xlsx"
    if not mapping_default.exists():
        mapping_default = root / "manifests" / "task_file_mapping.xlsx"
    parser = argparse.ArgumentParser(description="Additional leakage-free traditional ML text classifiers")
    parser.add_argument("--data-dir", type=Path, default=root / "supervised_data")
    parser.add_argument("--mapping", type=Path, default=mapping_default)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "traditional_ml_robustness")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    parser.add_argument("--start-task", type=int)
    parser.add_argument("--end-task", type=int)
    parser.add_argument("--tasks", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_task(
    algorithm: str,
    task_no: int,
    seed: int,
    args: argparse.Namespace,
    mapping: pd.DataFrame,
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.perf_counter()
    run_key = f"{algorithm}_task{task_no:03d}_seed{seed}"
    artifact_root = args.output_dir / algorithm
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
    fit, validation = stratified_development_split(train, args.split_seed, args.validation_fraction)
    split_frame = pd.concat(
        [
            fit[["sample_id", "label"]].assign(split="fit"),
            validation[["sample_id", "label"]].assign(split="validation"),
        ],
        ignore_index=True,
    )
    (artifact_root / "splits").mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(artifact_root / "splits" / f"{run_key}.csv", index=False, encoding="utf-8-sig")

    trial_rows: list[dict[str, Any]] = []
    for param_set_no, parameters in enumerate(PARAMETER_GRIDS[algorithm], start=1):
        model = build_model(algorithm, parameters, seed, args.n_jobs)
        trial_started = time.perf_counter()
        model.fit(fit["text"], fit["label"])
        prediction = model.predict(validation["text"]).astype(int)
        metrics = metric_bundle(validation["label"], prediction, labels)
        trial = {
            "algorithm": algorithm,
            "run_key": run_key,
            "task_no": task_no,
            "seed": seed,
            "split_seed": args.split_seed,
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
        trial_rows.append(trial)
        append_or_replace_csv(
            artifact_root / "hyperparameter_trials.csv",
            trial,
            ["run_key", "param_set_no"],
        )
    best = max(trial_rows, key=validation_key)

    final_model = build_model(algorithm, best["parameters"], seed, args.n_jobs)
    final_started = time.perf_counter()
    final_model.fit(train["text"], train["label"])
    test_prediction = final_model.predict(test["text"]).astype(int)
    test_metrics = metric_bundle(test["label"], test_prediction, labels)
    intervals = bootstrap_metric_intervals(
        test["label"],
        test_prediction,
        labels,
        args.bootstrap_reps,
        seed + 300000 + task_no,
    )
    score_type, scores_json = score_columns(final_model, test["text"])
    class_order = final_model.named_steps["classifier"].classes_.astype(int).tolist()
    save_evaluation_artifacts(
        artifact_root,
        run_key,
        test,
        test_prediction,
        labels,
        {
            "score_type": score_type,
            "class_order_json": json.dumps(class_order),
            "model_scores_json": scores_json,
        },
    )

    row = {
        **mapping_fields(mapping, task_no),
        "algorithm": algorithm,
        "run_key": run_key,
        "model_family": "traditional_ml",
        "model_name": MODEL_NAMES[algorithm],
        "seed": seed,
        "split_seed": args.split_seed,
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
        artifact_root / "run_metadata" / f"{run_key}.json",
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
    completed = (
        completed_keys(summary_path, ["algorithm", "task_no", "seed"])
        if not args.overwrite
        else set()
    )
    print(
        f"Algorithms={args.algorithms} tasks={len(numbers)} seeds={args.seeds} "
        f"split_seed={args.split_seed} output={args.output_dir}"
    )
    for algorithm in args.algorithms:
        for task_no in numbers:
            for seed in args.seeds:
                key = (str(algorithm), str(task_no), str(seed))
                if key in completed:
                    print(f"[skip] algorithm={algorithm} task={task_no} seed={seed}")
                    continue
                try:
                    print(f"[start] algorithm={algorithm} task={task_no} seed={seed}")
                    row = run_task(algorithm, task_no, seed, args, mapping)
                    append_or_replace_csv(summary_path, row, ["algorithm", "task_no", "seed"])
                    print(
                        f"[done] algorithm={algorithm} task={task_no} seed={seed} "
                        f"kappa={row['kappa']:.4f}"
                    )
                except Exception as exc:
                    error = {
                        "algorithm": algorithm,
                        "task_no": task_no,
                        "seed": seed,
                        "time_utc": utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    append_jsonl(args.output_dir / "errors.jsonl", error)
                    print(f"[error] algorithm={algorithm} task={task_no} seed={seed}: {exc}")


if __name__ == "__main__":
    main()
