from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef

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


MODEL_TYPE = "debertav2"
_LIBRARY_CACHE = None


def parse_args() -> argparse.Namespace:
    root = package_root()
    parser = argparse.ArgumentParser(description="Leakage-free SimpleTransformers DeBERTa-v3 experiments")
    parser.add_argument("--data-dir", type=Path, default=root / "supervised_data")
    parser.add_argument("--mapping", type=Path, default=root / "manifests" / "task_file_mapping.xlsx")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "deberta_v3")
    parser.add_argument("--model", default=os.environ.get("DEBERTA_MODEL", "/hy-tmp/deberta-v3-base"))
    parser.add_argument("--start-task", type=int)
    parser.add_argument("--end-task", type=int)
    parser.add_argument("--tasks", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--keep-final-model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def patch_datasets_torchvision_compatibility() -> None:
    """Disable optional video dependencies that are irrelevant to text training."""
    try:
        import datasets.config as datasets_config

        datasets_config.TORCHVISION_AVAILABLE = False
        sys.modules.pop("torchvision", None)
    except Exception as exc:
        print(f"datasets/torchvision compatibility patch skipped: {exc}")


def patch_simpletransformers_transformers_compatibility() -> None:
    """Restore legacy SequenceSummary aliases imported by SimpleTransformers."""
    try:
        from transformers.modeling_utils import SequenceSummary

        patches = [
            ("transformers.models.xlnet.modeling_xlnet", "XLNetSequenceSummary"),
            ("transformers.models.xlm.modeling_xlm", "XLMSequenceSummary"),
            ("transformers.models.flaubert.modeling_flaubert", "FlaubertSequenceSummary"),
        ]
        for module_name, symbol_name in patches:
            try:
                module = __import__(module_name, fromlist=[symbol_name])
                if not hasattr(module, symbol_name):
                    setattr(module, symbol_name, SequenceSummary)
            except Exception as inner_exc:
                print(f"Compatibility patch skipped for {module_name}.{symbol_name}: {inner_exc}")
    except ImportError:
        # Recent Transformers releases removed SequenceSummary. The installed
        # SimpleTransformers path used for DeBERTa does not require this legacy alias.
        return
    except Exception as exc:
        print(f"Optional SimpleTransformers compatibility patch skipped: {exc}")


def require_libraries():
    global _LIBRARY_CACHE
    if _LIBRARY_CACHE is not None:
        return _LIBRARY_CACHE
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch before running DeBERTa") from exc
    patch_datasets_torchvision_compatibility()
    patch_simpletransformers_transformers_compatibility()
    try:
        from simpletransformers.classification import ClassificationModel
    except ImportError as exc:
        raise RuntimeError(
            'Install SimpleTransformers with: pip install "transformers>=4.31,<5" "simpletransformers==0.70.8"'
        ) from exc
    _LIBRARY_CACHE = (torch, ClassificationModel)
    return _LIBRARY_CACHE


def validate_torch_model_loading(torch: Any, model_path_or_id: str) -> None:
    from packaging.version import Version

    torch_version = Version(str(torch.__version__).split("+")[0])
    local_model = Path(model_path_or_id)
    has_safetensors = local_model.exists() and any(local_model.glob("*.safetensors"))
    if torch_version < Version("2.6") and not has_safetensors:
        raise RuntimeError(
            "Environment preflight failed before task training: "
            f"torch={torch.__version__}, model={model_path_or_id}, safetensors_found={has_safetensors}. "
            "Current Transformers versions require torch>=2.6 for legacy PyTorch .bin weights. "
            "Add the verified model.safetensors file to the local model directory (preferred), "
            "or install torch>=2.6 with a compatible official CUDA wheel, then rerun."
        )


def kappa_metric(labels, predictions) -> float:
    return float(cohen_kappa_score(labels, predictions))


def macro_f1_metric(labels, predictions) -> float:
    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def mcc_metric(labels, predictions) -> float:
    return float(matthews_corrcoef(labels, predictions))


def accuracy_metric(labels, predictions) -> float:
    return float(accuracy_score(labels, predictions))


def weighted_f1_metric(labels, predictions) -> float:
    return float(f1_score(labels, predictions, average="weighted", zero_division=0))


def simpletransformers_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[["text", "label"]].copy()
    output = output.rename(columns={"label": "labels"})
    output["labels"] = output["labels"].astype(int)
    return output


def model_arguments(
    output_dir: Path,
    seed: int,
    epochs: int,
    args: argparse.Namespace,
    evaluate_each_epoch: bool,
    no_save: bool,
    torch: Any,
) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "best_model_dir": str(output_dir / "best_model"),
        "cache_dir": str(output_dir / "cache"),
        "overwrite_output_dir": True,
        "reprocess_input_data": True,
        "use_cached_eval_features": False,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "save_steps": -1,
        "manual_seed": seed,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": int(epochs),
        "max_seq_length": args.max_length,
        "evaluate_during_training": evaluate_each_epoch,
        "evaluate_each_epoch": evaluate_each_epoch,
        "evaluate_during_training_steps": 10**12,
        "evaluate_during_training_verbose": evaluate_each_epoch,
        "evaluate_during_training_silent": not evaluate_each_epoch,
        "use_early_stopping": False,
        "early_stopping_metric": "kappa",
        "early_stopping_metric_minimize": False,
        "early_stopping_delta": 0.0,
        "wandb_project": None,
        "silent": False,
        "no_cache": True,
        "no_save": no_save,
        "fp16": bool(torch.cuda.is_available()),
        "use_multiprocessing": False,
        "use_multiprocessing_for_evaluation": False,
        "process_count": 1,
    }


def validation_history(training_details: Any) -> list[dict[str, Any]]:
    progress = training_details.copy() if isinstance(training_details, pd.DataFrame) else pd.DataFrame(training_details)
    required = {"kappa", "macro_f1", "mcc"}
    missing = sorted(required - set(progress.columns))
    if missing:
        raise RuntimeError(f"SimpleTransformers did not return epoch validation metrics: {missing}")
    progress = progress.dropna(subset=["kappa"]).reset_index(drop=True)
    if progress.empty:
        raise RuntimeError("SimpleTransformers returned no epoch-level validation results")

    rows = []
    for index, item in progress.iterrows():
        rows.append(
            {
                "epoch": index + 1,
                "global_step": int(item["global_step"]) if "global_step" in item and pd.notna(item["global_step"]) else None,
                "train_loss": float(item["train_loss"]) if "train_loss" in item and pd.notna(item["train_loss"]) else None,
                "eval_loss": float(item["eval_loss"]) if "eval_loss" in item and pd.notna(item["eval_loss"]) else None,
                "validation_kappa": float(item["kappa"]),
                "validation_macro_f1": float(item["macro_f1"]),
                "validation_mcc": float(item["mcc"]),
                "validation_accuracy": float(item["accuracy"]) if "accuracy" in item and pd.notna(item["accuracy"]) else None,
                "validation_weighted_f1": float(item["weighted_f1"]) if "weighted_f1" in item and pd.notna(item["weighted_f1"]) else None,
            }
        )
    return rows


def best_epoch_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["validation_kappa"],
            row["validation_macro_f1"],
            row["validation_mcc"],
            -row["epoch"],
        ),
    )


def training_loss_value(details: Any) -> float | None:
    if isinstance(details, (int, float, np.integer, np.floating)):
        value = float(details)
        return value if np.isfinite(value) else None
    if isinstance(details, pd.DataFrame) and "train_loss" in details.columns:
        values = pd.to_numeric(details["train_loss"], errors="coerce").dropna()
        return float(values.iloc[-1]) if not values.empty else None
    if isinstance(details, dict) and "train_loss" in details:
        values = pd.to_numeric(pd.Series(details["train_loss"]), errors="coerce").dropna()
        return float(values.iloc[-1]) if not values.empty else None
    return None


def save_final_model(model: Any, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(str(destination))
    model.tokenizer.save_pretrained(str(destination))
    if getattr(model, "config", None) is not None:
        model.config.save_pretrained(str(destination))


def run_task(task_no: int, seed: int, args: argparse.Namespace, mapping: pd.DataFrame) -> dict[str, Any]:
    torch, ClassificationModel = require_libraries()
    started_at = utc_now()
    started = time.perf_counter()
    run_key = f"deberta_task{task_no:03d}_seed{seed}"
    task_work = args.output_dir / "_working" / run_key
    if task_work.exists():
        shutil.rmtree(task_work)
    task_work.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / f"train{task_no}.xlsx"
    test_path = args.data_dir / f"test{task_no}.xlsx"
    train = read_task_data(train_path)
    test = read_task_data(test_path)
    labels = sorted(train["label"].unique().astype(int).tolist())
    if labels != list(range(len(labels))):
        raise ValueError(f"Task {task_no}: continuous labels must be 0..K-1, got {labels}")
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

    use_cuda = bool(torch.cuda.is_available())
    print(
        f"  backend=SimpleTransformers model_type={MODEL_TYPE} cuda={use_cuda} "
        f"fit={len(fit)} validation={len(validation)} full_train={len(train)} test={len(test)}"
    )

    selection_args = model_arguments(
        task_work / "selection",
        seed,
        args.max_epochs,
        args,
        evaluate_each_epoch=True,
        no_save=True,
        torch=torch,
    )
    selection_model = ClassificationModel(
        MODEL_TYPE,
        args.model,
        num_labels=len(labels),
        args=selection_args,
        use_cuda=use_cuda,
    )
    selection_started = time.perf_counter()
    _, selection_details = selection_model.train_model(
        simpletransformers_frame(fit),
        eval_data=simpletransformers_frame(validation),
        kappa=kappa_metric,
        macro_f1=macro_f1_metric,
        mcc=mcc_metric,
        accuracy=accuracy_metric,
        weighted_f1=weighted_f1_metric,
    )
    history = validation_history(selection_details)
    best = best_epoch_row(history)
    selection_seconds = time.perf_counter() - selection_started
    history_frame = pd.DataFrame(history)
    history_frame.insert(0, "run_key", run_key)
    history_frame.insert(1, "task_no", task_no)
    history_frame.insert(2, "seed", seed)
    (args.output_dir / "validation_history").mkdir(parents=True, exist_ok=True)
    history_frame.to_csv(
        args.output_dir / "validation_history" / f"{run_key}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    del selection_model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    set_global_seed(seed)
    final_args = model_arguments(
        task_work / "final",
        seed,
        int(best["epoch"]),
        args,
        evaluate_each_epoch=False,
        no_save=True,
        torch=torch,
    )
    final_model = ClassificationModel(
        MODEL_TYPE,
        args.model,
        num_labels=len(labels),
        args=final_args,
        use_cuda=use_cuda,
    )
    final_started = time.perf_counter()
    _, final_details = final_model.train_model(simpletransformers_frame(train))
    final_fit_seconds = time.perf_counter() - final_started

    test_prediction, raw_outputs = final_model.predict(test["text"].tolist())
    test_prediction = np.asarray(test_prediction, dtype=int)
    logits = np.asarray(raw_outputs, dtype=float)
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    test_metrics = metric_bundle(test["label"], test_prediction, labels)
    intervals = bootstrap_metric_intervals(
        test["label"], test_prediction, labels, args.bootstrap_reps, seed + 200000 + task_no
    )
    save_evaluation_artifacts(
        args.output_dir,
        run_key,
        test,
        test_prediction,
        labels,
        {
            "confidence": probabilities.max(axis=1),
            "probabilities_json": [json.dumps(row.tolist()) for row in probabilities],
        },
    )
    (args.output_dir / "logits").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "logits" / f"{run_key}.npz",
        logits=logits,
        labels=test["label"].to_numpy(),
        sample_ids=test["sample_id"].to_numpy(),
    )
    if args.keep_final_model:
        save_final_model(final_model, args.output_dir / "final_models" / run_key)

    row = {
        **mapping_fields(mapping, task_no),
        "run_key": run_key,
        "model_family": "pretrained_language_model",
        "model_name": "DeBERTa-v3-base",
        "training_backend": "SimpleTransformers",
        "simpletransformers_model_type": MODEL_TYPE,
        "model_path_or_id": args.model,
        "seed": seed,
        "fit_n": len(fit),
        "validation_n": len(validation),
        "final_training_n": len(train),
        "test_n": len(test),
        "best_epoch": int(best["epoch"]),
        "validation_kappa": best["validation_kappa"],
        "validation_macro_f1": best["validation_macro_f1"],
        "validation_mcc": best["validation_mcc"],
        **test_metrics,
        **intervals,
        "selection_metric": "validation_kappa; ties: macro_f1, mcc, earlier epoch",
        "validation_fraction": args.validation_fraction,
        "max_candidate_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "fp16": bool(use_cuda),
        "bootstrap_repetitions": args.bootstrap_reps,
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "selection_train_loss": history[-1].get("train_loss"),
        "final_train_loss": training_loss_value(final_details),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "selection_seconds": selection_seconds,
        "final_fit_seconds": final_fit_seconds,
        "total_seconds": time.perf_counter() - started,
        "status": "completed",
    }
    write_json(
        args.output_dir / "run_metadata" / f"{run_key}.json",
        {"summary": row, "environment": environment_metadata(), "validation_history": history},
    )

    del final_model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    shutil.rmtree(task_work, ignore_errors=True)
    return row


def main() -> None:
    args = parse_args()
    torch, _ = require_libraries()
    validate_torch_model_loading(torch, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping(args.mapping)
    numbers = task_numbers(args.data_dir, args.start_task, args.end_task, args.tasks)
    if not numbers:
        raise RuntimeError("No supervised task files matched the requested range")
    summary_path = args.output_dir / "task_results.csv"
    completed = completed_keys(summary_path, ["task_no", "seed"]) if not args.overwrite else set()
    print(f"Tasks={len(numbers)} seeds={args.seeds} model={args.model} output={args.output_dir}")
    for task_no in numbers:
        for seed in args.seeds:
            if (str(task_no), str(seed)) in completed:
                print(f"[skip] task={task_no} seed={seed}")
                continue
            try:
                print(f"[start] task={task_no} seed={seed}")
                row = run_task(task_no, seed, args, mapping)
                append_or_replace_csv(summary_path, row, ["task_no", "seed"])
                print(f"[done] task={task_no} seed={seed} best_epoch={row['best_epoch']} kappa={row['kappa']:.4f}")
            except Exception as exc:
                append_jsonl(
                    args.output_dir / "errors.jsonl",
                    {
                        "task_no": task_no,
                        "seed": seed,
                        "time_utc": utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print(f"[error] task={task_no} seed={seed}: {exc}")


if __name__ == "__main__":
    main()
