from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split


DEFAULT_SEED = 20260713


def package_root() -> Path:
    configured = os.environ.get("EXPERIMENT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    candidates = [Path("/hy-tmp"), Path("/hy-tmp/hy_tmp_experiment_package"), Path.cwd()]
    return next((path for path in candidates if (path / "supervised_data").exists()), Path("/hy-tmp"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_value(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_value(payload), ensure_ascii=False) + "\n")


def append_or_replace_csv(path: Path, row: dict[str, Any], key_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {key: json.dumps(json_value(value), ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()}
    incoming = pd.DataFrame([normalized])
    if path.exists():
        existing = pd.read_csv(path)
        if all(column in existing.columns for column in key_columns):
            exact_match = pd.Series(True, index=existing.index)
            for column in key_columns:
                exact_match &= existing[column].astype(str) == str(normalized[column])
            existing = existing.loc[~exact_match]
        output = pd.concat([existing, incoming], ignore_index=True, sort=False)
    else:
        output = incoming
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def completed_keys(path: Path, key_columns: list[str]) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if not all(column in frame.columns for column in key_columns):
        return set()
    return {
        tuple(str(row[column]) for column in key_columns)
        for _, row in frame.iterrows()
    }


def read_task_data(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Data")
    required = {
        "sample_id",
        "text",
        "label",
        "original_label",
        "annotation_disagreement_rate",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["text"] = frame["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["original_label"] = pd.to_numeric(frame["original_label"], errors="raise").astype(int)
    frame["annotation_disagreement_rate"] = pd.to_numeric(
        frame["annotation_disagreement_rate"], errors="coerce"
    ).fillna(0.0)
    if (frame["text"] == "").any():
        raise ValueError(f"{path} contains empty text")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicated sample_id values")
    return frame


def stratified_development_split(
    frame: pd.DataFrame,
    seed: int,
    validation_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = frame["label"].value_counts()
    if counts.min() < 2:
        raise ValueError(f"Every class needs at least two rows for stratified splitting: {counts.to_dict()}")
    fit_index, validation_index = train_test_split(
        np.arange(len(frame)),
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=frame["label"].to_numpy(),
    )
    fit = frame.iloc[np.sort(fit_index)].reset_index(drop=True)
    validation = frame.iloc[np.sort(validation_index)].reset_index(drop=True)
    if set(fit["sample_id"]) & set(validation["sample_id"]):
        raise RuntimeError("Development split overlap detected")
    if set(fit["label"]) != set(frame["label"]) or set(validation["label"]) != set(frame["label"]):
        raise RuntimeError("A class is missing from the fit or validation split")
    return fit, validation


def metric_bundle(y_true: Iterable[int], y_pred: Iterable[int], labels: list[int]) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    return {
        "kappa": float(cohen_kappa_score(true, pred, labels=labels)),
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(true, pred)),
        "accuracy": float(accuracy_score(true, pred)),
        "weighted_f1": float(f1_score(true, pred, labels=labels, average="weighted", zero_division=0)),
    }


def bootstrap_metric_intervals(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: list[int],
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    if repetitions <= 0:
        return {}
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ["kappa", "macro_f1", "mcc"]}
    for _ in range(repetitions):
        index = rng.integers(0, len(true), size=len(true))
        sampled = metric_bundle(true[index], pred[index], labels)
        for name in values:
            if math.isfinite(sampled[name]):
                values[name].append(sampled[name])
    output: dict[str, float] = {}
    for name, samples in values.items():
        if samples:
            output[f"{name}_ci95_low"] = float(np.percentile(samples, 2.5))
            output[f"{name}_ci95_high"] = float(np.percentile(samples, 97.5))
    return output


def save_evaluation_artifacts(
    output_root: Path,
    run_key: str,
    source: pd.DataFrame,
    predictions: np.ndarray,
    labels: list[int],
    extra_prediction_columns: dict[str, Any] | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_frame = source.copy()
    prediction_frame["prediction"] = np.asarray(predictions, dtype=int)
    prediction_frame["correct"] = (prediction_frame["label"] == prediction_frame["prediction"]).astype(int)
    if extra_prediction_columns:
        for name, values in extra_prediction_columns.items():
            prediction_frame[name] = values
    (output_root / "predictions").mkdir(exist_ok=True)
    prediction_frame.to_csv(
        output_root / "predictions" / f"{run_key}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = classification_report(
        source["label"],
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    per_class = []
    for label in labels:
        values = report[str(label)]
        per_class.append(
            {
                "run_key": run_key,
                "continuous_label": label,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": int(values["support"]),
            }
        )
    (output_root / "per_class").mkdir(exist_ok=True)
    pd.DataFrame(per_class).to_csv(
        output_root / "per_class" / f"{run_key}.csv", index=False, encoding="utf-8-sig"
    )

    matrix = confusion_matrix(source["label"], predictions, labels=labels)
    confusion = pd.DataFrame(matrix, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    confusion.insert(0, "true_label", labels)
    (output_root / "confusion_matrices").mkdir(exist_ok=True)
    confusion.to_csv(
        output_root / "confusion_matrices" / f"{run_key}.csv", index=False, encoding="utf-8-sig"
    )


def task_numbers(data_dir: Path, start: int | None, end: int | None, requested: list[int] | None) -> list[int]:
    found = sorted(
        int(path.stem.removeprefix("train"))
        for path in data_dir.glob("train*.xlsx")
        if path.stem.removeprefix("train").isdigit()
    )
    if requested:
        requested_set = set(requested)
        found = [number for number in found if number in requested_set]
    if start is not None:
        found = [number for number in found if number >= start]
    if end is not None:
        found = [number for number in found if number <= end]
    return found


def load_mapping(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="TaskMapping")
    frame["task_no"] = pd.to_numeric(frame["task_no"], errors="raise").astype(int)
    return frame.set_index("task_no", drop=False)


def mapping_fields(mapping: pd.DataFrame, task_no: int) -> dict[str, Any]:
    if task_no not in mapping.index:
        return {"task_no": task_no}
    row = mapping.loc[task_no]
    keep = [
        "task_no",
        "original_task_id",
        "combination_id",
        "design_tier",
        "comparison_anchor",
        "label_number",
        "selected_label_ids",
        "selected_labels",
        "class_balance_design",
        "training_sample_size",
        "fixed_test_sample_size",
        "sampled_samples_per_label",
        "realized_class_imbalance_index",
        "sampled_annotation_disagreement_rate",
        "avg_pairwise_semantic_similarity",
        "max_pairwise_semantic_similarity",
        "sampling_seed",
    ]
    return {name: row[name] for name in keep if name in row.index}


def environment_metadata() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ["numpy", "pandas", "sklearn", "torch", "transformers", "openai"]:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[name] = "not_installed"
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
    }
    try:
        import torch

        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["cuda_version"] = torch.version.cuda
        metadata["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return metadata
