from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment_common import (
    append_jsonl,
    append_or_replace_csv,
    bootstrap_metric_intervals,
    completed_keys,
    environment_metadata,
    metric_bundle,
    package_root,
    save_evaluation_artifacts,
    utc_now,
    write_json,
)


STRATEGIES = ["zero_shot", "few_shot", "codebook_guided"]


def parse_args() -> argparse.Namespace:
    root = package_root()
    parser = argparse.ArgumentParser(description="Auditable DeepSeek LLM coding experiments")
    parser.add_argument("--data-dir", type=Path, default=root / "llm_data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "deepseek_llm")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--tasks", nargs="+", help="LLM task IDs or combination IDs")
    parser.add_argument("--start", type=int, help="First task in sorted LLM task order, 1-based")
    parser.add_argument("--end", type=int, help="Last task in sorted LLM task order, inclusive")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DEEPSEEK_BATCH_SIZE", "20")))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-seconds", type=float, default=5.0)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--no-json-response-format", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_tasks(data_dir: Path, requested: list[str] | None, start: int | None, end: int | None) -> list[dict]:
    tasks = []
    for path in sorted(data_dir.glob("LLM*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_source_path"] = path
        tasks.append(payload)
    if requested:
        selected = set(requested)
        tasks = [
            task
            for task in tasks
            if task["metadata"]["llm_task_id"] in selected or task["metadata"]["combination_id"] in selected
        ]
    start_index = max((start or 1) - 1, 0)
    end_index = end if end is not None else len(tasks)
    return tasks[start_index:end_index]


def label_block(codebook: list[dict]) -> str:
    return "\n".join(f"- {row['label_id']}: {row['label_name']}" for row in codebook)


def codebook_block(codebook: list[dict]) -> str:
    return "\n".join(
        f"- {row['label_id']} ({row['label_name']}): {row['description']}" for row in codebook
    )


def demonstration_block(demonstrations: list[dict]) -> str:
    lines = []
    for row in demonstrations:
        lines.append(
            f"Example {row['demonstration_order']}\nText: {row['normalized_text']}\nLabel ID: {row['label_id']}"
        )
    return "\n\n".join(lines)


def build_messages(strategy: str, task_id: str, codebook: list[dict], demonstrations: list[dict], batch: list[dict]) -> list[dict]:
    allowed = [int(row["label_id"]) for row in codebook]
    items = [{"id": row["sample_id"], "text": row["normalized_text"]} for row in batch]
    system = (
        "You are an expert coding assistant for knowledge-intensive engineering contract text. "
        "Assign exactly one allowed contractual-function label to every item. "
        "Return valid JSON only and do not explain your reasoning."
    )
    sections = [
        f"Task ID: {task_id}",
        "Allowed labels:\n" + label_block(codebook),
    ]
    if strategy == "few_shot":
        sections.append("Fixed labeled demonstrations:\n" + demonstration_block(demonstrations))
    elif strategy == "codebook_guided":
        sections.append("Codebook definitions:\n" + codebook_block(codebook))
        sections.append("Fixed labeled demonstrations:\n" + demonstration_block(demonstrations))
    sections.extend(
        [
            "Rules:\n"
            "1. Classify the primary contractual function of each text.\n"
            "2. Use only one of the allowed integer label IDs.\n"
            "3. Include every input ID exactly once.\n"
            "4. Do not add explanations or extra keys.",
            "Input items:\n" + json.dumps(items, ensure_ascii=False, indent=2),
            "Required JSON schema:\n"
            '{"results":[{"id":"exact_input_id","prediction":0}]}\n'
            f"The prediction value must be one of: {allowed}",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(sections)}]


def extract_json(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def normalize_prediction(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid label ID")
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    match = re.search(r"-?\d+", str(value))
    if not match:
        raise ValueError(f"Cannot parse label ID from {value!r}")
    return int(match.group())


def parse_predictions(content: str, expected_ids: list[str], allowed_labels: list[int]) -> dict[str, int]:
    payload = extract_json(content)
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Response does not contain a results list")
    predictions: dict[str, int] = {}
    for row in rows:
        item_id = str(row.get("id", ""))
        if not item_id or item_id in predictions:
            raise ValueError(f"Missing or duplicated response ID: {item_id!r}")
        prediction = normalize_prediction(row.get("prediction"))
        if prediction not in allowed_labels:
            raise ValueError(f"Prediction {prediction} is outside allowed labels {allowed_labels}")
        predictions[item_id] = prediction
    if set(predictions) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(predictions))
        extra = sorted(set(predictions) - set(expected_ids))
        raise ValueError(f"Response IDs mismatch; missing={missing[:5]}, extra={extra[:5]}")
    return predictions


def client_for(args: argparse.Namespace):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai, pandas, openpyxl, numpy, and scikit-learn") from exc
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY before running")
    return OpenAI(api_key=api_key, base_url=args.base_url)


def response_usage(response: Any) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {name: getattr(usage, name, None) for name in ["prompt_tokens", "completion_tokens", "total_tokens"]}


def request_batch(
    client: Any,
    args: argparse.Namespace,
    run_id: str,
    batch_index: int,
    messages: list[dict],
    expected_ids: list[str],
    allowed_labels: list[int],
    audit_path: Path,
) -> tuple[dict[str, int], dict]:
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        sent_at = utc_now()
        started = time.perf_counter()
        request_kwargs = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if not args.no_json_response_format:
            request_kwargs["response_format"] = {"type": "json_object"}
        if not args.enable_thinking:
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        try:
            response = client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or ""
            predictions = parse_predictions(content, expected_ids, allowed_labels)
            metadata = {
                "response_id": getattr(response, "id", None),
                "response_model": getattr(response, "model", None),
                "response_created": getattr(response, "created", None),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "usage": response_usage(response),
            }
            append_jsonl(
                audit_path,
                {
                    "run_id": run_id,
                    "batch_index": batch_index,
                    "attempt": attempt,
                    "sent_at_utc": sent_at,
                    "completed_at_utc": utc_now(),
                    "latency_seconds": time.perf_counter() - started,
                    "request": request_kwargs,
                    "raw_content": content,
                    "parsed_predictions": predictions,
                    "response_metadata": metadata,
                    "status": "success",
                },
            )
            return predictions, metadata
        except Exception as exc:
            last_error = exc
            append_jsonl(
                audit_path,
                {
                    "run_id": run_id,
                    "batch_index": batch_index,
                    "attempt": attempt,
                    "sent_at_utc": sent_at,
                    "completed_at_utc": utc_now(),
                    "latency_seconds": time.perf_counter() - started,
                    "request": request_kwargs,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "status": "failed",
                },
            )
            if attempt < args.max_retries:
                wait = args.retry_seconds * (2 ** (attempt - 1)) + random.random()
                time.sleep(wait)
    raise RuntimeError(f"Batch failed after {args.max_retries} attempts: {last_error}")


def configuration_hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def class_imbalance_index(labels: list[int], allowed_labels: list[int]) -> float:
    if len(allowed_labels) <= 1 or not labels:
        return 0.0
    counts = np.asarray([labels.count(label) for label in allowed_labels], dtype=float)
    proportions = counts[counts > 0] / counts.sum()
    entropy = -float(np.sum(proportions * np.log(proportions)))
    return 1.0 - entropy / math.log(len(allowed_labels))


def load_partial(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    output = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output[str(row["sample_id"])] = row
    return output


def summarize_successful_calls(path: Path) -> tuple[set[str], dict[str, int]]:
    models: set[str] = set()
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not path.exists():
        return models, totals
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "success":
                continue
            response = row.get("response_metadata", {})
            if response.get("response_model"):
                models.add(str(response["response_model"]))
            usage = response.get("usage", {})
            for name in totals:
                totals[name] += int(usage.get(name) or 0)
    return models, totals


def run_once(client: Any, task: dict, strategy: str, repeat: int, args: argparse.Namespace) -> dict:
    metadata = task["metadata"]
    run_id = f"{metadata['llm_task_id']}_{strategy}_R{repeat}"
    started_at = utc_now()
    started = time.perf_counter()
    config = {
        "run_id": run_id,
        "model": args.model,
        "base_url": args.base_url,
        "strategy": strategy,
        "repeat": repeat,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "thinking_enabled": args.enable_thinking,
        "json_response_format": not args.no_json_response_format,
        "source_file": task["_source_path"].name,
    }
    config_hash = configuration_hash(config)
    partial_path = args.output_dir / "partials" / run_id / f"{config_hash}.jsonl"
    audit_path = args.output_dir / "api_calls" / run_id / f"{config_hash}.jsonl"
    if args.overwrite:
        partial_path.unlink(missing_ok=True)
        audit_path.unlink(missing_ok=True)
    write_json(args.output_dir / "run_configs" / run_id / f"{config_hash}.json", config)

    codebook = task["codebook"]
    allowed_labels = sorted(int(row["label_id"]) for row in codebook)
    test_rows = task["fixed_test"]
    partial = load_partial(partial_path)
    pending = [row for row in test_rows if row["sample_id"] not in partial]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        batch_index = start // args.batch_size + 1
        messages = build_messages(strategy, metadata["llm_task_id"], codebook, task["demonstrations"], batch)
        expected_ids = [row["sample_id"] for row in batch]
        predictions, response_metadata = request_batch(
            client,
            args,
            run_id,
            batch_index,
            messages,
            expected_ids,
            allowed_labels,
            audit_path,
        )
        for row in batch:
            append_jsonl(
                partial_path,
                {
                    "sample_id": row["sample_id"],
                    "true_label": int(row["label"]),
                    "prediction": int(predictions[row["sample_id"]]),
                    "batch_index": batch_index,
                    "response_model": response_metadata.get("response_model"),
                    "response_id": response_metadata.get("response_id"),
                    "completed_at_utc": utc_now(),
                },
            )

    partial = load_partial(partial_path)
    missing = [row["sample_id"] for row in test_rows if row["sample_id"] not in partial]
    if missing:
        raise RuntimeError(f"Run {run_id} has {len(missing)} missing predictions")
    response_models, total_usage = summarize_successful_calls(audit_path)
    frame = pd.DataFrame(test_rows).copy()
    frame["label"] = frame["label"].astype(int)
    frame["original_label"] = frame["label"]
    frame["text"] = frame["normalized_text"]
    prediction = np.asarray([partial[row["sample_id"]]["prediction"] for row in test_rows], dtype=int)
    metrics = metric_bundle(frame["label"], prediction, allowed_labels)
    intervals = bootstrap_metric_intervals(
        frame["label"],
        prediction,
        allowed_labels,
        args.bootstrap_reps,
        int(hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8], 16),
    )
    artifact_key = f"{run_id}_{config_hash}"
    save_evaluation_artifacts(
        args.output_dir,
        artifact_key,
        frame,
        prediction,
        allowed_labels,
        {"prompt_strategy": strategy, "repeat_id": repeat, "config_hash": config_hash},
    )
    row = {
        "run_id": run_id,
        "config_hash": config_hash,
        "llm_task_id": metadata["llm_task_id"],
        "combination_id": metadata["combination_id"],
        "model_family": "large_language_model",
        "request_model": args.model,
        "response_models": sorted(response_models),
        "prompt_strategy": strategy,
        "repeat_id": repeat,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "thinking_enabled": args.enable_thinking,
        "label_number": metadata["label_number"],
        "selected_label_ids": metadata["selected_label_ids"],
        "selected_labels": metadata["selected_labels"],
        "test_n": len(frame),
        "demonstration_count": 0 if strategy == "zero_shot" else len(task["demonstrations"]),
        "fixed_test_class_imbalance_index": class_imbalance_index(
            frame["label"].astype(int).tolist(), allowed_labels
        ),
        "annotation_disagreement_rate": metadata["unweighted_label_disagreement_rate"],
        "avg_pairwise_semantic_similarity": metadata["avg_pairwise_semantic_similarity"],
        "complexity_quadrant": metadata["complexity_quadrant"],
        **metrics,
        **intervals,
        **total_usage,
        "bootstrap_repetitions": args.bootstrap_reps,
        "source_sha256": hashlib.sha256(task["_source_path"].read_bytes()).hexdigest(),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "total_seconds": time.perf_counter() - started,
        "status": "completed",
    }
    write_json(
        args.output_dir / "run_metadata" / f"{artifact_key}.json",
        {"summary": row, "configuration": config, "environment": environment_metadata()},
    )
    return row


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.data_dir, args.tasks, args.start, args.end)
    if not tasks:
        raise RuntimeError("No LLM tasks matched the requested selection")
    client = client_for(args)
    summary_path = args.output_dir / "run_results.csv"
    completed = completed_keys(summary_path, ["run_id", "config_hash"]) if not args.overwrite else set()
    print(
        f"Tasks={len(tasks)} strategies={args.strategies} repeats={args.repeats} "
        f"model={args.model} output={args.output_dir}"
    )
    for task in tasks:
        for strategy in args.strategies:
            for repeat in args.repeats:
                run_id = f"{task['metadata']['llm_task_id']}_{strategy}_R{repeat}"
                config_hash = configuration_hash(
                    {
                        "run_id": run_id,
                        "model": args.model,
                        "base_url": args.base_url,
                        "strategy": strategy,
                        "repeat": repeat,
                        "temperature": args.temperature,
                        "batch_size": args.batch_size,
                        "max_tokens": args.max_tokens,
                        "thinking_enabled": args.enable_thinking,
                        "json_response_format": not args.no_json_response_format,
                        "source_file": task["_source_path"].name,
                    }
                )
                if (run_id, config_hash) in completed:
                    print(f"[skip] {run_id} config={config_hash}")
                    continue
                try:
                    print(f"[start] {run_id} config={config_hash}")
                    row = run_once(client, task, strategy, repeat, args)
                    append_or_replace_csv(summary_path, row, ["run_id", "config_hash"])
                    print(f"[done] {run_id} kappa={row['kappa']:.4f}")
                except Exception as exc:
                    append_jsonl(
                        args.output_dir / "errors.jsonl",
                        {
                            "run_id": run_id,
                            "config_hash": config_hash,
                            "time_utc": utc_now(),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    print(f"[error] {run_id}: {exc}")


if __name__ == "__main__":
    main()
