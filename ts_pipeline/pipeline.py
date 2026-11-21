"""Staged deterministic pipeline orchestration."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import CacheStore
from .canonical import (
    atomic_write_json,
    file_sha256,
    fingerprint,
    to_builtin,
)
from .config import Config, load_config
from .errors import (
    E_CONFIG_INVALID,
    E_METRIC_FAILED,
    E_MODEL_FIT_FAILED,
    E_PREDICTION_FAILED,
    E_SAMPLE_INVALID,
    E_SNAPSHOT_CORRUPT,
    E_SNAPSHOT_STALE,
    E_WINDOW_BUILD_FAILED,
    Failure,
    PipelineFatalError,
    sort_failures,
)
from .injections import INJECTION_CODES, Injection, LoadedInjections, load_injections
from .models import ModelError, TrainingContext, model_for_name
from .quantization import QuantizationError, aggregate_metrics, evaluate_window
from .samples import LoadedSamples, SequenceSample, load_samples
from .scheduler import (
    ScheduleResult,
    TaskPlan,
    chunk_tasks_by_windows,
    execute_parallel,
    invalidation_report,
    make_task,
    schedule_tasks,
)
from .snapshot import SnapshotManager
from .windows import WindowBuildError, build_window, window_cache_key


@dataclass
class PipelineOptions:
    config_path: Path
    state_dir: Path
    report_path: Path
    samples_override: Path | None = None
    injection_override: Path | None = None
    stop_after_stage: str | None = None
    stop_event: threading.Event | None = None


@dataclass
class PipelineResult:
    exit_code: int
    report_path: Path
    report: dict[str, Any]


@dataclass
class RunState:
    config: Config
    samples: dict[str, SequenceSample]
    injections: LoadedInjections
    snapshot: SnapshotManager
    cache: CacheStore
    failures: list[Failure] = field(default_factory=list)
    windows: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    tasks: list[TaskPlan] = field(default_factory=list)
    schedule: ScheduleResult | None = None
    train_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    prediction_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    metric_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    invalidated: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, list[Any]] = field(default_factory=dict)
    hits: dict[str, dict[str, Any]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    stage_status: dict[str, str] = field(default_factory=dict)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _empty_audit() -> dict[str, list[Any]]:
    return {
        "window_reused": [],
        "window_computed": [],
        "training_reused": [],
        "training_computed": [],
        "training_failed": [],
        "prediction_reused": [],
        "prediction_computed": [],
        "prediction_failed": [],
        "metric_reused": [],
        "metric_computed": [],
        "metric_failed": [],
    }


def _record_hit(
    state: RunState,
    injection: Injection,
    *,
    stage: str,
    model_name: str | None,
    failure_code: str,
    message: str,
) -> None:
    record = state.hits.setdefault(
        injection.injection_id,
        {
            "injection_id": injection.injection_id,
            "kind": injection.kind,
            "sequence_id": injection.sequence_id,
            "window_index": injection.window_index,
            "mapped_business_code": INJECTION_CODES[injection.kind],
            "business_code": failure_code,
            "stages": [],
            "models": [],
            "messages": [],
            "parameters": to_builtin(injection.raw),
        },
    )
    if stage not in record["stages"]:
        record["stages"].append(stage)
    if model_name is not None and model_name not in record["models"]:
        record["models"].append(model_name)
    if message not in record["messages"]:
        record["messages"].append(message)


def _failure_from_exception(
    task: TaskPlan,
    stage: str,
    exc: Exception,
    default_code: str,
) -> Failure:
    code = getattr(exc, "code", default_code)
    return Failure(
        code=code,
        stage=stage,
        message=str(exc),
        sequence_id=task.series_id,
        window_index=task.window_index,
        model_name=task.model_name,
    )


def _task_fingerprints(tasks: list[TaskPlan]) -> dict[str, Any]:
    return {task.task_key: task.fingerprint_record() for task in tasks}


def _progress(
    state: RunState,
    stage: str,
    *,
    completed_windows: list[dict[str, Any]] | None = None,
    completed_tasks: list[str] | None = None,
    cursor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schedule = state.schedule
    return {
        "stage": stage,
        "completed_series": sorted(state.samples),
        "completed_windows": (
            completed_windows
            if completed_windows is not None
            else [
                {"series_id": sequence_id, "window_index": window_index}
                for sequence_id, window_index in sorted(state.windows)
            ]
        ),
        "completed_tasks": completed_tasks or [],
        "cursor": (
            cursor
            if cursor is not None
            else (schedule.pending_cursor if schedule else None)
        ),
        "budget_used": schedule.budget_used if schedule else 0,
        "task_fingerprints": _task_fingerprints(state.tasks),
    }


def _should_stop(options: PipelineOptions, stage: str) -> bool:
    if options.stop_event is not None and options.stop_event.is_set():
        return True
    return options.stop_after_stage == stage


def _stage_payload_load(state: RunState, previous_failures: list[Failure]) -> dict[str, Any]:
    return {
        "status": "complete",
        "config_hash": state.config.config_hash,
        "samples": [sample.to_summary() for _, sample in sorted(state.samples.items())],
        "injections": [injection.to_dict() for injection in state.injections.items],
        "failures": [failure.to_dict() for failure in sort_failures(previous_failures)],
    }


def _stage_payload_windows(state: RunState, window_failures: list[Failure]) -> dict[str, Any]:
    return {
        "status": "complete",
        "windows": [
            state.windows[key] for key in sorted(state.windows)
        ],
        "failures": [failure.to_dict() for failure in sort_failures(window_failures)],
    }


def _stage_payload_models(state: RunState, training_failures: list[Failure]) -> dict[str, Any]:
    return {
        "status": "complete",
        "training_artifacts": [
            state.train_artifacts[key] for key in sorted(state.train_artifacts)
        ],
        "failures": [failure.to_dict() for failure in sort_failures(training_failures)],
    }


def _stage_payload_metrics(state: RunState, metric_failures: list[Failure]) -> dict[str, Any]:
    return {
        "status": "complete",
        "metric_records": [
            state.metric_payloads[key] for key in sorted(state.metric_payloads)
        ],
        "failures": [failure.to_dict() for failure in sort_failures(metric_failures)],
    }


def _build_injection_report(state: RunState) -> dict[str, Any]:
    items = []
    hit_ids = set(state.hits)
    for injection in state.injections.items:
        item = injection.to_dict()
        item["hit"] = injection.injection_id in hit_ids
        item["hit_record"] = state.hits.get(injection.injection_id)
        items.append(item)
    return {
        "manifest_hash": state.injections.manifest_hash,
        "items": items,
        "hits": [state.hits[key] for key in sorted(state.hits)],
        "unhit_injection_ids": [
            injection.injection_id
            for injection in state.injections.items
            if injection.injection_id not in hit_ids
        ],
    }


def _build_report(
    state: RunState,
    *,
    status: str,
    exit_code: int,
    samples_hash: str,
) -> dict[str, Any]:
    failures = sort_failures(state.failures)
    models: dict[str, Any] = {}
    matrix: dict[tuple[str, int], dict[str, Any]] = {}
    over_tolerance_global: list[dict[str, Any]] = []
    successful_tasks = 0

    for model_name in state.config.models.enabled:
        model_windows: list[dict[str, Any]] = []
        model_metric_records: list[dict[str, Any]] = []
        over_tolerance: list[dict[str, Any]] = []
        task_list = sorted(
            (task for task in state.tasks if task.model_name == model_name),
            key=lambda task: task.sort_key,
        )
        for task in task_list:
            train_payload = state.train_artifacts.get(task.task_key)
            prediction_payload = state.prediction_payloads.get(task.task_key)
            metric_payload = state.metric_payloads.get(task.task_key)
            if not train_payload or not prediction_payload or not metric_payload:
                continue
            successful_tasks += 1
            artifact = train_payload["artifact"]
            metrics = metric_payload["metrics"]
            predictions = prediction_payload["predictions"]
            window = task.window
            record = {
                "series_id": task.series_id,
                "window_index": task.window_index,
                "model_name": model_name,
                "training_window": artifact["training_window"],
                "train_window": artifact["training_window"],
                "fitted_params": artifact["fitted_params"],
                "residual_sequence": artifact["residual_sequence"],
                "residual_fingerprint": artifact["residual_fingerprint"],
                "forecast_start_index": window["forecast_start_index"],
                "forecast_timestamps_ms": window["forecast_timestamps_ms"],
                "predictions": [
                    {
                        "step": point["step"],
                        "raw_value": point["prediction"],
                        "fixed_value": point["prediction_fixed"],
                        "quantized_value": point["prediction_quantized"],
                    }
                    for point in metrics["points"]
                ],
                "actual_values": window["actual_values"],
                "actual_missing": window["actual_missing"],
                "imputation_records": window["imputation_records"],
                "metrics": metrics,
                "cache_keys": {
                    "window_key": task.window_key,
                    "train_key": task.train_key,
                    "prediction_key": task.prediction_key,
                    "metric_key": task.metric_key,
                    "prediction_fingerprint": prediction_payload["prediction_fingerprint"],
                },
            }
            model_windows.append(record)
            model_metric_records.append(metrics)
            matrix.setdefault(
                (task.series_id, task.window_index),
                {
                    "series_id": task.series_id,
                    "window_index": task.window_index,
                    "models": {},
                },
            )["models"][model_name] = {
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "tolerance_exceeded": metrics["tolerance_exceeded"],
                "first_exceeded": metrics["first_exceeded"],
                "cache_keys": {
                    "prediction_key": task.prediction_key,
                    "metric_key": task.metric_key,
                },
            }
            if metrics["tolerance_exceeded"]:
                item = {
                    "series_id": task.series_id,
                    "window_index": task.window_index,
                    "model_name": model_name,
                    "first_exceeded": metrics["first_exceeded"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                }
                over_tolerance.append(item)
                over_tolerance_global.append(item)
        models[model_name] = {
            "aggregate": aggregate_metrics(model_metric_records),
            "windows": model_windows,
            "over_tolerance_windows": over_tolerance,
        }

    window_records = [state.windows[key] for key in sorted(state.windows)]
    schedule_report: dict[str, Any] | None = None
    if state.schedule is not None:
        schedule_report = state.schedule.to_dict()
        schedule_report.update(
            {
                "concurrency": state.config.scheduler.concurrency,
                "batch_point_budget": state.config.scheduler.batch_point_budget,
                "per_series_point_quota": state.config.scheduler.per_series_point_quota,
                "budget_used_points": state.schedule.budget_used,
                "budget_remaining_points": state.schedule.budget_remaining,
            }
        )

    return {
        "schema_version": 1,
        "report_version": "1.0",
        "status": status,
        "exit_code": exit_code,
        "config": state.config.to_dict(),
        "inputs": {
            "config_path": str(state.config.source_path),
            "samples_path": str(state.config.samples_path),
            "injection_manifest_path": str(state.config.injection_manifest_path),
            "config_hash": state.config.config_hash,
            "samples_hash": samples_hash,
            "injection_hash": state.injections.manifest_hash,
        },
        "snapshot": state.snapshot.snapshot_summary(),
        "stage_timings_ms": {key: round(value, 3) for key, value in state.timings.items()},
        "stage_status": state.stage_status,
        "summary": {
            "series_count": len(state.samples),
            "window_count": len(state.windows),
            "task_count": len(state.tasks),
            "successful_model_windows": successful_tasks,
            "failure_count": len(failures),
            "injection_hit_count": len(state.hits),
        },
        "samples": [sample.to_summary() for _, sample in sorted(state.samples.items())],
        "windows": window_records,
        "models": models,
        "metric_matrix": [matrix[key] for key in sorted(matrix)],
        "over_tolerance_windows": over_tolerance_global,
        "cache_plan": [task.cache_plan() for task in sorted(state.tasks, key=lambda item: item.sort_key)],
        "incremental_audit": {
            "invalidation": state.invalidated,
            **state.audit,
        },
        "scheduler": schedule_report,
        "failures": [failure.to_dict() for failure in failures],
        "injections": _build_injection_report(state),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    atomic_write_json(path, report)


def _fatal_report(
    *,
    options: PipelineOptions,
    failures: list[Failure],
    timings: dict[str, float],
    config: Config | None = None,
) -> PipelineResult:
    report = {
        "schema_version": 1,
        "report_version": "1.0",
        "status": "fatal",
        "exit_code": 2,
        "config": config.to_dict() if config else None,
        "inputs": {
            "config_path": str(options.config_path.resolve()),
        },
        "snapshot": None,
        "stage_timings_ms": {key: round(value, 3) for key, value in timings.items()},
        "stage_status": {"bootstrap": "failed"},
        "summary": {
            "series_count": 0,
            "window_count": 0,
            "task_count": 0,
            "successful_model_windows": 0,
            "failure_count": len(failures),
            "injection_hit_count": 0,
        },
        "samples": [],
        "windows": [],
        "models": {},
        "metric_matrix": [],
        "over_tolerance_windows": [],
        "cache_plan": [],
        "incremental_audit": {},
        "scheduler": None,
        "failures": [failure.to_dict() for failure in sort_failures(failures)],
        "injections": {"manifest_hash": None, "items": [], "hits": [], "unhit_injection_ids": []},
    }
    _write_report(options.report_path, report)
    return PipelineResult(2, options.report_path, report)


def run_pipeline(options: PipelineOptions) -> PipelineResult:
    overall_start = time.perf_counter()
    timings: dict[str, float] = {}
    load_start = time.perf_counter()
    loaded_config = load_config(
        options.config_path,
        samples_override=options.samples_override,
        injection_override=options.injection_override,
    )
    timings["load_config"] = _elapsed_ms(load_start)
    if loaded_config.config is None or loaded_config.failures:
        failures = list(loaded_config.failures)
        if not failures:
            failures.append(Failure(E_CONFIG_INVALID, "load_validation", "invalid configuration"))
        return _fatal_report(
            options=options,
            failures=failures,
            timings=timings,
            config=loaded_config.config,
        )
    config = loaded_config.config

    sample_start = time.perf_counter()
    loaded_samples = load_samples(config.samples_path)
    timings["load_samples"] = _elapsed_ms(sample_start)
    if loaded_samples.fatal:
        return _fatal_report(
            options=options,
            failures=loaded_samples.failures,
            timings=timings,
            config=config,
        )

    injection_start = time.perf_counter()
    loaded_injections = load_injections(
        config.injection_manifest_path,
        config,
        loaded_samples.samples,
    )
    timings["load_injections"] = _elapsed_ms(injection_start)
    if loaded_injections.fatal:
        return _fatal_report(
            options=options,
            failures=loaded_samples.failures + loaded_injections.failures,
            timings=timings,
            config=config,
        )
    if not loaded_samples.samples:
        failures = loaded_samples.failures + loaded_injections.failures
        if not failures:
            failures.append(Failure(E_SAMPLE_INVALID, "load_validation", "no valid samples"))
        return _fatal_report(options=options, failures=failures, timings=timings, config=config)

    samples_hash = file_sha256(config.samples_path)
    snapshot = SnapshotManager(
        options.state_dir,
        snapshot_version=config.snapshot.version,
        config_hash=config.config_hash,
        injection_hash=loaded_injections.manifest_hash,
        samples_hash=samples_hash,
    )
    prior_task_fingerprints = snapshot.previous_task_fingerprints()
    cache = CacheStore(options.state_dir / "cache")
    state = RunState(
        config=config,
        samples=loaded_samples.samples,
        injections=loaded_injections,
        snapshot=snapshot,
        cache=cache,
        failures=list(loaded_samples.failures) + list(loaded_injections.failures),
        audit=_empty_audit(),
    )
    if snapshot.status_code == E_SNAPSHOT_CORRUPT:
        state.failures.append(
            Failure(
                E_SNAPSHOT_CORRUPT,
                "snapshot",
                "; ".join(snapshot.stale_reasons) or "snapshot is corrupt",
            )
        )

    # Stage 1: load validation snapshot.
    snapshot.commit(
        stage="load",
        stage_payload=_stage_payload_load(state, state.failures),
        progress=_progress(state, "load"),
        status="complete",
    )
    state.stage_status["load"] = "complete"
    timings["stage_load"] = _elapsed_ms(load_start)
    if _should_stop(options, "load"):
        state.timings = timings
        report = _build_report(
            state,
            status="interrupted",
            exit_code=0,
            samples_hash=samples_hash,
        )
        _write_report(options.report_path, report)
        return PipelineResult(0, options.report_path, report)

    # Stage 2: deterministic windows and imputation.
    windows_start = time.perf_counter()
    window_failures: list[Failure] = []
    window_specs: list[tuple[str, int]] = []
    for sequence_id in sorted(state.samples):
        sample = state.samples[sequence_id]
        required = config.backtest.window_length + config.backtest.forecast_horizon
        if sample.length < required:
            window_failures.append(
                Failure(
                    E_SAMPLE_INVALID,
                    "window_build",
                    "sequence is shorter than window_length + forecast_horizon",
                    sequence_id=sequence_id,
                    detail={"sequence_length": sample.length, "required": required},
                )
            )
            continue
        count = (sample.length - required) // config.backtest.step + 1
        for window_index in range(count):
            window_specs.append((sequence_id, window_index))
    window_specs.sort()

    interval = config.snapshot.commit_interval_windows
    window_chunks = [
        window_specs[index : index + interval]
        for index in range(0, len(window_specs), interval)
    ]
    if not window_chunks:
        window_chunks = [[]]
    for chunk_index, chunk in enumerate(window_chunks):
        for sequence_id, window_index in chunk:
            sample = state.samples[sequence_id]
            injections_here = state.injections.for_window(sequence_id, window_index)
            key = window_cache_key(sample, window_index, config, injections_here)
            cached = cache.get("window", key, snapshot.committed_inventory_map)
            if cached is not None and cached.get("schema") == "window-artifact-v1":
                state.windows[(sequence_id, window_index)] = cached
                state.audit["window_reused"].append(
                    {"series_id": sequence_id, "window_index": window_index, "window_key": key}
                )
                continue
            try:
                window = build_window(sample, window_index, config, injections_here)
            except WindowBuildError as exc:
                failure = exc.as_failure(sequence_id, window_index)
                window_failures.append(failure)
                for injection in injections_here:
                    if injection.kind in {
                        "non_finite_observation",
                        "irregular_timestamp",
                        "window_too_long",
                    }:
                        _record_hit(
                            state,
                            injection,
                            stage="window_build",
                            model_name=None,
                            failure_code=exc.code,
                            message=exc.message,
                        )
                continue
            except Exception as exc:
                failure = Failure(
                    E_WINDOW_BUILD_FAILED,
                    "window_build",
                    f"unexpected window build failure: {exc}",
                    sequence_id=sequence_id,
                    window_index=window_index,
                )
                window_failures.append(failure)
                continue
            state.windows[(sequence_id, window_index)] = window
            state.audit["window_computed"].append(
                {"series_id": sequence_id, "window_index": window_index, "window_key": key}
            )
            entry = cache.put("window", key, window)
            snapshot.register_inventory([entry])
        is_last = chunk_index == len(window_chunks) - 1
        snapshot.commit(
            stage="windows",
            stage_payload=_stage_payload_windows(state, window_failures),
            progress=_progress(
                state,
                "windows",
                completed_windows=[
                    {"series_id": sequence_id, "window_index": window_index}
                    for sequence_id, window_index in sorted(state.windows)
                ],
            ),
            status="complete" if is_last else "in_progress",
        )

    state.failures.extend(window_failures)
    state.tasks = [
        make_task(config=config, window=window, model_name=model_name)
        for window in (state.windows[key] for key in sorted(state.windows))
        for model_name in config.models.enabled
    ]
    state.tasks.sort(key=lambda task: task.sort_key)
    state.invalidated = invalidation_report(prior_task_fingerprints, state.tasks)
    state.stage_status["windows"] = "complete"
    timings["stage_windows"] = _elapsed_ms(windows_start)
    if _should_stop(options, "windows"):
        state.timings = timings
        report = _build_report(
            state,
            status="interrupted",
            exit_code=0,
            samples_hash=samples_hash,
        )
        _write_report(options.report_path, report)
        return PipelineResult(0, options.report_path, report)

    # Stage 3: baseline model training.
    models_start = time.perf_counter()
    state.schedule = schedule_tasks(state.tasks, config)
    state.failures.extend(state.schedule.failures)
    training_failures: list[Failure] = []
    completed_training: list[str] = []
    train_chunks = chunk_tasks_by_windows(state.schedule.ready_tasks, interval)
    if not train_chunks:
        train_chunks = [[]]
    config_ref = config

    def train_worker(task: TaskPlan) -> tuple[str, dict[str, Any] | None, Exception | None]:
        try:
            model = model_for_name(task.model_name)
            window = task.window
            context = TrainingContext(
                series_id=task.series_id,
                window_index=task.window_index,
                train_start_index=window["train_start_index"],
                train_end_index=window["train_end_index_exclusive"],
                train_start_ms=window["train_timestamps_ms"][0],
                train_end_ms=window["train_timestamps_ms"][-1],
                values=tuple(float(value) for value in window["train_values"]),
            )
            artifact = model.fit(
                context.values,
                window["train_timestamps_ms"],
                context,
                config_ref.models,
            )
            payload = {
                "schema": "train-cache-v1",
                "task_key": task.task_key,
                "series_id": task.series_id,
                "window_index": task.window_index,
                "model_name": task.model_name,
                "window_key": task.window_key,
                "train_key": task.train_key,
                "artifact": artifact,
            }
            return task.task_key, payload, None
        except Exception as exc:
            return task.task_key, None, exc

    for chunk_index, chunk in enumerate(train_chunks):
        misses: list[TaskPlan] = []
        for task in chunk:
            cached = cache.get("train", task.train_key, snapshot.committed_inventory_map)
            if cached is not None and isinstance(cached.get("artifact"), dict):
                state.train_artifacts[task.task_key] = cached
                state.audit["training_reused"].append(task.task_key)
                completed_training.append(task.task_key)
            else:
                misses.append(task)
        for task_key, payload, error in execute_parallel(
            misses, train_worker, config.scheduler.concurrency
        ):
            task = next(item for item in chunk if item.task_key == task_key)
            if error is not None:
                failure = _failure_from_exception(task, "model_training", error, E_MODEL_FIT_FAILED)
                training_failures.append(failure)
                state.audit["training_failed"].append(task_key)
                continue
            assert payload is not None
            state.train_artifacts[task_key] = payload
            state.audit["training_computed"].append(task_key)
            completed_training.append(task_key)
            entry = cache.put("train", task.train_key, payload)
            snapshot.register_inventory([entry])
        is_last = chunk_index == len(train_chunks) - 1
        snapshot.commit(
            stage="models",
            stage_payload=_stage_payload_models(state, training_failures),
            progress=_progress(
                state,
                "models",
                completed_tasks=sorted(set(completed_training)),
            ),
            status="complete" if is_last else "in_progress",
        )
    state.failures.extend(training_failures)
    state.stage_status["models"] = "complete"
    timings["stage_models"] = _elapsed_ms(models_start)
    if _should_stop(options, "models"):
        state.timings = timings
        report = _build_report(
            state,
            status="interrupted",
            exit_code=0,
            samples_hash=samples_hash,
        )
        _write_report(options.report_path, report)
        return PipelineResult(0, options.report_path, report)

    # Stage 4: prediction, quantization and metrics.
    metrics_start = time.perf_counter()
    metric_failures: list[Failure] = []
    completed_metrics: list[str] = []
    metric_candidates = [
        task
        for task in state.schedule.ready_tasks
        if task.task_key in state.train_artifacts
    ]
    metric_chunks = chunk_tasks_by_windows(metric_candidates, interval)
    if not metric_chunks:
        metric_chunks = [[]]

    def prediction_worker(task: TaskPlan) -> tuple[str, dict[str, Any] | None, Exception | None]:
        try:
            model = model_for_name(task.model_name)
            artifact = state.train_artifacts[task.task_key]["artifact"]
            predictions = model.predict(artifact, config_ref.backtest.forecast_horizon)
            if len(predictions) != config_ref.backtest.forecast_horizon:
                raise ModelError(
                    E_PREDICTION_FAILED,
                    f"model returned {len(predictions)} predictions, expected "
                    f"{config_ref.backtest.forecast_horizon}",
                )
            if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in predictions):
                raise ModelError(E_PREDICTION_FAILED, "model returned a non-finite prediction")
            payload = {
                "schema": "prediction-cache-v1",
                "task_key": task.task_key,
                "series_id": task.series_id,
                "window_index": task.window_index,
                "model_name": task.model_name,
                "window_key": task.window_key,
                "train_key": task.train_key,
                "prediction_key": task.prediction_key,
                "predictions": [float(value) for value in predictions],
                "prediction_fingerprint": fingerprint(
                    {"schema": "prediction-values-v1", "values": [float(value) for value in predictions]}
                ),
            }
            return task.task_key, payload, None
        except Exception as exc:
            return task.task_key, None, exc

    def metric_worker(
        task: TaskPlan,
        predictions: list[float],
    ) -> tuple[str, dict[str, Any] | None, Exception | None]:
        try:
            metrics = evaluate_window(
                predictions,
                task.window["actual_values"],
                config_ref.quantization,
                config_ref.backtest.error_tolerance,
            )
            payload = {
                "schema": "metric-cache-v1",
                "task_key": task.task_key,
                "series_id": task.series_id,
                "window_index": task.window_index,
                "model_name": task.model_name,
                "window_key": task.window_key,
                "train_key": task.train_key,
                "prediction_key": task.prediction_key,
                "metric_key": task.metric_key,
                "actual_fingerprint": task.window["actual_fingerprint"],
                "metrics": metrics,
            }
            return task.task_key, payload, None
        except Exception as exc:
            return task.task_key, None, exc

    for chunk_index, chunk in enumerate(metric_chunks):
        blocked: set[str] = set()
        for task in chunk:
            matching_quant_injections = [
                injection
                for injection in state.injections.for_window(task.series_id, task.window_index)
                if injection.kind == "quantization_out_of_range"
                and (
                    injection.raw.get("model") is None
                    or injection.raw.get("model") == task.model_name
                )
            ]
            if matching_quant_injections:
                for injection in matching_quant_injections:
                    _record_hit(
                        state,
                        injection,
                        stage="metrics",
                        model_name=task.model_name,
                        failure_code=E_CONFIG_INVALID,
                        message="injected quantization level is outside the allowed enumeration",
                    )
                metric_failures.append(
                    Failure(
                        E_CONFIG_INVALID,
                        "quantization",
                        "injected quantization level is outside the allowed enumeration",
                        sequence_id=task.series_id,
                        window_index=task.window_index,
                        model_name=task.model_name,
                        injection_id=matching_quant_injections[0].injection_id,
                        detail={"quantization_level": matching_quant_injections[0].raw.get("quantization_level")},
                    )
                )
                state.audit["metric_failed"].append(task.task_key)
                blocked.add(task.task_key)

        prediction_misses: list[TaskPlan] = []
        for task in chunk:
            if task.task_key in blocked:
                continue
            cached = cache.get("prediction", task.prediction_key, snapshot.committed_inventory_map)
            if (
                cached is not None
                and isinstance(cached.get("predictions"), list)
                and len(cached["predictions"]) == config_ref.backtest.forecast_horizon
            ):
                state.prediction_payloads[task.task_key] = cached
                state.audit["prediction_reused"].append(task.task_key)
            else:
                prediction_misses.append(task)
        for task_key, payload, error in execute_parallel(
            prediction_misses, prediction_worker, config.scheduler.concurrency
        ):
            task = next(item for item in chunk if item.task_key == task_key)
            if error is not None:
                metric_failures.append(
                    _failure_from_exception(task, "prediction", error, E_PREDICTION_FAILED)
                )
                state.audit["prediction_failed"].append(task_key)
                blocked.add(task_key)
                continue
            assert payload is not None
            state.prediction_payloads[task_key] = payload
            state.audit["prediction_computed"].append(task_key)
            entry = cache.put("prediction", task.prediction_key, payload)
            snapshot.register_inventory([entry])

        for task in chunk:
            if task.task_key in blocked or task.task_key not in state.prediction_payloads:
                continue
            cached = cache.get("metric", task.metric_key, snapshot.committed_inventory_map)
            if cached is not None and isinstance(cached.get("metrics"), dict):
                state.metric_payloads[task.task_key] = cached
                state.audit["metric_reused"].append(task.task_key)
                completed_metrics.append(task.task_key)
                continue
            predictions = state.prediction_payloads[task.task_key]["predictions"]
            task_key, payload, error = metric_worker(task, predictions)
            if error is not None:
                metric_failures.append(
                    _failure_from_exception(task, "metrics", error, E_METRIC_FAILED)
                )
                state.audit["metric_failed"].append(task_key)
                continue
            assert payload is not None
            state.metric_payloads[task_key] = payload
            state.audit["metric_computed"].append(task_key)
            completed_metrics.append(task_key)
            entry = cache.put("metric", task.metric_key, payload)
            snapshot.register_inventory([entry])
        is_last = chunk_index == len(metric_chunks) - 1
        snapshot.commit(
            stage="metrics",
            stage_payload=_stage_payload_metrics(state, metric_failures),
            progress=_progress(
                state,
                "metrics",
                completed_tasks=sorted(set(completed_training + completed_metrics)),
            ),
            status="complete" if is_last else "in_progress",
        )
    state.failures.extend(metric_failures)
    state.stage_status["metrics"] = "complete"
    timings["stage_metrics"] = _elapsed_ms(metrics_start)
    timings["total"] = _elapsed_ms(overall_start)
    state.timings = timings
    report = _build_report(
        state,
        status="completed",
        exit_code=0,
        samples_hash=samples_hash,
    )
    _write_report(options.report_path, report)
    return PipelineResult(0, options.report_path, report)
