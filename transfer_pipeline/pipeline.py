"""Incremental orchestration, invalidation, concurrency, and reporting."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical import (
    ALGORITHM_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    atomic_write_json,
    fingerprint,
)
from .data import (
    AdapterConfigDocument,
    EffectiveAdapterConfig,
    EncoderPackage,
    FeatureLibrary,
    TaskDescription,
    failure_record,
    failure_sort_key,
    normalized_run_material,
    parse_encoder,
    parse_feature_library,
    validate_tasks,
)
from .encoder import RepresentationBuild, build_representations
from .errors import (
    AdapterConfigError,
    EncoderError,
    EvaluationError,
    FeatureLibraryError,
    PipelineConfigError,
    PipelineError,
    SnapshotError,
    SnapshotMismatchError,
    TrainingError,
)
from .injection import build_injection_plan
from .snapshot import SnapshotStore
from .training import evaluate_adapter, train_adapter


@dataclass(frozen=True)
class PipelineResult:
    report: Mapping[str, Any]
    exit_code: int


def _empty_incremental() -> dict[str, Any]:
    return {
        "adapter_recomputed_tasks": [],
        "adapter_reused_tasks": [],
        "deleted_tasks": [],
        "evaluation_recomputed_tasks": [],
        "evaluation_reused_tasks": [],
        "invalidated_tasks": [],
        "reused_tasks": [],
        "trained_tasks": [],
        "invalidated_task_count": 0,
        "reused_task_count": 0,
        "trained_task_count": 0,
    }


def _base_report(
    *,
    config_hash: str | None,
    run_id: str | None,
    snapshot_version: int,
    snapshot_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stage_versions = {}
    completed_tasks: list[str] = []
    completed_stages: list[str] = []
    if snapshot_payload is not None:
        stage_versions = copy.deepcopy(snapshot_payload.get("stage_versions", {}))
        completed_tasks = list(snapshot_payload.get("completed_tasks", []))
        completed_stages = list(snapshot_payload.get("completed_stages", []))
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "snapshot_version": snapshot_version,
        "run_state": "failed",
        "exit_code": 2,
        "interrupted": False,
        "resumed": False,
        "resumed_stages": [],
        "tasks": [],
        "incremental": _empty_incremental(),
        "cache": {
            "adapter_hits": 0,
            "adapter_misses": 0,
            "evaluation_hits": 0,
            "evaluation_misses": 0,
            "shared_representation_hits": 0,
            "shared_representation_recomputed": 0,
            "training_invocations": 0,
        },
        "shared_forward": {
            "recomputed_samples": [],
            "reused_samples": [],
            "referenced_samples": [],
        },
        "sample_references": {},
        "failures": [],
        "injection_hits": [],
        "snapshot": {
            "version": snapshot_version,
            "config_hash": config_hash,
            "run_id": run_id,
            "run_state": "failed",
            "completed_tasks": completed_tasks,
            "completed_stages": completed_stages,
            "stage_versions": stage_versions,
        },
    }


def _finish_report(report: dict[str, Any], report_path: str | Path | None) -> None:
    report["failures"].sort(key=failure_sort_key)
    if report_path is not None:
        atomic_write_json(report_path, report)


def _fatal(
    exc: BaseException,
    *,
    code: str,
    stage: str,
    config_hash: str | None,
    run_id: str | None,
    snapshot_version: int,
    snapshot_payload: Mapping[str, Any] | None,
    report_path: str | Path | None,
) -> PipelineResult:
    report = _base_report(
        config_hash=config_hash,
        run_id=run_id,
        snapshot_version=snapshot_version,
        snapshot_payload=snapshot_payload,
    )
    report["run_state"] = "failed"
    report["exit_code"] = 2
    report["failures"].append(
        failure_record(
            None,
            stage,
            code,
            type(exc).__name__,
            str(exc),
        )
    )
    report["snapshot"]["run_state"] = "failed"
    _finish_report(report, report_path)
    return PipelineResult(report=report, exit_code=2)


def _core_config_hash(features: FeatureLibrary, encoder: EncoderPackage) -> str:
    """Stable environment identity; tunable adapter fields are intentionally absent."""

    return fingerprint(
        {
            "algorithm_version": ALGORITHM_VERSION,
            "feature_dim": features.feature_dim,
            "encoder_fingerprint": encoder.fingerprint(),
            "representation_schema": "standardize_project_bias_v1",
        },
        domain="pipeline-core-config",
    )


def _adapter_cache_key(
    task: TaskDescription,
    features: FeatureLibrary,
    representations: Mapping[str, Mapping[str, Any]],
    config: EffectiveAdapterConfig,
) -> str:
    training_material = []
    for sample_id in task.train_ids:
        sample = features.samples[sample_id]
        training_material.append(
            {
                "sample_id": sample_id,
                "label": sample.label,
                "sample_fingerprint": sample.content_fingerprint(),
                "representation_key": representations[sample_id]["key"],
            }
        )
    return fingerprint(
        {
            "algorithm_version": ALGORITHM_VERSION,
            "task_name": task.name,
            "classes": list(task.classes),
            "training": training_material,
            "adapter_config": config.training_material(),
        },
        domain="task-adapter-cache-key",
    )


def _evaluation_cache_key(
    task: TaskDescription,
    features: FeatureLibrary,
    representations: Mapping[str, Mapping[str, Any]],
    config: EffectiveAdapterConfig,
    adapter_key: str,
) -> str:
    validation_material = []
    for sample_id in task.val_ids:
        sample = features.samples[sample_id]
        validation_material.append(
            {
                "sample_id": sample_id,
                "label": sample.label,
                "sample_fingerprint": sample.content_fingerprint(),
                "representation_key": representations[sample_id]["key"],
            }
        )
    return fingerprint(
        {
            "algorithm_version": ALGORITHM_VERSION,
            "task_name": task.name,
            "classes": list(task.classes),
            "adapter_cache_key": adapter_key,
            "validation": validation_material,
            "validation_threshold": config.validation_threshold,
        },
        domain="task-evaluation-cache-key",
    )


def _valid_adapter_entry(value: Any, key: str) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("key") == key
        and isinstance(value.get("parameters"), Mapping)
        and isinstance(value.get("summary"), Mapping)
        and isinstance(value.get("losses"), list)
    )


def _valid_evaluation_entry(value: Any, key: str) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("key") == key
        and isinstance(value.get("metrics"), Mapping)
    )


def _array_parameters(entry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    parameters_raw = entry.get("parameters")
    if not isinstance(parameters_raw, Mapping):
        raise TrainingError("adapter cache entry has no parameter table")
    parameters: dict[str, np.ndarray] = {}
    for name, values in parameters_raw.items():
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TrainingError(f"adapter parameter {name!r} is malformed") from exc
        if not np.isfinite(array).all():
            raise TrainingError(f"adapter parameter {name!r} is non-finite")
        parameters[str(name)] = array
    return parameters


@dataclass(frozen=True)
class _TaskExecution:
    task_name: str
    skipped: bool = False
    failure: Mapping[str, Any] | None = None
    adapter_entry: Mapping[str, Any] | None = None
    evaluation_entry: Mapping[str, Any] | None = None
    adapter_hit: bool = False
    evaluation_hit: bool = False
    trained: bool = False
    adapter_recomputed: bool = False
    evaluation_recomputed: bool = False


def _execute_task(
    task: TaskDescription,
    config: EffectiveAdapterConfig,
    features: FeatureLibrary,
    representations: Mapping[str, Mapping[str, Any]],
    adapter_cache: Mapping[str, Any],
    evaluation_cache: Mapping[str, Any],
    *,
    working_version: int,
    force_full: bool,
    injection_ids: Sequence[str],
    forced_errors: Sequence[str],
    stop_event: threading.Event,
    completion_callback: Any | None = None,
) -> _TaskExecution:
    def completed(result: _TaskExecution) -> _TaskExecution:
        if completion_callback is not None:
            completion_callback()
        return result

    if stop_event.is_set():
        return _TaskExecution(task_name=task.name, skipped=True)
    adapter_key = _adapter_cache_key(task, features, representations, config)
    adapter_entry: Mapping[str, Any] | None = None
    adapter_hit = False
    trained = False
    try:
        if "train" in forced_errors:
            raise RuntimeError("injected training-stage failure")
        cached_adapter = None if force_full else adapter_cache.get(adapter_key)
        if _valid_adapter_entry(cached_adapter, adapter_key):
            adapter_entry = cached_adapter
            adapter_hit = True
        else:
            result = train_adapter(task, features.samples, representations, config)
            adapter_entry = {
                "key": adapter_key,
                "task_name": task.name,
                "adapter_type": config.adapter_type,
                "rank": config.rank,
                "parameters": {
                    name: np.asarray(value, dtype=np.float64).tolist()
                    for name, value in result.parameters.items()
                },
                "losses": list(result.losses),
                "converged": bool(result.converged),
                "epochs_run": int(result.epochs_run),
                "summary": copy.deepcopy(dict(result.summary)),
                "artifact_version": working_version,
            }
            trained = True
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        code = exc.code if isinstance(exc, PipelineError) else TrainingError.code
        return completed(
            _TaskExecution(
                task_name=task.name,
                failure=failure_record(
                    task.name,
                    "adapter_train",
                    code,
                    type(exc).__name__,
                    str(exc),
                    injection_ids,
                ),
            )
        )

    evaluation_key = _evaluation_cache_key(
        task, features, representations, config, adapter_key
    )
    evaluation_entry: Mapping[str, Any] | None = None
    evaluation_hit = False
    try:
        if "evaluate" in forced_errors:
            raise RuntimeError("injected evaluation-stage failure")
        cached_evaluation = None if force_full else evaluation_cache.get(evaluation_key)
        if _valid_evaluation_entry(cached_evaluation, evaluation_key):
            evaluation_entry = cached_evaluation
            evaluation_hit = True
        else:
            parameters = _array_parameters(adapter_entry)
            metrics = evaluate_adapter(
                task, features.samples, representations, config, parameters
            )
            evaluation_entry = {
                "key": evaluation_key,
                "task_name": task.name,
                "adapter_cache_key": adapter_key,
                "metrics": copy.deepcopy(metrics),
                "artifact_version": working_version,
            }
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        code = exc.code if isinstance(exc, PipelineError) else EvaluationError.code
        return completed(
            _TaskExecution(
                task_name=task.name,
                failure=failure_record(
                    task.name,
                    "evaluation",
                    code,
                    type(exc).__name__,
                    str(exc),
                    injection_ids,
                ),
            )
        )
    return completed(
        _TaskExecution(
            task_name=task.name,
            adapter_entry=adapter_entry,
            evaluation_entry=evaluation_entry,
            adapter_hit=adapter_hit,
            evaluation_hit=evaluation_hit,
            trained=trained,
            adapter_recomputed=not adapter_hit,
            evaluation_recomputed=not evaluation_hit,
        )
    )


def _sample_references(
    features: FeatureLibrary, tasks: Sequence[TaskDescription]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {
        sample_id: [] for sample_id in features.sorted_ids()
    }
    for task in sorted(tasks, key=lambda item: item.stable_sort_key):
        for sample_id in sorted(set(task.train_ids) | set(task.val_ids)):
            result.setdefault(sample_id, []).append(task.name)
    for sample_id in result:
        result[sample_id] = sorted(set(result[sample_id]))
    return {sample_id: result[sample_id] for sample_id in sorted(result)}


def _snapshot_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": snapshot.get("global_version", 0),
        "config_hash": snapshot.get("config_hash"),
        "run_id": snapshot.get("run_id"),
        "run_state": snapshot.get("run_state"),
        "completed_tasks": list(snapshot.get("completed_tasks", [])),
        "completed_stages": list(snapshot.get("completed_stages", [])),
        "stage_versions": copy.deepcopy(snapshot.get("stage_versions", {})),
    }


def run_pipeline(
    tasks_document: Any,
    feature_document: Any,
    encoder_document: Any,
    adapter_config_document: Any,
    *,
    state_directory: str | Path,
    injection_document: Any | None = None,
    report_path: str | Path | None = None,
    stop_event: threading.Event | None = None,
    expected_version: int | None = None,
    expected_config_hash: str | None = None,
    require_resume: bool = False,
    force_full: bool = False,
    stop_after_completed: int | None = None,
) -> PipelineResult:
    """Run or incrementally update the full deterministic pipeline."""

    if expected_version is not None and (
        isinstance(expected_version, bool) or not isinstance(expected_version, int)
    ):
        exc = PipelineConfigError("expected_version must be an integer")
        return _fatal(
            exc,
            code=exc.code,
            stage="startup",
            config_hash=None,
            run_id=None,
            snapshot_version=0,
            snapshot_payload=None,
            report_path=report_path,
        )
    try:
        features = parse_feature_library(feature_document)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        code = exc.code if isinstance(exc, PipelineError) else FeatureLibraryError.code
        return _fatal(
            exc,
            code=code,
            stage="feature_library",
            config_hash=None,
            run_id=None,
            snapshot_version=0,
            snapshot_payload=None,
            report_path=report_path,
        )
    try:
        encoder = parse_encoder(encoder_document)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        code = exc.code if isinstance(exc, PipelineError) else EncoderError.code
        return _fatal(
            exc,
            code=code,
            stage="encoder",
            config_hash=None,
            run_id=None,
            snapshot_version=0,
            snapshot_payload=None,
            report_path=report_path,
        )
    try:
        adapter_config = AdapterConfigDocument.from_mapping(adapter_config_document)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        code = exc.code if isinstance(exc, PipelineError) else PipelineConfigError.code
        return _fatal(
            exc,
            code=code,
            stage="adapter_config",
            config_hash=None,
            run_id=None,
            snapshot_version=0,
            snapshot_payload=None,
            report_path=report_path,
        )

    core_hash = _core_config_hash(features, encoder)
    store = SnapshotStore(state_directory)
    try:
        store.ensure_writable()
        previous = store.load()
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        code = exc.code if isinstance(exc, PipelineError) else SnapshotError.code
        return _fatal(
            exc,
            code=code,
            stage="snapshot",
            config_hash=core_hash,
            run_id=None,
            snapshot_version=0,
            snapshot_payload=None,
            report_path=report_path,
    )
    previous_version = int(previous["global_version"]) if previous is not None else 0
    if previous is not None:
        stage_versions = previous.get("stage_versions")
        if (
            not isinstance(stage_versions, Mapping)
            or not stage_versions
            or any(
                isinstance(version, bool) or not isinstance(version, int)
                for version in stage_versions.values()
            )
            or max(stage_versions.values()) != previous_version
        ):
            exc = SnapshotMismatchError(
                "snapshot global_version is inconsistent with stage artifact versions"
            )
            return _fatal(
                exc,
                code=exc.code,
                stage="snapshot",
                config_hash=core_hash,
                run_id=previous.get("run_id"),
                snapshot_version=previous_version,
                snapshot_payload=previous,
                report_path=report_path,
            )
    if previous is not None and previous.get("config_hash") != core_hash:
        exc = SnapshotMismatchError("snapshot config hash does not match current core config")
        return _fatal(
            exc,
            code=exc.code,
            stage="snapshot",
            config_hash=core_hash,
            run_id=previous.get("run_id"),
            snapshot_version=previous_version,
            snapshot_payload=previous,
            report_path=report_path,
        )
    if expected_config_hash is not None and expected_config_hash != core_hash:
        exc = SnapshotMismatchError("expected config hash does not match current core config")
        return _fatal(
            exc,
            code=exc.code,
            stage="snapshot",
            config_hash=core_hash,
            run_id=previous.get("run_id") if previous else None,
            snapshot_version=previous_version,
            snapshot_payload=previous,
            report_path=report_path,
        )
    if expected_version is not None and expected_version != previous_version:
        exc = SnapshotMismatchError(
            f"expected snapshot version {expected_version}, found {previous_version}"
        )
        return _fatal(
            exc,
            code=exc.code,
            stage="snapshot",
            config_hash=core_hash,
            run_id=previous.get("run_id") if previous else None,
            snapshot_version=previous_version,
            snapshot_payload=previous,
            report_path=report_path,
        )

    try:
        injection_plan = build_injection_plan(
            tasks_document, adapter_config, injection_document
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        code = exc.code if isinstance(exc, PipelineError) else PipelineConfigError.code
        return _fatal(
            exc,
            code=code,
            stage="injection",
            config_hash=core_hash,
            run_id=previous.get("run_id") if previous else None,
            snapshot_version=previous_version,
            snapshot_payload=previous,
            report_path=report_path,
        )
    effective_tasks_document = injection_plan.tasks_document
    effective_config = injection_plan.adapter_config

    run_id = fingerprint(
        normalized_run_material(
            effective_tasks_document,
            features,
            encoder,
            effective_config,
            injection_plan.public_document,
        ),
        domain="pipeline-run",
    )
    same_run = bool(previous is not None and previous.get("run_id") == run_id)
    resume = bool(
        same_run
        and previous is not None
        and previous.get("run_state") in {"in_progress", "interrupted"}
        and not force_full
    )
    if require_resume and not resume:
        exc = SnapshotMismatchError(
            "resume requested but no matching interrupted snapshot/version was found"
        )
        return _fatal(
            exc,
            code=exc.code,
            stage="snapshot",
            config_hash=core_hash,
            run_id=run_id,
            snapshot_version=previous_version,
            snapshot_payload=previous,
            report_path=report_path,
        )
    working_version = previous_version if resume else previous_version + 1

    accepted_tasks, failures = validate_tasks(
        effective_tasks_document,
        features,
        encoder,
        injection_plan.injection_ids_by_task,
    )
    runnable_tasks: list[TaskDescription] = []
    effective_configs: dict[str, EffectiveAdapterConfig] = {}
    for task in accepted_tasks:
        try:
            config = effective_config.resolve(task.name)
        except AdapterConfigError as exc:
            failures.append(
                failure_record(
                    task.name,
                    "adapter_config",
                    exc.code,
                    type(exc).__name__,
                    str(exc),
                    injection_plan.injection_ids_by_task.get(task.name, ()),
                )
            )
            continue
        runnable_tasks.append(task)
        effective_configs[task.name] = config
    failures.sort(key=failure_sort_key)

    if force_full:
        representation_cache: dict[str, Any] = {}
        adapter_cache: dict[str, Any] = {}
        evaluation_cache: dict[str, Any] = {}
        previous_task_entries: dict[str, Any] = {}
    else:
        representation_cache = copy.deepcopy(
            dict(previous.get("representation_cache", {})) if previous else {}
        )
        adapter_cache = copy.deepcopy(
            dict(previous.get("adapter_cache", {})) if previous else {}
        )
        evaluation_cache = copy.deepcopy(
            dict(previous.get("evaluation_cache", {})) if previous else {}
        )
        previous_task_entries = copy.deepcopy(
            dict(previous.get("task_entries", {})) if previous else {}
        )
    previous_task_names = sorted(previous_task_entries)
    resumed_stages = sorted(previous.get("completed_stages", [])) if resume and previous else []

    stop_event = stop_event or threading.Event()
    accepted_references = sorted(
        {
            sample_id
            for task in accepted_tasks
            for sample_id in (*task.train_ids, *task.val_ids)
        }
    )
    try:
        representation_build: RepresentationBuild = build_representations(
            features.samples,
            encoder,
            accepted_references,
            representation_cache,
            force_full=force_full,
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        code = exc.code if isinstance(exc, PipelineError) else "E_SHARED_FORWARD_FAILED"
        return _fatal(
            exc,
            code=code,
            stage="shared_forward",
            config_hash=core_hash,
            run_id=run_id,
            snapshot_version=working_version,
            snapshot_payload=previous,
            report_path=report_path,
        )
    representation_cache = copy.deepcopy(dict(representation_build.cache))
    for entry in representation_cache.values():
        if isinstance(entry, dict) and entry.get("artifact_version") is None:
            entry["artifact_version"] = working_version
    current_representations = {
        sample_id: representation_cache[entry["key"]]
        for sample_id, entry in representation_build.current.items()
    }

    task_entries: dict[str, Any] = {}
    completed_task_names: list[str] = []
    completed_stages = ["shared_forward"]
    stage_versions = {
        "shared_forward": working_version,
        "adapter": working_version,
        "evaluation": working_version,
    }
    sample_references = _sample_references(features, accepted_tasks)

    def make_snapshot(run_state: str) -> dict[str, Any]:
        stages = list(completed_stages)
        if run_state == "committed":
            stages = ["shared_forward", "adapter", "evaluation"]
        unique_stages = sorted(set(stages))
        snapshot = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "global_version": working_version,
            "config_hash": core_hash,
            "run_id": run_id,
            "run_state": run_state,
            "completed_stages": unique_stages,
            "completed_tasks": sorted(set(completed_task_names)),
            "stage_versions": dict(stage_versions),
            "representation_cache": representation_cache,
            "adapter_cache": adapter_cache,
            "evaluation_cache": evaluation_cache,
            "task_entries": task_entries,
            "sample_references": sample_references,
            "failures": sorted(failures, key=failure_sort_key),
            "injection_hits": injection_plan.report_hits(failures),
        }
        return snapshot

    store.commit(make_snapshot("in_progress"))

    results: dict[str, _TaskExecution] = {}
    finished_count = 0
    worker_completed_count = 0
    completion_lock = threading.Lock()

    def completion_callback() -> None:
        nonlocal worker_completed_count
        with completion_lock:
            worker_completed_count += 1
            if (
                stop_after_completed is not None
                and worker_completed_count >= stop_after_completed
            ):
                stop_event.set()

    ordered_runnable = sorted(runnable_tasks, key=lambda task: task.stable_sort_key)
    futures: dict[Future[_TaskExecution], TaskDescription] = {}
    with ThreadPoolExecutor(max_workers=effective_config.concurrency) as executor:
        for task in ordered_runnable:
            future = executor.submit(
                _execute_task,
                task,
                effective_configs[task.name],
                features,
                current_representations,
                adapter_cache,
                evaluation_cache,
                working_version=working_version,
                force_full=force_full,
                injection_ids=injection_plan.injection_ids_by_task.get(task.name, ()),
                forced_errors=injection_plan.forced_errors.get(task.name, ()),
                stop_event=stop_event,
                completion_callback=completion_callback,
            )
            futures[future] = task
        for future in as_completed(futures):
            result = future.result()
            finished_count += 1
            if result.skipped:
                continue
            results[result.task_name] = result
            if result.failure is not None:
                failures.append(dict(result.failure))
            else:
                assert result.adapter_entry is not None
                assert result.evaluation_entry is not None
                adapter_cache[str(result.adapter_entry["key"])] = result.adapter_entry
                evaluation_cache[str(result.evaluation_entry["key"])] = result.evaluation_entry
                completed_task_names.append(result.task_name)
                task_entries[result.task_name] = {
                    "adapter_cache_key": result.adapter_entry["key"],
                    "evaluation_cache_key": result.evaluation_entry["key"],
                    "adapter_artifact_version": result.adapter_entry.get("artifact_version"),
                    "evaluation_artifact_version": result.evaluation_entry.get(
                        "artifact_version"
                    ),
                }
            failures.sort(key=failure_sort_key)
            if finished_count % effective_config.snapshot_commit_interval == 0:
                store.commit(make_snapshot("in_progress"))

    interrupted = bool(stop_event.is_set()) and (
        len(completed_task_names) < len(runnable_tasks)
    )
    run_state = "interrupted" if interrupted else "committed"
    if interrupted:
        completed_stages = ["shared_forward"]
    else:
        completed_stages = ["shared_forward", "adapter", "evaluation"]
    failures.sort(key=failure_sort_key)
    final_snapshot = make_snapshot(run_state)
    final_snapshot["injection_hits"] = injection_plan.report_hits(failures)
    store.commit(final_snapshot)

    report = _base_report(
        config_hash=core_hash,
        run_id=run_id,
        snapshot_version=working_version,
        snapshot_payload=final_snapshot,
    )
    report["run_state"] = run_state
    report["exit_code"] = 0
    report["interrupted"] = interrupted
    report["resumed"] = resume
    report["resumed_stages"] = resumed_stages
    report["failures"] = sorted(failures, key=failure_sort_key)
    report["injection_hits"] = injection_plan.report_hits(failures)
    report["sample_references"] = sample_references
    report["shared_forward"] = {
        "recomputed_samples": list(representation_build.recomputed_samples),
        "reused_samples": list(representation_build.reused_samples),
        "referenced_samples": accepted_references,
    }

    task_reports: list[dict[str, Any]] = []
    adapter_recomputed: list[str] = []
    adapter_reused: list[str] = []
    evaluation_recomputed: list[str] = []
    evaluation_reused: list[str] = []
    invalidated: list[str] = []
    fully_reused: list[str] = []
    trained_tasks: list[str] = []
    for task_name in sorted(results):
        result = results[task_name]
        if result.failure is not None:
            continue
        adapter_entry = result.adapter_entry
        evaluation_entry = result.evaluation_entry
        assert adapter_entry is not None and evaluation_entry is not None
        task_reports.append(
            {
                "name": task_name,
                "stable_sort_key": f"task:{task_name}",
                "status": "ok",
                "adapter_type": adapter_entry["adapter_type"],
                "rank": adapter_entry["rank"],
                "adapter_parameter_summary": copy.deepcopy(adapter_entry["summary"]),
                "training_losses": list(adapter_entry["losses"]),
                "converged": bool(adapter_entry["converged"]),
                "epochs_run": int(adapter_entry["epochs_run"]),
                "metrics": copy.deepcopy(evaluation_entry["metrics"]),
                "adapter_cache_key": adapter_entry["key"],
                "evaluation_cache_key": evaluation_entry["key"],
                "adapter_artifact_version": adapter_entry.get("artifact_version"),
                "evaluation_artifact_version": evaluation_entry.get("artifact_version"),
                "cache": {
                    "adapter_hit": result.adapter_hit,
                    "evaluation_hit": result.evaluation_hit,
                },
            }
        )
        if result.adapter_recomputed:
            adapter_recomputed.append(task_name)
        else:
            adapter_reused.append(task_name)
        if result.evaluation_recomputed:
            evaluation_recomputed.append(task_name)
        else:
            evaluation_reused.append(task_name)
        if result.trained:
            trained_tasks.append(task_name)
        if result.adapter_hit and result.evaluation_hit:
            fully_reused.append(task_name)
        else:
            invalidated.append(task_name)
    task_reports.sort(key=lambda item: (item["stable_sort_key"], item["name"]))
    report["tasks"] = task_reports
    report["adapter_parameter_table"] = [
        {
            "task_name": task["name"],
            "stable_sort_key": task["stable_sort_key"],
            "adapter_type": task["adapter_type"],
            "rank": task["rank"],
            "parameter_count": task["adapter_parameter_summary"]["parameter_count"],
            "digest": task["adapter_parameter_summary"]["digest"],
        }
        for task in task_reports
    ]
    current_task_names = sorted(
        task.name for task in accepted_tasks
    )
    deleted_tasks = sorted(set(previous_task_names) - set(current_task_names))
    report["incremental"] = {
        "adapter_recomputed_tasks": sorted(adapter_recomputed),
        "adapter_reused_tasks": sorted(adapter_reused),
        "deleted_tasks": deleted_tasks,
        "evaluation_recomputed_tasks": sorted(evaluation_recomputed),
        "evaluation_reused_tasks": sorted(evaluation_reused),
        "invalidated_tasks": sorted(invalidated),
        "reused_tasks": sorted(fully_reused),
        "trained_tasks": sorted(trained_tasks),
        "invalidated_task_count": len(invalidated),
        "reused_task_count": len(fully_reused),
        "trained_task_count": len(trained_tasks),
    }
    report["cache"] = {
        "adapter_hits": len(adapter_reused),
        "adapter_misses": len(adapter_recomputed),
        "evaluation_hits": len(evaluation_reused),
        "evaluation_misses": len(evaluation_recomputed),
        "shared_representation_hits": len(representation_build.reused_samples),
        "shared_representation_recomputed": len(representation_build.recomputed_samples),
        "training_invocations": len(trained_tasks),
        "cache_hit_task_count": len(fully_reused),
        "invalidated_task_count": len(invalidated),
    }
    report["failure_count"] = len(failures)
    report["snapshot"] = _snapshot_summary(final_snapshot)
    report["snapshot"]["run_state"] = run_state
    _finish_report(report, report_path)
    return PipelineResult(report=report, exit_code=0)
