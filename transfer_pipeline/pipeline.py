"""Incremental task-adaptation pipeline orchestration."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import train_and_validate
from .canonical import atomic_write_json, sha256_json, stable_task_sort_key
from .encoder import forward_fingerprint_inputs, run_shared_forward
from .errors import PipelineConfigMismatchError, SimulatedCrash
from .inputs import (
    Encoder,
    InputState,
    TaskLoadResult,
    TaskSpec,
    config_hash,
    feature_library_fingerprint,
    injection_ids,
    load_tasks,
    parse_encoder,
    runtime_injections,
)
from .snapshot import SnapshotData, SnapshotManager, generation_manifest_summary


@dataclass
class PipelineResult:
    report: dict[str, Any]
    snapshot: SnapshotData | None
    shared_forward: dict[str, Any] | None


def _task_sample_payload(features: dict[str, dict[str, Any]], sample_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    payload = []
    for sample_id in sorted(sample_ids):
        record = features[sample_id]
        payload.append(
            {
                "sample_id": sample_id,
                "features": [float(value) for value in record["features"]],
                "label": record.get("label", record.get("class")),
            }
        )
    return payload


def build_task_cache_key(
    task: TaskSpec,
    features: dict[str, dict[str, Any]],
    encoder: Encoder,
    config: dict[str, Any],
    injection_entries: list[dict[str, Any]] | None = None,
) -> str:
    target_index = encoder.layer_index(task.target_layer)
    if target_index is None:
        raise ValueError(f"target layer missing: {task.target_layer}")
    effective_rank = task.adapter_rank_override if task.adapter_rank_override is not None else config["rank"]
    referenced_layers = []
    for layer in encoder.layers[: target_index + 1]:
        referenced_layers.append(
            {
                "name": layer.name,
                "version": layer.version,
                "fingerprint": layer.fingerprint,
            }
        )
    payload = {
        "schema": "task-cache-v1",
        "task_name": task.name,
        "classes": sorted(task.classes),
        "target_layer": task.target_layer,
        "target_layer_index": target_index,
        "target_layer_whitelist": sorted(task.target_layer_whitelist),
        "train_samples": _task_sample_payload(features, task.train_ids),
        "validation_samples": _task_sample_payload(features, task.validation_ids),
        "encoder_normalization_fingerprint": encoder.normalization_fingerprint,
        "referenced_layers": referenced_layers,
        "activation": config["activation"],
        "adapter": {
            "type": config["adapter_type"],
            "rank": int(effective_rank),
            "scale": float(config["scale"]),
            "learning_rate": float(config["learning_rate"]),
            "epochs": int(config["epochs"]),
            "batch_size": int(config["batch_size"]),
            "seed_offset": int(config["seed_offset"]),
        },
        "validation": {
            "metric": config["validation_metric"],
            "threshold": float(config["validation_threshold"]),
        },
        "injection": {
            "task_injection_id": task.injection_id,
            "entries": injection_entries or [],
        },
    }
    return sha256_json(payload)


def _prefix_count(tasks: list[TaskSpec], ready: dict[str, dict[str, Any]]) -> int:
    count = 0
    for task in tasks:
        if task.name not in ready:
            break
        count += 1
    return count


def _sorted_failure(failure: dict[str, Any]) -> tuple[Any, ...]:
    return (
        failure.get("sort_key", ""),
        failure.get("task_name", ""),
        failure.get("phase", ""),
        failure.get("business_code", ""),
        failure.get("message", ""),
        failure.get("injection_id") or "",
    )


def _base_manifest(encoder: Encoder) -> dict[str, Any]:
    return {"fingerprint": encoder.fingerprint, "layers": encoder.layer_manifests()}


def _changed_base_layers(prior: SnapshotData | None, encoder: Encoder) -> list[str]:
    if prior is None:
        return []
    prior_layers = prior.manifest.get("base", {}).get("layers", [])
    prior_map = {
        str(layer.get("name")): (layer.get("version"), layer.get("fingerprint"))
        for layer in prior_layers
        if isinstance(layer, dict)
    }
    current_map = {
        layer.name: (layer.version, layer.fingerprint) for layer in encoder.layers
    }
    return sorted(
        name
        for name in set(prior_map) | set(current_map)
        if prior_map.get(name) != current_map.get(name)
    )


def _base_change_affects_task(task: TaskSpec, encoder: Encoder, changed_layers: list[str]) -> bool:
    if not changed_layers:
        return False
    target_index = encoder.layer_index(task.target_layer)
    if target_index is None:
        return True
    prefix = {layer.name for layer in encoder.layers[: target_index + 1]}
    return bool(prefix.intersection(changed_layers))


def _task_failure_payload(
    task: TaskSpec,
    cache_key: str,
    *,
    phase: str,
    code: str,
    message: str,
    exception_type: str | None,
    injection_id: str | None,
    adapter_type: str,
    rank: int,
    scale: float,
    reason_code: str | None = None,
) -> dict[str, Any]:
    failure = {
        "sort_key": task.sort_key,
        "task_name": task.name,
        "task_index": task.index,
        "target_layer": task.target_layer,
        "phase": phase,
        "business_code": code,
        "reason_code": reason_code,
        "message": message,
        "exception_type": exception_type,
        "injection_id": injection_id,
    }
    return {
        "task_name": task.name,
        "sort_key": task.sort_key,
        "target_layer": task.target_layer,
        "classes": list(task.effective_classes),
        "cache_key": cache_key,
        "status": "failure",
        "adapter_type": adapter_type,
        "rank": rank,
        "scale": scale,
        "failure": failure,
    }


def _exception_details(exc: BaseException) -> tuple[str, str, str | None]:
    code = getattr(exc, "business_code", "E_TRAINING_EXCEPTION")
    details = getattr(exc, "details", {}) or {}
    exception_type = str(details.get("exception_type") or type(exc).__name__)
    phase = str(details["phase"]) if details.get("phase") else None
    return str(code), exception_type, phase


def _injection_entries_by_id(injections: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = injections.get("injections", []) if isinstance(injections, dict) else []
    result: dict[str, dict[str, Any]] = {}
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get("id") is not None:
                result[str(entry["id"])] = entry
    return result


def _build_encoder_failure_report(
    state: InputState,
    normalized_config: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    code = getattr(error, "business_code", "E_ENCODER_INVALID")
    failure = {
        "sort_key": stable_task_sort_key("__encoder__", ""),
        "task_name": "__encoder__",
        "task_index": None,
        "target_layer": "",
        "phase": "load",
        "business_code": code,
        "reason_code": None,
        "message": str(error),
        "exception_type": type(error).__name__,
        "injection_id": None,
    }
    return {
        "schema_version": 1,
        "run_mode": "incremental",
        "input_state_hash": state.input_state_hash,
        "config_hash": config_hash(normalized_config),
        "config": normalized_config,
        "base": None,
        "task_results": [],
        "adapter_summary": [],
        "loss_trajectories": [],
        "validation_metrics": [],
        "validation_summary": {
            "task_count": 0,
            "mean_accuracy": None,
            "mean_macro_f1": None,
            "passed_count": 0,
            "threshold_failed_count": 0,
        },
        "threshold_failures": [],
        "failures": [failure],
        "lifecycle": {
            "invalidated": [],
            "reused": [],
            "retrained": [],
            "new_tasks": [],
            "removed": [],
            "not_trained": [],
        },
        "injection_hits": [],
        "injection_summary": {
            "expected_ids": injection_ids(state.injections),
            "hit_ids": [],
            "unmatched_ids": injection_ids(state.injections),
            "all_hit": not injection_ids(state.injections),
        },
        "snapshot": None,
        "recovery": {
            "resumed": False,
            "start_generation": None,
            "start_version": None,
            "recovered_completed_tasks": [],
            "shared_forward_reused": False,
            "config_migration": None,
        },
        "execution": {
            "configured_max_concurrency": int(normalized_config["max_concurrency"]),
            "observed_max_inflight": 0,
            "snapshot_interval": int(normalized_config["snapshot_interval"]),
        },
    }


def run_pipeline(
    input_source: str | Path | InputState,
    workspace: str | Path,
    *,
    mode: str = "incremental",
    fresh: bool = False,
    allow_config_migration: bool = False,
    expected_global_version: int | None = None,
    crash_after_completed: int | None = None,
    report_path: str | Path | None = None,
    injections_path: str | Path | None = None,
) -> PipelineResult:
    """Run the full or incremental pipeline and return a deterministic report."""
    if mode not in {"incremental", "full"}:
        raise ValueError("mode must be 'incremental' or 'full'")
    if isinstance(input_source, InputState):
        state = input_source
    else:
        state = InputState.from_directory(input_source, injections_path=injections_path)
    normalized_config = state.normalized_config
    current_config_hash = config_hash(normalized_config)
    manager = SnapshotManager(workspace)
    if fresh or mode == "full":
        manager.clear()

    expected_shared_dim: int | None = None
    try:
        raw_shared_dim = state.tasks.get("shared_feature_dim")
        if raw_shared_dim is not None:
            expected_shared_dim = int(raw_shared_dim)
    except (TypeError, ValueError):
        expected_shared_dim = None
    try:
        encoder = parse_encoder(state.encoder, expected_shared_dim=expected_shared_dim)
    except Exception as exc:
        report = _build_encoder_failure_report(state, normalized_config, exc)
        report["run_mode"] = mode if not fresh else "full"
        if report_path is not None:
            atomic_write_json(report_path, report)
        return PipelineResult(report=report, snapshot=None, shared_forward=None)

    load_result: TaskLoadResult = load_tasks(
        state.tasks,
        state.features,
        encoder,
    )
    tasks = load_result.tasks
    load_failures = list(load_result.failures)

    prior_loaded = None if (fresh or mode == "full") else manager.load_current()
    prior: SnapshotData | None = prior_loaded
    prior_for_lifecycle = prior_loaded
    config_migration: dict[str, Any] | None = None
    if expected_global_version is not None and prior is None:
        raise PipelineConfigMismatchError(
            f"no complete snapshot is available for expected global version {expected_global_version}",
            details={
                "reason": "snapshot_missing",
                "expected_global_version": expected_global_version,
            },
        )
    if prior is not None:
        prior_config_hash = str(prior.manifest.get("config_hash", ""))
        if expected_global_version is not None and int(prior.global_version) != int(expected_global_version):
            raise PipelineConfigMismatchError(
                f"snapshot global version {prior.global_version} != expected {expected_global_version}",
                details={
                    "reason": "global_version_mismatch",
                    "snapshot_version": prior.global_version,
                    "expected_global_version": expected_global_version,
                    "generation": prior.generation,
                },
            )
        if prior_config_hash != current_config_hash:
            if not allow_config_migration:
                raise PipelineConfigMismatchError(
                    f"snapshot config hash {prior_config_hash} != current {current_config_hash}",
                    details={
                        "reason": "config_hash_mismatch",
                        "snapshot_version": prior.global_version,
                        "snapshot_config_hash": prior_config_hash,
                        "current_config_hash": current_config_hash,
                        "generation": prior.generation,
                    },
                )
            config_migration = {
                "business_code": "E_CONFIG_MISMATCH",
                "reason": "config_hash_changed_explicit_new_lineage",
                "prior_config_hash": prior_config_hash,
                "current_config_hash": current_config_hash,
                "prior_generation": prior.generation,
                "prior_version": prior.global_version,
            }
            prior = None
            manager.clear()

    recovery = {
        "resumed": prior is not None,
        "start_generation": prior_for_lifecycle.generation if prior_for_lifecycle else None,
        "start_version": prior_for_lifecycle.global_version if prior_for_lifecycle else None,
        "recovered_completed_tasks": [
            str(entry.get("task_name"))
            for entry in (prior.completed_tasks if prior else [])
        ],
        "shared_forward_reused": False,
        "config_migration": config_migration,
    }

    current_input_fingerprint = forward_fingerprint_inputs(
        encoder, state.features, str(normalized_config["activation"])
    )
    shared_reused = bool(
        prior is not None
        and prior.shared_forward.get("input_fingerprint") == current_input_fingerprint
    )
    recovery["shared_forward_reused"] = shared_reused
    if shared_reused and prior is not None:
        shared_forward = copy.deepcopy(prior.shared_forward)
    else:
        shared_forward = run_shared_forward(
            encoder, state.features, str(normalized_config["activation"])
        )

    prior_cache = prior.cache_by_name() if prior is not None else {}
    prior_completed = prior.completed_by_name() if prior is not None else {}
    changed_base_layers = _changed_base_layers(prior_for_lifecycle, encoder)
    injection_entries_by_task: dict[str, list[dict[str, Any]]] = {}
    raw_injection_entries = (
        state.injections.get("injections", [])
        if isinstance(state.injections, dict)
        else []
    )
    if isinstance(raw_injection_entries, list):
        for entry in raw_injection_entries:
            if not isinstance(entry, dict) or entry.get("task") is None:
                continue
            injection_entries_by_task.setdefault(str(entry["task"]), []).append(entry)
    for entries in injection_entries_by_task.values():
        entries.sort(key=lambda item: str(item.get("id", "")))
    task_keys = {
        task.name: build_task_cache_key(
            task,
            state.features,
            encoder,
            normalized_config,
            injection_entries_by_task.get(task.name, []),
        )
        for task in tasks
    }

    ready: dict[str, dict[str, Any]] = {}
    reused: list[dict[str, Any]] = []
    invalidated: list[dict[str, Any]] = []
    new_tasks: list[dict[str, Any]] = []
    pending: list[TaskSpec] = []
    rank_failures: list[TaskSpec] = []
    for task in tasks:
        cache_key = task_keys[task.name]
        old = prior_cache.get(task.name)
        old_completed = prior_completed.get(task.name)
        if old is not None and old_completed is not None and str(old.get("cache_key")) == cache_key:
            payload = copy.deepcopy(old)
            payload["source"] = "reused"
            if isinstance(payload.get("failure"), dict):
                # Positional task index changes when unrelated tasks are added or
                # deleted; refresh it so cached failures match a clean rebuild.
                payload["failure"]["task_index"] = task.index
            ready[task.name] = payload
            reused.append(
                {
                    "task_name": task.name,
                    "sort_key": task.sort_key,
                    "cache_key": cache_key,
                }
            )
            continue

        prior_key = str(old.get("cache_key")) if old is not None else None
        if prior_for_lifecycle is not None:
            if old is None:
                reason = "config_hash_changed" if config_migration else "new_task"
            elif config_migration:
                reason = "config_hash_changed"
            elif _base_change_affects_task(task, encoder, changed_base_layers):
                reason = "referenced_base_layer_changed:" + ",".join(changed_base_layers)
            elif prior_key != cache_key:
                reason = "task_inputs_or_target_changed"
            else:
                reason = "artifact_missing"
            entry = {
                "task_name": task.name,
                "sort_key": task.sort_key,
                "reason": reason,
                "previous_cache_key": prior_key,
                "cache_key": cache_key,
            }
            if old is None and not config_migration:
                new_tasks.append(entry)
            else:
                invalidated.append(entry)
        else:
            new_tasks.append(
                {
                    "task_name": task.name,
                    "sort_key": task.sort_key,
                    "reason": "fresh_build",
                    "previous_cache_key": None,
                    "cache_key": cache_key,
                }
            )
        effective_rank = (
            task.adapter_rank_override
            if task.adapter_rank_override is not None
            else int(normalized_config["rank"])
        )
        if effective_rank <= 0:
            rank_failures.append(task)
            failure_payload = _task_failure_payload(
                task,
                cache_key,
                phase="adapter_construction",
                code="E_RANK_INVALID",
                message=f"adapter rank must be positive before construction (got {effective_rank})",
                exception_type="RankInvalidError",
                injection_id=task.injection_id,
                adapter_type=str(normalized_config["adapter_type"]),
                rank=int(effective_rank),
                scale=float(normalized_config["scale"]),
                reason_code=None,
            )
            failure_payload["source"] = "fresh"
            ready[task.name] = failure_payload
        else:
            pending.append(task)

    version_counter = prior.global_version if prior is not None else 0
    last_snapshot: SnapshotData | None = prior
    last_committed_count = 0

    def commit_prefix(count: int) -> SnapshotData:
        nonlocal version_counter, last_snapshot, last_committed_count
        version_counter += 1
        prefix_payloads = [ready[task.name] for task in tasks[:count]]
        snapshot = manager.commit(
            version=version_counter,
            config_hash=current_config_hash,
            base_fingerprint=encoder.fingerprint,
            base_layers=encoder.layer_manifests(),
            input_state_hash=state.input_state_hash,
            shared_forward=shared_forward,
            completed_tasks=prefix_payloads,
            input_fingerprints={
                "encoder": encoder.fingerprint,
                "features": feature_library_fingerprint(state.features),
                "shared_forward": current_input_fingerprint,
            },
        )
        last_snapshot = snapshot
        last_committed_count = count
        if crash_after_completed is not None and count >= int(crash_after_completed):
            raise SimulatedCrash(
                f"simulated crash after committed stable prefix of {count} tasks",
                details={"generation": snapshot.generation, "global_version": snapshot.global_version},
            )
        return snapshot

    initial_prefix = _prefix_count(tasks, ready)
    commit_prefix(initial_prefix)

    interval = int(normalized_config["snapshot_interval"])
    runtime = runtime_injections(state.injections)
    injection_by_id = _injection_entries_by_id(state.injections)
    inflight_lock = threading.Lock()
    inflight = 0
    observed_max_inflight = 0

    def execute_task(task: TaskSpec) -> dict[str, Any]:
        return train_and_validate(
            task=task,
            cache_key=task_keys[task.name],
            shared_forward=shared_forward,
            features=state.features,
            config=normalized_config,
            runtime=runtime,
        )

    def wrapped_execute(task: TaskSpec) -> dict[str, Any]:
        nonlocal inflight, observed_max_inflight
        with inflight_lock:
            inflight += 1
            observed_max_inflight = max(observed_max_inflight, inflight)
        try:
            return execute_task(task)
        finally:
            with inflight_lock:
                inflight -= 1

    max_concurrency = max(1, int(normalized_config["max_concurrency"]))
    worker_count = min(max_concurrency, len(pending)) if pending else 0
    attempted_names: list[str] = []
    with ThreadPoolExecutor(max_workers=worker_count or 1, thread_name_prefix="task-adapter") as executor:
        futures = {executor.submit(wrapped_execute, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            attempted_names.append(task.name)
            try:
                result = future.result()
                result["source"] = "fresh"
            except Exception as exc:  # task boundary: isolate and continue
                code, exception_type, exception_phase = _exception_details(exc)
                runtime_entry = runtime.get((task.name, "training")) or runtime.get(
                    (task.name, "validation")
                )
                phase = exception_phase or "training"
                if exception_phase is None:
                    if (
                        runtime.get((task.name, "validation")) is not None
                        and runtime.get((task.name, "training")) is None
                    ):
                        phase = "validation"
                injection_id = (
                    str(runtime_entry.get("id"))
                    if runtime_entry and runtime_entry.get("id") is not None
                    else task.injection_id
                )
                result = _task_failure_payload(
                    task,
                    task_keys[task.name],
                    phase=phase,
                    code=code,
                    message=str(exc),
                    exception_type=exception_type,
                    injection_id=injection_id,
                    adapter_type=str(normalized_config["adapter_type"]),
                    rank=int(
                        task.adapter_rank_override
                        if task.adapter_rank_override is not None
                        else normalized_config["rank"]
                    ),
                    scale=float(normalized_config["scale"]),
                    reason_code=None,
                )
                result["source"] = "fresh"
            ready[task.name] = result

            prefix = _prefix_count(tasks, ready)
            while last_committed_count + interval <= prefix:
                commit_prefix(last_committed_count + interval)

    final_prefix = _prefix_count(tasks, ready)
    if final_prefix != len(tasks):
        missing = [task.name for task in tasks if task.name not in ready]
        raise RuntimeError(f"task execution did not produce results for: {missing}")
    if last_committed_count < len(tasks):
        commit_prefix(len(tasks))
    if last_snapshot is None:
        last_snapshot = manager.load_current()

    task_results = [ready[task.name] for task in tasks]
    execution_failures = [
        copy.deepcopy(task["failure"])
        for task in task_results
        if task.get("status") == "failure" and "failure" in task
    ]
    failures = load_failures + execution_failures
    failures.sort(key=_sorted_failure)

    successful_tasks = [task for task in task_results if task.get("status") == "success"]
    adapter_summary = [
        {
            "task_name": task["task_name"],
            "sort_key": task["sort_key"],
            "target_layer": task["target_layer"],
            "adapter_type": task["adapter_type"],
            "rank": task["rank"],
            "scale": task["scale"],
            "parameter_shapes": task["parameter_shapes"],
            "parameter_fingerprint": task["parameter_fingerprint"],
            "cache_key": task["cache_key"],
        }
        for task in successful_tasks
    ]
    loss_trajectories = [
        {
            "task_name": task["task_name"],
            "sort_key": task["sort_key"],
            "losses": task["loss_trajectory"],
        }
        for task in successful_tasks
    ]
    validation_metrics = [
        {
            "task_name": task["task_name"],
            "sort_key": task["sort_key"],
            "target_layer": task["target_layer"],
            "accuracy": task["validation"]["accuracy"],
            "macro_f1": task["validation"]["macro_f1"],
            "metric": task["validation"]["metric"],
            "metric_value": task["validation"]["metric_value"],
            "threshold": task["validation"]["threshold"],
            "passed": task["validation"]["passed"],
            "sample_count": task["validation"]["sample_count"],
            "cache_key": task["cache_key"],
        }
        for task in successful_tasks
    ]
    threshold_failures = [
        {
            "task_name": item["task_name"],
            "sort_key": item["sort_key"],
            "business_code": "E_VALIDATION_THRESHOLD",
            "metric": item["metric"],
            "metric_value": item["metric_value"],
            "threshold": item["threshold"],
        }
        for item in validation_metrics
        if not item["passed"]
    ]
    validation_summary = {
        "task_count": len(validation_metrics),
        "mean_accuracy": (
            sum(item["accuracy"] for item in validation_metrics) / len(validation_metrics)
            if validation_metrics
            else None
        ),
        "mean_macro_f1": (
            sum(item["macro_f1"] for item in validation_metrics) / len(validation_metrics)
            if validation_metrics
            else None
        ),
        "passed_count": sum(1 for item in validation_metrics if item["passed"]),
        "threshold_failed_count": len(threshold_failures),
    }

    current_names = {
        str(task.get("name", task.get("task_name", "")))
        for task in state.tasks.get("tasks", [])
        if isinstance(task, dict)
    }
    removed = sorted(
        set(prior_completed) - current_names
    ) if prior_for_lifecycle is not None else []

    injection_hits = []
    for failure in failures:
        injection_id = failure.get("injection_id")
        if not injection_id:
            continue
        entry = injection_by_id.get(str(injection_id), {})
        injection_hits.append(
            {
                "injection_id": str(injection_id),
                "kind": entry.get("kind", failure.get("phase")),
                "task_name": failure.get("task_name"),
                "phase": failure.get("phase"),
                "business_code": failure.get("business_code"),
                "reason_code": failure.get("reason_code"),
                "hit": True,
            }
        )
    injection_hits.sort(
        key=lambda item: (item["injection_id"], str(item["task_name"]), str(item["phase"]))
    )
    expected_injection_ids = injection_ids(state.injections)
    hit_ids = sorted({item["injection_id"] for item in injection_hits})
    injection_summary = {
        "expected_ids": expected_injection_ids,
        "hit_ids": hit_ids,
        "unmatched_ids": sorted(set(expected_injection_ids) - set(hit_ids)),
        "all_hit": set(expected_injection_ids) == set(hit_ids),
    }

    retrained = [
        {
            "task_name": task.name,
            "sort_key": task.sort_key,
            "cache_key": task_keys[task.name],
        }
        for task in tasks
        if task.name in attempted_names
    ]
    not_trained = [
        {
            "task_name": task.name,
            "sort_key": task.sort_key,
            "business_code": "E_RANK_INVALID",
        }
        for task in rank_failures
    ]
    invalidated.sort(key=lambda item: (item["sort_key"], item["task_name"]))
    reused.sort(key=lambda item: (item["sort_key"], item["task_name"]))
    new_tasks.sort(key=lambda item: (item["sort_key"], item["task_name"]))
    retrained.sort(key=lambda item: (item["sort_key"], item["task_name"]))

    snapshot_summary = generation_manifest_summary(last_snapshot)
    report = {
        "schema_version": 1,
        "run_mode": mode if not fresh or mode == "full" else "incremental",
        "input_state_hash": state.input_state_hash,
        "config_hash": current_config_hash,
        "config": normalized_config,
        "base": {
            **_base_manifest(encoder),
            "changed_layers_since_prior": changed_base_layers,
        },
        "task_results": task_results,
        "adapter_summary": adapter_summary,
        "loss_trajectories": loss_trajectories,
        "validation_metrics": validation_metrics,
        "validation_summary": validation_summary,
        "threshold_failures": threshold_failures,
        "failures": failures,
        "lifecycle": {
            "invalidated": invalidated,
            "reused": reused,
            "retrained": retrained,
            "new_tasks": new_tasks,
            "removed": removed,
            "not_trained": not_trained,
            "config_migration": config_migration,
        },
        "injection_hits": injection_hits,
        "injection_summary": injection_summary,
        "snapshot": snapshot_summary,
        "recovery": {
            **recovery,
            "final_generation": last_snapshot.generation if last_snapshot else None,
            "final_version": last_snapshot.global_version if last_snapshot else None,
        },
        "execution": {
            "configured_max_concurrency": max_concurrency,
            "observed_max_inflight": observed_max_inflight,
            "snapshot_interval": interval,
        },
    }
    if report_path is not None:
        atomic_write_json(report_path, report)
    return PipelineResult(report=report, snapshot=last_snapshot, shared_forward=shared_forward)
