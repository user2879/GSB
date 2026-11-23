from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .cache import stage_cache_key
from .encoding import (
    EncodedSequence,
    encode_sequence,
    encoding_cache_entry,
    vector_arrays_from_payload,
)
from .models import (
    AttentionSample,
    PositionConfig,
    ValidationIssue,
    canonical_json,
    sha256_value,
    with_active_schemes,
)
from .scheduler import BudgetPlan, WorkTask, build_budget_plan
from .scoring import (
    ERROR_STAGES,
    NO_ERROR_STAGE,
    aggregate_error_distributions,
    aggregate_layer_statistics,
    attention_cache_entry,
    attention_result_from_payload,
    attention_scores,
    boundary_reasons,
    error_cache_entry,
    trace_errors,
)
from .snapshot import SnapshotStore


STAGES = ("loading", "encoding", "attention", "error_trace")
STAGE_FOR_CACHE = {"encoding": "encoding", "attention": "attention", "error_trace": "error_trace"}


@dataclass(frozen=True)
class PipelineRun:
    report: dict[str, Any]
    config: PositionConfig
    samples: tuple[AttentionSample, ...]
    workspace: Path
    report_path: Path | None = None

    @property
    def deterministic_report(self) -> dict[str, Any]:
        return deterministic_view(self.report)


def deterministic_view(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "run_metadata"}


def _clean_float(value: Any, decimals: int) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if type(value) is int:
        return value
    if not isinstance(value, float):
        return value
    rounded = round(float(value), decimals)
    if rounded == 0.0:
        return 0.0
    return rounded


def _rounded_list(values: Iterable[Any], decimals: int) -> list[Any]:
    return [_clean_float(value, decimals) for value in values]


def _rounded_summary(summary: Mapping[str, Any], decimals: int) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, list):
            rendered[key] = _rounded_list(value, decimals)
        elif isinstance(value, dict):
            rendered[key] = _rounded_summary(value, decimals)
        else:
            rendered[key] = _clean_float(value, decimals)
    return rendered


def _chunks(values: Sequence[WorkTask], size: int) -> Iterable[Sequence[WorkTask]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class _ConcurrencyTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.maximum = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


def _entry_is_compatible(entry: Mapping[str, Any], stage: str, sample: AttentionSample, scheme: str) -> bool:
    return (
        entry.get("stage") == stage
        and entry.get("sample_id") == sample.sample_id
        and entry.get("scheme") == scheme
    )


def _atomic_write_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output


def _snapshot_status(stop_event: threading.Event | None, complete: bool) -> str:
    if stop_event is not None and stop_event.is_set() and not complete:
        return "interrupted"
    return "complete" if complete else "running"


def _loading_payload(
    config: PositionConfig,
    config_hash: str,
    samples: Sequence[AttentionSample],
    sample_issues: Sequence[ValidationIssue],
    edit_issues: Sequence[ValidationIssue],
    budget_plan: BudgetPlan,
) -> dict[str, Any]:
    return {
        "stage": "loading",
        "schema_version": 1,
        "config_hash": config_hash,
        "config": config.as_dict(),
        "sample_fingerprints": {sample.sample_id: sample.fingerprint() for sample in samples},
        "sample_errors": [issue.as_dict() for issue in sample_issues],
        "edit_errors": [issue.as_dict() for issue in edit_issues],
        "budget_plan": budget_plan.as_dict(),
        "entries": {},
    }


def _stage_payload(
    stage: str,
    config_hash: str,
    entries: Mapping[tuple[str, str], Mapping[str, Any]],
    keys_by_identity: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    rendered_entries = {
        keys_by_identity[identity]: dict(entry)
        for identity, entry in sorted(entries.items(), key=lambda item: item[0])
    }
    return {
        "stage": stage,
        "schema_version": 1,
        "config_hash": config_hash,
        "entries": rendered_entries,
    }


def _budget_snapshot_dict(plan: BudgetPlan) -> dict[str, Any]:
    return {
        "consumed_points": plan.consumed_points,
        "consumed_mac": plan.consumed_mac,
        "remaining_points": plan.remaining_points,
        "remaining_mac": plan.remaining_mac,
        "global_point_capacity": plan.global_point_capacity,
        "global_mac_budget": plan.global_mac_budget,
    }


def _serialise_encoded_for_trace(entry: Mapping[str, Any]) -> EncodedSequence:
    query, keys = vector_arrays_from_payload(dict(entry))
    return EncodedSequence(
        scheme=str(entry["scheme"]),
        query_position=int(entry["query_position"]),
        key_positions=tuple(int(value) for value in entry["key_positions"]),
        query_vector=query,
        key_vectors=keys,
        query_effective_position=float(entry["query_effective_position"]),
        key_effective_positions=tuple(float(value) for value in entry["key_effective_positions"]),
        frequency_fingerprint=str(entry["frequency_fingerprint"]),
    )


def run_pipeline(
    config: PositionConfig,
    samples: Sequence[AttentionSample],
    workspace: str | Path,
    *,
    sample_issues: Sequence[ValidationIssue] = (),
    edit_issues: Sequence[ValidationIssue] = (),
    edit_plan_fingerprint: str | None = None,
    applied_edit_operations: Sequence[Mapping[str, Any]] = (),
    report_path: str | Path | None = None,
    stop_event: threading.Event | None = None,
    stage_complete_callback: Callable[[str], None] | None = None,
    run_id: str | None = None,
    schemes_override: Iterable[str] | None = None,
) -> PipelineRun:
    if schemes_override is not None:
        config = with_active_schemes(config, schemes_override)
    ordered_samples = tuple(sorted(samples, key=lambda item: item.sample_id))
    active_schemes = config.active_schemes
    config_hash = config.fingerprint()
    run_id = run_id or uuid.uuid4().hex
    started_monotonic = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    workspace_path = Path(workspace)
    store = SnapshotStore(workspace_path)
    store.cleanup_temporary_directories()

    previous_manifest = store.load_manifest()
    cached_entries, cache_read_errors = store.load_cached_entries()
    previous_compatible = False
    if previous_manifest is not None:
        try:
            previous_compatible = SnapshotStore.compatible(
                previous_manifest,
                global_version=config.global_version,
                config_hash=config_hash,
            )
        except (TypeError, ValueError):
            previous_compatible = False
    diagnostic_codes: set[str] = set(cache_read_errors)
    if previous_manifest is not None and not previous_compatible:
        diagnostic_codes.add("E_SNAPSHOT_STALE")

    budget_plan = build_budget_plan(config, ordered_samples)
    scheduled_tasks = budget_plan.scheduled_tasks
    keys_by_stage: dict[str, dict[tuple[str, str], str]] = {
        stage: {
            task.identity: stage_cache_key(stage, config, task.sample, task.scheme)
            for task in budget_plan.tasks
        }
        for stage in ("encoding", "attention", "error_trace")
    }

    stage_inputs: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        "encoding": {},
        "attention": {},
        "error_trace": {},
    }
    cache_stats: dict[str, dict[str, int]] = {
        stage: {"computed": 0, "reused": 0, "scheduled": len(scheduled_tasks)}
        for stage in ("encoding", "attention", "error_trace")
    }
    tracker = _ConcurrencyTracker()
    interrupted = False
    manifest: dict[str, Any] | None = None

    def commit(
        stage: str,
        payload: Mapping[str, Any],
        *,
        completed_identities: Sequence[tuple[str, str]],
        status: str,
    ) -> None:
        nonlocal manifest
        completed_samples = sorted({identity[0] for identity in completed_identities})
        completed_schemes = sorted({identity[1] for identity in completed_identities})
        manifest = store.commit_stage(
            stage,
            payload,
            global_version=config.global_version,
            config_hash=config_hash,
            completed_samples=completed_samples,
            completed_schemes=completed_schemes,
            consumed_budget=_budget_snapshot_dict(budget_plan),
            status=status,
            run_id=run_id,
            current_stage=stage,
        )

    loading_payload = _loading_payload(
        config,
        config_hash,
        ordered_samples,
        sample_issues,
        edit_issues,
        budget_plan,
    )
    all_sample_identities = [(sample.sample_id, scheme) for sample in ordered_samples for scheme in active_schemes]
    commit("loading", loading_payload, completed_identities=all_sample_identities, status="running")
    if stage_complete_callback is not None:
        stage_complete_callback("loading")
    if stop_event is not None and stop_event.is_set():
        interrupted = True

    def execute_stage(
        stage: str,
        compute: Callable[[WorkTask], dict[str, Any]],
    ) -> None:
        nonlocal interrupted
        if interrupted:
            return
        stage_entries: dict[tuple[str, str], dict[str, Any]] = {}
        cache_slot = cached_entries[stage]
        for task_batch in _chunks(scheduled_tasks, config.snapshot_commit_interval):
            misses: list[WorkTask] = []
            for task in task_batch:
                key = keys_by_stage[stage][task.identity]
                cached = cache_slot.get(key)
                if cached is not None and _entry_is_compatible(cached, stage, task.sample, task.scheme):
                    stage_entries[task.identity] = cached
                    cache_stats[stage]["reused"] += 1
                else:
                    misses.append(task)

            if misses:
                max_workers = min(config.thread_pool_concurrency, len(misses))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for task in misses:
                        def guarded(current_task: WorkTask = task) -> dict[str, Any]:
                            tracker.enter()
                            try:
                                return compute(current_task)
                            finally:
                                tracker.leave()

                        futures[executor.submit(guarded)] = task
                    computed_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
                    for future, task in sorted(futures.items(), key=lambda item: item[1].sort_key):
                        computed_by_identity[task.identity] = future.result()
                for task in misses:
                    entry = computed_by_identity[task.identity]
                    key = keys_by_stage[stage][task.identity]
                    cache_slot[key] = entry
                    stage_inputs[stage][task.identity] = entry
                    stage_entries[task.identity] = entry
                    cache_stats[stage]["computed"] += 1

            payload = _stage_payload(stage, config_hash, stage_entries, keys_by_stage[stage])
            status = _snapshot_status(stop_event, complete=False)
            commit(stage, payload, completed_identities=tuple(stage_entries), status=status)

        stage_inputs[stage] = stage_entries
        if stage_complete_callback is not None:
            stage_complete_callback(stage)
        if stop_event is not None and stop_event.is_set():
            interrupted = True

    def compute_encoding(task: WorkTask) -> dict[str, Any]:
        encoded = encode_sequence(config, task.sample, task.scheme)
        return encoding_cache_entry(config, task.sample, task.scheme, encoded)

    def compute_attention(task: WorkTask) -> dict[str, Any]:
        encoding_entry = stage_inputs["encoding"][task.identity]
        query, keys = vector_arrays_from_payload(encoding_entry)
        result = attention_scores(
            query,
            keys,
            int(encoding_entry["query_position"]),
            [int(value) for value in encoding_entry["key_positions"]],
            config.training_context_length,
        )
        return attention_cache_entry(config, task.sample, task.scheme, result)

    def compute_error_trace(task: WorkTask) -> dict[str, Any]:
        encoding_entry = stage_inputs["encoding"][task.identity]
        attention_entry = stage_inputs["attention"][task.identity]
        encoded = _serialise_encoded_for_trace(encoding_entry)
        attention_result = attention_result_from_payload(attention_entry)
        trace = trace_errors(config, task.sample, task.scheme, encoded, attention_result)
        return error_cache_entry(config, task.sample, task.scheme, trace)

    execute_stage("encoding", compute_encoding)
    execute_stage("attention", compute_attention)
    execute_stage("error_trace", compute_error_trace)

    complete = not interrupted
    final_stage = "error_trace" if stage_inputs["error_trace"] else (
        "attention" if stage_inputs["attention"] else ("encoding" if stage_inputs["encoding"] else "loading")
    )
    final_payload = _stage_payload(
        final_stage,
        config_hash,
        stage_inputs[final_stage] if final_stage in stage_inputs else {},
        keys_by_stage.get(final_stage, {}),
    )
    final_identities: Sequence[tuple[str, str]] = tuple(stage_inputs[final_stage]) if final_stage in stage_inputs else all_sample_identities
    commit(
        final_stage,
        final_payload,
        completed_identities=final_identities,
        status="complete" if complete else "interrupted",
    )

    report = _build_report(
        config,
        config_hash,
        ordered_samples,
        sample_issues,
        edit_issues,
        budget_plan,
        keys_by_stage,
        stage_inputs,
        cache_stats,
        tracker.maximum,
        interrupted,
        run_id=run_id,
        started_at=started_at,
        elapsed_ms=(time.perf_counter() - started_monotonic) * 1000.0,
        workspace=workspace_path,
        resumed=previous_compatible,
        stale=previous_manifest is not None and not previous_compatible,
        diagnostic_codes=sorted(diagnostic_codes),
        edit_plan_fingerprint=edit_plan_fingerprint,
        applied_edit_operations=applied_edit_operations,
        snapshot=store.snapshot_view(),
    )
    output_path: Path | None = None
    if report_path is not None:
        output_path = _atomic_write_report(report_path, report)
    return PipelineRun(
        report=report,
        config=config,
        samples=ordered_samples,
        workspace=workspace_path,
        report_path=output_path,
    )


def _build_report(
    config: PositionConfig,
    config_hash: str,
    samples: Sequence[AttentionSample],
    sample_issues: Sequence[ValidationIssue],
    edit_issues: Sequence[ValidationIssue],
    budget_plan: BudgetPlan,
    keys_by_stage: Mapping[str, Mapping[tuple[str, str], str]],
    stage_inputs: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
    cache_stats: Mapping[str, Mapping[str, int]],
    observed_max_concurrency: int,
    interrupted: bool,
    *,
    run_id: str,
    started_at: str,
    elapsed_ms: float,
    workspace: Path,
    resumed: bool,
    stale: bool,
    diagnostic_codes: Sequence[str],
    edit_plan_fingerprint: str | None,
    applied_edit_operations: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    decimals = config.precision_decimals
    tasks_by_identity = {task.identity: task for task in budget_plan.tasks}
    sample_results: list[dict[str, Any]] = []
    attention_entries: list[dict[str, Any]] = []
    trace_entries: list[dict[str, Any]] = []
    over_tolerance: list[dict[str, Any]] = []
    successful_tasks = 0

    for sample in samples:
        scheme_results: list[dict[str, Any]] = []
        reasons = boundary_reasons(config, sample)
        for scheme in config.active_schemes:
            identity = (sample.sample_id, scheme)
            task = tasks_by_identity[identity]
            stage_cache_keys = {
                stage: keys_by_stage[stage][identity]
                for stage in ("encoding", "attention", "error_trace")
            }
            result: dict[str, Any] = {
                "scheme": scheme,
                "status": task.status,
                "error_code": task.error_code,
                "cache_keys": stage_cache_keys,
                "references": {
                    stage: {
                        "sample_id": sample.sample_id,
                        "scheme": scheme,
                        "stage": stage,
                        "sample_fingerprint": sample.fingerprint(),
                    }
                    for stage in ("encoding", "attention", "error_trace")
                },
            }
            if task.scheduled:
                encoding_entry = stage_inputs["encoding"].get(identity)
                attention_entry = stage_inputs["attention"].get(identity)
                trace_entry = stage_inputs["error_trace"].get(identity)
                if encoding_entry is not None and attention_entry is not None and trace_entry is not None:
                    successful_tasks += 1
                    result["status"] = "ok"
                    result["position_summary"] = {
                        "digest": encoding_entry["position_digest"],
                        "query_position": int(encoding_entry["query_position"]),
                        "key_positions": list(encoding_entry["key_positions"]),
                        "query_effective_position": _clean_float(
                            encoding_entry["query_effective_position"], decimals
                        ),
                        "key_effective_positions": _rounded_list(
                            encoding_entry["key_effective_positions"], decimals
                        ),
                        "query_norm": _clean_float(encoding_entry["query_norm"], decimals),
                        "key_norms": _rounded_list(encoding_entry["key_norms"], decimals),
                        "frequency_fingerprint": encoding_entry["frequency_fingerprint"],
                        "rotation_pair_count": int(encoding_entry["rotation_pair_count"]),
                    }
                    result["attention"] = {
                        "scores": _rounded_list(attention_entry["scores"], decimals),
                        "normalized_weights": _rounded_list(attention_entry["weights"], decimals),
                        "distances": [int(value) for value in attention_entry["distances"]],
                        "buckets": list(attention_entry["buckets"]),
                        "tier_summary": _rounded_summary(attention_entry["tier_summary"], decimals),
                    }
                    trace_deviations = _rounded_summary(trace_entry["deviations"], decimals)
                    result["error_trace"] = {
                        "deviations": trace_deviations,
                        "first_exceeded": trace_entry["first_exceeded"],
                        "reference_query_position": int(trace_entry["reference_query_position"]),
                        "reference_key_positions": [int(value) for value in trace_entry["reference_key_positions"]],
                        "external_key_count": int(trace_entry["external_key_count"]),
                        "reference_rule": trace_entry["reference_rule"],
                    }
                    attention_entries.append(dict(attention_entry))
                    trace_entries.append(dict(trace_entry))
                    if trace_entry["first_exceeded"] != NO_ERROR_STAGE:
                        over_tolerance.append(
                            {
                                "sample_id": sample.sample_id,
                                "scheme": scheme,
                                "first_exceeded": trace_entry["first_exceeded"],
                                "deviations": trace_deviations,
                                "tolerance": _clean_float(config.tolerance, decimals),
                            }
                        )
                else:
                    result["status"] = "incomplete"
            scheme_results.append(result)

        sample_results.append(
            {
                "sample_id": sample.sample_id,
                "source_line": sample.source_line,
                "content_fingerprint": sample.fingerprint(),
                "query_position": sample.query_position,
                "key_positions": list(sample.key_positions),
                "key_count": len(sample.key_positions),
                "point_units": sample.point_units,
                "flags": {
                    "empty_keys": sample.empty_keys,
                    "self_attention": sample.self_attention,
                    "boundary": bool(reasons),
                    "boundary_reasons": reasons,
                },
                "schemes": scheme_results,
            }
        )

    boundary_samples = [
        {
            "sample_id": result["sample_id"],
            "reasons": result["flags"]["boundary_reasons"],
        }
        for result in sample_results
        if result["flags"]["boundary"]
    ]
    layer_statistics = aggregate_layer_statistics(attention_entries)
    error_distributions = aggregate_error_distributions(trace_entries)
    over_tolerance.sort(key=lambda item: (item["sample_id"], item["scheme"]))
    boundary_samples.sort(key=lambda item: item["sample_id"])
    budget_usage = budget_plan.as_dict()
    validation = {
        "sample_errors": [issue.as_dict() for issue in sample_issues],
        "edit_errors": [issue.as_dict() for issue in edit_issues],
    }
    summary = {
        "valid_sample_count": len(samples),
        "invalid_sample_count": len(sample_issues),
        "task_count": len(budget_plan.tasks),
        "successful_task_count": successful_tasks,
        "budget_exhausted_task_count": len(budget_plan.exhausted_tasks),
        "pending_task_count": len(budget_plan.pending_queue),
        "over_tolerance_count": len(over_tolerance),
        "boundary_sample_count": len(boundary_samples),
        "active_schemes": list(config.active_schemes),
    }
    status = "interrupted" if interrupted else "complete"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "config": config.as_dict(),
        "config_hash": config_hash,
        "summary": summary,
        "samples": sample_results,
        "layer_statistics": layer_statistics,
        "error_distributions": error_distributions,
        "over_tolerance_samples": over_tolerance,
        "boundary_samples": boundary_samples,
        "budget_usage": budget_usage,
        "validation": validation,
        "run_metadata": {
            "run_id": run_id,
            "started_at_utc": started_at,
            "elapsed_ms": round(elapsed_ms, 3),
            "workspace": str(workspace),
            "resumed": resumed,
            "stale": stale,
            "diagnostic_codes": list(diagnostic_codes),
            "cache": {stage: dict(stats) for stage, stats in sorted(cache_stats.items())},
            "execution": {
                "configured_max_concurrency": config.thread_pool_concurrency,
                "observed_max_concurrency": observed_max_concurrency,
            },
            "edit_plan": {
                "fingerprint": edit_plan_fingerprint,
                "applied_operations": [dict(operation) for operation in applied_edit_operations],
            },
            "snapshot": dict(snapshot),
        },
    }
    # Validate canonicalisability before returning so a report can never be written with NaN/inf.
    canonical_json(deterministic_view(report))
    return report


def build_failure_report(
    issues: Sequence[ValidationIssue],
    *,
    workspace: str | Path,
    run_id: str | None = None,
    report_path: str | Path | None = None,
    extra_diagnostic_codes: Sequence[str] = (),
) -> None:
    run_id = run_id or uuid.uuid4().hex
    codes = sorted({issue.code for issue in issues} | set(extra_diagnostic_codes))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed",
        "config": None,
        "config_hash": None,
        "summary": {
            "valid_sample_count": 0,
            "invalid_sample_count": 0,
            "task_count": 0,
            "successful_task_count": 0,
            "budget_exhausted_task_count": 0,
            "pending_task_count": 0,
            "over_tolerance_count": 0,
            "boundary_sample_count": 0,
            "active_schemes": [],
        },
        "samples": [],
        "layer_statistics": {},
        "error_distributions": {},
        "over_tolerance_samples": [],
        "boundary_samples": [],
        "budget_usage": {},
        "validation": {
            "config_errors": [issue.as_dict() for issue in issues],
            "sample_errors": [],
            "edit_errors": [],
        },
        "run_metadata": {
            "run_id": run_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": 0.0,
            "workspace": str(workspace),
            "resumed": False,
            "stale": False,
            "diagnostic_codes": codes,
            "cache": {},
            "execution": {},
            "edit_plan": {},
            "snapshot": {},
        },
    }
    if report_path is not None:
        _atomic_write_report(report_path, report)
