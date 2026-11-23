from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .attention import ScoreResult, score_attention
from .cache import ResultCache
from .config import DiagnosticConfig, load_config
from .errors import DiagnosticError, ErrorCode, Failure
from .fingerprints import (
    position_sample_key,
    scheme_parameter_fingerprint,
    scoring_sample_key,
    sensitivity_key,
    stage_key_map,
    trace_key,
)
from .injections import Injection, load_injections
from .models import EncodedSample
from .pruning import load_pruning_policies, policies_public
from .rope import (
    apply_input_precision_and_quantization,
    apply_rotation,
    apply_rotation_batch,
    generate_position_encoding,
)
from .samples import SampleDataset, load_samples
from .scheduler import SampleTask, ScheduleResult, schedule_sample_tasks
from .sensitivity import scan_scheme_sensitivity
from .snapshot import SnapshotManager, SnapshotState
from .statistics import compute_statistics
from .strategy import select_strategy
from .trace import compute_error_trace
from .util import atomic_write_json, sha256_json, sorted_failures


SCHEMA_VERSION = "1.0"


@dataclass
class TaskRun:
    task: SampleTask
    encoded: EncodedSample | None = None
    score_result: ScoreResult | None = None
    failure: Failure | None = None
    injection_records: list[dict[str, Any]] = field(default_factory=list)
    computed_position: bool = False
    computed_scoring: bool = False
    position_cache_payload: dict[str, Any] | None = None
    scoring_cache_payload: dict[str, Any] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failure_record(
    injection: Injection,
    *,
    scheme: str | None,
    sample_id: str | None,
) -> tuple[Failure, dict[str, Any]]:
    failure = Failure(
        code=ErrorCode.INJECTED_FAULT,
        message=injection.message,
        sample_id=sample_id or "",
        stage=injection.stage,
        injection_id=injection.injection_id,
        details={
            "scheme": scheme or "*",
            "action": injection.action,
            "injection_stage": injection.stage,
        },
    )
    record = {
        "injection_id": injection.injection_id,
        "scheme": scheme or "*",
        "sample_id": sample_id or "*",
        "stage": injection.stage,
        "action": injection.action,
        "message": injection.message,
        "status": "fault_captured",
    }
    return failure, record


def _matching_injections(
    injections: Sequence[Injection],
    *,
    scheme: str | None,
    sample_id: str | None,
    stage: str,
) -> list[Injection]:
    return [
        injection
        for injection in injections
        if injection.matches(scheme=scheme, sample_id=sample_id, stage=stage)
    ]


class DiagnosticRunner:
    def __init__(
        self,
        *,
        config_path: Path,
        samples_path: Path,
        pruning_path: Path,
        injections_path: Path,
        workspace: Path,
        output_path: Path | None = None,
        force_full: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        self.samples_path = Path(samples_path)
        self.pruning_path = Path(pruning_path)
        self.injections_path = Path(injections_path)
        self.workspace = Path(workspace).resolve()
        if output_path is None:
            self.output_path = self.workspace / "artifacts" / "latest" / "report.json"
        else:
            candidate = Path(output_path)
            self.output_path = candidate if candidate.is_absolute() else self.workspace / candidate
        self.force_full = bool(force_full)
        self.snapshot_manager = SnapshotManager(self.workspace)

    def run(self) -> dict[str, Any]:
        started_at = _now_iso()
        start_time = time.perf_counter()
        try:
            report = self._run_inner()
        except Exception as exc:  # pragma: no cover - defensive audit boundary
            report = self._abort_report(
                config=None,
                failures=[
                    Failure(
                        code=ErrorCode.INTERNAL_ERROR,
                        message=f"未处理内部异常: {exc}",
                        stage="pipeline",
                        details={"traceback": traceback.format_exc(limit=8)},
                    )
                ],
                config_hash="",
                injection_hash="",
                snapshot_state=SnapshotState(),
            )
        report["run_metadata"] = {
            "started_at_utc": started_at,
            "finished_at_utc": _now_iso(),
            "duration_ms": round((time.perf_counter() - start_time) * 1000.0, 3),
            "mode": "full_rebuild" if self.force_full else "incremental",
            "workspace": self.workspace.as_posix(),
            "output_path": self.output_path.resolve().as_posix(),
        }
        atomic_write_json(self.output_path, report)
        return report

    def _run_inner(self) -> dict[str, Any]:
        snapshot_state = self.snapshot_manager.load()
        cache = ResultCache() if self.force_full else snapshot_state.cache
        startup_failures: list[Failure] = []
        if snapshot_state.cache_failure is not None and not self.force_full:
            startup_failures.append(snapshot_state.cache_failure)

        config, config_failures, config_hash = load_config(self.config_path)
        if config is None:
            failures = startup_failures + config_failures
            return self._abort_report(
                config=None,
                failures=failures,
                config_hash=config_hash,
                injection_hash="",
                snapshot_state=snapshot_state,
            )

        policies, pruning_failures, pruning_hash = load_pruning_policies(self.pruning_path)
        injections, injection_failures, injection_hash = load_injections(self.injections_path)
        setup_failures = startup_failures + pruning_failures + injection_failures
        if setup_failures:
            return self._abort_report(
                config=config,
                failures=setup_failures,
                config_hash=config_hash,
                injection_hash=injection_hash,
                snapshot_state=snapshot_state,
                pruning_hash=pruning_hash,
            )

        dataset = load_samples(self.samples_path, dimension=config.dimension)
        validation_failures = list(dataset.failures)
        if dataset.sample_count == 0:
            return self._abort_report(
                config=config,
                failures=validation_failures,
                config_hash=config_hash,
                injection_hash=injection_hash,
                snapshot_state=snapshot_state,
                dataset=dataset,
                pruning_hash=pruning_hash,
                config_valid=True,
            )

        try:
            strategy = select_strategy(
                config,
                sample_count=dataset.sample_count,
                total_points=dataset.total_points,
                policies=policies,
            )
        except DiagnosticError as exc:
            return self._abort_report(
                config=config,
                failures=validation_failures + [exc.as_failure()],
                config_hash=config_hash,
                injection_hash=injection_hash,
                snapshot_state=snapshot_state,
                dataset=dataset,
                pruning_hash=pruning_hash,
            )

        dimension_indices = tuple(int(value) for value in strategy["dimension_indices"])
        chunk_size = int(strategy["scoring_chunk_size"])
        pruning_fingerprint = strategy["pruning_fingerprint"]
        strategy_fp = strategy["strategy_fingerprint"]
        selected_schemes = list(strategy["scheme_priority"])

        previous_manifest = snapshot_state.manifest if not self.force_full else {}
        current_stage_keys: dict[str, str] = {}
        tasks: list[SampleTask] = []
        for scheme in selected_schemes:
            position_keys: list[str] = []
            scoring_keys: list[str] = []
            for sample in sorted(dataset.samples, key=lambda item: item.sample_id):
                position_key = position_sample_key(
                    scheme, sample.content_fingerprint, config, pruning_fingerprint
                )
                scoring_key = scoring_sample_key(
                    position_key, sample.content_fingerprint, config, strategy_fp
                )
                position_keys.append(position_key)
                scoring_keys.append(scoring_key)
                injection_force = bool(
                    _matching_injections(
                        injections, scheme=scheme, sample_id=sample.sample_id, stage="position"
                    )
                    or _matching_injections(
                        injections, scheme=scheme, sample_id=sample.sample_id, stage="scoring"
                    )
                )
                tasks.append(
                    SampleTask(
                        scheme=scheme,
                        sample=sample,
                        position_key=position_key,
                        scoring_key=scoring_key,
                        position_hit=(not self.force_full) and cache.contains(position_key),
                        scoring_hit=(not self.force_full) and cache.contains(scoring_key),
                        force_execution=injection_force,
                        sort_key=f"{scheme}|{sample.sample_id}|sample_diagnostic",
                        stage_sort_keys={
                            "position": f"{scheme}|{sample.sample_id}|position",
                            "scoring": f"{scheme}|{sample.sample_id}|scoring",
                        },
                    )
                )
            stage_trace_key = trace_key(
                scheme, dataset.dataset_fingerprint, config, pruning_fingerprint, strategy_fp
            )
            stage_sensitivity_key = sensitivity_key(
                scheme, dataset.dataset_fingerprint, config, pruning_fingerprint, strategy_fp
            )
            current_stage_keys.update(
                stage_key_map(
                    scheme,
                    position_keys=position_keys,
                    scoring_keys=scoring_keys,
                    trace=stage_trace_key,
                    sensitivity=stage_sensitivity_key,
                )
            )

        resume_compatible = bool(
            previous_manifest
            and previous_manifest.get("global_version") == config.global_version
            and previous_manifest.get("config_hash") == config_hash
            and previous_manifest.get("dataset_fingerprint") == dataset.dataset_fingerprint
            and previous_manifest.get("injection_hash") == injection_hash
        )
        previously_completed: set[str] = set()
        if resume_compatible and not self.force_full:
            previously_completed = {
                str(value) for value in previous_manifest.get("completed_tasks", [])
            }

        schedule = schedule_sample_tasks(
            tasks,
            budget=config.point_budget,
            previously_completed=previously_completed,
        )

        completed_tasks: set[str] = set()
        task_runs: dict[str, TaskRun] = {}
        encoded_by_scheme: dict[str, dict[str, EncodedSample]] = {
            scheme: {} for scheme in selected_schemes
        }
        scores_by_scheme: dict[str, dict[str, ScoreResult]] = {
            scheme: {} for scheme in selected_schemes
        }
        failures: list[Failure] = list(validation_failures) + list(schedule.pending_failures)
        triggered_injections: list[dict[str, Any]] = []
        commit_seq = int(previous_manifest.get("commit_seq", 0) or 0)

        def make_manifest() -> dict[str, Any]:
            completed_schemes = []
            completed_samples_by_scheme: dict[str, list[str]] = {}
            for scheme in selected_schemes:
                scheme_sort_keys = {task.sort_key for task in tasks if task.scheme == scheme}
                if scheme_sort_keys and scheme_sort_keys.issubset(completed_tasks):
                    completed_schemes.append(scheme)
                completed_samples_by_scheme[scheme] = sorted(
                    task.sample.sample_id
                    for task in tasks
                    if task.scheme == scheme and task.sort_key in completed_tasks
                )
            uncompleted = sorted(
                task.sort_key for task in tasks if task.sort_key not in completed_tasks
            )
            return {
                "schema": "rope-diag-snapshot-v1",
                "global_version": config.global_version,
                "config_hash": config_hash,
                "injection_hash": injection_hash,
                "pruning_hash": pruning_hash,
                "dataset_fingerprint": dataset.dataset_fingerprint,
                "strategy_fingerprint": strategy_fp,
                "completed_tasks": sorted(completed_tasks),
                "completed_schemes": sorted(completed_schemes),
                "completed_samples_by_scheme": completed_samples_by_scheme,
                "pending_or_failed_tasks": uncompleted,
                "stage_keys": current_stage_keys,
                "snapshot_interval": config.snapshot_interval,
                "commit_seq": commit_seq + 1,
            }

        def commit_snapshot() -> None:
            nonlocal commit_seq
            manifest = make_manifest()
            committed = self.snapshot_manager.commit(manifest, cache)
            commit_seq = int(committed.get("commit_seq", commit_seq + 1))

        def ingest_run(run: TaskRun) -> None:
            task = run.task
            task_runs[task.sort_key] = run
            if run.injection_records:
                triggered_injections.extend(run.injection_records)
            if run.failure is not None:
                failures.append(run.failure)
                return
            if run.computed_position and run.position_cache_payload is not None:
                cache.put(task.position_key, run.position_cache_payload)
            if run.computed_scoring and run.scoring_cache_payload is not None:
                cache.put(task.scoring_key, run.scoring_cache_payload)
            if run.encoded is None or run.score_result is None:
                failures.append(
                    Failure(
                        code=ErrorCode.INTERNAL_ERROR,
                        message="任务成功但缺少位置编码或打分结果",
                        sample_id=task.sample.sample_id,
                        stage="pipeline",
                        details={"scheme": task.scheme, "sort_key": task.sort_key},
                    )
                )
                return
            encoded_by_scheme[task.scheme][task.sample.sample_id] = run.encoded
            scores_by_scheme[task.scheme][task.sample.sample_id] = run.score_result
            completed_tasks.add(task.sort_key)

        # Cache-only tasks do not consume the point budget.
        for task in schedule.reused:
            position_payload = cache.get(task.position_key)
            scoring_payload = cache.get(task.scoring_key)
            try:
                if position_payload is None or scoring_payload is None:
                    raise ValueError("cache entry vanished")
                encoded = EncodedSample.from_dict(position_payload["payload"])
                score_result = ScoreResult.from_mapping(scoring_payload["payload"])
                ingest_run(
                    TaskRun(
                        task=task,
                        encoded=encoded,
                        score_result=score_result,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(
                    Failure(
                        code=ErrorCode.CACHE_CORRUPT,
                        message=f"缓存结果无法解码: {exc}",
                        sample_id=task.sample.sample_id,
                        stage="cache",
                        details={"scheme": task.scheme, "sort_key": task.sort_key},
                    )
                )

        task_inputs: dict[str, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
        for task in schedule.approved:
            task_inputs[task.sort_key] = (
                cache.get(task.position_key) if task.position_hit else None,
                cache.get(task.scoring_key) if task.scoring_hit else None,
            )

        def execute_task(task: SampleTask) -> TaskRun:
            cached_position, cached_scoring = task_inputs[task.sort_key]
            return self._execute_task(
                task,
                config=config,
                injections=injections,
                dimension_indices=dimension_indices,
                chunk_size=chunk_size,
                cached_position=cached_position,
                cached_scoring=cached_scoring,
            )

        completed_since_commit = 0
        if schedule.approved:
            with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
                for run in executor.map(execute_task, list(schedule.approved)):
                    ingest_run(run)
                    completed_since_commit += 1
                    if completed_since_commit >= config.snapshot_interval:
                        commit_snapshot()
                        completed_since_commit = 0

        scheme_payloads: dict[str, Any] = {}
        aggregate_stage_status: dict[tuple[str, str], dict[str, Any]] = {}
        for scheme in selected_schemes:
            scheme_tasks = [task for task in tasks if task.scheme == scheme]
            scheme_runs = [task_runs.get(task.sort_key) for task in scheme_tasks]
            scheme_failures = [
                failure
                for failure in failures
                if failure.details.get("scheme") == scheme
                and failure.code != ErrorCode.POINT_BUDGET_PENDING
            ]

            stats_injections = _matching_injections(
                injections, scheme=scheme, sample_id=None, stage="statistics"
            )
            if stats_injections:
                statistics_payload: dict[str, Any] = {
                    "status": ErrorCode.INJECTED_FAULT,
                    "error": {
                        "code": ErrorCode.INJECTED_FAULT,
                        "message": "统计阶段命中异常注入",
                    },
                }
                for injection in stats_injections:
                    failure, record = _failure_record(
                        injection, scheme=scheme, sample_id=None
                    )
                    scheme_failures.append(failure)
                    failures.append(failure)
                    triggered_injections.append(record)
            else:
                statistics_payload = compute_statistics(
                    list(dataset.samples), scores_by_scheme[scheme], config
                )
                if statistics_payload["status"] == ErrorCode.NO_EXTRAPOLATION_SAMPLE:
                    no_ext_failure = Failure(
                        code=ErrorCode.NO_EXTRAPOLATION_SAMPLE,
                        message="外推区间内样本数为零，已终止该方案统计汇总",
                        sample_id="",
                        stage="statistics",
                        details={"scheme": scheme},
                    )
                    scheme_failures.append(no_ext_failure)
                    failures.append(no_ext_failure)

            aggregate_status: dict[str, dict[str, Any]] = {}
            trace_payload, trace_status = self._aggregate_trace(
                scheme=scheme,
                dataset=dataset,
                encoded_samples=encoded_by_scheme[scheme],
                score_results=scores_by_scheme[scheme],
                config=config,
                dimension_indices=dimension_indices,
                chunk_size=chunk_size,
                cache=cache,
                key=current_stage_keys[f"{scheme}:trace"],
                injections=injections,
            )
            aggregate_status["trace"] = trace_status
            for record in trace_status.pop("injection_records", []):
                triggered_injections.append(record)
            for failure in trace_status.pop("injection_failures", []):
                scheme_failures.append(failure)
                failures.append(failure)

            sensitivity_payload, sensitivity_status = self._aggregate_sensitivity(
                scheme=scheme,
                dataset=dataset,
                score_results=scores_by_scheme[scheme],
                config=config,
                dimension_indices=dimension_indices,
                chunk_size=chunk_size,
                cache=cache,
                key=current_stage_keys[f"{scheme}:sensitivity"],
                injections=injections,
            )
            aggregate_status["sensitivity"] = sensitivity_status
            for record in sensitivity_status.pop("injection_records", []):
                triggered_injections.append(record)
            for failure in sensitivity_status.pop("injection_failures", []):
                scheme_failures.append(failure)
                failures.append(failure)

            pending_count = sum(1 for task in scheme_tasks if task in schedule.pending)
            failed_sample_count = sum(
                1
                for run in scheme_runs
                if run is not None and run.failure is not None
            )
            if pending_count == len(scheme_tasks):
                status = "pending"
            elif pending_count > 0:
                status = "partial_pending"
            elif statistics_payload["status"] == ErrorCode.NO_EXTRAPOLATION_SAMPLE:
                status = "statistics_error"
            elif scheme_failures or failed_sample_count:
                status = "partial_fault"
            else:
                status = "ok"

            scheme_payloads[scheme] = {
                "scheme": scheme,
                "status": status,
                "parameters": {
                    "parameter_fingerprint": scheme_parameter_fingerprint(scheme, config),
                    "base": config.base,
                    "scaling_factor": config.scaling_factor,
                    "training_context_length": config.training_context_length,
                    "extrapolation_target_length": config.extrapolation_target_length,
                    "dimension": config.dimension,
                },
                "sample_summary": {
                    "planned": len(scheme_tasks),
                    "completed": sum(1 for task in scheme_tasks if task.sort_key in completed_tasks),
                    "pending": pending_count,
                    "failed": failed_sample_count,
                },
                "statistics": statistics_payload,
                "trace": trace_payload,
                "sensitivity": sensitivity_payload,
                "errors": [
                    failure.to_dict()
                    for failure in sorted(
                        {
                            (failure.code, failure.message, failure.sample_id, failure.stage): failure
                            for failure in scheme_failures
                        }.values(),
                        key=lambda item: item.sort_key,
                    )
                ],
            }
            aggregate_stage_status[(scheme, "trace")] = aggregate_status["trace"]
            aggregate_stage_status[(scheme, "sensitivity")] = aggregate_status["sensitivity"]

        cache_payload = self._build_cache_report(
            selected_schemes=selected_schemes,
            tasks=tasks,
            schedule=schedule,
            task_runs=task_runs,
            current_stage_keys=current_stage_keys,
            previous_manifest=previous_manifest,
            aggregate_stage_status=aggregate_stage_status,
        )

        comparison = self._build_comparison(selected_schemes, scheme_payloads)
        final_manifest: dict[str, Any] | None = None
        if not self.force_full or cache.entries:
            final_manifest = self.snapshot_manager.commit(make_manifest(), cache)

        report = {
            "schema_version": SCHEMA_VERSION,
            "inputs": {
                "config_path": self.config_path.resolve().as_posix(),
                "config_hash": config_hash,
                "samples_path": self.samples_path.resolve().as_posix(),
                "samples_file_sha256": dataset.file_sha256,
                "dataset_fingerprint": dataset.dataset_fingerprint,
                "pruning_path": self.pruning_path.resolve().as_posix(),
                "pruning_hash": pruning_hash,
                "injections_path": self.injections_path.resolve().as_posix(),
                "injection_hash": injection_hash,
            },
            "validation": {
                "config_valid": True,
                "sample_valid_count": dataset.sample_count,
                "sample_invalid_count": len(dataset.failures),
                "sample_failures": sorted_failures(list(dataset.failures)),
                "failures": sorted_failures(failures),
            },
            "strategy": strategy,
            "pruning_policies": policies_public(policies, config.dimension),
            "scheduler": schedule.public_dict(),
            "schemes": scheme_payloads,
            "comparison": comparison,
            "cache": cache_payload,
            "injections": {
                "injection_hash": injection_hash,
                "configured": [
                    {
                        "injection_id": injection.injection_id,
                        "enabled": injection.enabled,
                        "scheme": injection.scheme,
                        "sample_id": injection.sample_id,
                        "stage": injection.stage,
                        "action": injection.action,
                    }
                    for injection in injections
                ],
                "triggered": sorted(
                    triggered_injections,
                    key=lambda item: (
                        item["injection_id"],
                        item["scheme"],
                        item["sample_id"],
                        item["stage"],
                    ),
                ),
            },
            "snapshot": {
                "resume_compatible": resume_compatible,
                "resumed_completed_task_count": len(previously_completed),
                "commit_seq": commit_seq,
                "manifest": final_manifest,
            },
            "failures": sorted_failures(failures),
        }
        report["semantic_digest"] = sha256_json(
            {
                "strategy_fingerprint": strategy_fp,
                "schemes": scheme_payloads,
                "comparison": comparison,
                "sample_failures": sorted_failures(list(dataset.failures)),
            }
        )
        return report

    def _execute_task(
        self,
        task: SampleTask,
        *,
        config: DiagnosticConfig,
        injections: Sequence[Injection],
        dimension_indices: Sequence[int],
        chunk_size: int,
        cached_position: dict[str, Any] | None,
        cached_scoring: dict[str, Any] | None,
    ) -> TaskRun:
        position_injections = _matching_injections(
            injections, scheme=task.scheme, sample_id=task.sample.sample_id, stage="position"
        )
        scoring_injections = _matching_injections(
            injections, scheme=task.scheme, sample_id=task.sample.sample_id, stage="scoring"
        )
        if position_injections or scoring_injections:
            selected = position_injections or scoring_injections
            first = selected[0]
            failure, record = _failure_record(
                first, scheme=task.scheme, sample_id=task.sample.sample_id
            )
            records = [record]
            for injection in selected[1:]:
                _, extra = _failure_record(
                    injection, scheme=task.scheme, sample_id=task.sample.sample_id
                )
                records.append(extra)
            return TaskRun(task=task, failure=failure, injection_records=records)

        try:
            if cached_position is not None:
                encoded = EncodedSample.from_dict(cached_position["payload"])
                computed_position = False
                new_position_payload = None
            else:
                query_prepared = apply_input_precision_and_quantization(
                    task.sample.query_features, config
                )
                keys_prepared = np.vstack(
                    [
                        apply_input_precision_and_quantization(row, config)
                        for row in task.sample.key_features
                    ]
                )
                query_encoding = generate_position_encoding(
                    task.scheme, task.sample.query_position, config
                )
                key_encodings = tuple(
                    generate_position_encoding(task.scheme, position, config)
                    for position in task.sample.key_positions
                )
                encoded = EncodedSample(
                    scheme=task.scheme,
                    sample_id=task.sample.sample_id,
                    query_encoding=query_encoding,
                    key_encodings=key_encodings,
                    query_rotated=apply_rotation(query_prepared, query_encoding),
                    key_rotated=apply_rotation_batch(keys_prepared, key_encodings),
                )
                computed_position = True
                new_position_payload = {
                    "kind": "position",
                    "scheme": task.scheme,
                    "sample_id": task.sample.sample_id,
                    "payload": encoded.to_dict(),
                }

            if cached_scoring is not None:
                score_result = ScoreResult.from_mapping(cached_scoring["payload"])
                computed_scoring = False
                new_scoring_payload = None
            else:
                score_result = score_attention(
                    encoded.query_rotated,
                    encoded.key_rotated,
                    dimension_indices,
                    chunk_size,
                )
                computed_scoring = True
                new_scoring_payload = {
                    "kind": "scoring",
                    "scheme": task.scheme,
                    "sample_id": task.sample.sample_id,
                    "payload": score_result.to_dict(),
                }
            return TaskRun(
                task=task,
                encoded=encoded,
                score_result=score_result,
                computed_position=computed_position,
                computed_scoring=computed_scoring,
                position_cache_payload=new_position_payload,
                scoring_cache_payload=new_scoring_payload,
            )
        except DiagnosticError as exc:
            return TaskRun(task=task, failure=exc.as_failure())
        except (KeyError, TypeError, ValueError) as exc:
            return TaskRun(
                task=task,
                failure=Failure(
                    code=ErrorCode.CACHE_CORRUPT,
                    message=f"缓存或任务数据解码失败: {exc}",
                    sample_id=task.sample.sample_id,
                    stage="scoring",
                    details={"scheme": task.scheme},
                ),
            )

    def _aggregate_trace(
        self,
        *,
        scheme: str,
        dataset: SampleDataset,
        encoded_samples: Mapping[str, EncodedSample],
        score_results: Mapping[str, ScoreResult],
        config: DiagnosticConfig,
        dimension_indices: Sequence[int],
        chunk_size: int,
        cache: ResultCache,
        key: str,
        injections: Sequence[Injection],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        injection_matches = _matching_injections(
            injections, scheme=scheme, sample_id=None, stage="trace"
        )
        if injection_matches:
            failures = []
            records = []
            for injection in injection_matches:
                failure, record = _failure_record(injection, scheme=scheme, sample_id=None)
                failures.append(failure)
                records.append(record)
            return (
                {"status": ErrorCode.INJECTED_FAULT, "samples": [], "stages": []},
                {
                    "status": "injected_fault",
                    "cache_key": key,
                    "injection_failures": failures,
                    "injection_records": records,
                },
            )
        if not encoded_samples or not score_results:
            return (
                {"status": "not_run", "samples": [], "stages": []},
                {"status": "not_run", "cache_key": key},
            )
        if not self.force_full and cache.contains(key):
            payload = cache.get(key)
            if payload is not None and isinstance(payload.get("payload"), dict):
                return payload["payload"], {"status": "reused", "cache_key": key}
        payload = compute_error_trace(
            scheme,
            list(dataset.samples),
            encoded_samples,
            score_results,
            config,
            dimension_indices,
            chunk_size,
        )
        cache.put(key, {"kind": "trace", "scheme": scheme, "payload": payload})
        return payload, {"status": "recomputed", "cache_key": key}

    def _aggregate_sensitivity(
        self,
        *,
        scheme: str,
        dataset: SampleDataset,
        score_results: Mapping[str, ScoreResult],
        config: DiagnosticConfig,
        dimension_indices: Sequence[int],
        chunk_size: int,
        cache: ResultCache,
        key: str,
        injections: Sequence[Injection],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        injection_matches = _matching_injections(
            injections, scheme=scheme, sample_id=None, stage="sensitivity"
        )
        if injection_matches:
            failures = []
            records = []
            for injection in injection_matches:
                failure, record = _failure_record(injection, scheme=scheme, sample_id=None)
                failures.append(failure)
                records.append(record)
            return (
                {"status": ErrorCode.INJECTED_FAULT, "entries": []},
                {
                    "status": "injected_fault",
                    "cache_key": key,
                    "injection_failures": failures,
                    "injection_records": records,
                },
            )
        if not score_results:
            return (
                {"status": "not_run", "entries": []},
                {"status": "not_run", "cache_key": key},
            )
        if not self.force_full and cache.contains(key):
            payload = cache.get(key)
            if payload is not None and isinstance(payload.get("payload"), dict):
                return payload["payload"], {"status": "reused", "cache_key": key}
        payload = scan_scheme_sensitivity(
            scheme,
            list(dataset.samples),
            score_results,
            config,
            dimension_indices,
            chunk_size,
        )
        cache.put(key, {"kind": "sensitivity", "scheme": scheme, "payload": payload})
        return payload, {"status": "recomputed", "cache_key": key}

    def _build_cache_report(
        self,
        *,
        selected_schemes: Sequence[str],
        tasks: Sequence[SampleTask],
        schedule: ScheduleResult,
        task_runs: Mapping[str, TaskRun],
        current_stage_keys: Mapping[str, str],
        previous_manifest: Mapping[str, Any],
        aggregate_stage_status: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> dict[str, Any]:
        previous_keys = previous_manifest.get("stage_keys", {}) if previous_manifest else {}
        events: list[dict[str, Any]] = []
        decision_by_key = {decision.task.sort_key: decision for decision in schedule.decisions}
        for scheme in selected_schemes:
            scheme_tasks = [task for task in tasks if task.scheme == scheme]
            for stage in ("position", "scoring"):
                hit_count = 0
                computed_count = 0
                failed_count = 0
                pending_count = 0
                for task in scheme_tasks:
                    decision = decision_by_key.get(task.sort_key)
                    run = task_runs.get(task.sort_key)
                    if decision is not None and decision.action == "pending":
                        pending_count += 1
                        continue
                    if run is not None and run.failure is not None:
                        failed_count += 1
                        continue
                    hit = task.position_hit if stage == "position" else task.scoring_hit
                    if hit:
                        hit_count += 1
                    if run is not None:
                        computed = (
                            run.computed_position
                            if stage == "position"
                            else run.computed_scoring
                        )
                        if computed:
                            computed_count += 1
                if pending_count:
                    status = "partial_pending"
                elif failed_count:
                    status = "partial_fault"
                elif hit_count == len(scheme_tasks) and hit_count > 0:
                    status = "reused"
                elif hit_count > 0 and computed_count > 0:
                    status = "partial"
                elif computed_count > 0:
                    status = "recomputed"
                elif hit_count > 0:
                    status = "reused"
                else:
                    status = "not_run"
                key_name = f"{scheme}:{stage}"
                current_key = current_stage_keys.get(key_name, "")
                previous_key = str(previous_keys.get(key_name, ""))
                events.append(
                    self._cache_event(
                        scheme=scheme,
                        stage=stage,
                        status=status,
                        current_key=current_key,
                        previous_key=previous_key,
                        hit_count=hit_count,
                        computed_count=computed_count,
                        failed_count=failed_count,
                        pending_count=pending_count,
                    )
                )
            for stage in ("trace", "sensitivity"):
                status_payload = aggregate_stage_status.get((scheme, stage), {"status": "not_run"})
                key_name = f"{scheme}:{stage}"
                current_key = current_stage_keys.get(key_name, "")
                previous_key = str(previous_keys.get(key_name, ""))
                events.append(
                    self._cache_event(
                        scheme=scheme,
                        stage=stage,
                        status=str(status_payload.get("status", "not_run")),
                        current_key=current_key,
                        previous_key=previous_key,
                        hit_count=1 if status_payload.get("status") == "reused" else 0,
                        computed_count=1 if status_payload.get("status") == "recomputed" else 0,
                        failed_count=1 if status_payload.get("status") == "injected_fault" else 0,
                        pending_count=0,
                    )
                )
        events.sort(key=lambda item: (item["scheme"], item["stage"]))
        return {
            "stage_events": events,
            "invalidated": [event for event in events if event["invalidated"]],
            "reused": [event for event in events if event["status"] == "reused"],
            "sample_decisions": [decision.to_dict() for decision in schedule.decisions],
            "rule": (
                "stage keys derive from scheme parameters, sample/dataset content, precision, "
                "quantization, pruning policy and strategy; only misses are recomputed"
            ),
        }

    @staticmethod
    def _cache_event(
        *,
        scheme: str,
        stage: str,
        status: str,
        current_key: str,
        previous_key: str,
        hit_count: int,
        computed_count: int,
        failed_count: int,
        pending_count: int,
    ) -> dict[str, Any]:
        key_changed = bool(previous_key and current_key != previous_key)
        if key_changed:
            reason = "input_fingerprint_changed"
        elif status == "reused":
            reason = "cache_hit"
        elif status == "recomputed":
            reason = "cache_miss"
        elif status == "partial_pending":
            reason = "point_budget_pending"
        elif status == "partial_fault":
            reason = "injected_or_execution_fault"
        elif status == "injected_fault":
            reason = "injection_forced"
        elif status == "not_run":
            reason = "not_run"
        else:
            reason = "no_cache"
        invalidated = bool(
            key_changed
            or status
            in {"recomputed", "partial", "partial_pending", "partial_fault", "injected_fault"}
        )
        return {
            "scheme": scheme,
            "stage": stage,
            "status": status,
            "cache_key": current_key,
            "previous_cache_key": previous_key or None,
            "cache_hit_count": hit_count,
            "computed_count": computed_count,
            "failed_count": failed_count,
            "pending_count": pending_count,
            "reason": reason,
            "invalidated": invalidated,
            "reused": status == "reused",
        }

    @staticmethod
    def _build_comparison(
        schemes: Sequence[str], scheme_payloads: Mapping[str, Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for scheme in schemes:
            payload = scheme_payloads.get(scheme, {})
            trace = payload.get("trace", {})
            trace_stages = {item["stage"]: item for item in trace.get("stages", [])}
            stats = payload.get("statistics", {})
            rows.append(
                {
                    "scheme": scheme,
                    "status": payload.get("status", "not_run"),
                    "extrapolation_score_count": stats.get("extrapolation", {}).get(
                        "score_count", 0
                    ),
                    "extrapolation_score_mean": stats.get("extrapolation", {}).get("mean"),
                    "mean_position_vector_deviation": trace_stages.get(
                        "position_vector", {}
                    ).get("mean_deviation"),
                    "mean_score_deviation": trace_stages.get("score", {}).get(
                        "mean_deviation"
                    ),
                    "mean_attention_deviation": trace_stages.get("attention", {}).get(
                        "mean_deviation"
                    ),
                    "overall_first_exceeded": trace.get("overall_first_exceeded"),
                    "lower_is_better": True,
                }
            )
        ranked = sorted(
            rows,
            key=lambda row: (
                row["mean_score_deviation"] is None,
                row["mean_score_deviation"]
                if row["mean_score_deviation"] is not None
                else float("inf"),
                row["scheme"],
            ),
        )
        rank_map = {row["scheme"]: rank for rank, row in enumerate(ranked, start=1)}
        for row in rows:
            row["score_deviation_rank"] = rank_map[row["scheme"]]
        return rows

    def _abort_report(
        self,
        *,
        config: DiagnosticConfig | None,
        failures: Iterable[Failure],
        config_hash: str,
        injection_hash: str,
        snapshot_state: SnapshotState,
        dataset: SampleDataset | None = None,
        pruning_hash: str = "",
        config_valid: bool | None = None,
    ) -> dict[str, Any]:
        failure_list = list(failures)
        if config_valid is None:
            config_valid = config is not None and not any(
                failure.stage in {"config", "pruning", "injection_validation"}
                and failure.code in {ErrorCode.CONFIG_INVALID, ErrorCode.IO_ERROR}
                for failure in failure_list
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "inputs": {
                "config_path": self.config_path.resolve().as_posix(),
                "config_hash": config_hash,
                "samples_path": self.samples_path.resolve().as_posix(),
                "samples_file_sha256": dataset.file_sha256 if dataset else "",
                "dataset_fingerprint": dataset.dataset_fingerprint if dataset else "",
                "pruning_path": self.pruning_path.resolve().as_posix(),
                "pruning_hash": pruning_hash,
                "injections_path": self.injections_path.resolve().as_posix(),
                "injection_hash": injection_hash,
            },
            "config": config.public_dict() if config is not None else None,
            "validation": {
                "config_valid": config_valid,
                "sample_valid_count": dataset.sample_count if dataset else 0,
                "sample_invalid_count": len(dataset.failures) if dataset else 0,
                "sample_failures": sorted_failures(list(dataset.failures)) if dataset else [],
                "failures": sorted_failures(failure_list),
            },
            "strategy": None,
            "scheduler": None,
            "schemes": {},
            "comparison": [],
            "cache": {
                "stage_events": [],
                "invalidated": [],
                "reused": [],
                "sample_decisions": [],
            },
            "injections": {
                "injection_hash": injection_hash,
                "configured": [],
                "triggered": [],
            },
            "snapshot": {
                "resume_compatible": False,
                "resumed_completed_task_count": 0,
                "commit_seq": int(snapshot_state.manifest.get("commit_seq", 0) or 0),
                "manifest": snapshot_state.manifest or None,
            },
            "failures": sorted_failures(failure_list),
            "semantic_digest": sha256_json({"failures": sorted_failures(failure_list)}),
        }
