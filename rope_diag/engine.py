"""Diagnostic pipeline orchestration with deterministic concurrent stages."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .attention import AttentionResult, aggregate_stratified_statistics, compute_attention
from .cache import STAGE_RANK, StageCache, stage_cache_key
from .config import (
    SCHEME_RANK,
    DiagnosticConfig,
    parse_config,
)
from .encoding import (
    SampleEncoding,
    encode_sample,
    encoding_parameters,
    reference_encoding,
    validate_scheme_numeric_inputs,
)
from .errors import (
    ConfigError,
    E_BASE_ZERO,
    E_CONFIG_INVALID,
    E_INTERNAL,
    E_NONFINITE_INPUT,
    E_POSITION_OUT_OF_RANGE,
    E_SCALE_INVALID,
    WorkbenchError,
)
from .injection import Injection, apply_injections, load_injections
from .samples import Sample, load_samples
from .sensitivity import run_sensitivity_scan
from .snapshot import SnapshotManager
from .tracing import summarize_traces, trace_attention
from .util import atomic_write_json, hash_object, read_json


STAGE_NAMES = ("encoding", "scoring", "tracing", "sensitivity")
TRACE_POLICY_VERSION = 1
SCORE_POLICY_VERSION = 1


@dataclass(frozen=True)
class EngineOptions:
    state_dir: Path = Path(".workbench_state")
    resume: bool = False
    use_cache: bool = True
    stop_after_stage: str | None = None
    stop_event: threading.Event | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_dir", Path(self.state_dir))
        if self.stop_after_stage not in (None, *STAGE_NAMES):
            raise ConfigError(
                "unsupported stop_after_stage value",
                {"value": self.stop_after_stage, "supported": list(STAGE_NAMES)},
            )


@dataclass
class Task:
    scheme: str
    sample: Sample
    effective_sample: Sample
    base: float
    scaling_factor: float
    defects: list[dict[str, Any]]

    @property
    def identity(self) -> tuple[str, str]:
        return self.scheme, self.sample.sample_id

    @property
    def sort_key(self) -> tuple[int, str]:
        return SCHEME_RANK[self.scheme], self.sample.sample_id


@dataclass
class StageJob:
    sort_key: tuple[int, str, int]
    operation: Callable[[], Any]


class DiagnosticEngine:
    def __init__(
        self,
        *,
        config_path: str | Path,
        samples_path: str | Path,
        injections_path: str | Path,
        options: EngineOptions | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.samples_path = Path(samples_path)
        self.injections_path = Path(injections_path)
        self.options = options or EngineOptions()

    def _should_interrupt(self, completed_stage: str) -> bool:
        if self.options.stop_after_stage == completed_stage:
            return True
        return bool(self.options.stop_event and self.options.stop_event.is_set())

    def _run_jobs(self, jobs: list[StageJob], concurrency: int) -> dict[tuple[int, str, int], Any]:
        if not jobs:
            return {}
        ordered = sorted(jobs, key=lambda item: item.sort_key)
        results: dict[tuple[int, str, int], Any] = {}
        if concurrency <= 1 or len(ordered) <= 1:
            for job in ordered:
                try:
                    results[job.sort_key] = job.operation()
                except WorkbenchError as exc:
                    results[job.sort_key] = exc
                except Exception as exc:  # pragma: no cover - defensive isolation
                    results[job.sort_key] = WorkbenchError(
                        E_INTERNAL,
                        "unexpected stage worker failure",
                        details={"error_type": type(exc).__name__, "error": str(exc)},
                    )
            return results

        with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="rope-diag") as pool:
            futures = {pool.submit(job.operation): job for job in ordered}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    results[job.sort_key] = future.result()
                except WorkbenchError as exc:
                    results[job.sort_key] = exc
                except Exception as exc:  # pragma: no cover - defensive isolation
                    results[job.sort_key] = WorkbenchError(
                        E_INTERNAL,
                        "unexpected stage worker failure",
                        details={"error_type": type(exc).__name__, "error": str(exc)},
                    )
        return results

    @staticmethod
    def _task_stage_key(task: Task, stage: str) -> str:
        return f"{task.scheme}|{task.sample.sample_id}|{stage}"

    @staticmethod
    def _stage_sort_key(task: Task, stage: str) -> tuple[int, str, int]:
        return SCHEME_RANK[task.scheme], task.sample.sample_id, STAGE_RANK[stage]

    def _generic_defect(
        self,
        task: Task,
        defects: list[dict[str, Any]],
        *,
        code: str,
        message: str,
        stage: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if any(item.get("code") == code for item in defects):
            return
        defects.append(
            {
                "injection_id": None,
                "injection_index": None,
                "fault": None,
                "code": code,
                "stage": stage,
                "message": message,
                "details": details or {},
            }
        )

    def _validate_effective_task(self, task: Task) -> None:
        sample = task.effective_sample
        defects = task.defects
        query = sample.query_array
        keys = sample.key_array
        if not np.all(np.isfinite(query)) or not np.all(np.isfinite(keys)):
            self._generic_defect(
                task,
                defects,
                code=E_NONFINITE_INPUT,
                message="query or key feature contains NaN/Inf",
                stage="encoding",
                details={"sample_id": sample.sample_id},
            )
        try:
            validate_scheme_numeric_inputs(
                task.scheme,
                task.base,
                len(sample.query_features),
                task.scaling_factor,
                self.config.precision,
            )
        except WorkbenchError as exc:
            self._generic_defect(
                task,
                defects,
                code=exc.code,
                message=exc.message,
                stage=exc.stage or "encoding",
                details=exc.details,
            )
        invalid_positions: list[dict[str, Any]] = []
        if sample.query_position < 0 or sample.query_position >= self.config.extrapolation_target_length:
            invalid_positions.append(
                {"location": "query_position", "position": sample.query_position}
            )
        for index, position in enumerate(sample.key_positions):
            if position < 0 or position >= self.config.extrapolation_target_length:
                invalid_positions.append(
                    {"location": "key_position", "key_index": index, "position": position}
                )
        if invalid_positions:
            self._generic_defect(
                task,
                defects,
                code=E_POSITION_OUT_OF_RANGE,
                message="position falls outside [0, extrapolation_target_length)",
                stage="scoring",
                details={
                    "sample_id": sample.sample_id,
                    "extrapolation_target_length": self.config.extrapolation_target_length,
                    "positions": invalid_positions,
                },
            )

    def _prepare_tasks(
        self,
        samples: list[Sample],
        injections: list[Injection],
    ) -> list[Task]:
        injections_by_sample: dict[str, list[Injection]] = {}
        for injection in injections:
            injections_by_sample.setdefault(injection.sample_id, []).append(injection)
        tasks: list[Task] = []
        for scheme in self.config.schemes:
            for sample in samples:
                effective, base, scaling_factor, defects = apply_injections(
                    sample,
                    scheme,
                    self.config.base,
                    self.config.scaling_factor,
                    injections_by_sample.get(sample.sample_id, []),
                    self.config.extrapolation_target_length,
                )
                task = Task(
                    scheme=scheme,
                    sample=sample,
                    effective_sample=effective,
                    base=base,
                    scaling_factor=scaling_factor,
                    defects=defects,
                )
                if effective.query_array.size != self.config.dimension:
                    self._generic_defect(
                        task,
                        defects,
                        code=E_CONFIG_INVALID,
                        message="query feature dimension does not match configuration",
                        stage="encoding",
                        details={
                            "sample_id": sample.sample_id,
                            "expected": self.config.dimension,
                            "actual": int(effective.query_array.size),
                        },
                    )
                elif effective.key_array.ndim != 2 or effective.key_array.shape[1] != self.config.dimension:
                    self._generic_defect(
                        task,
                        defects,
                        code=E_CONFIG_INVALID,
                        message="key feature dimension does not match configuration",
                        stage="encoding",
                        details={
                            "sample_id": sample.sample_id,
                            "expected": self.config.dimension,
                            "actual": list(effective.key_array.shape),
                        },
                    )
                self._validate_effective_task(task)
                defects.sort(
                    key=lambda item: (
                        item.get("injection_index") is None,
                        item.get("injection_index") or 0,
                        str(item.get("code")),
                        str(item.get("message")),
                    )
                )
                tasks.append(task)
        tasks.sort(key=lambda item: item.sort_key)
        return tasks

    @staticmethod
    def _failure_from_defect(task: Task, defect: dict[str, Any]) -> dict[str, Any]:
        details = dict(defect.get("details") or {})
        if defect.get("fault") is not None:
            details["fault"] = defect["fault"]
        return {
            "code": defect["code"],
            "scheme": task.scheme,
            "sample_id": task.sample.sample_id,
            "stage": defect.get("stage"),
            "injection_id": defect.get("injection_id"),
            "injection_index": defect.get("injection_index"),
            "message": defect.get("message", ""),
            "details": details,
        }

    @staticmethod
    def _failure_from_error(task: Task, stage: str, error: WorkbenchError) -> dict[str, Any]:
        return {
            "code": error.code,
            "scheme": task.scheme,
            "sample_id": task.sample.sample_id,
            "stage": error.stage or stage,
            "injection_id": None,
            "injection_index": None,
            "message": error.message,
            "details": error.details,
        }

    def _classification_fingerprint(self, sample: Sample) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index, position in enumerate(sample.key_positions):
            distance = abs(int(position) - int(sample.query_position))
            is_boundary = distance == self.config.train_context_length
            result.append(
                {
                    "key_index": index,
                    "key_position": int(position),
                    "relative_distance": int(distance),
                    "layer": "in_train" if distance <= self.config.train_context_length else "out_of_train",
                    "boundary_marked": bool(is_boundary),
                }
            )
        return result

    def _prepare_failures(self, tasks: list[Task]) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for task in tasks:
            for defect in task.defects:
                failures.append(self._failure_from_defect(task, defect))
        return failures

    def _encoding_stage(
        self,
        tasks: list[Task],
        cache: StageCache,
    ) -> tuple[dict[tuple[str, str], SampleEncoding], dict[tuple[str, str], str], list[dict[str, Any]]]:
        results: dict[tuple[str, str], SampleEncoding] = {}
        keys: dict[tuple[str, str], str] = {}
        failures: list[dict[str, Any]] = []
        jobs: list[StageJob] = []
        job_tasks: dict[tuple[int, str, int], tuple[Task, str, dict[str, Any]]] = {}
        attempted: list[Task] = []

        for task in tasks:
            if task.defects:
                continue
            attempted.append(task)
            parameters = encoding_parameters(
                task.scheme,
                self.config.base,
                self.config.dimension,
                self.config.scaling_factor,
            )
            key, material = stage_cache_key(
                stage="encoding",
                scheme_parameters=parameters,
                sample_fingerprint=task.sample.fingerprint,
                precision=self.config.precision,
            )
            payload = cache.load(
                key=key,
                stage="encoding",
                scheme=task.scheme,
                sample_id=task.sample.sample_id,
                precision=self.config.precision,
            )
            if payload is not None:
                try:
                    encoding = SampleEncoding.from_payload(payload)
                    if (
                        encoding.scheme == task.scheme
                        and encoding.sample_id == task.sample.sample_id
                        and encoding.dimension == self.config.dimension
                    ):
                        results[task.identity] = encoding
                        keys[task.identity] = key
                        cache.record_reuse(
                            key=key,
                            stage="encoding",
                            scheme=task.scheme,
                            sample_id=task.sample.sample_id,
                            precision=self.config.precision,
                            material=material,
                        )
                        continue
                except (KeyError, TypeError, ValueError):
                    pass
            sort_key = self._stage_sort_key(task, "encoding")
            jobs.append(
                StageJob(
                    sort_key=sort_key,
                    operation=lambda task=task: encode_sample(
                        task.scheme,
                        task.effective_sample,
                        self.config.dimension,
                        task.base,
                        task.scaling_factor,
                        self.config.precision,
                    ),
                )
            )
            job_tasks[sort_key] = (task, key, material)

        job_results = self._run_jobs(jobs, self.config.concurrency)
        for sort_key in sorted(job_results):
            task, key, material = job_tasks[sort_key]
            value = job_results[sort_key]
            if isinstance(value, WorkbenchError):
                failures.append(self._failure_from_error(task, "encoding", value))
                continue
            encoding = value
            assert isinstance(encoding, SampleEncoding)
            results[task.identity] = encoding
            keys[task.identity] = key
            cache.store(
                key=key,
                stage="encoding",
                scheme=task.scheme,
                sample_id=task.sample.sample_id,
                precision=self.config.precision,
                payload=encoding.to_payload(),
            )
            cache.record_recompute(
                key=key,
                stage="encoding",
                scheme=task.scheme,
                sample_id=task.sample.sample_id,
                precision=self.config.precision,
                material=material,
            )
        return results, keys, failures

    def _scoring_stage(
        self,
        tasks: list[Task],
        encodings: dict[tuple[str, str], SampleEncoding],
        encoding_keys: dict[tuple[str, str], str],
        cache: StageCache,
    ) -> tuple[dict[tuple[str, str], AttentionResult], dict[tuple[str, str], str], list[dict[str, Any]]]:
        results: dict[tuple[str, str], AttentionResult] = {}
        keys: dict[tuple[str, str], str] = {}
        failures: list[dict[str, Any]] = []
        jobs: list[StageJob] = []
        job_tasks: dict[tuple[int, str, int], tuple[Task, str, dict[str, Any]]] = {}

        for task in tasks:
            if task.identity not in encodings:
                continue
            parameters = encoding_parameters(
                task.scheme,
                self.config.base,
                self.config.dimension,
                self.config.scaling_factor,
            )
            dependency_fingerprint = hash_object(
                {
                    "encoding_cache_key": encoding_keys[task.identity],
                    "classification": self._classification_fingerprint(task.sample),
                    "dimension": self.config.dimension,
                    "score_policy_version": SCORE_POLICY_VERSION,
                }
            )
            key, material = stage_cache_key(
                stage="scoring",
                scheme_parameters=parameters,
                sample_fingerprint=task.sample.fingerprint,
                precision=self.config.precision,
                dependency_fingerprint=dependency_fingerprint,
            )
            payload = cache.load(
                key=key,
                stage="scoring",
                scheme=task.scheme,
                sample_id=task.sample.sample_id,
                precision=self.config.precision,
            )
            if payload is not None:
                try:
                    attention = AttentionResult.from_payload(payload)
                    if attention.scheme == task.scheme and attention.sample_id == task.sample.sample_id:
                        results[task.identity] = attention
                        keys[task.identity] = key
                        cache.record_reuse(
                            key=key,
                            stage="scoring",
                            scheme=task.scheme,
                            sample_id=task.sample.sample_id,
                            precision=self.config.precision,
                            material=material,
                        )
                        continue
                except (KeyError, TypeError, ValueError):
                    pass
            sort_key = self._stage_sort_key(task, "scoring")

            def operation(
                task: Task = task,
                encoding: SampleEncoding = encodings[task.identity],
            ) -> AttentionResult:
                reference = reference_encoding(
                    task.effective_sample,
                    self.config.dimension,
                    task.base,
                    self.config.precision,
                )
                return compute_attention(
                    task.effective_sample,
                    task.scheme,
                    encoding,
                    reference,
                    self.config.dimension,
                    self.config.train_context_length,
                )

            jobs.append(StageJob(sort_key=sort_key, operation=operation))
            job_tasks[sort_key] = (task, key, material)

        job_results = self._run_jobs(jobs, self.config.concurrency)
        for sort_key in sorted(job_results):
            task, key, material = job_tasks[sort_key]
            value = job_results[sort_key]
            if isinstance(value, WorkbenchError):
                failures.append(self._failure_from_error(task, "scoring", value))
                continue
            attention = value
            assert isinstance(attention, AttentionResult)
            results[task.identity] = attention
            keys[task.identity] = key
            cache.store(
                key=key,
                stage="scoring",
                scheme=task.scheme,
                sample_id=task.sample.sample_id,
                precision=self.config.precision,
                payload=attention.to_payload(),
            )
            cache.record_recompute(
                key=key,
                stage="scoring",
                scheme=task.scheme,
                sample_id=task.sample.sample_id,
                precision=self.config.precision,
                material=material,
            )
        return results, keys, failures

    def _tracing_stage(
        self,
        tasks: list[Task],
        encodings: dict[tuple[str, str], SampleEncoding],
        attentions: dict[tuple[str, str], AttentionResult],
        scoring_keys: dict[tuple[str, str], str],
        cache: StageCache,
    ) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
        results: dict[tuple[str, str], list[dict[str, Any]]] = {}
        failures: list[dict[str, Any]] = []
        jobs: list[StageJob] = []
        job_tasks: dict[tuple[int, str, int], tuple[Task, str, dict[str, Any]]] = {}

        for task in tasks:
            if task.identity not in attentions or task.identity not in encodings:
                continue
            parameters = encoding_parameters(
                task.scheme,
                self.config.base,
                self.config.dimension,
                self.config.scaling_factor,
            )
            dependency_fingerprint = hash_object(
                {
                    "scoring_cache_key": scoring_keys[task.identity],
                    "tolerances": self.config.tolerances.as_dict(),
                    "trace_policy_version": TRACE_POLICY_VERSION,
                }
            )
            key, material = stage_cache_key(
                stage="tracing",
                scheme_parameters=parameters,
                sample_fingerprint=task.sample.fingerprint,
                precision=self.config.precision,
                dependency_fingerprint=dependency_fingerprint,
            )
            payload = cache.load(
                key=key,
                stage="tracing",
                scheme=task.scheme,
                sample_id=task.sample.sample_id,
                precision=self.config.precision,
            )
            if payload is not None and isinstance(payload.get("traces"), list):
                traces = payload["traces"]
                results[task.identity] = traces
                cache.record_reuse(
                    key=key,
                    stage="tracing",
                    scheme=task.scheme,
                    sample_id=task.sample.sample_id,
                    precision=self.config.precision,
                    material=material,
                )
                continue
            sort_key = self._stage_sort_key(task, "tracing")
            jobs.append(
                StageJob(
                    sort_key=sort_key,
                    operation=lambda task=task: trace_attention(
                        attentions[task.identity],
                        encodings[task.identity],
                        self.config.tolerances,
                        self.config.train_context_length,
                    ),
                )
            )
            job_tasks[sort_key] = (task, key, material)

        job_results = self._run_jobs(jobs, self.config.concurrency)
        for sort_key in sorted(job_results):
            task, key, material = job_tasks[sort_key]
            value = job_results[sort_key]
            if isinstance(value, WorkbenchError):
                failures.append(self._failure_from_error(task, "tracing", value))
                continue
            traces = value
            assert isinstance(traces, list)
            results[task.identity] = traces
            cache.store(
                key=key,
                stage="tracing",
                scheme=task.scheme,
                sample_id=task.sample.sample_id,
                precision=self.config.precision,
                payload={"traces": traces},
            )
            cache.record_recompute(
                key=key,
                stage="tracing",
                scheme=task.scheme,
                sample_id=task.sample.sample_id,
                precision=self.config.precision,
                material=material,
            )
        return results, failures

    def _record_bypasses(self, tasks: list[Task], cache: StageCache) -> None:
        for task in tasks:
            if not task.defects:
                continue
            has_injection = any(defect.get("injection_id") for defect in task.defects)
            reason = "injection_bypass" if has_injection else "invalid_input_bypass"
            for stage in ("encoding", "scoring", "tracing"):
                cache.record_bypass(
                    stage=stage,
                    scheme=task.scheme,
                    sample_id=task.sample.sample_id,
                    reason=reason,
                )

    def _interrupt(
        self,
        snapshot: SnapshotManager,
        completed_stage: str,
        cache: StageCache,
    ) -> dict[str, Any]:
        cache.write_index()
        completed_schemes = list(self.config.schemes) if completed_stage in {"tracing", "sensitivity"} else []
        completed_samples = (
            [sample.sample_id for sample in self.samples]
            if completed_stage in {"tracing", "sensitivity"}
            else []
        )
        snapshot.interrupt(completed_schemes, completed_samples)
        return {
            "status": "interrupted",
            "completed_stage": completed_stage,
            "exit_code": 0,
            "config_hash": self.config_hash,
            "injection_hash": self.injection_hash,
            "manifest": str(snapshot.manifest_path),
        }

    def _build_report(
        self,
        *,
        tasks: list[Task],
        failures: list[dict[str, Any]],
        attentions: dict[tuple[str, str], AttentionResult],
        traces: dict[tuple[str, str], list[dict[str, Any]]],
        sensitivity: dict[str, Any],
        cache: StageCache,
    ) -> dict[str, Any]:
        stratified: list[dict[str, Any]] = []
        for scheme in self.config.schemes:
            scheme_results = [
                attentions[(scheme, sample.sample_id)]
                for sample in self.samples
                if (scheme, sample.sample_id) in attentions
            ]
            rows = aggregate_stratified_statistics(scheme_results, self.config.train_context_length)
            stratified.append({"scheme": scheme, **rows})

        flat_traces = [
            trace
            for scheme in self.config.schemes
            for sample in self.samples
            for trace in traces.get((scheme, sample.sample_id), [])
        ]
        flat_traces.sort(
            key=lambda item: (
                SCHEME_RANK[str(item["scheme"])],
                str(item["sample_id"]),
                int(item["key_index"]),
            )
        )

        failures.sort(
            key=lambda item: (
                SCHEME_RANK.get(str(item.get("scheme")), 99),
                str(item.get("sample_id")),
                STAGE_RANK.get(str(item.get("stage")), 99),
                item.get("injection_index") is None,
                int(item.get("injection_index") or 0),
                str(item.get("code")),
                str(item.get("message")),
            )
        )
        injection_hits = [
            {
                "injection_id": failure["injection_id"],
                "injection_index": failure["injection_index"],
                "scheme": failure["scheme"],
                "sample_id": failure["sample_id"],
                "stage": failure["stage"],
                "fault": failure.get("details", {}).get("fault"),
                "code": failure["code"],
                "message": failure["message"],
            }
            for failure in failures
            if failure.get("injection_id") is not None
        ]

        hits_by_sample_map: dict[str, dict[str, Any]] = {}
        for hit in injection_hits:
            group = hits_by_sample_map.setdefault(
                str(hit["sample_id"]),
                {"sample_id": hit["sample_id"], "hits": []},
            )
            group["hits"].append(hit)
        injection_hits_by_sample: list[dict[str, Any]] = []
        for sample_id in sorted(hits_by_sample_map):
            group = hits_by_sample_map[sample_id]
            group["hits"].sort(
                key=lambda item: (
                    SCHEME_RANK[str(item["scheme"])],
                    int(item["injection_index"] or 0),
                    str(item["injection_id"]),
                )
            )
            group["injection_ids"] = sorted(
                {str(item["injection_id"]) for item in group["hits"]}
            )
            group["codes"] = sorted({str(item["code"]) for item in group["hits"]})
            injection_hits_by_sample.append(group)

        scheme_summaries: list[dict[str, Any]] = []
        for scheme in self.config.schemes:
            scheme_tasks = [task for task in tasks if task.scheme == scheme]
            scheme_failures = [failure for failure in failures if failure.get("scheme") == scheme]
            scheme_attentions = [value for key, value in attentions.items() if key[0] == scheme]
            scheme_traces = [
                trace
                for sample in self.samples
                for trace in traces.get((scheme, sample.sample_id), [])
            ]
            scheme_summaries.append(
                {
                    "scheme": scheme,
                    "sample_count": len(scheme_tasks),
                    "successful_sample_count": len(scheme_attentions),
                    "failure_count": len(scheme_failures),
                    "injection_hit_count": sum(
                        1 for failure in scheme_failures if failure.get("injection_id") is not None
                    ),
                    "out_of_train_trace_count": len(scheme_traces),
                    "extrapolation_sample_count": len(
                        {
                            str(trace["sample_id"])
                            for trace in scheme_traces
                        }
                    ),
                }
            )

        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "config_hash": self.config_hash,
            "samples_hash": self.samples_hash,
            "injection_hash": self.injection_hash,
            "configuration": self.config.as_dict(),
            "execution": {
                "concurrency": self.config.concurrency,
                "cache_enabled": self.options.use_cache,
                "precision_policy": "stage arrays use the configured precision; all deviations are computed in float64",
            },
            "stratified_statistics": stratified,
            "error_traces": flat_traces,
            "trace_summary": summarize_traces(flat_traces),
            "sensitivity": sensitivity,
            "injection_hits": injection_hits,
            "injection_hits_by_sample": injection_hits_by_sample,
            "failures": failures,
            "scheme_summaries": scheme_summaries,
            "reuse_manifest": cache.audit(),
            "report_contract": {
                "business_codes": [
                    "E_CONFIG_INVALID",
                    "E_CONFIG_MISMATCH",
                    "E_NONFINITE_INPUT",
                    "E_POSITION_OUT_OF_RANGE",
                    "E_BASE_ZERO",
                    "E_SCALE_INVALID",
                ],
                "diagnostic_digest_scope": "all fields except reuse_manifest and diagnostic_digest",
            },
        }
        digest_material = dict(report)
        digest_material.pop("reuse_manifest", None)
        report["diagnostic_digest"] = hash_object(digest_material)
        return report

    def run(self, report_path: str | Path) -> dict[str, Any]:
        raw_config = read_json(self.config_path)
        self.config: DiagnosticConfig = parse_config(raw_config)
        self.config_hash = hash_object(raw_config)
        self.samples, self.samples_hash = load_samples(self.samples_path)
        self.injections, raw_injections, self.injection_hash = load_injections(self.injections_path)
        sample_ids = {sample.sample_id for sample in self.samples}
        for injection in self.injections:
            if injection.sample_id not in sample_ids:
                raise ConfigError(
                    "injection references an unknown sample_id",
                    {
                        "injection_id": injection.injection_id,
                        "sample_id": injection.sample_id,
                    },
                )
            if injection.scheme is not None and injection.scheme not in self.config.schemes:
                raise ConfigError(
                    "injection references a scheme that is not enabled in configuration",
                    {
                        "injection_id": injection.injection_id,
                        "scheme": injection.scheme,
                        "enabled_schemes": list(self.config.schemes),
                    },
                )

        snapshot = SnapshotManager(
            self.options.state_dir,
            config_hash=self.config_hash,
            injection_hash=self.injection_hash,
            samples_hash=self.samples_hash,
            resume_requested=self.options.resume,
        )
        snapshot.start()
        cache = StageCache(self.options.state_dir, enabled=self.options.use_cache)

        tasks = self._prepare_tasks(self.samples, self.injections)
        failures = self._prepare_failures(tasks)
        self._record_bypasses(tasks, cache)
        active_tasks = [task for task in tasks if not task.defects]

        encodings, encoding_keys, encoding_failures = self._encoding_stage(active_tasks, cache)
        failures.extend(encoding_failures)
        self._mark_stage(snapshot, cache, "encoding", active_tasks)
        if self._should_interrupt("encoding"):
            return self._interrupt(snapshot, "encoding", cache)

        attentions, scoring_keys, scoring_failures = self._scoring_stage(
            active_tasks,
            encodings,
            encoding_keys,
            cache,
        )
        failures.extend(scoring_failures)
        self._mark_stage(snapshot, cache, "scoring", active_tasks)
        if self._should_interrupt("scoring"):
            return self._interrupt(snapshot, "scoring", cache)

        traces, tracing_failures = self._tracing_stage(
            active_tasks,
            encodings,
            attentions,
            scoring_keys,
            cache,
        )
        failures.extend(tracing_failures)
        self._mark_stage(snapshot, cache, "tracing", active_tasks)
        if self._should_interrupt("tracing"):
            return self._interrupt(snapshot, "tracing", cache)

        sensitivity = run_sensitivity_scan(self.config, self.samples, attentions)
        self._mark_stage(snapshot, cache, "sensitivity", active_tasks)
        if self._should_interrupt("sensitivity"):
            return self._interrupt(snapshot, "sensitivity", cache)

        report = self._build_report(
            tasks=tasks,
            failures=failures,
            attentions=attentions,
            traces=traces,
            sensitivity=sensitivity,
            cache=cache,
        )
        cache.write_index()
        atomic_write_json(Path(report_path), report)
        snapshot.complete(list(self.config.schemes), [sample.sample_id for sample in self.samples])
        return report

    def _mark_stage(
        self,
        snapshot: SnapshotManager,
        cache: StageCache,
        stage: str,
        active_tasks: list[Task],
    ) -> None:
        cache.write_index()
        completed_keys = [self._task_stage_key(task, stage) for task in active_tasks]
        snapshot.mark_stage(stage, completed_keys)
