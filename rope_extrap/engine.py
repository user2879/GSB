from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping

from .cache import ResultCache, StageKeys, build_stage_keys
from .encoding import encode_sample, position_vector_summary
from .errors import (
    E_COMPUTE_NONFINITE,
    E_INTERNAL,
    STAGE_ORDER,
    BusinessError,
    Failure,
)
from .models import (
    STAGES,
    AttentionSample,
    CacheEntry,
    PositionEncodingConfig,
    ScoreResult,
    TraceResult,
    config_summary,
    stage_sort_key,
)
from .scoring import (
    aggregate_dimension_sensitivity,
    aggregate_score_statistics,
    calculate_scores,
    score_entry_dict,
)
from .tracing import (
    aggregate_external_trace_statistics,
    key_trace_dict,
    trace_sample,
)


@dataclass(frozen=True)
class StageJob:
    stage: str
    sample_id: str
    scheme: str
    cache_key: str
    feature_line: int
    source_ref: str
    stage_keys: StageKeys

    @property
    def stable_sort_key(self) -> str:
        return stage_sort_key(self.scheme, self.sample_id, self.stage)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_order": STAGE_ORDER[self.stage],
            "cache_key": self.cache_key,
            "sample_id": self.sample_id,
            "scheme": self.scheme,
            "feature_line": self.feature_line,
            "source_ref": self.source_ref,
            "stable_sort_key": self.stable_sort_key,
        }


@dataclass
class ExecutionPlan:
    config_fingerprint: str
    stage_keys_by_sample: dict[str, StageKeys] = field(default_factory=dict)
    reused: list[CacheEntry] = field(default_factory=list)
    jobs_by_stage: dict[str, list[StageJob]] = field(
        default_factory=lambda: {stage: [] for stage in STAGES}
    )
    blocked_samples: set[str] = field(default_factory=set)
    blocked_reasons: dict[str, dict[str, Any]] = field(default_factory=dict)
    recomputed: list[StageJob] = field(default_factory=list)
    failed: list[Failure] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def remaining_jobs(self) -> list[StageJob]:
        recomputed_keys = {job.stable_sort_key for job in self.recomputed}
        skipped_keys = {item["stable_sort_key"] for item in self.skipped}
        return [
            job
            for stage in STAGES
            for job in self.jobs_by_stage[stage]
            if job.stable_sort_key not in recomputed_keys and job.stable_sort_key not in skipped_keys
        ]

    def is_complete(self) -> bool:
        return not self.remaining_jobs()

    def audit_dict(self) -> dict[str, Any]:
        reused = sorted(self.reused, key=lambda entry: entry.stable_sort_key())
        recomputed = sorted(self.recomputed, key=lambda job: job.stable_sort_key)
        failed = sorted(self.failed, key=lambda failure: failure.sort_key())
        skipped = sorted(self.skipped, key=lambda item: item["stable_sort_key"])
        return {
            "reused": [entry.audit_dict() for entry in reused],
            "recomputed": [job.audit_dict() for job in recomputed],
            "failed": [failure.to_dict() for failure in failed],
            "skipped": skipped,
            "summary": {
                "reused_stage_count": len(reused),
                "recomputed_stage_count": len(recomputed),
                "failed_stage_count": len(failed),
                "skipped_stage_count": len(skipped),
                "reused_samples": sorted({entry.sample_id for entry in reused}),
                "recomputed_samples": sorted({job.sample_id for job in recomputed}),
                "failed_samples": sorted({failure.sample_id for failure in failed if failure.sample_id}),
            },
        }


def _failure_from_exception(exc: Exception, job: StageJob) -> Failure:
    if isinstance(exc, BusinessError):
        return Failure(
            code=exc.code,
            message=exc.message,
            stage=exc.stage or job.stage,
            sample_id=exc.sample_id or job.sample_id,
            scope="runtime",
            detail=repr(exc),
        )
    return Failure(
        code=E_INTERNAL,
        message=f"unexpected {job.stage} error: {exc}",
        stage=job.stage,
        sample_id=job.sample_id,
        scope="runtime",
        detail=repr(exc),
    )


class DiagnosticEngine:
    """Deterministic incremental diagnostic engine.

    The engine is deliberately synchronous and explicit about its cache.  A
    caller can build a plan, execute it under a deadline, and continue the same
    content-addressed work on a later interaction; no hidden background thread
    mutates results after a response has been returned.
    """

    def __init__(
        self,
        config: PositionEncodingConfig,
        samples: Mapping[str, AttentionSample] | Iterable[AttentionSample],
        *,
        loader_failures: Iterable[Failure] = (),
        cache: ResultCache | None = None,
        stage_hooks: Mapping[str, Callable[[AttentionSample], None]] | None = None,
        stage_failures: MutableMapping[tuple[str, str, str], Failure] | None = None,
    ) -> None:
        self.config = config
        if isinstance(samples, Mapping):
            self.samples: dict[str, AttentionSample] = dict(sorted(samples.items()))
        else:
            self.samples = {sample.sample_id: sample for sample in sorted(samples, key=lambda item: item.sample_id)}
        self.loader_failures = sorted(loader_failures, key=lambda failure: failure.sort_key())
        self.cache = cache or ResultCache()
        self.stage_failures: MutableMapping[tuple[str, str, str], Failure] = (
            stage_failures if stage_failures is not None else {}
        )
        self.stage_hooks = dict(stage_hooks or {})
        self.last_plan: ExecutionPlan | None = None
        self.last_elapsed_ms: float | None = None
        self.max_observed_concurrency = 0

    def set_config(self, config: PositionEncodingConfig) -> None:
        self.config = config

    def replace_sample(self, sample: AttentionSample) -> None:
        self.samples[sample.sample_id] = sample
        self.samples = dict(sorted(self.samples.items()))

    def build_plan(
        self,
        *,
        config: PositionEncodingConfig | None = None,
        yield_callback: Callable[[], None] | None = None,
    ) -> ExecutionPlan:
        config = config or self.config
        plan = ExecutionPlan(config_fingerprint=config.config_fingerprint())
        for sample_id in sorted(self.samples):
            if yield_callback is not None:
                yield_callback()
            sample = self.samples[sample_id]
            keys = build_stage_keys(config, sample)
            plan.stage_keys_by_sample[sample_id] = keys
            blocked = False
            for stage in STAGES:
                cache_key = keys.as_dict()[stage]
                cached = self.cache.get(stage, cache_key)
                if cached is not None:
                    plan.reused.append(cached)
                    continue
                failure = self.stage_failures.get((sample_id, stage, cache_key))
                if failure is not None:
                    plan.blocked_samples.add(sample_id)
                    plan.blocked_reasons.setdefault(
                        sample_id,
                        {"stage": stage, "cache_key": cache_key, "code": failure.code},
                    )
                    blocked = True
                    break
                if blocked:
                    plan.skipped.append(
                        {
                            "stage": stage,
                            "cache_key": cache_key,
                            "sample_id": sample_id,
                            "scheme": config.scheme,
                            "feature_line": sample.feature_line,
                            "source_ref": sample.source_ref,
                            "stable_sort_key": stage_sort_key(config.scheme, sample_id, stage),
                            "reason": "upstream_stage_blocked",
                        }
                    )
                    continue
                plan.jobs_by_stage[stage].append(
                    StageJob(
                        stage=stage,
                        sample_id=sample_id,
                        scheme=config.scheme,
                        cache_key=cache_key,
                        feature_line=sample.feature_line,
                        source_ref=sample.source_ref,
                        stage_keys=keys,
                    )
                )
        for stage in STAGES:
            plan.jobs_by_stage[stage].sort(key=lambda job: job.stable_sort_key)
        self.last_plan = plan
        return plan

    def _job_result(self, job: StageJob, config: PositionEncodingConfig) -> Any:
        sample = self.samples[job.sample_id]
        hook = self.stage_hooks.get(job.stage)
        if hook is not None:
            hook(sample)
        if job.stage == "position_encoding":
            return encode_sample(config, sample)
        encoding_entry = self.cache.get("position_encoding", job.stage_keys.position_encoding)
        if encoding_entry is None:
            raise BusinessError(
                E_INTERNAL,
                "position encoding dependency is unavailable",
                stage=job.stage,
                sample_id=job.sample_id,
            )
        if job.stage == "attention_score":
            return calculate_scores(config, sample, encoding_entry.result)
        score_entry = self.cache.get("attention_score", job.stage_keys.attention_score)
        if score_entry is None:
            raise BusinessError(
                E_INTERNAL,
                "attention score dependency is unavailable",
                stage=job.stage,
                sample_id=job.sample_id,
            )
        return trace_sample(config, sample, encoding_entry.result, score_entry.result)

    def execute_plan(
        self,
        plan: ExecutionPlan,
        *,
        deadline_monotonic: float | None = None,
        batch_size: int | None = None,
        config: PositionEncodingConfig | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> bool:
        config = config or self.config
        start = time.monotonic()
        batch_size = batch_size or config.concurrency
        executor = ThreadPoolExecutor(max_workers=config.concurrency, thread_name_prefix="rope-diagnostic")
        try:
            for stage in STAGES:
                jobs = plan.jobs_by_stage[stage]
                for offset in range(0, len(jobs), batch_size):
                    if cancel_check is not None and cancel_check():
                        self.last_elapsed_ms = (time.monotonic() - start) * 1000.0
                        return False
                    batch = jobs[offset : offset + batch_size]
                    live: list[StageJob] = []
                    for job in batch:
                        if job.sample_id in plan.blocked_samples:
                            plan.skipped.append(
                                {
                                    "stage": job.stage,
                                    "cache_key": job.cache_key,
                                    "sample_id": job.sample_id,
                                    "scheme": job.scheme,
                                    "feature_line": job.feature_line,
                                    "source_ref": job.source_ref,
                                    "stable_sort_key": job.stable_sort_key,
                                    "reason": "upstream_stage_failed",
                                }
                            )
                        else:
                            live.append(job)
                    if not live:
                        continue
                    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                        self.last_elapsed_ms = (time.monotonic() - start) * 1000.0
                        return False
                    if cancel_check is not None and cancel_check():
                        self.last_elapsed_ms = (time.monotonic() - start) * 1000.0
                        return False
                    futures = {executor.submit(self._job_result, job, config): job for job in live}
                    self.max_observed_concurrency = max(self.max_observed_concurrency, len(futures))
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:  # isolation boundary per sample/stage
                            failure = _failure_from_exception(exc, job)
                            failure = Failure(
                                code=failure.code,
                                message=failure.message,
                                stage=failure.stage,
                                sample_id=failure.sample_id,
                                line=job.feature_line,
                                scope=failure.scope,
                                detail=failure.detail,
                            )
                            plan.failed.append(failure)
                            plan.blocked_samples.add(job.sample_id)
                            plan.blocked_reasons.setdefault(
                                job.sample_id,
                                {"stage": job.stage, "cache_key": job.cache_key, "code": failure.code},
                            )
                            self.stage_failures[(job.sample_id, job.stage, job.cache_key)] = failure
                        else:
                            entry = CacheEntry(
                                stage=job.stage,
                                cache_key=job.cache_key,
                                sample_id=job.sample_id,
                                scheme=job.scheme,
                                feature_line=job.feature_line,
                                source_ref=job.source_ref,
                                result=result,
                            )
                            self.cache.put(entry)
                            self.cache.mark_recomputed(job.stage)
                            plan.recomputed.append(job)
                    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                        self.last_elapsed_ms = (time.monotonic() - start) * 1000.0
                        return False
        finally:
            executor.shutdown(wait=True)
        self.last_elapsed_ms = (time.monotonic() - start) * 1000.0
        return plan.is_complete()

    def run_full(self, *, config: PositionEncodingConfig | None = None) -> ExecutionPlan:
        config = config or self.config
        plan = self.build_plan(config=config)
        self.execute_plan(plan, config=config)
        return plan

    def _sample_status(self, sample: AttentionSample, keys: StageKeys) -> tuple[str, list[Failure]]:
        key_map = keys.as_dict()
        failures: list[Failure] = []
        for stage in STAGES:
            cache_key = key_map[stage]
            if self.cache.get(stage, cache_key) is None:
                failure = self.stage_failures.get((sample.sample_id, stage, cache_key))
                if failure is not None:
                    failures.append(failure)
                    return "failed", failures
                return "incomplete", failures
        return "complete", failures

    def build_report(
        self,
        *,
        plan: ExecutionPlan | None = None,
        execution_audit: dict[str, Any] | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self.config
        plan = plan or self.build_plan()
        score_results: dict[str, ScoreResult] = {}
        trace_results: dict[str, TraceResult] = {}
        sample_rows: list[dict[str, Any]] = []
        failures: list[Failure] = list(self.loader_failures)
        incomplete_samples: list[str] = []
        failed_samples: set[str] = set()
        for sample_id in sorted(self.samples):
            sample = self.samples[sample_id]
            keys = plan.stage_keys_by_sample.get(sample_id) or build_stage_keys(config, sample)
            status, current_failures = self._sample_status(sample, keys)
            failures.extend(current_failures)
            if status == "failed":
                failed_samples.add(sample_id)
                continue
            if status == "incomplete":
                incomplete_samples.append(sample_id)
                continue
            encoded_entry = self.cache.get("position_encoding", keys.position_encoding)
            score_entry = self.cache.get("attention_score", keys.attention_score)
            trace_entry = self.cache.get("error_trace", keys.error_trace)
            assert encoded_entry is not None and score_entry is not None and trace_entry is not None
            encoded = encoded_entry.result
            score_result: ScoreResult = score_entry.result
            trace_result: TraceResult = trace_entry.result
            score_results[sample_id] = score_result
            trace_results[sample_id] = trace_result
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "query_position": sample.query_position,
                    "key_positions": list(sample.key_positions),
                    "feature_line": sample.feature_line,
                    "source_ref": sample.source_ref,
                    "sample_interval": score_result.sample_interval,
                    "has_extrapolation": score_result.has_extrapolation,
                    "position_vector_summary": position_vector_summary(encoded),
                    "scores": [
                        score_entry_dict(entry)
                        for entry in sorted(
                            score_result.entries,
                            key=lambda item: (item.relative_distance, item.key_position),
                        )
                    ],
                    "dominant_dimension": score_result.dominant_dimension,
                    "dominant_contribution_abs": score_result.dominant_contribution_abs,
                    "trace": {
                        "reference_query_position": trace_result.reference_query_position,
                        "stage_max_deviations": dict(trace_result.stage_max_deviations),
                        "first_exceeded_stage": trace_result.first_exceeded_stage,
                        "external_key_traces": [
                            key_trace_dict(trace)
                            for trace in trace_result.key_traces
                            if trace.interval == "extrapolation"
                        ],
                        "violation": trace_result.violation,
                    },
                    "cache_keys": keys.as_dict(),
                    "stage_fingerprints": {
                        "position_fingerprint": encoded.position_fingerprint,
                        "score_fingerprint": score_result.score_fingerprint,
                        "trace_fingerprint": trace_result.trace_fingerprint,
                    },
                    "scheme": config.scheme,
                }
            )

        score_statistics = aggregate_score_statistics(score_results)
        error_statistics = aggregate_external_trace_statistics(trace_results, score_results)
        dimension_sensitivity = aggregate_dimension_sensitivity(score_results, config.dimension)

        violations: list[dict[str, Any]] = []
        for sample_id in sorted(trace_results):
            violation = trace_results[sample_id].violation
            if violation is not None:
                violations.append(dict(violation))
        violations.sort(
            key=lambda item: (
                -float(item["max_deviation"]),
                str(item["sample_id"]),
                int(item["relative_distance"]),
                int(item["key_position"]),
            )
        )
        for rank, violation in enumerate(violations, start=1):
            violation["rank"] = rank

        failures = _dedupe_failures(failures)
        code_counts: dict[str, int] = {}
        for failure in failures:
            code_counts[failure.code] = code_counts.get(failure.code, 0) + 1
        total_samples = len(self.samples)
        resolved_samples = len(sample_rows) + len(failed_samples)
        complete = not incomplete_samples
        summary = {
            "completed_samples": len(sample_rows),
            "failed_samples": len(failed_samples),
            "resolved_samples": resolved_samples,
            "total_valid_samples": total_samples,
            "incomplete_samples": incomplete_samples,
            "progress": (float(resolved_samples) / total_samples) if total_samples else 1.0,
            "current_scheme": config.scheme,
            "extrapolation_over_tolerance_samples": len(violations),
            "is_complete": complete,
        }
        audit = execution_audit if execution_audit is not None else (plan.audit_dict() if plan else None)
        report = {
            "schema_version": "1.0.0",
            "config": config_summary(config),
            "summary": summary,
            "position_encoding": {
                "scheme": config.scheme,
                "effective_base": _effective_base_from_cache(config, self.cache, plan),
                "description": {
                    "standard_rope": "pairwise rotation at position/base^(2i/d)",
                    "linear_interpolation": "pairwise rotation at (position/scale)/base^(2i/d)",
                    "ntk_aware": "pairwise rotation with base adjusted by scale^(d/(d-2))",
                }[config.scheme],
            },
            "reference_semantics": {
                "reference_positions": "min(position, training_context_length)",
                "reference_mask": "same retained dimension chunks as the measured encoding",
                "attention_normalization": "softmax(scores - max(scores))",
                "tolerance": config.error_tolerance,
            },
            "samples": sample_rows,
            "score_statistics": score_statistics,
            "error_statistics": error_statistics,
            "over_tolerance_samples": violations,
            "dimension_sensitivity_ranking": dimension_sensitivity,
            "failures": [failure.to_dict() for failure in failures],
            "business_code_counts": dict(sorted(code_counts.items())),
            "cache": {
                "stats": self.cache.stats(),
                "audit": audit,
            },
            "run": {
                "engine_max_observed_concurrency": self.max_observed_concurrency,
                "configured_concurrency": config.concurrency,
                **(run_metadata or {}),
            },
        }
        return report


def _effective_base_from_cache(
    config: PositionEncodingConfig,
    cache: ResultCache,
    plan: ExecutionPlan,
) -> float | None:
    for sample_id in sorted(plan.stage_keys_by_sample):
        keys = plan.stage_keys_by_sample[sample_id]
        entry = cache.get("position_encoding", keys.position_encoding)
        if entry is not None:
            return float(entry.result.effective_base)
    return None


def _dedupe_failures(failures: Iterable[Failure]) -> list[Failure]:
    unique: dict[tuple[object, ...], Failure] = {}
    for failure in failures:
        key = (
            failure.code,
            failure.stage,
            failure.sample_id,
            failure.line,
            failure.scope,
            failure.message,
        )
        unique[key] = failure
    return sorted(unique.values(), key=lambda failure: failure.sort_key())


def run_full_diagnosis(
    config: PositionEncodingConfig,
    samples: Mapping[str, AttentionSample] | Iterable[AttentionSample],
    *,
    loader_failures: Iterable[Failure] = (),
    stage_hooks: Mapping[str, Callable[[AttentionSample], None]] | None = None,
) -> dict[str, Any]:
    engine = DiagnosticEngine(
        config,
        samples,
        loader_failures=loader_failures,
        stage_hooks=stage_hooks,
    )
    plan = engine.run_full()
    return engine.build_report(plan=plan, run_metadata={"mode": "full"})
