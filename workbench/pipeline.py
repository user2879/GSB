"""End-to-end workbench orchestration."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import align_score, retrieve, retrieval_quality
from .config import AlignmentConfig
from .errors import (
    E_BOUNDARY_SENSITIVE,
    E_CACHE_CORRUPT,
    E_CHUNK_MISMATCH,
    E_CONFIG_THRESHOLD,
    E_FEATURE_NONFINITE,
    E_POINT_BUDGET_PENDING,
    E_SAMPLE_INVALID,
    E_SNAPSHOT_STALE,
    Failure,
    WorkbenchError,
)
from .features import (
    assemble_visual_artifact,
    extract_text_features,
    extract_visual_block,
    prepare_visual_context,
)
from .hashing import canonical_json, hash_json
from .injections import (
    CHUNK_NOT_DIVISIBLE,
    MISSING_ASSERTION_FIELD,
    NONFINITE_FEATURE,
    THRESHOLD_OUT_OF_RANGE,
    TRUNCATE_PNG,
    Injection,
    InjectionSet,
)
from .samples import LoadedSample, load_samples
from .scheduler import Task, run_task_batch
from .snapshot import SnapshotStore, atomic_write_json


CACHE_SCHEMA_VERSION = 3


@dataclass
class SampleState:
    sample: LoadedSample
    chunk_size: int
    visual_key: str
    text_key: str
    visual_injection: Injection | None = None
    text_injection: Injection | None = None
    text_nonfinite_injection: Injection | None = None
    align_injection: Injection | None = None
    chunk_injection: Injection | None = None
    visual: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    alignment_key: str | None = None
    alignment: dict[str, Any] | None = None
    visual_reused: bool = False
    text_reused: bool = False
    alignment_reused: bool = False
    pending_stages: set[str] = field(default_factory=set)


@dataclass
class RunOutcome:
    report: dict[str, Any]
    execution_trace: dict[str, Any]


def _enabled_injection_dict(injections: InjectionSet) -> list[dict[str, Any]]:
    return [item.to_dict() for item in injections.items if item.enabled]


def _injection_affects(item: Injection, sample: str, stage: str) -> bool:
    if not item.enabled or (item.sample != "*" and item.sample != sample):
        return False
    if item.type == TRUNCATE_PNG:
        return stage == "visual"
    if item.type == MISSING_ASSERTION_FIELD:
        return stage == "text"
    if item.type == NONFINITE_FEATURE:
        if "align" in item.stage or item.stage == "alignment":
            return stage == "alignment"
        if "text" in item.stage:
            return stage == "text"
        return stage == "visual"
    if item.type == CHUNK_NOT_DIVISIBLE:
        return stage == "visual"
    if item.type == THRESHOLD_OUT_OF_RANGE:
        return stage == "retrieval"
    return False


def _stage_injection_material(
    injections: InjectionSet, sample: str, stage: str
) -> list[dict[str, Any]]:
    return [
        item.to_dict()
        for item in injections.items
        if _injection_affects(item, sample, stage)
    ]


def _visual_cache_key(
    sample: LoadedSample,
    config: AlignmentConfig,
    chunk_size: int,
    injections: InjectionSet,
) -> str:
    params = config.stage_visual_params()
    params["chunk_size"] = chunk_size
    return hash_json(
        {
            "cache_schema": CACHE_SCHEMA_VERSION,
            "stage": "visual",
            "sample": sample.name,
            "image_fingerprint": sample.image_fingerprint,
            "regions_fingerprint": sample.regions_fingerprint,
            "params": params,
            "quantization_level": config.quantization_level,
            "effective_decimal_bits": config.effective_decimal_bits,
            "injections": _stage_injection_material(injections, sample.name, "visual"),
        }
    )


def _text_cache_key(
    sample: LoadedSample,
    config: AlignmentConfig,
    injections: InjectionSet,
) -> str:
    return hash_json(
        {
            "cache_schema": CACHE_SCHEMA_VERSION,
            "stage": "text",
            "sample": sample.name,
            "text_fingerprint": sample.text_fingerprint,
            "params": config.stage_text_params(),
            "quantization_level": config.quantization_level,
            "effective_decimal_bits": config.effective_decimal_bits,
            "injections": _stage_injection_material(injections, sample.name, "text"),
        }
    )


def _alignment_cache_key(state: SampleState, config: AlignmentConfig) -> str:
    assert state.visual_key and state.text_key
    return hash_json(
        {
            "cache_schema": CACHE_SCHEMA_VERSION,
            "stage": "alignment",
            "sample": state.sample.name,
            "visual_key": state.visual_key,
            "text_key": state.text_key,
            "params": config.stage_alignment_params(),
            "quantization_level": config.quantization_level,
            "effective_decimal_bits": config.effective_decimal_bits,
            "injections": _stage_injection_material(
                InjectionSet([]), state.sample.name, "alignment"
            )
            + (
                [state.align_injection.to_dict()]
                if state.align_injection is not None
                else []
            ),
        }
    )


def _failure_from_exception(
    sample: str,
    stage: str,
    exc: BaseException,
    injection: Injection | None = None,
) -> Failure:
    if isinstance(exc, WorkbenchError):
        code = exc.code
        message = exc.message
        details = dict(exc.details)
    else:
        code = E_SAMPLE_INVALID
        message = f"{type(exc).__name__}: {exc}"
        details = {}
    return Failure(
        sample=sample,
        code=code,
        message=message,
        stage=stage,
        parent_code=E_SAMPLE_INVALID if stage in {"visual", "text", "alignment"} else None,
        injection_id=injection.id if injection else None,
        details=details,
    )


class Workbench:
    def __init__(
        self,
        *,
        samples_dir: Path | str,
        config: AlignmentConfig | Path | str,
        output_dir: Path | str,
        injections: InjectionSet | Path | str | None = None,
        mode: str = "incremental",
        report_path: Path | str | None = None,
    ) -> None:
        self.samples_dir = Path(samples_dir)
        self.config = config if isinstance(config, AlignmentConfig) else AlignmentConfig.load(config)
        self.output_dir = Path(output_dir)
        if isinstance(injections, InjectionSet):
            self.injections = injections
        elif injections is None:
            self.injections = InjectionSet.empty()
        else:
            self.injections = InjectionSet.load(injections)
        if mode not in {"incremental", "full"}:
            raise WorkbenchError("E_CONFIG_INVALID", f"unknown run mode: {mode}")
        self.mode = mode
        self.report_path = (
            Path(report_path) if report_path is not None else self.output_dir / "report.json"
        )
        self.last_execution_trace: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        return self.run_with_trace().report

    def run_with_trace(self) -> RunOutcome:
        self.injections.reset_audit()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        failures: list[Failure] = []
        run_codes: set[str] = set()
        pending_tasks: list[dict[str, Any]] = []
        committed_versions: list[int] = []

        store = SnapshotStore(
            self.output_dir,
            config_hash=self.config.config_hash,
            full_rebuild=self.mode == "full",
        )
        if store.status == E_SNAPSHOT_STALE:
            run_codes.add(E_SNAPSHOT_STALE)

        threshold_injection = self.injections.threshold_fault("*")
        threshold_fault: dict[str, Any] | None = None
        if threshold_injection is not None:
            bad_value = threshold_injection.params.get("value", 1.5)
            try:
                bad_raw = dict(self.config.raw)
                bad_raw["similarity_threshold"] = bad_value
                AlignmentConfig.from_mapping(bad_raw)
            except WorkbenchError as exc:
                if exc.code != E_CONFIG_THRESHOLD:
                    raise
                threshold_fault = {
                    "injection_id": threshold_injection.id,
                    "injected_value": bad_value,
                    "fallback_threshold": self.config.similarity_threshold,
                    "code": E_CONFIG_THRESHOLD,
                }
                run_codes.add(E_CONFIG_THRESHOLD)
                self.injections.mark_hit(
                    threshold_injection,
                    sample="*",
                    stage="retrieval",
                    code=E_CONFIG_THRESHOLD,
                    proof={
                        "injected_value": bad_value,
                        "valid_range": [0.0, 1.0],
                        "fallback_threshold": self.config.similarity_threshold,
                    },
                )

        load_result = load_samples(self.samples_dir, self.injections)
        failures.extend(load_result.failures)
        samples = load_result.samples
        injection_context_hash = hash_json(_enabled_injection_dict(self.injections))

        states: dict[str, SampleState] = {}
        for sample in samples:
            chunk_size, chunk_injection = self.injections.sample_chunk_override(
                sample.name, self.config.chunk_size
            )
            visual_injection = self.injections.sample_nonfinite(
                sample.name, {"feature", "visual", "visual_feature"}
            )
            align_injection = self.injections.sample_nonfinite(
                sample.name, {"align", "alignment"}
            )
            text_nonfinite_injection = self.injections.sample_nonfinite(
                sample.name, {"text", "text_feature"}
            )
            state = SampleState(
                sample=sample,
                chunk_size=chunk_size,
                chunk_injection=chunk_injection,
                visual_injection=visual_injection,
                align_injection=align_injection,
                text_nonfinite_injection=text_nonfinite_injection,
                visual_key=_visual_cache_key(
                    sample, self.config, chunk_size, self.injections
                ),
                text_key=_text_cache_key(sample, self.config, self.injections),
            )
            states[sample.name] = state

        def cache_get(stage: str, key: str) -> dict[str, Any] | None:
            try:
                return store.get(stage, key)
            except ValueError:
                run_codes.add(E_CACHE_CORRUPT)
                return None

        for name in sorted(states):
            state = states[name]
            injection_forces_recompute = _stage_injection_material(
                self.injections, name, "visual"
            ) != []
            if not injection_forces_recompute:
                cached = cache_get("visual", state.visual_key)
                if cached is not None:
                    state.visual = cached
                    state.visual_reused = True
            if not _stage_injection_material(self.injections, name, "text"):
                cached = cache_get("text", state.text_key)
                if cached is not None:
                    state.text = cached
                    state.text_reused = True

        tasks: list[Task] = []
        visual_contexts: dict[str, dict[str, Any]] = {}
        visual_expected_keys: dict[str, list[str]] = {}
        for name in sorted(states):
            state = states[name]
            if state.visual is None:
                context: dict[str, Any] | None = None
                try:
                    context = prepare_visual_context(
                        state.sample,
                        self.config,
                        chunk_size=state.chunk_size,
                    )
                except WorkbenchError as exc:
                    if exc.code == E_CHUNK_MISMATCH and state.chunk_injection is not None:
                        self.injections.mark_hit(
                            state.chunk_injection,
                            sample=name,
                            stage="visual_feature",
                            code=E_CHUNK_MISMATCH,
                            proof=dict(exc.details),
                        )
                    failures.append(
                        _failure_from_exception(
                            name,
                            "visual",
                            exc,
                            state.chunk_injection if exc.code == E_CHUNK_MISMATCH else None,
                        )
                    )
                    run_codes.add(exc.code)
                if context is not None:
                    visual_contexts[name] = context
                    expected: list[str] = []
                    for row in range(context["rows"]):
                        for col in range(context["cols"]):
                            task = Task(
                                sample=name,
                                stage="visual",
                                row=row,
                                col=col,
                                tail=f"block-{row:04d}-{col:04d}",
                                points=self.config.visual_dim,
                                fn=lambda c=context, r=row, k=col, s=name: extract_visual_block(
                                    c,
                                    r,
                                    k,
                                    self.config,
                                    inject_nonfinite=states[s].visual_injection is not None,
                                ),
                            )
                            tasks.append(task)
                            expected.append(task.stable_key)
                    visual_expected_keys[name] = expected
            if state.text is None:
                assertion_count = len(state.sample.text.assertions)
                present = {
                    (assertion.entity, assertion.attribute)
                    for assertion in state.sample.text.assertions
                }
                missing = sum(
                    1
                    for entity in state.sample.text.entities
                    for attribute in self.config.required_attributes
                    if (entity.id, attribute) not in present
                )
                entry_count = max(1, assertion_count + missing)
                tasks.append(
                    Task(
                        sample=name,
                        stage="text",
                        row=0,
                        col=0,
                        tail="assertions",
                        points=self.config.text_dim * entry_count,
                        fn=lambda s=state.sample, n=name: extract_text_features(
                            s,
                            self.config,
                            inject_nonfinite=states[n].text_nonfinite_injection is not None,
                        ),
                    )
                )

        feature_batch = run_task_batch(
            tasks,
            worker_limit=self.config.thread_pool_size,
            point_budget=self.config.feature_point_budget,
        )
        remaining_budget = max(
            0, self.config.feature_point_budget - feature_batch.consumed_points
        )
        pending_tasks.extend(task.to_pending_dict() for task in feature_batch.pending)
        for task in feature_batch.pending:
            states[task.sample].pending_stages.add(task.stage)
            run_codes.add(E_POINT_BUDGET_PENDING)

        for name in sorted(visual_contexts):
            state = states[name]
            expected = visual_expected_keys[name]
            errors = [feature_batch.errors[key] for key in expected if key in feature_batch.errors]
            if errors:
                exc = errors[0]
                injection = state.visual_injection if isinstance(exc, WorkbenchError) and exc.code == E_FEATURE_NONFINITE else None
                if injection is not None:
                    self.injections.mark_hit(
                        injection,
                        sample=name,
                        stage="visual_feature",
                        code=E_FEATURE_NONFINITE,
                        proof=dict(exc.details) if isinstance(exc, WorkbenchError) else {},
                    )
                failures.append(_failure_from_exception(name, "visual", exc, injection))
                run_codes.add(getattr(exc, "code", E_SAMPLE_INVALID))
                continue
            if any(key not in feature_batch.results for key in expected):
                continue
            blocks = [feature_batch.results[key] for key in expected]
            artifact = assemble_visual_artifact(visual_contexts[name], blocks, self.config)
            store.put("visual", state.visual_key, artifact)
            state.visual = artifact

        for name in sorted(states):
            state = states[name]
            if state.text is not None:
                continue
            task_key = f"{name}|text|r000000|c000000|assertions"
            if task_key in feature_batch.results:
                artifact = feature_batch.results[task_key]
                store.put("text", state.text_key, artifact)
                state.text = artifact
            elif task_key in feature_batch.errors:
                exc = feature_batch.errors[task_key]
                injection = (
                    state.text_nonfinite_injection
                    if isinstance(exc, WorkbenchError)
                    and exc.code == E_FEATURE_NONFINITE
                    else None
                )
                if injection is not None:
                    self.injections.mark_hit(
                        injection,
                        sample=name,
                        stage="text_feature",
                        code=E_FEATURE_NONFINITE,
                        proof=dict(exc.details) if isinstance(exc, WorkbenchError) else {},
                    )
                failures.append(
                    _failure_from_exception(name, "text", exc, injection)
                )
                run_codes.add(getattr(exc, "code", E_SAMPLE_INVALID))
            elif "text" not in state.pending_stages:
                failures.append(
                    Failure(
                        sample=name,
                        code=E_SAMPLE_INVALID,
                        message="text feature task was not admitted",
                        stage="text",
                        parent_code=E_SAMPLE_INVALID,
                    )
                )
                run_codes.add(E_SAMPLE_INVALID)

        checkpoint_names = sorted(
            name
            for name, state in states.items()
            if state.visual is not None and state.text is not None
        )
        checkpoints = self._checkpoint_commits(
            store,
            checkpoint_names,
            self.config,
            injection_context_hash,
            sorted(failure.sample for failure in failures),
        )
        committed_versions.extend(checkpoints)

        align_tasks: list[Task] = []
        for name in checkpoint_names:
            state = states[name]
            state.alignment_key = _alignment_cache_key(state, self.config)
            force_recompute = state.align_injection is not None and (
                "align" in state.align_injection.stage
                or state.align_injection.stage == "alignment"
            )
            if not force_recompute:
                cached = cache_get("alignment", state.alignment_key)
                if cached is not None:
                    state.alignment = cached
                    state.alignment_reused = True
            if state.alignment is None:
                align_tasks.append(
                    Task(
                        sample=name,
                        stage="align",
                        row=0,
                        col=0,
                        tail="self-alignment",
                        points=max(self.config.visual_dim, self.config.text_dim),
                        fn=lambda s=state: self._compute_alignment(s),
                    )
                )

        align_batch = run_task_batch(
            align_tasks,
            worker_limit=self.config.thread_pool_size,
            point_budget=remaining_budget,
        )
        pending_tasks.extend(task.to_pending_dict() for task in align_batch.pending)
        for task in align_batch.pending:
            states[task.sample].pending_stages.add("alignment")
            run_codes.add(E_POINT_BUDGET_PENDING)

        for name in sorted(checkpoint_names):
            state = states[name]
            if state.alignment is not None:
                continue
            task_key = f"{name}|align|r000000|c000000|self-alignment"
            if task_key in align_batch.results:
                artifact = align_batch.results[task_key]
                assert state.alignment_key is not None
                store.put("alignment", state.alignment_key, artifact)
                state.alignment = artifact
            elif task_key in align_batch.errors:
                exc = align_batch.errors[task_key]
                injection = state.align_injection if isinstance(exc, WorkbenchError) and exc.code == E_FEATURE_NONFINITE else None
                if injection is not None:
                    self.injections.mark_hit(
                        injection,
                        sample=name,
                        stage="alignment",
                        code=E_FEATURE_NONFINITE,
                    )
                failures.append(_failure_from_exception(name, "alignment", exc, injection))
                run_codes.add(getattr(exc, "code", E_SAMPLE_INVALID))

        eligible_names = sorted(
            name
            for name, state in states.items()
            if state.visual is not None
            and state.text is not None
            and state.alignment is not None
            and not state.pending_stages
        )
        retrieval_artifact: dict[str, Any] | None = None
        retrieval_key: str | None = None
        retrieval_reused = False
        retrieval_blocked = any(
            state.pending_stages for state in states.values()
        ) or E_POINT_BUDGET_PENDING in run_codes
        if not retrieval_blocked:
            visual_artifacts = {name: states[name].visual for name in eligible_names}  # type: ignore[misc]
            text_artifacts = {name: states[name].text for name in eligible_names}  # type: ignore[misc]
            failure_signature = sorted(
                [
                    {
                        "sample": failure.sample,
                        "code": failure.code,
                        "parent_code": failure.parent_code,
                        "stage": failure.stage,
                        "injection_id": failure.injection_id,
                    }
                    for failure in failures
                ],
                key=canonical_json,
            )
            retrieval_key = hash_json(
                {
                    "cache_schema": CACHE_SCHEMA_VERSION,
                    "stage": "retrieval",
                    "eligible_samples": eligible_names,
                    "alignment_keys": [
                        states[name].alignment_key for name in eligible_names
                    ],
                    "visual_keys": [states[name].visual_key for name in eligible_names],
                    "text_keys": [states[name].text_key for name in eligible_names],
                    "failures": failure_signature,
                    "injection_context_hash": injection_context_hash,
                    "params": self.config.stage_retrieval_params(),
                    "quantization_level": self.config.quantization_level,
                    "effective_decimal_bits": self.config.effective_decimal_bits,
                }
            )
            cached_retrieval = cache_get("retrieval", retrieval_key)
            if cached_retrieval is not None:
                retrieval_artifact = cached_retrieval
                retrieval_reused = True
            else:
                retrieval_artifact = retrieve(text_artifacts, visual_artifacts, self.config)
                store.put("retrieval", retrieval_key, retrieval_artifact)
        else:
            run_codes.add(E_POINT_BUDGET_PENDING)

        report_retrieval = copy.deepcopy(retrieval_artifact) if retrieval_artifact else None
        boundary_diagnostics = self._quantization_diagnostics(
            states=states,
            eligible_names=eligible_names,
            retrieval=report_retrieval,
            baseline_config=self.config,
        )
        if boundary_diagnostics["business_codes"]:
            run_codes.update(boundary_diagnostics["business_codes"])

        if self.injections.audit()["missed_count"]:
            for item in self.injections.audit()["missed"]:
                if item["status"] == "missed":
                    run_codes.add("E_INJECTION_MISSED")

        completed_samples = sorted(
            set(eligible_names)
            | {failure.sample for failure in failures if failure.code != E_POINT_BUDGET_PENDING}
        )
        manifest = store.commit(
            completed_samples=completed_samples,
            extra={
                "injection_manifest_hash": injection_context_hash,
                "failed_samples": sorted(failure.sample for failure in failures),
                "pending_tasks": pending_tasks,
            },
        )
        committed_versions.append(int(manifest["global_version"]))

        pending_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for pending in pending_tasks:
            pending_groups.setdefault((pending["sample"], pending["stage"]), []).append(
                pending
            )
        for (sample_name, stage), grouped in sorted(pending_groups.items()):
            failures.append(
                Failure(
                    sample=sample_name,
                    code=E_POINT_BUDGET_PENDING,
                    message="stable scheduler budget exhausted before task admission",
                    stage=stage,
                    details={
                        "task_count": len(grouped),
                        "pending_points": sum(item["points"] for item in grouped),
                        "first_sort_key": grouped[0]["sort_key"],
                        "last_sort_key": grouped[-1]["sort_key"],
                    },
                )
            )
        failures.sort(
            key=lambda failure: (
                failure.sample,
                failure.code,
                failure.stage,
                canonical_json(failure.details),
            )
        )

        sample_payloads = self._sample_payloads(states, eligible_names)
        retrieval_queries = report_retrieval["queries"] if report_retrieval else []
        quality = retrieval_quality(retrieval_artifact) if retrieval_artifact else {
            "queries": 0,
            "self_hit_at_k_count": 0,
            "self_hit_at_k_rate": 0.0,
            "top1_self_hit_count": 0,
            "top1_self_hit_rate": 0.0,
            "mrr": 0.0,
            "per_query": [],
        }
        reuse_ledger = self._reuse_ledger(states, retrieval_reused, retrieval_key)
        core_payload = {
            "samples": sample_payloads,
            "retrieval_queries": retrieval_queries,
            "retrieval_quality": quality,
            "boundary_diagnostics": boundary_diagnostics,
            "failures": [failure.to_dict() for failure in failures],
        }
        core_digest = hash_json(core_payload)
        business_codes = self._business_codes(failures, run_codes)
        status = "ok"
        if failures or pending_tasks or boundary_diagnostics["business_codes"] or run_codes:
            status = "degraded"
        if not eligible_names and not failures:
            status = "failed"

        execution_trace = {
            "feature_batch": feature_batch.ledger(),
            "alignment_batch": align_batch.ledger(),
            "retrieval_executed": retrieval_key is not None,
        }
        report = {
            "schema_version": 1,
            "status": status,
            "mode": self.mode,
            "reproducibility_sha256": core_digest,
            "config": self.config.to_dict(),
            "config_hash": self.config.config_hash,
            "injection_manifest_hash": injection_context_hash,
            "inputs": {
                "samples_dir": str(self.samples_dir),
                "discovered_sample_count": len(samples) + len(load_result.failures),
                "valid_sample_count": len(samples),
                "failed_sample_count": len({failure.sample for failure in failures}),
                "sample_names": sorted(states),
                "eligible_sample_names": eligible_names,
            },
            "snapshot": store.snapshot_report(),
            "snapshot_commits": committed_versions,
            "scheduler": {
                "worker_limit": self.config.thread_pool_size,
                "point_budget": self.config.feature_point_budget,
                "consumed_points": feature_batch.consumed_points + align_batch.consumed_points,
                "pending_task_count": len(pending_tasks),
                "thread_pool_used": True,
            },
            "pending_tasks": pending_tasks,
            "failures": [failure.to_dict() for failure in failures],
            "samples": sample_payloads,
            "retrieval_index": {
                "cache_key": retrieval_key,
                "reused": retrieval_reused,
                "threshold": self.config.similarity_threshold,
                "top_k": self.config.top_k,
                "tie_breaker": "score_fixed_desc_then_sample_asc",
                "queries": retrieval_queries,
            },
            "retrieval_quality": quality,
            "boundary_diagnostics": boundary_diagnostics,
            "reuse_ledger": reuse_ledger,
            "injection_results": self.injections.audit(),
            "threshold_injection": threshold_fault,
            "business_codes": business_codes,
        }
        report["artifacts"] = self._write_artifacts(
            report=report,
            states=states,
            eligible_names=eligible_names,
            retrieval_artifact=report_retrieval,
            failures=failures,
            execution_trace=execution_trace,
        )
        atomic_write_json(self.report_path, report)
        self.last_execution_trace = execution_trace
        store.cleanup_uncommitted()
        return RunOutcome(report=report, execution_trace=execution_trace)

    def _compute_alignment(self, state: SampleState) -> dict[str, Any]:
        assert state.visual is not None and state.text is not None
        artifact = align_score(state.visual, state.text, self.config)
        if state.align_injection is not None and (
            "align" in state.align_injection.stage
            or state.align_injection.stage == "alignment"
        ):
            raise WorkbenchError(
                E_FEATURE_NONFINITE,
                "alignment score was injected as non-finite",
                {"sample": state.sample.name, "stage": "alignment"},
            )
        return artifact

    def _checkpoint_commits(
        self,
        store: SnapshotStore,
        names: list[str],
        config: AlignmentConfig,
        injection_context_hash: str,
        failures: list[str],
    ) -> list[int]:
        versions: list[int] = []
        interval = config.snapshot_commit_interval
        for end in range(interval, len(names) + 1, interval):
            manifest = store.commit(
                completed_samples=names[:end],
                extra={
                    "injection_manifest_hash": injection_context_hash,
                    "failed_samples": failures,
                    "checkpoint_stage": "features",
                },
            )
            versions.append(int(manifest["global_version"]))
        return versions

    def _sample_payloads(
        self, states: dict[str, SampleState], eligible_names: list[str]
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for name in sorted(states):
            state = states[name]
            visual_refs = (
                [entry["block_id"] for entry in state.visual["blocks"]]
                if state.visual
                else []
            )
            text_refs = (
                [entry["assertion_id"] for entry in state.text["entries"]]
                if state.text
                else []
            )
            payloads.append(
                {
                    "sample": name,
                    "status": "eligible" if name in eligible_names else "incomplete",
                    "visual_cache_key": state.visual_key,
                    "text_cache_key": state.text_key,
                    "alignment_cache_key": state.alignment_key,
                    "visual_feature_refs": visual_refs,
                    "text_feature_refs": text_refs,
                    "padding_positions": state.text["padding_positions"] if state.text else [],
                    "self_alignment": (
                        {
                            "score_fixed": state.alignment["score_fixed"],
                            "score": state.alignment["score"],
                            "contributions": state.alignment["contributions"],
                            "fixed_decimal_bits": state.alignment["fixed_decimal_bits"],
                        }
                        if state.alignment
                        else None
                    ),
                    "pending_stages": sorted(state.pending_stages),
                }
            )
        return payloads

    def _reuse_ledger(
        self,
        states: dict[str, SampleState],
        retrieval_reused: bool,
        retrieval_key: str | None,
    ) -> dict[str, Any]:
        ledger: dict[str, Any] = {}
        for stage in ("visual", "text", "alignment"):
            reused: list[dict[str, Any]] = []
            recomputed: list[dict[str, Any]] = []
            for name in sorted(states):
                state = states[name]
                artifact = {
                    "visual": state.visual,
                    "text": state.text,
                    "alignment": state.alignment,
                }[stage]
                if artifact is None:
                    continue
                key = {
                    "visual": state.visual_key,
                    "text": state.text_key,
                    "alignment": state.alignment_key,
                }[stage]
                was_reused = {
                    "visual": state.visual_reused,
                    "text": state.text_reused,
                    "alignment": state.alignment_reused,
                }[stage]
                item = {"sample": name, "cache_key": key}
                if stage == "visual":
                    item["references"] = [entry["block_id"] for entry in artifact["blocks"]]
                elif stage == "text":
                    item["references"] = [entry["assertion_id"] for entry in artifact["entries"]]
                    item["padding_positions"] = artifact["padding_positions"]
                else:
                    item["references"] = [state.visual_key, state.text_key]
                (reused if was_reused else recomputed).append(item)
            ledger[stage] = {
                "reused": reused,
                "recomputed": recomputed,
                "reused_count": len(reused),
                "recomputed_count": len(recomputed),
            }
        ledger["retrieval"] = {
            "reused": (
                [{"cache_key": retrieval_key}]
                if retrieval_reused and retrieval_key is not None
                else []
            ),
            "recomputed": (
                [{"cache_key": retrieval_key}]
                if retrieval_key is not None and not retrieval_reused
                else []
            ),
            "reused_count": 1 if retrieval_reused and retrieval_key else 0,
            "recomputed_count": 1 if retrieval_key and not retrieval_reused else 0,
        }
        return ledger

    def _quantization_diagnostics(
        self,
        *,
        states: dict[str, SampleState],
        eligible_names: list[str],
        retrieval: dict[str, Any] | None,
        baseline_config: AlignmentConfig,
    ) -> dict[str, Any]:
        levels = baseline_config.adjacent_quantization_levels()
        result: dict[str, Any] = {
            "baseline_level": baseline_config.quantization_level,
            "baseline_fixed_decimal_bits": baseline_config.effective_decimal_bits,
            "adjacent_levels": list(levels),
            "tolerance": baseline_config.diagnostic_tolerance,
            "threshold_neighborhood": baseline_config.boundary_neighborhood,
            "sensitive_count": 0,
            "sensitive": [],
            "errors": [],
            "business_codes": [],
        }
        if retrieval is None or not eligible_names or not levels:
            return result
        baseline_queries = {query["query"]: query for query in retrieval["queries"]}
        comparisons: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for level in levels:
            alt_config = baseline_config.with_quantization_level(level)
            try:
                alt_visual, alt_text, alt_alignment, alt_retrieval = self._build_level_run(
                    states, eligible_names, alt_config
                )
            except Exception as exc:
                result["errors"].append(
                    {
                        "adjacent_level": level,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            alt_queries = {query["query"]: query for query in alt_retrieval["queries"]}
            for query_name in sorted(baseline_queries):
                base_query = baseline_queries[query_name]
                alt_query = alt_queries.get(query_name)
                if alt_query is None:
                    continue
                base_all = {item["sample"]: item for item in base_query["all_candidates"]}
                alt_all = {item["sample"]: item for item in alt_query["all_candidates"]}
                base_returned = {item["sample"]: item for item in base_query["results"]}
                alt_returned = {item["sample"]: item for item in alt_query["results"]}
                for sample_name in sorted(set(base_returned) | set(alt_returned)):
                    base_item = base_all.get(sample_name)
                    alt_item = alt_all.get(sample_name)
                    if base_item is None or alt_item is None:
                        continue
                    base_rank = base_returned.get(sample_name, {}).get("rank")
                    alt_rank = alt_returned.get(sample_name, {}).get("rank")
                    threshold_flip = base_item["passes_threshold"] != alt_item["passes_threshold"]
                    rank_flip = base_rank != alt_rank
                    flipped = threshold_flip or rank_flip
                    boundary_sensitive = flipped and (
                        abs(base_item["score"] - baseline_config.similarity_threshold)
                        <= baseline_config.boundary_neighborhood
                        or abs(alt_item["score"] - baseline_config.similarity_threshold)
                        <= baseline_config.boundary_neighborhood
                    )
                    visual_diff = self._vector_difference(
                        states[sample_name].visual,
                        alt_visual[sample_name],
                    )
                    text_diff = self._vector_difference(
                        states[query_name].text,
                        alt_text[query_name],
                    )
                    score_diff = abs(base_item["score"] - alt_item["score"])
                    first_stage = None
                    if visual_diff is not None and visual_diff > baseline_config.diagnostic_tolerance:
                        first_stage = "visual_feature_quantization"
                    elif text_diff is not None and text_diff > baseline_config.diagnostic_tolerance:
                        first_stage = "text_feature_quantization"
                    elif score_diff > baseline_config.diagnostic_tolerance:
                        first_stage = "alignment_score"
                    elif flipped:
                        first_stage = "retrieval_threshold_or_order"
                    comparison = {
                        "baseline_level": baseline_config.quantization_level,
                        "adjacent_level": level,
                        "adjacent_fixed_decimal_bits": alt_config.effective_decimal_bits,
                        "baseline_score": base_item["score"],
                        "adjacent_score": alt_item["score"],
                        "baseline_score_fixed": base_item["score_fixed"],
                        "adjacent_score_fixed": alt_item["score_fixed"],
                        "score_delta": alt_item["score"] - base_item["score"],
                        "absolute_score_delta": score_diff,
                        "baseline_rank": base_rank,
                        "adjacent_rank": alt_rank,
                        "threshold_flip": threshold_flip,
                        "rank_flip": rank_flip,
                        "flipped": flipped,
                        "boundary_sensitive": boundary_sensitive,
                        "visual_vector_difference": visual_diff,
                        "text_vector_difference": text_diff,
                        "first_exceeded_stage": first_stage,
                    }
                    if sample_name in base_returned:
                        comparisons.setdefault((query_name, sample_name), []).append(
                            comparison
                        )
                    if boundary_sensitive:
                        result["sensitive"].append(
                            {
                                "query": query_name,
                                "sample": sample_name,
                                **comparison,
                            }
                        )
        for query in retrieval["queries"]:
            annotated_results: list[dict[str, Any]] = []
            for item in query["results"]:
                annotated = copy.deepcopy(item)
                annotated["quantization_comparisons"] = comparisons.get(
                    (query["query"], item["sample"]), []
                )
                annotated_results.append(annotated)
            query["results"] = annotated_results
        result["sensitive"].sort(
            key=lambda item: (
                item["query"],
                item["sample"],
                item["adjacent_level"],
        )
        )
        result["sensitive_count"] = len(result["sensitive"])
        if result["sensitive"]:
            result["business_codes"] = [E_BOUNDARY_SENSITIVE]
        return result

    def _build_level_run(
        self,
        states: dict[str, SampleState],
        names: list[str],
        config: AlignmentConfig,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, Any],
    ]:
        visuals: dict[str, dict[str, Any]] = {}
        texts: dict[str, dict[str, Any]] = {}
        alignments: dict[str, dict[str, Any]] = {}
        for name in names:
            state = states[name]
            visual = self._extract_visual_for_level(state, config)
            text = extract_text_features(state.sample, config)
            alignment = align_score(visual, text, config)
            visuals[name] = visual
            texts[name] = text
            alignments[name] = alignment
        retrieval = retrieve(texts, visuals, config)
        return visuals, texts, alignments, retrieval

    def _extract_visual_for_level(
        self, state: SampleState, config: AlignmentConfig
    ) -> dict[str, Any]:
        context = prepare_visual_context(
            state.sample, config, chunk_size=state.chunk_size
        )
        blocks = [
            extract_visual_block(context, row, col, config)
            for row in range(context["rows"])
            for col in range(context["cols"])
        ]
        return assemble_visual_artifact(context, blocks, config)

    def _vector_difference(
        self,
        left: dict[str, Any] | None,
        right: dict[str, Any] | None,
    ) -> float | None:
        if left is None or right is None:
            return None
        left_entries = left.get("blocks") or left.get("entries") or []
        right_entries = right.get("blocks") or right.get("entries") or []
        if len(left_entries) != len(right_entries):
            return None
        maximum = 0.0
        for left_entry, right_entry in zip(left_entries, right_entries):
            left_scale = float(left_entry["fixed_scale"])
            right_scale = float(right_entry["fixed_scale"])
            left_values = np.asarray(left_entry["quantized"], dtype=np.float64) / left_scale
            right_values = np.asarray(right_entry["quantized"], dtype=np.float64) / right_scale
            common = min(left_values.size, right_values.size)
            if common == 0:
                continue
            maximum = max(
                maximum,
                float(np.max(np.abs(left_values[:common] - right_values[:common]))),
            )
        return maximum

    def _business_codes(self, failures: list[Failure], run_codes: set[str]) -> list[str]:
        codes = set(run_codes)
        for failure in failures:
            codes.add(failure.code)
            if failure.parent_code:
                codes.add(failure.parent_code)
        for hit in self.injections.audit()["hits"]:
            if hit.get("code"):
                codes.add(hit["code"])
        for item in self.injections.audit()["missed"]:
            if item.get("code"):
                codes.add(item["code"])
        return sorted(code for code in codes if isinstance(code, str) and code.startswith("E_"))

    def _write_artifacts(
        self,
        *,
        report: dict[str, Any],
        states: dict[str, SampleState],
        eligible_names: list[str],
        retrieval_artifact: dict[str, Any] | None,
        failures: list[Failure],
        execution_trace: dict[str, Any],
    ) -> dict[str, str]:
        artifacts_dir = self.output_dir / "artifacts"
        paths = {
            "visual_features": artifacts_dir / "visual_features.json",
            "text_features": artifacts_dir / "text_features.json",
            "alignment_scores": artifacts_dir / "alignment_scores.json",
            "retrieval_index": artifacts_dir / "retrieval_index.json",
            "failures": artifacts_dir / "failures.json",
            "execution_trace": artifacts_dir / "execution_trace.json",
        }
        atomic_write_json(
            paths["visual_features"],
            {
                "samples": {
                    name: states[name].visual
                    for name in sorted(states)
                    if states[name].visual is not None
                },
                "cache_keys": {
                    name: states[name].visual_key for name in sorted(states)
                },
            },
        )
        atomic_write_json(
            paths["text_features"],
            {
                "samples": {
                    name: states[name].text
                    for name in sorted(states)
                    if states[name].text is not None
                },
                "cache_keys": {
                    name: states[name].text_key for name in sorted(states)
                },
            },
        )
        atomic_write_json(
            paths["alignment_scores"],
            {
                "samples": {
                    name: states[name].alignment
                    for name in sorted(states)
                    if states[name].alignment is not None
                },
                "cache_keys": {
                    name: states[name].alignment_key
                    for name in sorted(states)
                    if states[name].alignment_key is not None
                },
            },
        )
        atomic_write_json(
            paths["retrieval_index"],
            retrieval_artifact
            if retrieval_artifact is not None
            else {"status": "pending", "code": E_POINT_BUDGET_PENDING, "queries": []},
        )
        atomic_write_json(
            paths["failures"],
            [failure.to_dict() for failure in failures],
        )
        atomic_write_json(paths["execution_trace"], execution_trace)
        return {
            key: str(path.relative_to(self.output_dir)) for key, path in paths.items()
        }
