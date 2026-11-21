from __future__ import annotations

import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import InferenceConfig, load_config, sha256_canonical
from .decoder import verify_path_score, viterbi_decode
from .entities import merge_entities
from .errors import (
    E_GLOBAL_BUDGET_EXHAUSTED,
    E_INTERNAL_ERROR,
    E_LABEL_LENGTH_MISMATCH,
    E_MAX_ITERATIONS_EXCEEDED,
    E_NONFINITE_SCORE,
    E_PARAM_SHAPE_MISMATCH,
    E_SAMPLE_QUOTA_EXCEEDED,
    WorkbenchError,
    failure_record,
    failure_sort_key,
)
from .features import block_signature, emission_scores_for_positions, hash_collision_report
from .injector import InjectionSet, InjectionSpec, injection_hit, load_injections
from .loader import CorpusLoadResult, load_corpus
from .params import ParameterPackage, load_parameter_package, replace_hash_buckets, validate_runtime_bucket_shape
from .reporting import build_report, write_report_bundle
from .scheduling import (
    BudgetDecision,
    build_batch_plans,
    build_cache_events,
    cache_event_id,
    cache_events_for_executed,
    plan_budget,
    simulate_lru_cache,
)
from .snapshot import SnapshotManager
from .types import CorpusSample


def _elapsed_ms(start_ns: int) -> float:
    return round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 6)


@dataclass
class SampleExecution:
    sample_id: str
    decision: BudgetDecision
    status: str
    result: dict[str, Any] | None
    failure: dict[str, Any] | None
    injection_hits: list[dict[str, Any]] = field(default_factory=list)
    executed_cache_event_ids: list[str] = field(default_factory=list)
    emission_artifact: dict[str, Any] | None = None
    decode_artifact: dict[str, Any] | None = None
    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.status in {"success", "failed"}


@dataclass(frozen=True)
class RunOutcome:
    exit_code: int
    status: str
    report_path: Path
    dashboard_path: Path
    report: dict[str, Any]
    snapshot_manifest: dict[str, Any] | None


@dataclass
class PriorState:
    completed_ids: list[str]
    consumed_budget: int
    manifest: dict[str, Any] | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    injection_hits: list[dict[str, Any]] = field(default_factory=list)
    stage_timings: dict[str, dict[str, float]] = field(default_factory=dict)
    cache_event_ids: list[str] = field(default_factory=list)


class Workbench:
    def __init__(
        self,
        samples_dir: Path | str,
        output_dir: Path | str,
        *,
        workers: int | None = None,
        resume: bool = False,
        expected_version: int | None = None,
        delay_ms: int = 0,
        stop_after: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.samples_dir = Path(samples_dir)
        self.output_dir = Path(output_dir)
        self.workers = workers
        self.resume = bool(resume)
        self.expected_version = expected_version
        self.delay_ms = max(0, int(delay_ms))
        self.stop_after = stop_after
        self.stop_event = stop_event or threading.Event()
        self.snapshot = SnapshotManager(self.output_dir)

    def run(self) -> RunOutcome:
        run_started_ns = time.perf_counter_ns()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cfg = load_config(self.samples_dir / "inference.yaml", workers_override=self.workers)
        injections = load_injections(self.samples_dir / "injections.json")
        corpus = load_corpus(self.samples_dir / "corpus.jsonl")
        params_path = self.samples_dir / "model_params.json"
        try:
            params = load_parameter_package(params_path, cfg)
        except WorkbenchError as exc:
            return self._model_failure_outcome(run_started_ns, cfg, corpus, injections, exc, params_path)

        prior = PriorState(completed_ids=[], consumed_budget=0)
        if self.resume:
            manifest, artifacts = self.snapshot.validate_resume(
                snapshot_format_version=cfg.snapshot_format_version,
                config_hash=cfg.config_hash,
                param_version=params.model_version,
                param_hash=params.raw_hash,
                corpus_hash=corpus.corpus_hash,
                expected_version=self.expected_version,
            )
            prior = self._load_prior_state(manifest, artifacts, corpus)
        else:
            self._commit_initial(cfg, corpus, params, injections)

        active_samples = [sample for sample in corpus.samples if sample.sample_id not in set(prior.completed_ids)]
        initial_remaining = max(0, cfg.global_char_budget - prior.consumed_budget)
        decisions = plan_budget(
            active_samples,
            cfg,
            injections,
            initial_remaining=initial_remaining,
            sequence_start=len(prior.ledger) + 1,
        )
        decision_by_id = {decision.sample_id: decision for decision in decisions}
        sample_by_id = {sample.sample_id: sample for sample in corpus.samples}

        schedule_failures: dict[str, dict[str, Any]] = {}
        schedule_hits: dict[str, list[dict[str, Any]]] = {}
        for decision in decisions:
            if decision.admitted:
                continue
            injection: InjectionSpec | None = None
            if decision.injection_id:
                injection = next(
                    (item for item in injections.all if item.injection_id == decision.injection_id),
                    None,
                )
            failure = {
                "sample_id": decision.sample_id,
                "stage": "schedule",
                "code": decision.code or E_INTERNAL_ERROR,
                "exception_type": (
                    "QuotaExceededError"
                    if decision.code == E_SAMPLE_QUOTA_EXCEEDED
                    else "GlobalBudgetExhaustedError"
                ),
                "message": decision.reason,
                "injection_id": decision.injection_id,
            }
            schedule_failures[decision.sample_id] = failure
            if injection is not None:
                schedule_hits[decision.sample_id] = [
                    injection_hit(injection, decision.code or E_INTERNAL_ERROR, decision.reason)
                ]

        executions: dict[str, SampleExecution] = {}
        pending = [sample_by_id[decision.sample_id] for decision in decisions if decision.admitted]
        pending.sort(key=lambda sample: sample.sample_id)
        completed_after_start = 0
        last_commit_count = len(prior.completed_ids)
        interrupted = False
        index = 0

        def record_execution(execution: SampleExecution) -> None:
            nonlocal completed_after_start
            executions[execution.sample_id] = execution
            completed_after_start += 1
            if self.stop_after is not None and completed_after_start >= self.stop_after:
                self.stop_event.set()

        def commit_prefix(status: str, force: bool = False) -> None:
            nonlocal last_commit_count
            prefix = self._terminal_prefix(
                corpus.samples,
                prior.completed_ids,
                executions,
                schedule_failures,
            )
            if force or len(prefix) - last_commit_count >= cfg.snapshot_interval_batches:
                artifacts = self._build_snapshot_artifacts(
                    cfg,
                    corpus,
                    params,
                    injections,
                    prior,
                    prefix,
                    decisions,
                    executions,
                    schedule_failures,
                    schedule_hits,
                    status,
                )
                consumed = self._prefix_budget_consumed(prior, prefix, decisions)
                self.snapshot.commit(
                    artifacts,
                    snapshot_format_version=cfg.snapshot_format_version,
                    config_hash=cfg.config_hash,
                    param_version=params.model_version,
                    param_hash=params.raw_hash,
                    corpus_hash=corpus.corpus_hash,
                    completed_sample_ids=prefix,
                    consumed_char_budget=consumed,
                    recovery_point_sample_id=prefix[-1] if prefix else None,
                    status=status,
                )
                last_commit_count = len(prefix)

        if pending:
            with ThreadPoolExecutor(max_workers=cfg.max_workers, thread_name_prefix="ner-slot") as pool:
                inflight: dict[Future[SampleExecution], CorpusSample] = {}
                while index < len(pending) or inflight:
                    while (
                        index < len(pending)
                        and len(inflight) < cfg.max_workers
                        and not self.stop_event.is_set()
                    ):
                        sample = pending[index]
                        index += 1
                        decision = decision_by_id[sample.sample_id]
                        future = pool.submit(self._execute_sample, sample, decision, cfg, params, injections)
                        inflight[future] = sample

                    if not inflight:
                        break
                    done, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=lambda item: inflight[item].sample_id):
                        sample = inflight.pop(future)
                        try:
                            execution = future.result()
                        except BaseException as exc:
                            decision = decision_by_id[sample.sample_id]
                            execution = SampleExecution(
                                sample_id=sample.sample_id,
                                decision=decision,
                                status="failed",
                                result=None,
                                failure=failure_record(
                                    sample.sample_id,
                                    "emission",
                                    E_INTERNAL_ERROR,
                                    type(exc).__name__,
                                    str(exc),
                                ),
                            )
                        record_execution(execution)
                        commit_prefix("running")
                    if self.stop_event.is_set():
                        interrupted = True
        if index < len(pending):
            interrupted = True
        if self.stop_event.is_set() and (len(executions) + len(schedule_failures) < len(decisions)):
            interrupted = True

        final_status = "interrupted" if interrupted else "complete"
        commit_prefix(final_status, force=True)
        manifest = self.snapshot.current_manifest()
        if manifest is None:
            raise WorkbenchError(E_INTERNAL_ERROR, "snapshot", "snapshot was not committed")
        artifacts = self.snapshot.read_artifacts(manifest)
        report = self._build_final_report(
            run_started_ns,
            cfg,
            corpus,
            params,
            injections,
            manifest,
            artifacts,
        )
        report_path, dashboard_path = write_report_bundle(
            report,
            self.output_dir,
            cfg,
            ctx={"config_hash": cfg.config_hash, "param_summary": params.summary()},
        )
        return RunOutcome(
            exit_code=0,
            status=final_status,
            report_path=report_path,
            dashboard_path=dashboard_path,
            report=report,
            snapshot_manifest=manifest,
        )

    def _execute_sample(
        self,
        sample: CorpusSample,
        decision: BudgetDecision,
        cfg: InferenceConfig,
        params: ParameterPackage,
        injections: InjectionSet,
    ) -> SampleExecution:
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        task_started_ns = time.perf_counter_ns()
        timings: dict[str, float] = {}
        hits: list[dict[str, Any]] = []
        executed_cache_ids: list[str] = []
        emission_artifact: dict[str, Any] | None = None
        decode_artifact: dict[str, Any] | None = None
        current_stage = "emission"
        stage_started_ns = time.perf_counter_ns()

        try:
            emission_injections = injections.for_sample(sample.sample_id, "emission")
            hash_injections = [item for item in emission_injections if item.kind == "hash_bucket_mismatch"]
            if hash_injections:
                for injection in hash_injections:
                    hits.append(
                        injection_hit(
                            injection,
                            E_PARAM_SHAPE_MISMATCH,
                            "injected hash bucket count no longer matches emission weight tensor",
                        )
                    )
                modified = replace_hash_buckets(params, max(0, params.hash_buckets - 1))
                validate_runtime_bucket_shape(modified)

            nonfinite_injections = [item for item in emission_injections if item.kind == "nonfinite_score"]
            batch_plans = build_batch_plans(sample, cfg)
            if len(batch_plans) > cfg.max_iterations:
                raise WorkbenchError(
                    E_MAX_ITERATIONS_EXCEEDED,
                    "emission",
                    f"batch iterations {len(batch_plans)} exceed max_iterations {cfg.max_iterations}",
                )
            nonfinite_at: int | None = None
            if nonfinite_injections:
                first_positions = batch_plans[0].positions if batch_plans else (0,)
                nonfinite_at = int(first_positions[0])
                for injection in nonfinite_injections:
                    hits.append(
                        injection_hit(
                            injection,
                            E_NONFINITE_SCORE,
                            "runtime score for position was replaced with non-finite sentinel",
                            position=nonfinite_at,
                        )
                    )

            emission_matrix: list[list[int]] = [
                [0 for _ in params.labels] for _ in range(len(sample.chars))
            ]
            scored_chars = 0
            emission_batch_keys: list[str] = []
            for batch in batch_plans:
                partial, scored = emission_scores_for_positions(
                    sample,
                    params,
                    batch.positions,
                    max_scored_chars=max(0, decision.effective_quota - scored_chars),
                    nonfinite_at=nonfinite_at,
                )
                for position in batch.positions:
                    emission_matrix[position] = partial[position]
                scored_chars += scored
                emission_batch_keys.append(batch.stable_key)
                for block_key in batch.block_keys:
                    executed_cache_ids.append(
                        cache_event_id(sample.sample_id, batch.batch_index, "emission", block_key)
                    )
            timings["emission"] = _elapsed_ms(stage_started_ns)
            emission_artifact = {
                "sample_id": sample.sample_id,
                "scored_chars": scored_chars,
                "scores": emission_matrix,
                "batch_stable_keys": emission_batch_keys,
                "cache_event_ids": list(executed_cache_ids),
            }

            current_stage = "decode"
            stage_started_ns = time.perf_counter_ns()
            for batch in batch_plans:
                for block_key in batch.block_keys:
                    event_id = cache_event_id(sample.sample_id, batch.batch_index, "decode", block_key)
                    if event_id not in executed_cache_ids:
                        executed_cache_ids.append(event_id)

            decode_injections = injections.for_sample(sample.sample_id, "decode")
            truncated = [item for item in decode_injections if item.kind == "truncated_labels"]
            expected_length = len(sample.gold_labels)
            if truncated:
                expected_length = max(0, expected_length - len(truncated))
                for injection in truncated:
                    hits.append(
                        injection_hit(
                            injection,
                            E_LABEL_LENGTH_MISMATCH,
                            f"injected target label sequence length {expected_length}",
                            position=expected_length,
                        )
                    )
            decoded = viterbi_decode(
                emission_matrix,
                params.labels,
                params.transition_weights,
                params.start_weights,
                expected_length=expected_length,
                max_iterations=cfg.max_iterations,
            )
            verified_score = verify_path_score(
                decoded.labels,
                params.labels,
                emission_matrix,
                params.transition_weights,
                params.start_weights,
            )
            if verified_score != decoded.path_score:
                raise WorkbenchError(
                    E_INTERNAL_ERROR,
                    "decode",
                    f"path score verification failed: {verified_score} != {decoded.path_score}",
                )
            timings["decode"] = _elapsed_ms(stage_started_ns)
            decode_artifact = {
                "sample_id": sample.sample_id,
                "labels": list(decoded.labels),
                "path_score": decoded.path_score,
                "best_scores": [list(row) for row in decoded.best_scores],
                "backpointers": [list(row) for row in decoded.backpointers],
                "cache_event_ids": [item for item in executed_cache_ids if "|decode|" in item],
            }

            current_stage = "merge"
            stage_started_ns = time.perf_counter_ns()
            merged = merge_entities(sample.sample_id, sample.chars, decoded.labels)
            timings["merge"] = _elapsed_ms(stage_started_ns)
            result = {
                "sample_id": sample.sample_id,
                "task_key": f"{sample.sample_id}|b000000|pipeline",
                "char_count": len(sample.chars),
                "chars": list(sample.chars),
                "scored_chars": scored_chars,
                "predicted_labels": list(decoded.labels),
                "entities": [entity.as_dict() for entity in merged.entities],
                "path_score": decoded.path_score,
                "path_score_verified": True,
                "fixed_scale": params.scale,
                "position_candidates": [
                    {
                        "position": position,
                        "scores": {
                            label: int(decoded.best_scores[position][index])
                            for index, label in enumerate(params.labels)
                        },
                        "backpointer": (
                            None
                            if decoded.backpointers[position][index] < 0
                            else params.labels[decoded.backpointers[position][index]]
                        ),
                    }
                    for position in range(len(sample.chars))
                    for index in [params.label_to_index[decoded.labels[position]]]
                ],
                "repairs": [repair.as_dict() for repair in merged.repairs],
                "budget": decision.as_dict(),
                "batch_stable_keys": emission_batch_keys,
                "cache_access_count": len(executed_cache_ids),
                "cache_stable_keys": list(executed_cache_ids),
            }
            timings["total"] = _elapsed_ms(task_started_ns)
            return SampleExecution(
                sample_id=sample.sample_id,
                decision=decision,
                status="success",
                result=result,
                failure=None,
                injection_hits=hits,
                executed_cache_event_ids=executed_cache_ids,
                emission_artifact=emission_artifact,
                decode_artifact=decode_artifact,
                stage_timings_ms=timings,
            )
        except WorkbenchError as exc:
            timings[current_stage] = _elapsed_ms(stage_started_ns)
            timings["total"] = _elapsed_ms(task_started_ns)
            matching = [
                hit
                for hit in hits
                if hit.get("stage") == exc.stage and hit.get("code") == exc.code
            ]
            injection_id = matching[0]["injection_id"] if matching else None
            return SampleExecution(
                sample_id=sample.sample_id,
                decision=decision,
                status="failed",
                result=None,
                failure=exc.as_failure(sample.sample_id, injection_id=injection_id),
                injection_hits=hits,
                executed_cache_event_ids=executed_cache_ids,
                emission_artifact=emission_artifact,
                decode_artifact=decode_artifact,
                stage_timings_ms=timings,
            )
        except BaseException as exc:
            timings[current_stage] = _elapsed_ms(stage_started_ns)
            timings["total"] = _elapsed_ms(task_started_ns)
            return SampleExecution(
                sample_id=sample.sample_id,
                decision=decision,
                status="failed",
                result=None,
                failure=failure_record(
                    sample.sample_id,
                    current_stage,
                    E_INTERNAL_ERROR,
                    type(exc).__name__,
                    str(exc),
                ),
                injection_hits=hits,
                executed_cache_event_ids=executed_cache_ids,
                emission_artifact=emission_artifact,
                decode_artifact=decode_artifact,
                stage_timings_ms=timings,
            )

    def _commit_initial(
        self,
        cfg: InferenceConfig,
        corpus: CorpusLoadResult,
        params: ParameterPackage,
        injections: InjectionSet,
    ) -> None:
        load = {
            "corpus_hash": corpus.corpus_hash,
            "valid_sample_ids": [sample.sample_id for sample in corpus.samples],
            "failures": list(corpus.failures),
            "parameter_version": params.model_version,
            "parameter_hash": params.raw_hash,
            "injection_hash": injections.source_hash,
        }
        runtime = {
            "budget_ledger": [],
            "failures": [],
            "injection_hits": [],
            "stage_timings_ms": {},
            "cache_executed_event_ids": [],
        }
        self.snapshot.commit(
            {"load": load, "emission": {}, "decode": {}, "merge": {}, "runtime": runtime},
            snapshot_format_version=cfg.snapshot_format_version,
            config_hash=cfg.config_hash,
            param_version=params.model_version,
            param_hash=params.raw_hash,
            corpus_hash=corpus.corpus_hash,
            completed_sample_ids=[],
            consumed_char_budget=0,
            recovery_point_sample_id=None,
            status="running",
        )

    def _load_prior_state(
        self,
        manifest: dict[str, Any],
        artifacts: dict[str, Any],
        corpus: CorpusLoadResult,
    ) -> PriorState:
        completed = [str(item) for item in manifest.get("completed_sample_ids", [])]
        valid_ids = [sample.sample_id for sample in corpus.samples]
        if completed != valid_ids[: len(completed)]:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "snapshot",
                "completed sample list is not a stable prefix of the valid corpus order",
            )
        runtime = artifacts.get("runtime", {})
        ledger = list(runtime.get("budget_ledger", []))
        consumed = int(manifest.get("consumed_char_budget", 0))
        ledger_charge = sum(int(entry.get("allocated", 0)) for entry in ledger)
        if ledger_charge != consumed:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "snapshot",
                f"consumed budget {consumed} != ledger charges {ledger_charge}",
            )
        merge_payload = artifacts.get("merge", {})
        if not isinstance(merge_payload, dict):
            raise WorkbenchError("E_SNAPSHOT_STALE", "snapshot", "merge artifact must be an object")
        results = {
            str(sample_id): result
            for sample_id, result in merge_payload.items()
            if str(sample_id) in set(completed)
        }
        failures = [
            item
            for item in runtime.get("failures", [])
            if str(item.get("sample_id")) in set(completed)
        ]
        terminal_ids = set(results) | {str(item.get("sample_id")) for item in failures}
        if set(completed) - terminal_ids:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "snapshot",
                "snapshot lists completed samples without merge result or terminal failure",
            )
        return PriorState(
            completed_ids=completed,
            consumed_budget=consumed,
            manifest=manifest,
            artifacts=artifacts,
            results=results,
            ledger=ledger,
            failures=failures,
            injection_hits=list(runtime.get("injection_hits", [])),
            stage_timings=dict(runtime.get("stage_timings_ms", {})),
            cache_event_ids=list(runtime.get("cache_executed_event_ids", [])),
        )

    def _terminal_prefix(
        self,
        samples: Sequence[CorpusSample],
        prior_completed: Sequence[str],
        executions: dict[str, SampleExecution],
        schedule_failures: dict[str, dict[str, Any]],
    ) -> list[str]:
        prefix: list[str] = []
        prior_set = set(prior_completed)
        for sample in samples:
            sample_id = sample.sample_id
            if sample_id in prior_set:
                prefix.append(sample_id)
                continue
            execution = executions.get(sample_id)
            if execution is not None and execution.terminal:
                prefix.append(sample_id)
                continue
            if sample_id in schedule_failures:
                prefix.append(sample_id)
                continue
            break
        return prefix

    def _prefix_budget_consumed(
        self,
        prior: PriorState,
        prefix: Sequence[str],
        decisions: Sequence[BudgetDecision],
    ) -> int:
        prefix_set = set(prefix)
        new_charge = sum(
            decision.charge
            for decision in decisions
            if decision.sample_id in prefix_set and decision.sample_id not in set(prior.completed_ids)
        )
        return prior.consumed_budget + int(new_charge)

    def _build_snapshot_artifacts(
        self,
        cfg: InferenceConfig,
        corpus: CorpusLoadResult,
        params: ParameterPackage,
        injections: InjectionSet,
        prior: PriorState,
        prefix: Sequence[str],
        decisions: Sequence[BudgetDecision],
        executions: dict[str, SampleExecution],
        schedule_failures: dict[str, dict[str, Any]],
        schedule_hits: dict[str, list[dict[str, Any]]],
        status: str,
    ) -> dict[str, Any]:
        prefix_set = set(prefix)
        prior_set = set(prior.completed_ids)
        prior_emission = dict(prior.artifacts.get("emission", {})) if prior.artifacts else {}
        prior_decode = dict(prior.artifacts.get("decode", {})) if prior.artifacts else {}
        prior_merge = dict(prior.artifacts.get("merge", {})) if prior.artifacts else {}
        emission = dict(prior_emission)
        decode = dict(prior_decode)
        merge = dict(prior_merge)
        ledger = list(prior.ledger)
        failures = list(prior.failures)
        hits = list(prior.injection_hits)
        stage_timings = dict(prior.stage_timings)
        cache_event_ids = list(prior.cache_event_ids)

        decision_by_id = {decision.sample_id: decision for decision in decisions}

        for sample_id in prefix:
            if sample_id in prior_set:
                continue
            execution = executions.get(sample_id)
            if execution is not None:
                ledger.append(execution.decision.as_dict())
                if execution.emission_artifact is not None:
                    emission[sample_id] = execution.emission_artifact
                if execution.decode_artifact is not None:
                    decode[sample_id] = execution.decode_artifact
                if execution.result is not None:
                    merge[sample_id] = execution.result
                if execution.failure is not None:
                    failures.append(execution.failure)
                hits.extend(execution.injection_hits)
                stage_timings[sample_id] = execution.stage_timings_ms
                cache_event_ids.extend(execution.executed_cache_event_ids)
                continue

            failure = schedule_failures.get(sample_id)
            if failure is not None:
                failures.append(failure)
                hits.extend(schedule_hits.get(sample_id, []))
                stage_timings[sample_id] = {"schedule": 0.0, "total": 0.0}

        ledger_ids = {str(entry.get("sample_id")) for entry in ledger}
        for sample_id in prefix:
            if sample_id in prior_set or sample_id in ledger_ids:
                continue
            execution = executions.get(sample_id)
            if execution is not None:
                continue
            failure = schedule_failures.get(sample_id)
            if failure is not None:
                decision = decision_by_id.get(sample_id)
                if decision is not None:
                    ledger.append(decision.as_dict())

        ledger.sort(key=lambda item: int(item.get("sequence", 0)))
        failures.sort(key=failure_sort_key)
        hits.sort(key=lambda item: (str(item.get("sample_id")), str(item.get("stage")), str(item.get("injection_id"))))
        load = {
            "corpus_hash": corpus.corpus_hash,
            "valid_sample_ids": [sample.sample_id for sample in corpus.samples],
            "failures": list(corpus.failures),
            "parameter_version": params.model_version,
            "parameter_hash": params.raw_hash,
            "injection_hash": injections.source_hash,
        }
        runtime = {
            "status": status,
            "budget_ledger": ledger,
            "failures": failures,
            "injection_hits": hits,
            "stage_timings_ms": stage_timings,
            "cache_executed_event_ids": sorted(set(cache_event_ids)),
        }
        return {
            "load": load,
            "emission": {key: emission[key] for key in sorted(emission) if key in prefix_set or key in prior_set},
            "decode": {key: decode[key] for key in sorted(decode) if key in prefix_set or key in prior_set},
            "merge": {key: merge[key] for key in sorted(merge) if key in prefix_set or key in prior_set},
            "runtime": runtime,
        }

    def _build_final_report(
        self,
        run_started_ns: int,
        cfg: InferenceConfig,
        corpus: CorpusLoadResult,
        params: ParameterPackage,
        injections: InjectionSet,
        manifest: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> dict[str, Any]:
        merge = artifacts.get("merge", {})
        runtime = artifacts.get("runtime", {})
        failures = list(corpus.failures) + list(runtime.get("failures", []))
        failures.sort(key=failure_sort_key)
        results = [merge[key] for key in sorted(merge)]
        ledger = list(runtime.get("budget_ledger", []))
        ledger.sort(key=lambda item: int(item.get("sequence", 0)))
        hits = list(runtime.get("injection_hits", []))
        hits.sort(key=lambda item: (str(item.get("sample_id")), str(item.get("stage")), str(item.get("injection_id"))))

        all_events = build_cache_events(corpus.samples, cfg)
        executed_ids = set(runtime.get("cache_executed_event_ids", []))
        executed_events = cache_events_for_executed(corpus.samples, all_events, executed_ids)
        sample_map = {sample.sample_id: sample for sample in corpus.samples}

        def signature_for_block(block_key: str) -> str:
            sample_id, marker, index_text = block_key.partition(":block:")
            if not marker:
                raise WorkbenchError(E_INTERNAL_ERROR, "snapshot", f"bad block key: {block_key}")
            sample = sample_map[sample_id]
            start = int(index_text) * cfg.char_block_size
            end = min(len(sample.chars), start + cfg.char_block_size)
            return block_signature(sample, params, start, end)

        cache = simulate_lru_cache(executed_events, cfg.cache_block_budget, signature_for_block)
        collisions = hash_collision_report(corpus.samples, params)
        timings = dict(runtime.get("stage_timings_ms", {}))
        total_wall_ms = _elapsed_ms(run_started_ns)
        report = build_report(
            cfg=cfg,
            corpus=corpus,
            params=params,
            injections=injections,
            manifest=manifest,
            artifacts=artifacts,
            results=results,
            failures=failures,
            ledger=ledger,
            injection_hits=hits,
            cache=cache.as_dict(),
            collisions=collisions,
            stage_timings_ms=timings,
            total_wall_ms=total_wall_ms,
        )
        return report

    def _model_failure_outcome(
        self,
        run_started_ns: int,
        cfg: InferenceConfig,
        corpus: CorpusLoadResult,
        injections: InjectionSet,
        exc: WorkbenchError,
        params_path: Path,
    ) -> RunOutcome:
        from .config import sha256_file

        model_failure = failure_record("@model", "load", exc.code, exc.exception_type or type(exc).__name__, exc.detail)
        failures = list(corpus.failures) + [model_failure]
        failures.sort(key=failure_sort_key)
        placeholder_params = {
            "model_version": "@invalid",
            "model_hash": sha256_file(params_path) if params_path.exists() else "",
            "labels": [],
            "label_count": 0,
            "feature_dim": 0,
            "hash_buckets": 0,
            "fixed_decimals": cfg.fixed_decimals,
            "quantization_level": cfg.quantization_level,
            "quantization_step": 1 << cfg.quantization_level,
            "fixed_scale": 10 ** cfg.fixed_decimals,
            "start_weight_shape": [0],
            "transition_weight_shape": [0, 0],
            "emission_weight_shape": [0, 0, 0],
            "char_embedding_count": 0,
            "vocab_size": 0,
        }
        manifest = {
            "global_version": 0,
            "snapshot_format_version": cfg.snapshot_format_version,
            "config_hash": cfg.config_hash,
            "param_version": "@invalid",
            "param_hash": placeholder_params["model_hash"],
            "corpus_hash": corpus.corpus_hash,
            "completed_sample_ids": [],
            "consumed_char_budget": 0,
            "stage_versions": {},
            "artifact_sha256": {},
            "recovery_point_sample_id": None,
            "status": "model_invalid",
        }
        report = build_report(
            cfg=cfg,
            corpus=corpus,
            params=None,
            injections=injections,
            manifest=manifest,
            artifacts={"load": {"failures": failures}, "emission": {}, "decode": {}, "merge": {}, "runtime": {}},
            results=[],
            failures=failures,
            ledger=[],
            injection_hits=[],
            cache={
                "accesses": 0,
                "hits": 0,
                "misses": 0,
                "recomputations": 0,
                "eviction_count": 0,
                "evictions": [],
                "recompilation_count": 0,
                "recompilation_log": [],
                "resident_blocks": [],
                "checksum_mismatches": 0,
            },
            collisions={
                "hash_buckets": 0,
                "distinct_feature_keys": 0,
                "total_feature_hits": 0,
                "occupied_buckets": 0,
                "collision_events": 0,
                "colliding_bucket_count": 0,
                "colliding_buckets": [],
            },
            stage_timings_ms={"@model": {"load": _elapsed_ms(run_started_ns), "total": _elapsed_ms(run_started_ns)}},
            total_wall_ms=_elapsed_ms(run_started_ns),
            param_summary_override=placeholder_params,
        )
        report_path, dashboard_path = write_report_bundle(
            report,
            self.output_dir,
            cfg,
            ctx={"config_hash": cfg.config_hash, "param_summary": placeholder_params},
        )
        return RunOutcome(
            exit_code=2,
            status="model_invalid",
            report_path=report_path,
            dashboard_path=dashboard_path,
            report=report,
            snapshot_manifest=None,
        )


def load_report(path: Path | str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("report root must be an object")
    return value


def _difference_paths(left: object, right: object, path: str = "$", limit: int = 20) -> list[str]:
    if len(path) > 5000:
        return [f"{path}: nested structure exceeds comparison depth"]
    if type(left) is not type(right):
        return [f"{path}: type {type(left).__name__} != {type(right).__name__}"]
    if isinstance(left, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left:
                differences.append(f"{path}.{key}: missing on left")
            elif key not in right:
                differences.append(f"{path}.{key}: missing on right")
            else:
                differences.extend(_difference_paths(left[key], right[key], f"{path}.{key}", limit))
            if len(differences) >= limit:
                break
        return differences[:limit]
    if isinstance(left, list):
        differences = []
        if len(left) != len(right):
            differences.append(f"{path}: list length {len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(_difference_paths(left_item, right_item, f"{path}[{index}]", limit))
            if len(differences) >= limit:
                break
        return differences[:limit]
    if left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []


def compare_reports(left_path: Path | str, right_path: Path | str) -> dict[str, Any]:
    left = load_report(left_path)
    right = load_report(right_path)
    left_det = left.get("deterministic")
    right_det = right.get("deterministic")
    if left_det is None or right_det is None:
        raise ValueError("both reports must contain a deterministic section")
    differences = _difference_paths(left_det, right_det, "$.deterministic")
    left_digest = sha256_canonical(left_det)
    right_digest = sha256_canonical(right_det)
    return {
        "equal": not differences,
        "left_digest": left_digest,
        "right_digest": right_digest,
        "differences": differences,
        "left_workers": left.get("run_metadata", {}).get("max_workers"),
        "right_workers": right.get("run_metadata", {}).get("max_workers"),
    }
