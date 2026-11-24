"""End-to-end inference, scheduling, ledger commit, recovery and replay."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import atomic_write_json, canonical_json, digest_json
from .client import WorkerPool
from .errors import DiagnosticError, fail
from .invalidation import InvalidationReport
from .ledger import Ledger, LedgerAssignment, LedgerBatch, LedgerTrace
from .parameters import (
    batch_output_digest,
    parameter_digest,
    parameter_summary,
    parameters_for_worker,
)
from .planning import BatchPlan, ExecutionPlan, plan_batches
from .receipts import (
    NormalizationContext,
    NormalizedReceipt,
    decode_receipt_batch,
    encode_receipt,
    normalize_batch,
)
from .specs import SampleSpec, WorkerSpec


INJECTIONS = (
    "none",
    "receipt_corruption",
    "dependency_dangling",
    "ledger_digest_mismatch",
)


@dataclass(frozen=True)
class AssignmentExecution:
    request_id: str
    rank: int
    duplicate_refs: tuple[str, ...]
    vector_digest: str
    conclusion_digest: str
    normalized: NormalizedReceipt


@dataclass(frozen=True)
class BatchExecution:
    plan: BatchPlan
    batch_version: int
    param_version: int
    param_digest: str
    output_digest: str
    receipt_digest: str
    duration_ms: int
    wait_start_ms: float
    wait_end_ms: float
    run_start_ms: float
    run_end_ms: float
    worker_pid: int
    socket_evidence: Mapping[str, Any]
    assignments: tuple[AssignmentExecution, ...]
    source_format: str

    def to_ledger_batch(self, run_id: str) -> LedgerBatch:
        return LedgerBatch(
            batch_id=self.plan.batch_id,
            worker_id=self.plan.worker_id,
            batch_seq=self.plan.batch_seq,
            batch_version=self.batch_version,
            request_ids=self.plan.request_ids,
            arrival_seqs=self.plan.arrival_seqs,
            param_version=self.param_version,
            param_digest=self.param_digest,
            output_digest=self.output_digest,
            receipt_digest=self.receipt_digest,
            elapsed_ms=self.duration_ms,
            wait_start_ms=self.wait_start_ms,
            wait_end_ms=self.wait_end_ms,
            run_start_ms=self.run_start_ms,
            run_end_ms=self.run_end_ms,
            status="committed",
            run_id=run_id,
            worker_pid=self.worker_pid,
            assignments=tuple(
                LedgerAssignment(
                    request_id=assignment.request_id,
                    rank=assignment.rank,
                    duplicate_refs=assignment.duplicate_refs,
                    vector_digest=assignment.vector_digest,
                    conclusion_digest=assignment.conclusion_digest,
                    receipt_json=assignment.normalized.to_dict(),
                )
                for assignment in self.assignments
            ),
        )


@dataclass(frozen=True)
class SchedulerOutcome:
    results: tuple[BatchExecution, ...]
    observed_global_max: int
    observed_per_worker_max: Mapping[str, int]
    quota_ok: bool


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"
    suffix = digest_json({"value": value})[:10]
    return f"{cleaned[:80]}-{suffix}"


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.params_dir = self.root / "params"
        self.batches_dir = self.root / "batches"

    def prepare_fresh(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for managed_directory in (self.params_dir, self.batches_dir):
            if managed_directory.exists():
                shutil.rmtree(managed_directory)
        for managed_file in (self.root / "run_report.json", self.root / "replay_report.json"):
            try:
                managed_file.unlink()
            except FileNotFoundError:
                pass
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)

    def prepare_incremental(self) -> None:
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)

    def param_path(self, worker_id: str) -> Path:
        return self.params_dir / f"{_safe_filename(worker_id)}.json"

    def batch_path(self, batch_id: str) -> Path:
        return self.batches_dir / f"{_safe_filename(batch_id)}.json"

    def write_parameter_summary(self, worker: WorkerSpec, param_digest: str) -> Path:
        path = self.param_path(worker.worker_id)
        value = parameter_summary(worker, param_digest)
        value.update(
            {
                "schema": "receipt-diag-param-summary-v1",
                "param_digest_path": str(path),
                "host": worker.host,
                "port": worker.port,
            }
        )
        atomic_write_json(path, value)
        return path

    def write_batch(self, execution: BatchExecution) -> Path:
        output_path = self.batch_path(execution.plan.batch_id)
        value = {
            "schema": "receipt-diag-batch-summary-v1",
            "batch_id": execution.plan.batch_id,
            "batch_version": execution.batch_version,
            "worker_id": execution.plan.worker_id,
            "batch_seq": execution.plan.batch_seq,
            "request_ids": list(execution.plan.request_ids),
            "arrival_seqs": list(execution.plan.arrival_seqs),
            "param_version": execution.param_version,
            "param_digest": execution.param_digest,
            "param_digest_path": str(self.param_path(execution.plan.worker_id)),
            "output_digest": execution.output_digest,
            "output_summary_path": str(output_path),
            "receipt_digest": execution.receipt_digest,
            "source_format": execution.source_format,
            "duration_ms": execution.duration_ms,
            "socket_evidence": dict(execution.socket_evidence),
            "assignments": [
                {
                    "request_id": assignment.request_id,
                    "rank": assignment.rank,
                    "duplicate_refs": list(assignment.duplicate_refs),
                    "vector_digest": assignment.vector_digest,
                    "conclusion_digest": assignment.conclusion_digest,
                }
                for assignment in execution.assignments
            ],
        }
        atomic_write_json(output_path, value)
        return output_path

    def remove_batch(self, batch_id: str) -> None:
        path = self.batch_path(batch_id)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def write_exit_report(self, report: Mapping[str, Any]) -> Path:
        path = self.root / "run_report.json"
        atomic_write_json(path, report)
        return path

    def write_replay_report(self, report: Mapping[str, Any]) -> Path:
        path = self.root / "replay_report.json"
        atomic_write_json(path, report)
        return path


class DiagnosticsEngine:
    def __init__(
        self,
        spec: SampleSpec,
        artifact_dir: str | Path,
        *,
        receipt_format: str | None = None,
    ) -> None:
        self.spec = spec
        self.plan = plan_batches(spec)
        self.receipt_format = receipt_format or spec.rules.receipt_format
        self.artifacts = ArtifactStore(artifact_dir)

    def run_id(self, purpose: str) -> str:
        return digest_json(
            {
                "schema": "receipt-diag-run-v1",
                "purpose": purpose,
                "plan": self.plan.fingerprint,
                "spec": self.spec.fingerprint(),
                "receipt_format": self.receipt_format,
            }
        )

    def _validate_receipt_injected(
        self,
        payloads: list[str],
        injection: str,
    ) -> list[str]:
        if injection in ("none", "ledger_digest_mismatch"):
            return payloads
        if injection == "receipt_corruption":
            document = json.loads(payloads[0])
            document["fields"][0]["length"] = int(document["fields"][0]["length"]) + 1
            payloads[0] = canonical_json(document)
            return payloads
        if injection == "dependency_dangling":
            if self.receipt_format != "r3":
                fail(
                    "E_CONFIG",
                    "inject.dependency_dangling",
                    "dependency dangling injection requires r3 receipts",
                )
            document = json.loads(payloads[0])
            duplicate_field = next(
                field for field in document["fields"] if field["name"] == "duplicate_refs"
            )
            duplicate_field["value"] = list(duplicate_field["value"]) + [
                "injected-missing-request"
            ]
            duplicate_field["length"] = len(
                canonical_json(duplicate_field["value"]).encode("utf-8")
            )
            payloads[0] = canonical_json(document)
            return payloads
        fail("E_CONFIG", "inject", f"unsupported receipt injection: {injection}")

    def _build_batch_execution(
        self,
        batch: BatchPlan,
        worker_response: Mapping[str, Any],
        *,
        batch_version: int,
        wait_start_ms: float,
        wait_end_ms: float,
        run_start_ms: float,
        run_end_ms: float,
        injection: str,
    ) -> BatchExecution:
        worker = self.spec.worker_by_id[batch.worker_id]
        local_parameters = parameters_for_worker(worker)
        local_param_digest = parameter_digest(worker, local_parameters)
        response_param_digest = str(worker_response.get("param_digest", ""))
        if response_param_digest != local_param_digest:
            fail(
                "E_WORKER",
                f"worker:{worker.worker_id}/param_digest",
                f"worker returned {response_param_digest!r}, expected {local_param_digest!r}",
            )

        raw_outputs = worker_response.get("outputs")
        if not isinstance(raw_outputs, list):
            fail("E_WORKER", f"worker:{worker.worker_id}/outputs", "expected an array")
        output_map: dict[str, str] = {}
        for item in raw_outputs:
            if not isinstance(item, Mapping):
                fail("E_WORKER", f"worker:{worker.worker_id}/outputs", "invalid output row")
            request_id = item.get("request_id")
            vector_digest = item.get("vector_digest")
            if not isinstance(request_id, str) or not isinstance(vector_digest, str):
                fail("E_WORKER", f"worker:{worker.worker_id}/outputs", "invalid output row")
            if request_id in output_map:
                fail("E_WORKER", f"worker:{worker.worker_id}/outputs", "duplicate request output")
            output_map[request_id] = vector_digest
        if set(output_map) != set(batch.request_ids):
            fail(
                "E_WORKER",
                f"worker:{worker.worker_id}/outputs",
                "worker outputs do not match batch request_ids",
            )

        expected_output_digest = batch_output_digest(
            batch_id=batch.batch_id,
            request_ids=batch.request_ids,
            param_version=worker.param_version,
            param_digest=local_param_digest,
            vector_digests=[(request_id, output_map[request_id]) for request_id in batch.request_ids],
        )
        response_output_digest = str(worker_response.get("output_digest", ""))
        if response_output_digest != expected_output_digest:
            fail(
                "E_WORKER",
                f"worker:{worker.worker_id}/output_digest",
                "output digest mismatch between worker and coordinator",
            )
        duration_ms = worker_response.get("duration_ms")
        if type(duration_ms) is not int or duration_ms < 0:
            fail("E_WORKER", f"worker:{worker.worker_id}/duration_ms", "expected non-negative integer")

        contexts: dict[str, NormalizationContext] = {}
        for rank, request in enumerate(batch.requests):
            contexts[request.request_id] = NormalizationContext(
                worker_id=worker.worker_id,
                param_version=worker.param_version,
                duration_ms=duration_ms,
                rank=rank,
                duplicate_refs=request.duplicate_refs,
                param_digest=local_param_digest,
            )

        payloads: list[str] = []
        for request in batch.requests:
            values: dict[str, Any] = {
                "batch_id": batch.batch_id,
                "request_id": request.request_id,
                "output_digest": expected_output_digest,
                "param_version": worker.param_version,
                "duration_ms": duration_ms,
                "worker_id": worker.worker_id,
                "rank": contexts[request.request_id].rank,
                "duplicate_refs": list(request.duplicate_refs),
                "param_digest": local_param_digest,
            }
            payloads.append(encode_receipt(self.receipt_format, values))
        payloads = self._validate_receipt_injected(payloads, injection)
        decoded = decode_receipt_batch(
            payloads,
            known_request_ids=self.spec.request_by_id.keys(),
            batch_size=len(batch.requests),
        )
        normalized = normalize_batch(decoded, contexts)
        if len(normalized) != len(batch.requests):
            fail("E_FIELD", "receipt", "receipt count does not match batch size")
        by_request = {receipt.request_id: receipt for receipt in normalized}
        if set(by_request) != set(batch.request_ids):
            fail("E_FIELD", "receipt.request_id", "normalized request set does not match batch")
        for rank, request in enumerate(batch.requests):
            receipt = by_request[request.request_id]
            if receipt.rank != rank:
                fail(
                    "E_FIELD",
                    f"receipt[{request.request_id}].rank",
                    f"rank {receipt.rank!r} does not match plan rank {rank}",
                )
            if tuple(receipt.duplicate_refs) != request.duplicate_refs:
                fail(
                    "E_FIELD",
                    f"receipt[{request.request_id}].duplicate_refs",
                    "normalized duplicate refs do not match authoritative request",
                )

        assignments = tuple(
            AssignmentExecution(
                request_id=request.request_id,
                rank=rank,
                duplicate_refs=request.duplicate_refs,
                vector_digest=output_map[request.request_id],
                conclusion_digest=by_request[request.request_id].attribution_digest(),
                normalized=by_request[request.request_id],
            )
            for rank, request in enumerate(batch.requests)
        )
        receipt_digest = digest_json(
            {
                "schema": "receipt-diag-batch-receipt-v1",
                "batch_id": batch.batch_id,
                "worker_id": worker.worker_id,
                "batch_seq": batch.batch_seq,
                "param_version": worker.param_version,
                "param_digest": local_param_digest,
                "output_digest": expected_output_digest,
                "assignments": [
                    {
                        "request_id": assignment.request_id,
                        "rank": assignment.rank,
                        "duplicate_refs": list(assignment.duplicate_refs),
                        "conclusion_digest": assignment.conclusion_digest,
                    }
                    for assignment in assignments
                ],
            }
        )
        evidence = {
            "transport": worker_response.get("transport", "tcp-loopback"),
            "host": worker_response.get("host"),
            "port": worker_response.get("port"),
            "peer": worker_response.get("peer"),
            "pid": worker_response.get("pid"),
        }
        return BatchExecution(
            plan=batch,
            batch_version=batch_version,
            param_version=worker.param_version,
            param_digest=local_param_digest,
            output_digest=expected_output_digest,
            receipt_digest=receipt_digest,
            duration_ms=duration_ms,
            wait_start_ms=wait_start_ms,
            wait_end_ms=wait_end_ms,
            run_start_ms=run_start_ms,
            run_end_ms=run_end_ms,
            worker_pid=int(worker_response.get("pid", 0)),
            socket_evidence=evidence,
            assignments=assignments,
            source_format=self.receipt_format,
        )

    def _initial_trace(
        self,
        execution: BatchExecution,
        run_id: str,
        *,
        reason: str = "initial_execution",
    ) -> LedgerTrace:
        path = [f"worker:{execution.plan.worker_id}", f"batch:{execution.plan.batch_id}"]
        path.extend(f"request:{request_id}" for request_id in sorted(execution.plan.request_ids))
        return LedgerTrace(
            run_id=run_id,
            worker_id=execution.plan.worker_id,
            batch_id=execution.plan.batch_id,
            reason=reason,
            dependency_path=tuple(path),
            request_ids=tuple(sorted(execution.plan.request_ids)),
            meta={"batch_version": execution.batch_version},
        )

    def _invalidation_trace(
        self,
        execution: BatchExecution,
        run_id: str,
        report: InvalidationReport,
    ) -> LedgerTrace:
        entry = report.entry_by_batch_id[execution.plan.batch_id]
        return LedgerTrace(
            run_id=run_id,
            worker_id=execution.plan.worker_id,
            batch_id=execution.plan.batch_id,
            reason=entry.reasons[0] if entry.reasons else "dependency_invalidated",
            dependency_path=entry.dependency_path,
            request_ids=entry.request_ids,
            meta={
                "all_reasons": list(entry.reasons),
                "direct": entry.direct,
                "propagation": [dict(step) for step in entry.propagation],
            },
        )

    def _execute_schedule(
        self,
        batches: Sequence[BatchPlan],
        batch_versions: Mapping[str, int],
        *,
        worker_pool: WorkerPool,
        ledger: Ledger | None,
        run_id: str,
        injection: str,
        crash_after_commits: int | None,
        invalidation: InvalidationReport | None,
        write_artifacts: bool,
        trace_reason: str = "initial_execution",
    ) -> SchedulerOutcome:
        global_limit = self.spec.rules.global_concurrency
        global_semaphore = threading.Semaphore(global_limit)
        worker_semaphores = {
            worker.worker_id: threading.Semaphore(worker.concurrency)
            for worker in self.spec.workers
        }
        state_lock = threading.Lock()
        active_global = 0
        active_per_worker: dict[str, int] = {
            worker.worker_id: 0 for worker in self.spec.workers
        }
        observed_global_max = 0
        observed_per_worker_max = {worker_id: 0 for worker_id in active_per_worker}
        commit_count = 0
        scheduler_start_ns = time.monotonic_ns()

        def elapsed_ms() -> float:
            return round((time.monotonic_ns() - scheduler_start_ns) / 1_000_000.0, 3)

        results: list[BatchExecution] = []

        def execute_one(batch: BatchPlan) -> BatchExecution:
            nonlocal active_global, commit_count, observed_global_max
            wait_start_ms = elapsed_ms()
            worker_semaphore = worker_semaphores[batch.worker_id]
            with worker_semaphore:
                with global_semaphore:
                    wait_end_ms = elapsed_ms()
                    with state_lock:
                        active_global += 1
                        active_per_worker[batch.worker_id] += 1
                        observed_global_max = max(observed_global_max, active_global)
                        observed_per_worker_max[batch.worker_id] = max(
                            observed_per_worker_max[batch.worker_id],
                            active_per_worker[batch.worker_id],
                        )
                    run_start_ms = elapsed_ms()
                    try:
                        worker_response = worker_pool.forward_batch(
                            batch.batch_id,
                            batch.worker_id,
                            [
                                {
                                    "request_id": request.request_id,
                                    "arrival_seq": request.arrival_seq,
                                    "input": list(request.input_vector),
                                }
                                for request in batch.requests
                            ],
                        )
                        run_end_ms = elapsed_ms()
                        execution = self._build_batch_execution(
                            batch,
                            worker_response,
                            batch_version=batch_versions.get(batch.batch_id, 1),
                            wait_start_ms=wait_start_ms,
                            wait_end_ms=wait_end_ms,
                            run_start_ms=run_start_ms,
                            run_end_ms=run_end_ms,
                            injection=injection,
                        )
                        if write_artifacts:
                            self.artifacts.write_batch(execution)
                        if ledger is not None:
                            trace = (
                                self._invalidation_trace(execution, run_id, invalidation)
                                if invalidation is not None
                                else self._initial_trace(execution, run_id, reason=trace_reason)
                            )
                            ledger.commit_batch(execution.to_ledger_batch(run_id), [trace])
                            with state_lock:
                                commit_count += 1
                                current_commits = commit_count
                            if (
                                crash_after_commits is not None
                                and current_commits >= crash_after_commits
                            ):
                                # Deliberate hard termination for the recovery drill.
                                # SQLite is already durably committed for this batch.
                                os._exit(137)
                        return execution
                    finally:
                        with state_lock:
                            active_global -= 1
                            active_per_worker[batch.worker_id] -= 1

        if not batches:
            return SchedulerOutcome(
                results=(),
                observed_global_max=0,
                observed_per_worker_max=observed_per_worker_max,
                quota_ok=True,
            )

        failure: BaseException | None = None
        with ThreadPoolExecutor(
            max_workers=min(global_limit, max(1, len(batches))),
            thread_name_prefix="receipt-batch",
        ) as executor:
            futures: dict[Future[BatchExecution], BatchPlan] = {
                executor.submit(execute_one, batch): batch for batch in batches
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except BaseException as exc:  # preserve the first stable worker/field error
                    if failure is None:
                        failure = exc
                    for pending in futures:
                        if pending is not future:
                            pending.cancel()
        if failure is not None:
            raise failure
        results.sort(key=lambda item: (item.plan.worker_id, item.plan.batch_seq, item.plan.batch_id))
        quota_ok = (
            observed_global_max <= global_limit
            and all(
                observed_per_worker_max[worker.worker_id] <= worker.concurrency
                for worker in self.spec.workers
            )
        )
        if not quota_ok:
            fail(
                "E_CONFIG",
                "scheduler.concurrency",
                f"observed global={observed_global_max}, per_worker={observed_per_worker_max}",
            )
        return SchedulerOutcome(
            results=tuple(results),
            observed_global_max=observed_global_max,
            observed_per_worker_max=dict(observed_per_worker_max),
            quota_ok=quota_ok,
        )

    def _write_parameter_artifacts(self, worker_pool: WorkerPool) -> None:
        for worker in self.spec.workers:
            parameters = parameters_for_worker(worker)
            digest = parameter_digest(worker, parameters)
            self.artifacts.write_parameter_summary(worker, digest)

    def _report_payload(
        self,
        *,
        run_id: str,
        purpose: str,
        outcome: SchedulerOutcome,
        ledger: Ledger,
        invalidation: InvalidationReport | None,
        recovery: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        batches = ledger.batches_as_dict()
        batch_rows = [
            {
                "batch_id": batch_id,
                "worker_id": record["worker_id"],
                "batch_seq": record["batch_seq"],
                "batch_version": record["batch_version"],
                "param_digest": record["param_digest"],
                "output_digest": record["output_digest"],
                "receipt_digest": record["receipt_digest"],
                "wait_start_ms": record["wait_start_ms"],
                "wait_end_ms": record["wait_end_ms"],
                "elapsed_ms": record["elapsed_ms"],
                "param_digest_path": str(self.artifacts.param_path(record["worker_id"])),
                "output_summary_path": str(self.artifacts.batch_path(batch_id)),
            }
            for batch_id, record in sorted(
                batches.items(),
                key=lambda pair: (
                    pair[1]["worker_id"],
                    pair[1]["batch_seq"],
                    pair[0],
                ),
            )
        ]
        return {
            "schema": "receipt-diag-exit-report-v1",
            "purpose": purpose,
            "ok": True,
            "run_id": run_id,
            "plan_fingerprint": self.plan.fingerprint,
            "spec_fingerprint": self.spec.fingerprint(),
            "receipt_format": self.receipt_format,
            "ledger_content_digest": ledger.content_digest,
            "ledger_snapshot_version": ledger.snapshot_version,
            "completed_batches": len(batches),
            "planned_batches": len(self.plan.batches),
            "concurrency": {
                "global_limit": self.spec.rules.global_concurrency,
                "observed_global_max": outcome.observed_global_max,
                "per_worker_limit": {
                    worker.worker_id: worker.concurrency for worker in self.spec.workers
                },
                "observed_per_worker_max": dict(outcome.observed_per_worker_max),
                "quota_ok": outcome.quota_ok,
            },
            "batches": batch_rows,
            "invalidation": invalidation.to_dict() if invalidation is not None else None,
            "recovery": dict(recovery) if recovery is not None else None,
            "artifacts": {
                "root": str(self.artifacts.root),
                "params": str(self.artifacts.params_dir),
                "batches": str(self.artifacts.batches_dir),
            },
        }

    def run(
        self,
        ledger: Ledger,
        *,
        injection: str = "none",
        crash_after_commits: int | None = None,
    ) -> dict[str, Any]:
        if injection not in INJECTIONS:
            fail("E_CONFIG", "inject", f"expected one of {', '.join(INJECTIONS)}")
        if crash_after_commits is not None and crash_after_commits < 1:
            fail("E_CONFIG", "crash_after_commits", "must be a positive integer")
        run_id = self.run_id("initial")
        self.artifacts.prepare_fresh()
        ledger.begin_session(
            run_id=run_id,
            plan_fingerprint=self.plan.fingerprint,
            spec_fingerprint=self.spec.fingerprint(),
            receipt_format=self.receipt_format,
            total_count=len(self.plan.batches),
            reset=True,
            new_generation=True,
        )
        with WorkerPool(self.spec) as worker_pool:
            self._write_parameter_artifacts(worker_pool)
            outcome = self._execute_schedule(
                self.plan.sorted_batches(),
                {batch.batch_id: 1 for batch in self.plan.batches},
                worker_pool=worker_pool,
                ledger=ledger,
                run_id=run_id,
                injection=injection,
                crash_after_commits=crash_after_commits,
                invalidation=None,
                write_artifacts=True,
            )
        if injection == "ledger_digest_mismatch":
            ledger.corrupt_content_digest_for_injection()
        report = self._report_payload(
            run_id=run_id,
            purpose="initial_run",
            outcome=outcome,
            ledger=ledger,
            invalidation=None,
        )
        ledger.finalize_session(report)
        report["ledger_content_digest"] = ledger.content_digest
        report["exit_report_path"] = str(self.artifacts.root / "run_report.json")
        ledger.update_exit_report(report)
        self.artifacts.write_exit_report(report)
        return report

    def apply_invalidation(
        self,
        ledger: Ledger,
        old_spec: SampleSpec,
        old_plan: ExecutionPlan,
        report_data: InvalidationReport,
        *,
        injection: str = "none",
    ) -> dict[str, Any]:
        if led := ledger.plan_fingerprint:
            if led != old_plan.fingerprint:
                fail(
                    "E_LEDGER",
                    "ledger.plan_fingerprint",
                    f"ledger has {led!r}, expected old plan {old_plan.fingerprint!r}",
                )
        ledger_before = ledger.batches_as_dict()
        versions_before = {
            batch_id: int(record["batch_version"])
            for batch_id, record in ledger_before.items()
        }
        keep = report_data.unaffected_batch_ids
        run_id = self.run_id("invalidation")
        removed_traces = tuple(
            LedgerTrace(
                run_id=run_id,
                worker_id=entry.worker_id,
                batch_id=entry.batch_id,
                reason=entry.reasons[0] if entry.reasons else "batch_removed",
                dependency_path=entry.dependency_path,
                request_ids=entry.request_ids,
                meta={
                    "all_reasons": list(entry.reasons),
                    "propagation": [dict(step) for step in entry.propagation],
                },
            )
            for entry in report_data.removed_entries
        )
        ledger.prune_for_rebuild(
            keep_batch_ids=keep,
            run_id=run_id,
            plan_fingerprint=self.plan.fingerprint,
            spec_fingerprint=self.spec.fingerprint(),
            receipt_format=self.receipt_format,
            total_count=len(self.plan.batches),
            removed_traces=removed_traces,
        )
        self.artifacts.prepare_incremental()
        for entry in report_data.removed_entries:
            self.artifacts.remove_batch(entry.batch_id)
        rerun_batches = tuple(
            self.plan.by_id[batch_id]
            for batch_id in report_data.rerun_batch_ids
            if batch_id in self.plan.by_id
        )
        versions = {
            batch.batch_id: versions_before.get(batch.batch_id, 0) + 1
            for batch in rerun_batches
        }
        with WorkerPool(self.spec) as worker_pool:
            self._write_parameter_artifacts(worker_pool)
            outcome = self._execute_schedule(
                rerun_batches,
                versions,
                worker_pool=worker_pool,
                ledger=ledger,
                run_id=run_id,
                injection=injection,
                crash_after_commits=None,
                invalidation=report_data,
                write_artifacts=True,
                trace_reason="invalidation_rerun",
            )
        if injection == "ledger_digest_mismatch":
            ledger.corrupt_content_digest_for_injection()
        after = ledger.batches_as_dict()
        for batch_id in keep:
            if ledger_before[batch_id] != after.get(batch_id):
                fail(
                    "E_LEDGER",
                    f"ledger.batches.{batch_id}",
                    "unaffected batch changed during invalidation",
                )
        report = self._report_payload(
            run_id=run_id,
            purpose="invalidation_rerun",
            outcome=outcome,
            ledger=ledger,
            invalidation=report_data,
        )
        ledger.finalize_session(report)
        report["ledger_content_digest"] = ledger.content_digest
        report["exit_report_path"] = str(self.artifacts.root / "run_report.json")
        ledger.update_exit_report(report)
        self.artifacts.write_exit_report(report)
        return report

    def _verify_ledger_ownership(self, ledger: Ledger) -> None:
        records = ledger.batches_as_dict()
        for batch in self.plan.sorted_batches():
            record = records.get(batch.batch_id)
            if record is None:
                self._replay_fail(batch.batch_id, "", "missing_batch", "ledger", "absent")
            comparisons = {
                "worker_id": (record["worker_id"], batch.worker_id),
                "batch_seq": (record["batch_seq"], batch.batch_seq),
                "request_ids": (tuple(record["request_ids"]), batch.request_ids),
                "arrival_seqs": (tuple(record["arrival_seqs"]), batch.arrival_seqs),
            }
            for field, (actual, expected) in comparisons.items():
                if actual != expected:
                    self._replay_fail(batch.batch_id, "", field, actual, expected)
            assignment_map = {
                assignment["request_id"]: assignment for assignment in record["assignments"]
            }
            if set(assignment_map) != set(batch.request_ids):
                missing = sorted(set(batch.request_ids) - set(assignment_map))
                extra = sorted(set(assignment_map) - set(batch.request_ids))
                self._replay_fail(
                    batch.batch_id,
                    (missing or extra or [""])[0],
                    "request_ids",
                    sorted(assignment_map),
                    sorted(batch.request_ids),
                )
            for rank, request in enumerate(batch.requests):
                assignment = assignment_map[request.request_id]
                if assignment["rank"] != rank:
                    self._replay_fail(
                        batch.batch_id,
                        request.request_id,
                        "rank",
                        assignment["rank"],
                        rank,
                    )
                if tuple(sorted(assignment["duplicate_refs"])) != request.duplicate_refs:
                    self._replay_fail(
                        batch.batch_id,
                        request.request_id,
                        "duplicate_refs",
                        sorted(assignment["duplicate_refs"]),
                        list(request.duplicate_refs),
                    )

    @staticmethod
    def _replay_fail(
        batch_id: str,
        request_id: str,
        field: str,
        actual: Any,
        expected: Any,
    ) -> "None":
        fail(
            "E_REPLAY",
            f"{batch_id}/{request_id or '*'}/{field}",
            f"actual={actual!r}, expected={expected!r}",
        )

    def recovery_status(self, ledger: Ledger) -> dict[str, Any]:
        status = ledger.recovery_status()
        unknown = []
        if ledger.plan_fingerprint == self.plan.fingerprint:
            committed = set(status["committed_batch_ids"])
            unknown = [
                batch.batch_id
                for batch in self.plan.sorted_batches()
                if batch.batch_id not in committed
            ]
        return {
            **status,
            "unknown_batch_ids": unknown,
            "requires_rerun": bool(unknown),
            "plan_matches_ledger": ledger.plan_fingerprint == self.plan.fingerprint,
        }

    def recover(self, ledger: Ledger) -> dict[str, Any]:
        status = self.recovery_status(ledger)
        if status["session_state"] == "completed":
            return {
                "schema": "receipt-diag-recovery-v1",
                "recovered": False,
                "reason": "session already completed",
                "status": status,
            }
        if not status["plan_matches_ledger"]:
            fail(
                "E_LEDGER",
                "ledger.plan_fingerprint",
                "current sample plan differs from the interrupted ledger; use invalidation",
            )
        unknown = tuple(status["unknown_batch_ids"])
        committed_ids = set(status["committed_batch_ids"])
        extra = committed_ids - {batch.batch_id for batch in self.plan.batches}
        if extra:
            fail(
                "E_REPLAY",
                f"{sorted(extra)[0]}/*/batch_id",
                "committed batch is absent from the recovery plan",
            )
        run_id = self.run_id("initial")
        ledger.resume_session(run_id, len(self.plan.batches))
        rerun_batches = tuple(
            batch for batch in self.plan.sorted_batches() if batch.batch_id in set(unknown)
        )
        versions = ledger.batch_versions()
        self.artifacts.prepare_incremental()
        with WorkerPool(self.spec) as worker_pool:
            self._write_parameter_artifacts(worker_pool)
            outcome = self._execute_schedule(
                rerun_batches,
                {batch.batch_id: versions.get(batch.batch_id, 1) for batch in rerun_batches},
                worker_pool=worker_pool,
                ledger=ledger,
                run_id=run_id,
                injection="none",
                crash_after_commits=None,
                invalidation=None,
                write_artifacts=True,
                trace_reason="recovery_rerun",
            )
        self._verify_ledger_ownership(ledger)
        replay_verification = self.replay(ledger, write_report=False)
        recovery_info = {
            "recovered": True,
            "prior_unclean_termination": bool(status["prior_unclean_termination"]),
            "unknown_batch_ids": list(unknown),
            "rerun_batch_ids": list(unknown),
            "can_emit_previous_exit_report": False,
            "replay_verified_batches": list(replay_verification["verified_batches"]),
        }
        report = self._report_payload(
            run_id=run_id,
            purpose="crash_recovery",
            outcome=outcome,
            ledger=ledger,
            invalidation=None,
            recovery=recovery_info,
        )
        ledger.finalize_session(report)
        report["ledger_content_digest"] = ledger.content_digest
        report["exit_report_path"] = str(self.artifacts.root / "run_report.json")
        ledger.update_exit_report(report)
        self.artifacts.write_exit_report(report)
        return report

    def replay(self, ledger: Ledger, *, write_report: bool = True) -> dict[str, Any]:
        self._verify_ledger_ownership(ledger)
        records = ledger.batches_as_dict()
        versions = ledger.batch_versions()
        with WorkerPool(self.spec) as worker_pool:
            outcome = self._execute_schedule(
                self.plan.sorted_batches(),
                versions,
                worker_pool=worker_pool,
                ledger=None,
                run_id=self.run_id("replay"),
                injection="none",
                crash_after_commits=None,
                invalidation=None,
                write_artifacts=False,
            )
        for execution in outcome.results:
            record = records[execution.plan.batch_id]
            comparisons: list[tuple[str, str, Any, Any]] = [
                ("worker_id", "", record["worker_id"], execution.plan.worker_id),
                ("batch_seq", "", record["batch_seq"], execution.plan.batch_seq),
                ("batch_version", "", record["batch_version"], execution.batch_version),
                ("request_ids", "", tuple(record["request_ids"]), execution.plan.request_ids),
                ("param_version", "", record["param_version"], execution.param_version),
                ("param_digest", "", record["param_digest"], execution.param_digest),
                ("output_digest", "", record["output_digest"], execution.output_digest),
                ("receipt_digest", "", record["receipt_digest"], execution.receipt_digest),
            ]
            for field, request_id, actual, expected in comparisons:
                if actual != expected:
                    self._replay_fail(
                        execution.plan.batch_id, request_id, field, actual, expected
                    )
            ledger_assignments = {
                assignment["request_id"]: assignment
                for assignment in record["assignments"]
            }
            result_assignments = {
                assignment.request_id: assignment for assignment in execution.assignments
            }
            all_requests = sorted(set(ledger_assignments) | set(result_assignments))
            for request_id in all_requests:
                actual = ledger_assignments.get(request_id)
                expected = result_assignments.get(request_id)
                if actual is None or expected is None:
                    self._replay_fail(
                        execution.plan.batch_id,
                        request_id,
                        "request_id",
                        actual is not None,
                        expected is not None,
                    )
                for field in (
                    "rank",
                    "duplicate_refs",
                    "vector_digest",
                    "conclusion_digest",
                ):
                    actual_value = actual[field]
                    expected_value = getattr(expected, field)
                    if field == "duplicate_refs":
                        actual_value = tuple(sorted(actual_value))
                    if actual_value != expected_value:
                        self._replay_fail(
                            execution.plan.batch_id,
                            request_id,
                            field,
                            actual_value,
                            expected_value,
                        )
        report = {
            "schema": "receipt-diag-replay-report-v1",
            "ok": True,
            "plan_fingerprint": self.plan.fingerprint,
            "ledger_content_digest": ledger.content_digest,
            "verified_batches": [batch.batch_id for batch in self.plan.sorted_batches()],
            "concurrency": {
                "observed_global_max": outcome.observed_global_max,
                "observed_per_worker_max": dict(outcome.observed_per_worker_max),
                "quota_ok": outcome.quota_ok,
            },
        }
        if write_report:
            report["report_path"] = str(self.artifacts.write_replay_report(report))
        return report
