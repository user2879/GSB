"""End-to-end orchestration for execution, previews, authority checks, and recovery."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from .backends import BackendManager
from .errors import GatewayError, PreviewError, ReplayError
from .ledger import (
    ActivationResult,
    LedgerStore,
    PreviewCache,
    PreviewEntry,
    PreviewSnapshot,
    RecoveryReport,
)
from .model import (
    Backend,
    RequestSample,
    Spec,
    request_stream_digest,
)
from .parameters import BackendParameters, generate_parameters
from .routing import (
    BatchPlan,
    RoutingPlan,
    batch_for_request,
    plan_routes,
)
from .scheduling import BatchExecution, QuotaScheduler, SchedulerSummary


@dataclass(frozen=True)
class RunReport:
    revision: int
    activation: ActivationResult
    total_batches: int
    committed_before: int
    newly_committed: int
    skipped_batches: int
    global_limit: int
    scheduler: SchedulerSummary
    batches: tuple[Mapping[str, Any], ...]
    ledger_content_digest: str
    recovery: RecoveryReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "operation": "run",
            "revision": self.revision,
            "spec_changed": self.activation.changed,
            "replacement_diff": self.activation.diff,
            "total_batches": self.total_batches,
            "committed_before": self.committed_before,
            "newly_committed": self.newly_committed,
            "skipped_batches": self.skipped_batches,
            "global_limit": self.global_limit,
            "scheduler": {
                "global_limit": self.global_limit,
                "max_global_observed": self.scheduler.max_global_observed,
                "max_backend_observed": dict(self.scheduler.max_backend_observed),
                "dispatched_batches": self.scheduler.dispatched_batches,
                "worker_threads": list(self.scheduler.worker_threads),
            },
            "batches": list(self.batches),
            "ledger_content_digest": self.ledger_content_digest,
            "recovery": self.recovery.to_dict(),
        }


@dataclass(frozen=True)
class AuthorityReport:
    revision: int
    replayed_batches: int
    replayed_requests: int
    canonical_bytes: int
    canonical_digest: str
    scheduler: SchedulerSummary
    ledger_content_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "operation": "authoritative-verify",
            "revision": self.revision,
            "replayed_batches": self.replayed_batches,
            "replayed_requests": self.replayed_requests,
            "preview_canonical_bytes": self.canonical_bytes,
            "preview_canonical_digest": self.canonical_digest,
            "scheduler": {
                "max_global_observed": self.scheduler.max_global_observed,
                "max_backend_observed": dict(self.scheduler.max_backend_observed),
                "dispatched_batches": self.scheduler.dispatched_batches,
                "worker_threads": list(self.scheduler.worker_threads),
            },
            "ledger_content_digest": self.ledger_content_digest,
            "consistent": True,
        }


@dataclass(frozen=True)
class ReplayReport:
    revision: int
    replayed_batches: int
    replayed_requests: int
    ledger_integrity_ok: bool
    ledger_content_digest: str
    scheduler: SchedulerSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "operation": "replay-verify",
            "revision": self.revision,
            "replayed_batches": self.replayed_batches,
            "replayed_requests": self.replayed_requests,
            "ledger_integrity_ok": self.ledger_integrity_ok,
            "ledger_content_digest": self.ledger_content_digest,
            "scheduler": {
                "max_global_observed": self.scheduler.max_global_observed,
                "max_backend_observed": dict(self.scheduler.max_backend_observed),
                "dispatched_batches": self.scheduler.dispatched_batches,
                "worker_threads": list(self.scheduler.worker_threads),
            },
            "consistent": True,
        }


@dataclass(frozen=True)
class RecoveryOutcome:
    initial: RecoveryReport
    final: RecoveryReport
    newly_committed: int
    replay: ReplayReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "operation": "recover",
            "initial_recovery": self.initial.to_dict(),
            "final_recovery": self.final.to_dict(),
            "newly_committed": self.newly_committed,
            "replay": self.replay.to_dict(),
        }


def _first_difference(left: Any, right: Any, path: str = "stats") -> str | None:
    if type(left) is not type(right):
        return path
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            differing = sorted(left_keys ^ right_keys)
            return f"{path}.{differing[0]}"
        for key in sorted(left_keys):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.length"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return path
    return None


class DiagnosticEngine:
    _ROUTE_FIELDS = (
        "client_key",
        "arrival_ms",
        "rule_id",
        "backend_id",
        "target",
        "bucket",
        "weight_bp",
        "weight_version",
        "routing_version",
        "batch_id",
        "output_digest",
        "parameter_digest",
        "receipt_digest",
        "wait_ms",
        "exec_ms",
    )

    def __init__(
        self,
        spec: Spec,
        requests: Sequence[RequestSample],
        store: LedgerStore,
        *,
        parameters: Mapping[str, BackendParameters] | None = None,
        startup_timeout: float = 8.0,
        receipt_timeout: float = 20.0,
        worker_command_factory: Callable[[Backend], list[str]] | None = None,
    ) -> None:
        self.spec = spec
        self.requests = tuple(requests)
        self.store = store
        self.parameters = dict(parameters or {})
        for backend in spec.backends:
            self.parameters.setdefault(backend.id, generate_parameters(backend))
        self.startup_timeout = startup_timeout
        self.receipt_timeout = receipt_timeout
        self.worker_command_factory = worker_command_factory

    def plan(self) -> RoutingPlan:
        return plan_routes(self.spec, self.requests)

    def _manager(self) -> BackendManager:
        return BackendManager(
            self.spec,
            self.parameters,
            startup_timeout=self.startup_timeout,
            receipt_timeout=self.receipt_timeout,
            worker_command_factory=self.worker_command_factory,
        )

    def _run_batches(
        self,
        batches: Sequence[BatchPlan],
        *,
        on_complete: Callable[[BatchExecution], None] | None = None,
    ) -> tuple[list[BatchExecution], SchedulerSummary]:
        if not batches:
            return [], SchedulerSummary(0, {}, 0, ())
        with self._manager() as manager:
            scheduler = QuotaScheduler(
                manager,
                self.spec.limits.backend_concurrency,
                self.spec.limits.global_concurrency,
                on_complete=on_complete,
            )
            return scheduler.run(batches)

    @staticmethod
    def _execution_summary(execution: BatchExecution) -> dict[str, Any]:
        return {
            "batch_id": execution.plan.batch_id,
            "rule_id": execution.plan.rule_id,
            "backend_id": execution.plan.backend_id,
            "request_ids": list(execution.plan.request_ids),
            "wait_start_ms": execution.wait_start_ms,
            "wait_end_ms": execution.wait_end_ms,
            "wait_ms": execution.wait_ms,
            "exec_start_ms": execution.exec_start_ms,
            "exec_end_ms": execution.exec_end_ms,
            "exec_ms": execution.elapsed_ms,
            "worker_elapsed_ms": execution.receipt.elapsed_ms,
            "worker_pid": execution.receipt.worker_pid,
            "parameter_digest": execution.receipt.parameter_digest,
            "receipt_digest": execution.receipt.receipt_digest,
            "output_digest": execution.receipt.output_digest,
            "thread_name": execution.thread_name,
            "global_limit": execution.global_limit,
            "backend_quota": execution.backend_quota,
        }

    def _activate(self, plan: RoutingPlan) -> ActivationResult:
        return self.store.activate_spec(self.spec, plan, self.requests)

    def run(
        self,
        *,
        on_batch: Callable[[BatchExecution], None] | None = None,
    ) -> RunReport:
        plan = self.plan()
        activation = self._activate(plan)
        committed_before = set(self.store.committed_batch_ids(self.spec.revision))
        self.store.recovery_report(plan, self.requests)
        pending = [batch for batch in plan.batches if batch.batch_id not in committed_before]
        executions: list[BatchExecution] = []

        def commit(execution: BatchExecution) -> None:
            self.store.commit_batch(execution)
            executions.append(execution)
            if on_batch is not None:
                on_batch(execution)

        scheduler_summary = SchedulerSummary(0, {}, 0, ())
        if pending:
            _, scheduler_summary = self._run_batches(pending, on_complete=commit)
        final_recovery = self.store.recovery_report(plan, self.requests)
        integrity_ok, ledger_digest = self.store.verify_ledger(self.spec.revision)
        if not integrity_ok:
            raise GatewayError(
                "ledger content digest does not match its committed content",
                code="E_LEDGER",
                field="ledger_content_digest",
            )
        return RunReport(
            revision=self.spec.revision,
            activation=activation,
            total_batches=len(plan.batches),
            committed_before=len(committed_before),
            newly_committed=len(executions),
            skipped_batches=len(committed_before),
            global_limit=self.spec.limits.global_concurrency,
            scheduler=scheduler_summary,
            batches=tuple(
                self._execution_summary(execution)
                for execution in sorted(executions, key=lambda item: item.plan.batch_id)
            ),
            ledger_content_digest=ledger_digest,
            recovery=final_recovery,
        )

    def preview(self) -> PreviewCache:
        return self.store.load_preview_cache(self.spec.revision)

    def _require_active_ledger(self) -> None:
        latest = self.store.latest_spec_row()
        if latest is None:
            raise ReplayError("ledger has no active revision", field="revision")
        if int(latest["revision"]) != self.spec.revision:
            raise ReplayError(
                f"spec revision {self.spec.revision} is not the active revision {latest['revision']}",
                field="revision",
            )
        if latest["fingerprint"] != self.spec.fingerprint:
            raise ReplayError(
                "spec fingerprint differs from the active ledger revision",
                field="spec",
            )
        if latest["request_stream_digest"] != request_stream_digest(self.requests):
            raise ReplayError(
                "request stream digest differs from the active ledger revision",
                field="request_stream_digest",
            )

    def _replay_outputs(
        self, plan: RoutingPlan
    ) -> tuple[dict[int, str], dict[str, Any], SchedulerSummary, int]:
        executions, summary = self._run_batches(plan.batches)
        outputs: dict[int, str] = {}
        batch_receipts: dict[str, Any] = {}
        for execution in executions:
            batch_receipts[execution.plan.batch_id] = execution.receipt
            for request_id in execution.plan.request_ids:
                outputs[request_id] = execution.receipt.output_digests[request_id]
        missing = [request_id for request_id in plan.request_ids if request_id not in outputs]
        if missing:
            first = min(missing)
            batch = batch_for_request(plan, first)
            raise ReplayError(
                "authoritative replay did not produce an output for every request",
                request_id=first,
                backend_id=batch.backend_id,
                batch_id=batch.batch_id,
                field="output_digest",
            )
        return outputs, batch_receipts, summary, len(executions)

    def verify_authoritative(self, preview: PreviewCache | None = None) -> AuthorityReport:
        self._require_active_ledger()
        plan = self.plan()
        ledger_cache = self.store.load_preview_cache(self.spec.revision)
        cached = preview or ledger_cache
        outputs, receipts, summary, batch_count = self._replay_outputs(plan)
        ledger_entries = {entry.request_id: entry for entry in ledger_cache.entries}
        samples = {request.request_id: request for request in self.requests}
        authority_entries: list[PreviewEntry] = []
        batch_by_request = {
            item.sample.request_id: batch
            for batch in plan.batches
            for item in batch.items
        }
        for decision in plan.decisions:
            ledger_entry = ledger_entries.get(decision.request_id)
            if ledger_entry is None:
                raise PreviewError(
                    "preview cache has no entry for a replayed request",
                    request_id=decision.request_id,
                    backend_id=decision.backend_id,
                    field="presence",
                )
            batch = batch_by_request[decision.request_id]
            authority_entries.append(
                replace(
                    ledger_entry,
                    client_key=samples[decision.request_id].client_key,
                    arrival_ms=samples[decision.request_id].arrival_ms,
                    rule_id=decision.rule_id,
                    target=decision.target,
                    backend_id=decision.backend_id,
                    bucket=decision.bucket,
                    weight_bp=decision.weight_bp,
                    weight_version=decision.weight_version,
                    routing_version=decision.routing_version,
                    batch_id=batch.batch_id,
                    output_digest=outputs[decision.request_id],
                    parameter_digest=receipts[batch.batch_id].parameter_digest,
                )
            )
        authority = self.store.snapshot_from_entries(self.spec.revision, authority_entries)
        self._compare_preview(cached.snapshot, authority)
        integrity_ok, ledger_digest = self.store.verify_ledger(self.spec.revision)
        if not integrity_ok:
            first = authority_entries[0] if authority_entries else None
            raise PreviewError(
                "ledger content digest is invalid",
                request_id=first.request_id if first else None,
                backend_id=first.backend_id if first else None,
                field="ledger_content_digest",
            )
        canonical = authority.canonical_bytes()
        import hashlib

        return AuthorityReport(
            revision=self.spec.revision,
            replayed_batches=batch_count,
            replayed_requests=len(outputs),
            canonical_bytes=len(canonical),
            canonical_digest=hashlib.sha256(canonical).hexdigest(),
            scheduler=summary,
            ledger_content_digest=ledger_digest,
        )

    def _compare_preview(
        self, preview: PreviewSnapshot, authority: PreviewSnapshot
    ) -> None:
        if preview.revision != authority.revision:
            first = authority.entries[0] if authority.entries else None
            raise PreviewError(
                f"preview revision {preview.revision} differs from authoritative revision {authority.revision}",
                request_id=first.request_id if first else None,
                backend_id=first.backend_id if first else None,
                field="revision",
            )
        preview_entries = {entry.request_id: entry for entry in preview.entries}
        authority_entries = {entry.request_id: entry for entry in authority.entries}
        request_ids = sorted(
            set(preview_entries) | set(authority_entries),
            key=lambda request_id: (
                (preview_entries.get(request_id) or authority_entries[request_id]).arrival_ms,
                request_id,
            ),
        )
        for request_id in request_ids:
            expected = preview_entries.get(request_id)
            actual = authority_entries.get(request_id)
            if expected is None or actual is None:
                present = expected or actual
                raise PreviewError(
                    "request presence differs between preview cache and authoritative replay",
                    request_id=request_id,
                    backend_id=present.backend_id if present else None,
                    field="presence",
                )
            expected_values = expected.to_dict()
            actual_values = actual.to_dict()
            for field in self._ROUTE_FIELDS:
                if expected_values[field] != actual_values[field]:
                    raise PreviewError(
                        f"preview field {field!r} differs from authoritative replay",
                        request_id=request_id,
                        backend_id=actual.backend_id,
                        batch_id=actual.batch_id,
                        field=field,
                        details={
                            "preview": expected_values[field],
                            "authoritative": actual_values[field],
                        },
                    )
        difference = _first_difference(preview.stats, authority.stats)
        if difference is not None:
            first = authority.entries[0] if authority.entries else None
            raise PreviewError(
                "preview timing or aggregate statistics differ from authoritative recomputation",
                request_id=first.request_id if first else None,
                backend_id=first.backend_id if first else None,
                field=difference,
            )
        if preview.canonical_bytes() != authority.canonical_bytes():
            first = authority.entries[0] if authority.entries else None
            raise PreviewError(
                "preview and authoritative canonical bytes differ",
                request_id=first.request_id if first else None,
                backend_id=first.backend_id if first else None,
                field="canonical_bytes",
            )

    def verify_replay(self) -> ReplayReport:
        self._require_active_ledger()
        plan = self.plan()
        outputs, receipts, summary, batch_count = self._replay_outputs(plan)
        raw_entries = {
            int(row["request_id"]): row for row in self.store.fetch_raw_entries(self.spec.revision)
        }
        expected_request_ids = set(plan.request_ids)
        missing = sorted(expected_request_ids - set(raw_entries))
        if missing:
            request_id = missing[0]
            batch = batch_for_request(plan, request_id)
            raise ReplayError(
                "ledger has no committed entry for a planned request",
                request_id=request_id,
                backend_id=batch.backend_id,
                batch_id=batch.batch_id,
                field="ledger_status",
            )
        extra = sorted(set(raw_entries) - expected_request_ids)
        if extra:
            request_id = extra[0]
            row = raw_entries[request_id]
            raise ReplayError(
                "ledger contains a request that is absent from the replayed plan",
                request_id=request_id,
                backend_id=str(row["backend_id"]),
                batch_id=str(row["batch_id"]),
                field="unexpected_request",
            )
        batch_by_request = {
            item.sample.request_id: batch
            for batch in plan.batches
            for item in batch.items
        }
        for decision in plan.decisions:
            row = raw_entries[decision.request_id]
            batch = batch_by_request[decision.request_id]
            expected_values = {
                "rule_id": decision.rule_id,
                "backend_id": decision.backend_id,
                "target": decision.target,
                "bucket": decision.bucket,
                "weight_bp": decision.weight_bp,
                "weight_version": decision.weight_version,
                "routing_version": decision.routing_version,
                "batch_id": batch.batch_id,
                "output_digest": outputs[decision.request_id],
                "parameter_digest": receipts[batch.batch_id].parameter_digest,
            }
            for field in (
                "rule_id",
                "backend_id",
                "target",
                "bucket",
                "weight_bp",
                "weight_version",
                "routing_version",
                "batch_id",
                "output_digest",
                "parameter_digest",
            ):
                actual_value = row[field]
                if actual_value != expected_values[field]:
                    raise ReplayError(
                        f"ledger field {field!r} differs from deterministic replay",
                        request_id=decision.request_id,
                        backend_id=decision.backend_id,
                        batch_id=batch.batch_id,
                        field=field,
                        details={
                            "ledger": actual_value,
                            "replay": expected_values[field],
                        },
                    )
        integrity_ok, ledger_digest = self.store.verify_ledger(self.spec.revision)
        if not integrity_ok:
            first = plan.decisions[0]
            batch = batch_by_request[first.request_id]
            raise ReplayError(
                "committed ledger content digest is invalid",
                request_id=first.request_id,
                backend_id=batch.backend_id,
                batch_id=batch.batch_id,
                field="ledger_content_digest",
            )
        return ReplayReport(
            revision=self.spec.revision,
            replayed_batches=batch_count,
            replayed_requests=len(outputs),
            ledger_integrity_ok=integrity_ok,
            ledger_content_digest=ledger_digest,
            scheduler=summary,
        )

    def recover(self) -> RecoveryOutcome:
        self._require_active_ledger()
        plan = self.plan()
        initial = self.store.recovery_report(plan, self.requests)
        committed = set(self.store.committed_batch_ids(self.spec.revision))
        pending = [batch for batch in plan.batches if batch.batch_id not in committed]
        executions: list[BatchExecution] = []

        def commit(execution: BatchExecution) -> None:
            self.store.commit_batch(execution)
            executions.append(execution)

        if pending:
            self._run_batches(pending, on_complete=commit)
        final = self.store.recovery_report(plan, self.requests)
        replay = self.verify_replay()
        return RecoveryOutcome(
            initial=initial,
            final=final,
            newly_committed=len(executions),
            replay=replay,
        )
