from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .breaker import BreakerManager, CircuitBreaker
from .budget import BudgetScheduler
from .errors import MeshError
from .ledger import (
    AttemptCommit,
    AttemptIntent,
    FinalOutcome,
    Ledger,
    RecoverySnapshot,
)
from .logging_utils import StructuredLogger
from .model import MeshSpec, RequestSpec, RouteRevision, requests_digest
from .protocol import AttemptResult, send_attempt
from .selector import WeightedSelector
from .snapshot import (
    SnapshotEnvelope,
    SnapshotState,
    SnapshotStore,
    empty_diff,
)
from .stats import build_stats
from .upstream_manager import UpstreamManager

SELECTION_SEQ_STRIDE = 1_000_000


def mono_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def wall_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class RequestExecution:
    request: RequestSpec
    request_seq: int
    run_epoch: int


@dataclass
class RequestResult:
    request_id: str
    request_seq: int
    service: str
    final_outcome: str
    error_code: str | None
    attempts: int
    total_duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_seq": self.request_seq,
            "service": self.service,
            "final_outcome": self.final_outcome,
            "error_code": self.error_code,
            "attempts": self.attempts,
            "total_duration_ms": self.total_duration_ms,
        }


class MeshRuntime:
    """Coordinates upstream subprocesses, the dual-budget scheduler and ledger."""

    def __init__(
        self,
        spec: MeshSpec,
        revision: RouteRevision,
        ledger: Ledger,
        snapshot_store: SnapshotStore,
        logger: StructuredLogger,
        session_id: str,
    ) -> None:
        self.spec = spec
        self.revision = revision
        self.ledger = ledger
        self.snapshot_store = snapshot_store
        self.logger = logger
        existing_meta = ledger.meta_dict()
        if ledger.count_requests() > 0 and existing_meta.get("session_id"):
            self.session_id = existing_meta["session_id"]
        else:
            self.session_id = session_id
        self.selector = WeightedSelector(spec, revision)
        self.scheduler = BudgetScheduler(
            spec.session.global_concurrency,
            {service.id: service.quota for service in spec.ordered_services()},
        )
        instances = [
            instance
            for service in spec.ordered_services()
            for instance in service.instances
        ]
        self.breakers = BreakerManager(instances, spec.session.breaker)
        self.upstreams = UpstreamManager(spec)
        self._started = False
        self._stopped = False
        self._state_lock = threading.RLock()
        self._batch_counter = 0
        self._last_snapshot: SnapshotEnvelope | None = None
        self._snapshot_repair: dict[str, Any] | None = None
        self._recovery: RecoverySnapshot | None = None
        self._set_ledger_meta()

    def _set_ledger_meta(self) -> None:
        values: dict[str, Any] = {
            "session_id": self.session_id,
            "active_route_revision": self.revision.revision,
            "instance_version": self.spec.instance_version(),
            "config_digest": self.spec.config_digest(),
            "global_concurrency": self.spec.session.global_concurrency,
            "attempt_timeout_ms": self.spec.session.attempt_timeout_ms,
        }
        for service in self.spec.ordered_services():
            values[f"quota:{service.id}"] = service.quota
        self.ledger.set_meta(values)

    def apply_recovery(self, recovery: RecoverySnapshot) -> None:
        self._recovery = recovery
        self.scheduler.set_usage(
            recovery.budget_used_global, recovery.budget_used_by_service
        )
        self.breakers.restore(recovery.breakers, wall_ms())
        self.logger.warning(
            "state_restored_from_ledger",
            code="E_RECOVERY",
            budget_used_global=recovery.budget_used_global,
            completed_cursor=recovery.completed_cursor,
            unknown_attempts=len(recovery.unknown_attempts),
            exit_report_available=recovery.exit_report_available,
        )

    def apply_snapshot_state(self, state: SnapshotState) -> None:
        self.scheduler.set_usage(
            state.budget_used_global, state.budget_used_by_service
        )
        self.breakers.restore(state.breakers, wall_ms())
        self.logger.info(
            "state_restored_from_snapshot",
            budget_used_global=state.budget_used_global,
            completed_cursor=state.completed_cursor,
            route_revision=state.route_revision,
        )

    def restore_ledger_authoritative_state(self) -> None:
        global_used, service_used = self.ledger.budget_totals()
        self.scheduler.set_usage(global_used, service_used)
        self.breakers.restore(self.ledger.latest_breakers(), wall_ms())
        self.logger.warning(
            "state_restored_from_ledger",
            code="E_SNAPSHOT",
            budget_used_global=global_used,
            completed_cursor=self.ledger.completed_cursor(),
        )

    def start(self) -> None:
        if self._started:
            return
        self.upstreams.start()
        self._started = True
        self._stopped = False
        self.logger.info(
            "upstreams_started",
            session_id=self.session_id,
            instance_count=sum(len(service.instances) for service in self.spec.ordered_services()),
        )

    def stop(self) -> None:
        if self._stopped:
            return
        self.upstreams.stop()
        self._stopped = True
        self.logger.info("upstreams_stopped", session_id=self.session_id)

    def __enter__(self) -> "MeshRuntime":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def _existing_requests(self) -> dict[str, dict[str, Any]]:
        return {str(row["request_id"]): row for row in self.ledger.request_sequence()}

    def _prepare_execution(self, requests: list[RequestSpec]) -> list[RequestExecution]:
        with self._state_lock:
            existing = self._existing_requests()
            next_seq = 1 + max((int(row["request_seq"]) for row in existing.values()), default=0)
            executions: list[RequestExecution] = []
            seen: set[str] = set()
            for request in requests:
                if request.id in seen:
                    raise MeshError(
                        "E_SPEC",
                        f"duplicate request id in batch: {request.id!r}",
                        field="requests.id",
                        location=request.source_path or "$",
                        request_id=request.id,
                    )
                seen.add(request.id)
                row = existing.get(request.id)
                if row is None:
                    seq = self.ledger.register_request(request, next_seq)
                    next_seq += 1
                    epoch = self.ledger.request_epoch(request.id)
                else:
                    if row["status"] == "complete":
                        raise MeshError(
                            "E_SPEC",
                            f"request {request.id!r} is already complete; use replay for a fresh run",
                            field="requests.id",
                            location=request.source_path or "$",
                            request_id=request.id,
                            service=request.service,
                        )
                    seq = int(row["request_seq"])
                    if row["status"] == "pending":
                        epoch = int(row["run_epoch"])
                    else:
                        epoch = self.ledger.mark_rerun(request.id)
                executions.append(RequestExecution(request, seq, epoch))
            return sorted(executions, key=lambda item: (item.request_seq, item.request.id))

    def run_requests(self, requests: list[RequestSpec]) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("MeshRuntime.start() must be called before run_requests()")
        if not requests:
            raise MeshError("E_SPEC", "request batch is empty", field="requests", location="$")
        executions = self._prepare_execution(requests)
        self._batch_counter += 1
        batch_id = f"{self.session_id}-batch-{self._batch_counter:04d}"
        self.ledger.set_meta(
            {
                "request_sequence_digest": requests_digest([item.request for item in executions]),
                "last_batch_id": batch_id,
                "active_route_revision": self.revision.revision,
                "instance_version": self.spec.instance_version(),
            }
        )
        self.logger.info(
            "batch_started",
            session_id=self.session_id,
            batch_id=batch_id,
            request_count=len(executions),
            route_revision=self.revision.revision,
        )
        started = mono_ms()
        workers = min(
            len(executions),
            max(16, self.spec.session.global_concurrency * 8),
        )
        results: list[RequestResult] = []
        errors: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="mesh-attempt") as pool:
            futures: list[Future[RequestResult]] = [
                pool.submit(self._run_request, execution, batch_id)
                for execution in executions
            ]
            for future in futures:
                try:
                    results.append(future.result())
                except BaseException as exc:  # keep other in-flight requests and ledger coherent
                    errors.append(exc)
        if errors:
            first = errors[0]
            if isinstance(first, MeshError):
                raise first
            raise RuntimeError(f"request worker failed: {first}") from first
        finished = mono_ms()
        results.sort(key=lambda item: (item.request_seq, item.request_id))
        stats = self.current_stats()
        report = {
            "session_id": self.session_id,
            "batch_id": batch_id,
            "route_revision": self.revision.revision,
            "instance_version": self.spec.instance_version(),
            "request_count": len(results),
            "batch_duration_ms": finished - started,
            "results": [result.to_dict() for result in results],
            "budget": self.scheduler.snapshot(),
            "stats": stats,
            "ledger": self.ledger.summary(),
        }
        self.logger.info(
            "batch_finished",
            session_id=self.session_id,
            batch_id=batch_id,
            request_count=len(results),
            duration_ms=finished - started,
            successes=sum(1 for result in results if result.final_outcome == "success"),
            exhausted=sum(1 for result in results if result.error_code == "E_EXHAUST"),
            peak_global=report["budget"]["peak_global"],
        )
        return report

    def _run_request(self, execution: RequestExecution, batch_id: str) -> RequestResult:
        request = execution.request
        session = self.spec.session
        first_started = mono_ms()
        last_finished = first_started
        attempt_no = 1
        final_outcome = "exhausted"
        final_error: str | None = "E_EXHAUST"
        while True:
            selection = self.selector.select(request.service, request.id, attempt_no)
            instance = selection.instance
            select_seq = execution.request_seq * SELECTION_SEQ_STRIDE + attempt_no
            breaker = self.breakers.get(instance.id)
            decision = breaker.before_attempt()
            if not decision.allowed:
                finished = mono_ms()
                fast_fail_started = finished
                fast_fail_started_wall = wall_ms()
                last_finished = finished
                final_attempt = attempt_no >= session.max_attempts
                result = AttemptResult(
                    outcome="circuit_open",
                    transport_ok=False,
                    error_code="E_CIRCUIT_OPEN",
                    latency_ms=0,
                    detail=f"fast fail: {decision.reason}; remaining_cooldown_ms={decision.remaining_cooldown_ms}",
                )
                final = None
                if final_attempt:
                    final = FinalOutcome(
                        final_outcome="exhausted",
                        error_code="E_EXHAUST",
                        attempt_count=attempt_no,
                        total_duration_ms=max(0, finished - first_started),
                        completed_at_wall_ms=wall_ms(),
                    )
                self._commit_attempt(
                    execution,
                    attempt_no,
                    select_seq,
                    instance.id,
                    result,
                    breaker,
                    fast_fail_started,
                    fast_fail_started,
                    fast_fail_started_wall,
                    finished,
                    budget_used=False,
                    global_inflight=0,
                    service_inflight=0,
                    final_attempt=final_attempt,
                    final=final,
                    outcome_override="circuit_open",
                )
                self.logger.warning(
                    "attempt_circuit_fast_fail",
                    code="E_CIRCUIT_OPEN",
                    session_id=self.session_id,
                    batch_id=batch_id,
                    request_id=request.id,
                    service=request.service,
                    instance_id=instance.id,
                    attempt_no=attempt_no,
                    select_seq=select_seq,
                    breaker_state=breaker.state.value,
                    remaining_cooldown_ms=decision.remaining_cooldown_ms,
                )
                if final_attempt:
                    final_outcome = "exhausted"
                    final_error = "E_EXHAUST"
                    break
                self._backoff(attempt_no)
                attempt_no += 1
                continue
            wait_start = mono_ms()
            slot = self.scheduler.acquire(
                request.service, (execution.request_seq, attempt_no)
            )
            attempt_started = mono_ms()
            attempt_started_wall = wall_ms()
            intent = AttemptIntent(
                request_id=request.id,
                run_epoch=execution.run_epoch,
                attempt_no=attempt_no,
                service=request.service,
                instance_id=instance.id,
                select_seq=select_seq,
                started_wall_ms=attempt_started_wall,
                started_mono_ms=attempt_started,
            )
            try:
                self.ledger.begin_intent(intent)
                result = send_attempt(
                    instance.host,
                    instance.port,
                    instance.id,
                    request.service,
                    request,
                    attempt_no,
                    select_seq,
                    self.revision.revision,
                    session.attempt_timeout_ms,
                )
            except Exception as exc:  # defensive conversion, socket timeout remains primary
                result = AttemptResult(
                    outcome="host_error",
                    transport_ok=False,
                    error_code="E_HOST",
                    latency_ms=max(0, mono_ms() - attempt_started),
                    detail=f"unexpected attempt error: {exc}",
                )
            finally:
                self.scheduler.release(slot)
            finished = mono_ms()
            last_finished = finished
            previous_state = breaker.state
            breaker.record_result(decision.permit, result.success)  # type: ignore[arg-type]
            final_attempt = result.success or attempt_no >= session.max_attempts
            final: FinalOutcome | None = None
            if result.success:
                final_outcome = "success"
                final_error = None
                final = FinalOutcome(
                    final_outcome="success",
                    error_code=None,
                    attempt_count=attempt_no,
                    total_duration_ms=max(0, finished - first_started),
                    completed_at_wall_ms=wall_ms(),
                )
            elif final_attempt:
                final_outcome = "exhausted"
                final_error = "E_EXHAUST"
                final = FinalOutcome(
                    final_outcome="exhausted",
                    error_code="E_EXHAUST",
                    attempt_count=attempt_no,
                    total_duration_ms=max(0, finished - first_started),
                    completed_at_wall_ms=wall_ms(),
                )
            self._commit_attempt(
                execution,
                attempt_no,
                select_seq,
                instance.id,
                result,
                breaker,
                wait_start,
                attempt_started,
                attempt_started_wall,
                finished,
                budget_used=True,
                global_inflight=slot.global_inflight,
                service_inflight=slot.service_inflight,
                final_attempt=final_attempt,
                final=final,
            )
            if previous_state != breaker.state:
                self.logger.warning(
                    "circuit_transition",
                    code="E_CIRCUIT",
                    session_id=self.session_id,
                    request_id=request.id,
                    service=request.service,
                    instance_id=instance.id,
                    previous=previous_state.value,
                    current=breaker.state.value,
                    attempt_no=attempt_no,
                    remaining_cooldown_ms=breaker.cooldown_remaining_ms(),
                )
            self.logger.emit(
                "INFO" if result.success else "WARNING",
                "attempt_committed",
                code=result.error_code or "OK",
                session_id=self.session_id,
                batch_id=batch_id,
                request_id=request.id,
                service=request.service,
                instance_id=instance.id,
                attempt_no=attempt_no,
                select_seq=select_seq,
                outcome=result.outcome,
                latency_ms=result.latency_ms,
                budget_wait_ms=slot.wait_ms,
                breaker_state=breaker.state.value,
            )
            if final_attempt:
                break
            self._backoff(attempt_no)
            attempt_no += 1
        result_record = RequestResult(
            request_id=request.id,
            request_seq=execution.request_seq,
            service=request.service,
            final_outcome=final_outcome,
            error_code=final_error,
            attempts=attempt_no,
            total_duration_ms=max(0, last_finished - first_started),
        )
        self.logger.emit(
            "INFO" if final_outcome == "success" else "ERROR",
            "request_finished",
            code=final_error or "OK",
            session_id=self.session_id,
            batch_id=batch_id,
            request_id=request.id,
            request_seq=execution.request_seq,
            service=request.service,
            final_outcome=final_outcome,
            attempts=attempt_no,
            total_duration_ms=result_record.total_duration_ms,
        )
        return result_record

    def _commit_attempt(
        self,
        execution: RequestExecution,
        attempt_no: int,
        select_seq: int,
        instance_id: str,
        result: AttemptResult,
        breaker: CircuitBreaker,
        wait_start_mono: int,
        attempt_started_mono: int,
        started_wall_ms: int,
        finished_mono: int,
        *,
        budget_used: bool,
        global_inflight: int,
        service_inflight: int,
        final_attempt: bool,
        final: FinalOutcome | None,
        outcome_override: str | None = None,
    ) -> None:
        outcome = outcome_override or result.outcome
        wait_end = attempt_started_mono if budget_used else wait_start_mono
        self.ledger.commit_attempt(
            AttemptCommit(
                request_id=execution.request.id,
                request_seq=execution.request_seq,
                run_epoch=execution.run_epoch,
                attempt_no=attempt_no,
                select_seq=select_seq,
                service=execution.request.service,
                instance_id=instance_id,
                route_revision=self.revision.revision,
                instance_version=self.spec.instance_version(),
                outcome=outcome,
                status=result.status,
                error_code=result.error_code,
                transport_ok=result.transport_ok,
                started_wall_ms=started_wall_ms,
                started_mono_ms=attempt_started_mono,
                finished_mono_ms=finished_mono,
                wait_start_mono_ms=wait_start_mono,
                wait_end_mono_ms=wait_end,
                budget_wait_ms=max(0, wait_end - wait_start_mono) if budget_used else 0,
                duration_ms=max(0, finished_mono - attempt_started_mono),
                upstream_delay_ms=result.upstream_delay_ms,
                budget_used=budget_used,
                global_inflight=global_inflight,
                service_inflight=service_inflight,
                final_attempt=final_attempt,
                detail=result.detail,
                breaker_after=breaker.serialize(),
            ),
            final,
        )

    def _backoff(self, attempt_no: int) -> None:
        backoffs = self.spec.session.retry_backoff_ms
        index = attempt_no - 1
        if 0 <= index < len(backoffs):
            delay_ms = backoffs[index]
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

    def current_stats(self) -> dict[str, Any]:
        return build_stats(
            self.ledger.latest_outcomes(),
            self.ledger.latest_attempts(),
            self.spec.session.scale,
        )

    def snapshot_state(self) -> SnapshotState:
        global_used, service_used = self.ledger.budget_totals()
        return SnapshotState(
            route_revision=self.revision.revision,
            instance_version=self.spec.instance_version(),
            budget_used_global=global_used,
            budget_used_by_service=service_used,
            breakers=self.breakers.serialized(),
            completed_cursor=self.ledger.completed_cursor(),
            ledger_format_version=1,
            ledger_digest=self.ledger.content_digest(),
            request_digest=self.ledger.meta_dict().get("request_sequence_digest", ""),
        )

    def write_snapshot(self, reason: str = "batch_complete") -> SnapshotEnvelope:
        state = self.snapshot_state()
        envelope = self.snapshot_store.write(state)
        self._last_snapshot = envelope
        self.logger.info(
            "snapshot_state_captured",
            reason=reason,
            path=envelope.path,
            digest=envelope.digest,
            completed_cursor=state.completed_cursor,
        )
        return envelope

    def validate_or_replace_snapshot(self, *, apply_state: bool = True) -> dict[str, Any]:
        authoritative = self.snapshot_state()
        try:
            previous = self.snapshot_store.load_latest()
            if previous is None:
                if apply_state and self._recovery is None and self.ledger.count_attempts() > 0:
                    self.restore_ledger_authoritative_state()
                return {
                    "status": "none",
                    "restored": False,
                    "replacement": None,
                    "diff": empty_diff(),
                }
            self.snapshot_store.validate_against(
                previous,
                route_revision=self.revision.revision,
                instance_version=self.spec.instance_version(),
                ledger_digest=self.ledger.content_digest(),
            )
            if (
                previous.state.budget_used_global != authoritative.budget_used_global
                or previous.state.budget_used_by_service != authoritative.budget_used_by_service
                or previous.state.completed_cursor != authoritative.completed_cursor
            ):
                raise MeshError(
                    "E_SNAPSHOT",
                    "snapshot budget/cursor state does not match committed ledger",
                    field="snapshot.payload",
                    location=previous.path,
                )
            if apply_state and self._recovery is None:
                self.apply_snapshot_state(previous.state)
            return {
                "status": "valid",
                "restored": True,
                "snapshot": previous.path,
                "diff": empty_diff(),
            }
        except MeshError as exc:
            if exc.code != "E_SNAPSHOT":
                raise
            self.logger.error(
                "snapshot_rejected",
                code="E_SNAPSHOT",
                message=exc.message,
                field=exc.field,
                location=exc.location,
                details=exc.details,
            )
            try:
                previous = self.snapshot_store.load_latest()
            except MeshError:
                previous = self.snapshot_store.load_latest_unverified()
            if apply_state:
                self.restore_ledger_authoritative_state()
                authoritative = self.snapshot_state()
            replacement, diff = self.snapshot_store.replace_with(authoritative, previous)
            self._snapshot_repair = {
                "status": "replaced",
                "restored": False,
                "snapshot": replacement.path,
                "diff": diff,
                "error": exc.to_dict(),
            }
            return self._snapshot_repair

    def runtime_state(self) -> dict[str, Any]:
        budget = self.scheduler.snapshot()
        return {
            "session_id": self.session_id,
            "route_revision": self.revision.revision,
            "instance_version": self.spec.instance_version(),
            "budget": budget,
            "breakers": self.breakers.views(),
            "transitions": [
                {
                    "at_mono_ms": item.at_mono_ms,
                    "instance_id": item.instance_id,
                    "previous": item.previous,
                    "current": item.current,
                    "reason": item.reason,
                }
                for item in self.breakers.transitions()
            ],
            "snapshot_repair": self._snapshot_repair,
            "recovery": None
            if self._recovery is None
            else {
                "budget_used_global": self._recovery.budget_used_global,
                "completed_cursor": self._recovery.completed_cursor,
                "unknown_attempts": len(self._recovery.unknown_attempts),
                "rerun_requests": len(self._recovery.requests_to_rerun),
                "exit_report_available": self._recovery.exit_report_available,
            },
        }
