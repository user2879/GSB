from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any, Callable

from .errors import BusinessError
from .locks import LockManager, WaitRequest
from .models import (
    CommitRecord,
    EditScript,
    Operation,
    OperationResult,
    RunResult,
    SchedulerParams,
    SessionOutcome,
    SessionSpec,
    TraceEvent,
    WaitEvidence,
    canonical_json,
    digest_json,
)
from .storage import Storage


def _monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class _Runtime:
    spec: SessionSpec
    script_index: int
    status: str = "pending"
    start_seq: int = 0
    started_virtual_ms: int = 0
    finished_virtual_ms: int = 0
    started_monotonic_ms: int = 0
    finished_monotonic_ms: int = 0
    op_index: int = 0
    in_flight: int = 0
    wait_used_ms: int = 0
    pending_writes: list[dict[str, Any]] = field(default_factory=list)
    pending_write_rows: dict[str, Any] = field(default_factory=dict)
    rr_snapshot: dict[str, tuple[Any, int]] = field(default_factory=dict)
    rr_read_cache: dict[str, tuple[Any, int | None]] = field(default_factory=dict)
    operations: list[OperationResult] = field(default_factory=list)
    current_operation: OperationResult | None = None
    current_operation_spec: Operation | None = None
    current_wait: WaitRequest | None = None
    current_grant_seq: int | None = None
    current_grant_virtual_ms: int | None = None
    outcome: SessionOutcome | None = None

    @property
    def terminal(self) -> bool:
        return self.status in ("committed", "aborted")

    @property
    def active(self) -> bool:
        return self.status in ("running", "waiting_lock")


class Engine:
    def __init__(
        self,
        storage: Storage,
        params: SchedulerParams,
        observer: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.storage = storage
        self.params = params
        self.observer = observer
        self.trace: list[TraceEvent] = []
        self.runtimes: list[_Runtime] = []
        self.by_session: dict[str, _Runtime] = {}
        self.lock_manager = LockManager(event_callback=self._lock_event)
        self.run_id = ""
        self.script: EditScript | None = None
        self.now = 0
        self._next_start_seq = 1
        self._commit_records: list[CommitRecord] = []

    def _emit(
        self,
        event_type: str,
        session_id: str | None,
        op_no: int | None,
        payload: dict[str, Any] | None = None,
        now: int | None = None,
    ) -> None:
        event = TraceEvent(
            event_seq=len(self.trace) + 1,
            virtual_ms=self.now if now is None else now,
            monotonic_ms=_monotonic_ms(),
            session_id=session_id,
            op_no=op_no,
            event_type=event_type,
            payload=payload or {},
        )
        self.trace.append(event)
        if self.observer is not None:
            self.observer(event)

    def _lock_event(
        self,
        event_type: str,
        session_id: str | None,
        op_no: int | None,
        payload: dict[str, Any],
        now: int,
    ) -> None:
        self._emit(event_type, session_id, op_no, payload, now)

    def _session_phase(self, runtime: _Runtime, phase: str, now: int, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "phase": phase,
            "isolation_level": runtime.spec.isolation_level,
            "wait_used_ms": runtime.wait_used_ms,
            "lock_wait_budget_ms": runtime.spec.lock_wait_budget_ms,
            "concurrency_limit": runtime.spec.concurrency,
            "held_locks": self.lock_manager.held_by(runtime.spec.session_id),
        }
        if extra:
            payload.update(extra)
        self._emit("session.phase", runtime.spec.session_id, None, payload, now)

    def _active_count(self) -> int:
        return sum(1 for runtime in self.runtimes if runtime.active)

    def _start_session(self, runtime: _Runtime, now: int) -> None:
        runtime.status = "running"
        runtime.start_seq = self._next_start_seq
        self._next_start_seq += 1
        runtime.started_virtual_ms = now
        runtime.started_monotonic_ms = _monotonic_ms()
        if runtime.spec.isolation_level == "repeatable_read":
            runtime.rr_snapshot = {
                row_key: (row["value"], int(row["version"]))
                for row_key, row in self.storage.all_rows().items()
            }
        self._emit(
            "session.start",
            runtime.spec.session_id,
            None,
            {
                "start_seq": runtime.start_seq,
                "isolation_level": runtime.spec.isolation_level,
                "lock_wait_budget_ms": runtime.spec.lock_wait_budget_ms,
                "concurrency": runtime.spec.concurrency,
                "operation_timeout_ms": runtime.spec.operation_timeout_ms,
                "lock_order": list(runtime.spec.lock_order),
                "active_sessions": self._active_count(),
                "global_concurrency_limit": self.params.global_concurrency_limit,
            },
            now,
        )
        self._session_phase(runtime, "running", now)

    def _begin_operation(self, runtime: _Runtime, now: int) -> None:
        spec = runtime.spec
        operation = spec.operations[runtime.op_index]
        if runtime.in_flight + 1 > spec.concurrency:
            raise BusinessError(
                "E_ISOLATION",
                "per-session concurrency limit would be exceeded",
                path=f"$.sessions.{spec.session_id}.concurrency",
                session_id=spec.session_id,
                op_no=operation.op_no,
                details={"concurrency": spec.concurrency},
            )
        runtime.in_flight += 1
        result = OperationResult(
            op_no=operation.op_no,
            kind=operation.kind,
            row_key=operation.row_key,
            status="running",
            monotonic_start_ms=_monotonic_ms(),
        )
        runtime.current_operation = result
        runtime.current_operation_spec = operation
        runtime.operations.append(result)
        self._emit(
            "op.start",
            spec.session_id,
            operation.op_no,
            {
                "type": operation.kind,
                "row_key": operation.row_key,
                "op_index": runtime.op_index,
                "isolation_level": spec.isolation_level,
                "wait_used_ms": runtime.wait_used_ms,
            },
            now,
        )

        remaining_budget = max(0, spec.lock_wait_budget_ms - runtime.wait_used_ms)
        mode = "S" if operation.kind == "read" else "X"
        decision = self.lock_manager.acquire(
            session_id=spec.session_id,
            row_key=operation.row_key,
            mode=mode,
            isolation_level=spec.isolation_level,
            now=now,
            op_no=operation.op_no,
            op_started_virtual_ms=now,
            remaining_budget_ms=remaining_budget,
            operation_timeout_ms=spec.operation_timeout_ms,
        )
        if decision.status == "granted":
            result.grant_seq = decision.grant_seq
            runtime.current_grant_seq = decision.grant_seq
            runtime.current_grant_virtual_ms = now
            self._perform_operation(runtime, now)
            return

        result.status = "waiting_lock"
        runtime.current_wait = decision.request
        runtime.current_grant_seq = None
        runtime.current_grant_virtual_ms = None
        runtime.status = "waiting_lock"
        self._session_phase(
            runtime,
            "waiting_lock",
            now,
            {
                "waiting_row_key": operation.row_key,
                "request_seq": decision.request.request_seq if decision.request else None,
                "blocked_by": list(decision.blocked_by),
                "queue_position": decision.queue_position,
                "wait_limit_ms": decision.request.wait_limit_ms if decision.request else None,
            },
        )

    def _write_wait_evidence(self, runtime: _Runtime, granted_virtual_ms: int | None) -> None:
        result = runtime.current_operation
        request = runtime.current_wait
        if result is None or request is None:
            return
        wait_ms = 0 if granted_virtual_ms is None else max(0, granted_virtual_ms - request.requested_virtual_ms)
        evidence = WaitEvidence(
            request_seq=request.request_seq,
            row_key=request.row_key,
            mode=request.mode,
            started_virtual_ms=request.requested_virtual_ms,
            granted_virtual_ms=granted_virtual_ms,
            wait_ms=wait_ms,
            blocked_by=request.blocked_by,
            queue_position=request.queue_position,
        )
        result.wait = evidence
        result.wait_ms = wait_ms

    def _perform_operation(self, runtime: _Runtime, now: int) -> None:
        result = runtime.current_operation
        operation = runtime.current_operation_spec
        if result is None or operation is None:
            raise RuntimeError("operation state missing")
        spec = runtime.spec
        if operation.kind == "read":
            value, version, source = self._read_value(runtime, operation.row_key)
            result.value = value
            result.data_version = version
            self._emit(
                "op.read",
                spec.session_id,
                operation.op_no,
                {
                    "row_key": operation.row_key,
                    "value": value,
                    "data_version": version,
                    "read_source": source,
                    "isolation_level": spec.isolation_level,
                    "grant_seq": result.grant_seq,
                    "wait_ms": result.wait_ms,
                },
                now,
            )
            if spec.isolation_level == "read_committed":
                self.lock_manager.release_shared(
                    spec.session_id,
                    operation.row_key,
                    now,
                    reason="read_committed_transient_shared",
                )
        else:
            value = operation.value
            result.value = value
            result.data_version = None
            runtime.pending_writes.append(
                {
                    "op_no": operation.op_no,
                    "row_key": operation.row_key,
                    "value": value,
                }
            )
            runtime.pending_write_rows[operation.row_key] = value
            if spec.isolation_level == "repeatable_read":
                runtime.rr_read_cache[operation.row_key] = (value, None)
            self._emit(
                "op.write",
                spec.session_id,
                operation.op_no,
                {
                    "row_key": operation.row_key,
                    "value": value,
                    "pending": True,
                    "data_version": None,
                    "grant_seq": result.grant_seq,
                    "wait_ms": result.wait_ms,
                },
                now,
            )

        result.status = "success"
        result.monotonic_end_ms = _monotonic_ms()
        runtime.in_flight -= 1
        self._emit(
            "op.end",
            spec.session_id,
            operation.op_no,
            {
                "status": "success",
                "type": operation.kind,
                "row_key": operation.row_key,
                "wait_ms": result.wait_ms,
                "grant_seq": result.grant_seq,
                "held_locks": self.lock_manager.held_by(spec.session_id),
            },
            now,
        )
        runtime.current_operation = None
        runtime.current_operation_spec = None
        runtime.current_wait = None
        runtime.current_grant_seq = None
        runtime.current_grant_virtual_ms = None
        runtime.op_index += 1
        if runtime.op_index >= len(spec.operations):
            self._commit_session(runtime, now)
        else:
            runtime.status = "running"
            self._session_phase(runtime, "running", now)

    def _read_value(self, runtime: _Runtime, row_key: str) -> tuple[Any, int | None, str]:
        if row_key in runtime.pending_write_rows:
            return runtime.pending_write_rows[row_key], None, "own_pending_write"
        if runtime.spec.isolation_level == "repeatable_read":
            if row_key in runtime.rr_read_cache:
                value, version = runtime.rr_read_cache[row_key]
                return value, version, "repeatable_read_cache"
            value, version = runtime.rr_snapshot.get(row_key, (None, 0))
            runtime.rr_read_cache[row_key] = (value, version)
            return value, version, "repeatable_read_snapshot"
        value, version = self.storage.read_row(row_key)
        return value, version, "latest_committed"

    def _commit_session(self, runtime: _Runtime, now: int) -> None:
        spec = runtime.spec
        data_version, commit = self.storage.apply_commit(
            run_id=self.run_id,
            session_id=spec.session_id,
            virtual_ms=now,
            writes=runtime.pending_writes,
        )
        if commit is not None:
            self._commit_records.append(commit)
            for result in runtime.operations:
                if result.kind == "write":
                    result.data_version = data_version
        runtime.status = "committed"
        runtime.finished_virtual_ms = now
        runtime.finished_monotonic_ms = _monotonic_ms()
        if runtime.outcome is None:
            runtime.outcome = self._build_outcome(runtime, "COMMITTED", None, None, data_version)
        locked_entries = self.lock_manager.held_by(spec.session_id)
        self.lock_manager.release_session(spec.session_id, now, "commit")
        self._emit(
            "session.commit",
            spec.session_id,
            None,
            {
                "committed_data_version": data_version,
                "commit_seq": commit.commit_seq if commit else None,
                "held_at_commit": locked_entries,
                "wait_used_ms": runtime.wait_used_ms,
            },
            now,
        )
        self._session_phase(runtime, "committed", now, {"business_code": None})

    def _build_outcome(
        self,
        runtime: _Runtime,
        outcome: str,
        business_code: str | None,
        message: str | None,
        committed_data_version: int | None,
    ) -> SessionOutcome:
        return SessionOutcome(
            session_id=runtime.spec.session_id,
            start_seq=runtime.start_seq,
            isolation_level=runtime.spec.isolation_level,
            concurrency_limit=runtime.spec.concurrency,
            started_virtual_ms=runtime.started_virtual_ms,
            finished_virtual_ms=runtime.finished_virtual_ms,
            started_monotonic_ms=runtime.started_monotonic_ms,
            finished_monotonic_ms=runtime.finished_monotonic_ms,
            outcome=outcome,
            business_code=business_code,
            message=message,
            wait_used_ms=runtime.wait_used_ms,
            committed_data_version=committed_data_version,
            operations=runtime.operations,
        )

    def _abort_session(
        self,
        runtime: _Runtime,
        now: int,
        outcome: str,
        code: str,
        message: str,
        *,
        record_wait: bool = True,
        wait_ms_for_result: int | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        if runtime.terminal:
            return
        runtime.status = "aborted"
        runtime.finished_virtual_ms = now
        runtime.finished_monotonic_ms = _monotonic_ms()
        result = runtime.current_operation
        if result is not None and result.status in ("waiting_lock", "running"):
            result.status = "failed"
            result.business_code = code
            result.message = message
            result.monotonic_end_ms = _monotonic_ms()
            if record_wait and runtime.current_wait is not None:
                elapsed = (
                    wait_ms_for_result
                    if wait_ms_for_result is not None
                    else max(0, now - runtime.current_wait.requested_virtual_ms)
                )
                request = runtime.current_wait
                result.wait_ms = elapsed
                result.wait = WaitEvidence(
                    request_seq=request.request_seq,
                    row_key=request.row_key,
                    mode=request.mode,
                    started_virtual_ms=request.requested_virtual_ms,
                    granted_virtual_ms=None,
                    wait_ms=elapsed,
                    blocked_by=request.blocked_by,
                    queue_position=request.queue_position,
                )
        request = runtime.current_wait
        if request is not None:
            self.lock_manager.cancel_wait(
                runtime.spec.session_id,
                request.row_key,
                now,
                reason=f"session_abort:{code}",
            )
        held = self.lock_manager.held_by(runtime.spec.session_id)
        released = self.lock_manager.release_session(
            runtime.spec.session_id,
            now,
            f"abort:{code}",
        )
        runtime.pending_writes = []
        runtime.pending_write_rows = {}
        runtime.in_flight = 0
        if runtime.outcome is None:
            runtime.outcome = self._build_outcome(runtime, outcome, code, message, None)
        payload = {
            "business_code": code,
            "message": message,
            "wait_used_ms": runtime.wait_used_ms,
            "held_locks_released": held,
            "lock_entries_released": released,
        }
        if extra_payload:
            payload.update(extra_payload)
        self._emit("session.abort", runtime.spec.session_id, result.op_no if result else None, payload, now)
        self._session_phase(runtime, "aborted", now, {"business_code": code, "message": message})

    def _check_timeouts(self, now: int) -> None:
        for runtime in sorted(self.runtimes, key=lambda item: item.start_seq):
            if runtime.status != "waiting_lock" or runtime.current_wait is None:
                continue
            request = runtime.current_wait
            elapsed = max(0, now - request.requested_virtual_ms)
            remaining_budget = max(0, runtime.spec.lock_wait_budget_ms - runtime.wait_used_ms)
            budget_hit = elapsed >= remaining_budget
            operation_hit = now - request.op_started_virtual_ms >= runtime.spec.operation_timeout_ms
            if not budget_hit and not operation_hit:
                continue
            runtime.wait_used_ms += elapsed
            if budget_hit and operation_hit:
                reason = "lock_wait_budget_and_operation_timeout"
            elif operation_hit:
                reason = "operation_timeout"
            else:
                reason = "lock_wait_budget"
            message = (
                f"lock wait exceeded {reason}; session {runtime.spec.session_id} "
                f"op {request.op_no} waited {elapsed}ms "
                f"(budget remaining {remaining_budget}ms, operation timeout {runtime.spec.operation_timeout_ms}ms)"
            )
            self._emit(
                "session.lock_timeout",
                runtime.spec.session_id,
                request.op_no,
                {
                    "business_code": "E_LOCKTIMEOUT",
                    "row_key": request.row_key,
                    "request_seq": request.request_seq,
                    "wait_ms": elapsed,
                    "wait_used_ms": runtime.wait_used_ms,
                    "remaining_budget_ms": remaining_budget,
                    "operation_timeout_ms": runtime.spec.operation_timeout_ms,
                    "reason": reason,
                },
                now,
            )
            self._abort_session(
                runtime,
                now,
                "E_LOCKTIMEOUT",
                "E_LOCKTIMEOUT",
                message,
                wait_ms_for_result=elapsed,
                extra_payload={"timeout_reason": reason},
            )

    def _find_cycle(self, dependencies: dict[str, set[str]]) -> list[str] | None:
        active_ids = {runtime.spec.session_id for runtime in self.runtimes if runtime.active}
        nodes = sorted((set(dependencies) | {dep for values in dependencies.values() for dep in values}) & active_ids)
        color: dict[str, int] = {node: 0 for node in nodes}
        stack: list[str] = []

        def dfs(node: str) -> list[str] | None:
            color[node] = 1
            stack.append(node)
            for dependency in sorted(dependencies.get(node, set())):
                if dependency not in color:
                    continue
                if color[dependency] == 1:
                    index = stack.index(dependency)
                    return stack[index:] + [dependency]
                if color[dependency] == 0:
                    cycle = dfs(dependency)
                    if cycle:
                        return cycle
            stack.pop()
            color[node] = 2
            return None

        for node in nodes:
            if color[node] == 0:
                cycle = dfs(node)
                if cycle:
                    return cycle
        return None

    def _check_deadlocks(self, now: int) -> None:
        while True:
            dependencies = self.lock_manager.waiting_dependencies()
            cycle = self._find_cycle(dependencies)
            if cycle is None:
                return
            cycle_sessions = sorted(set(cycle[:-1]))
            candidates = [self.by_session[sid] for sid in cycle_sessions if sid in self.by_session]
            if not candidates:
                return
            victim = min(candidates, key=lambda runtime: (runtime.start_seq, runtime.spec.session_id))
            self._emit(
                "deadlock.detected",
                victim.spec.session_id,
                None,
                {
                    "cycle": cycle,
                    "cycle_sessions": cycle_sessions,
                    "dependencies": {sid: sorted(values) for sid, values in sorted(dependencies.items())},
                    "victim": victim.spec.session_id,
                    "arbitration_rule": "min(start_seq, session_id)",
                },
                now,
            )
            victim.wait_used_ms += 0 if victim.current_wait is None else max(
                0,
                now - victim.current_wait.requested_virtual_ms,
            )
            message = (
                f"deadlock cycle {cycle_sessions}; victim {victim.spec.session_id} "
                f"selected by min(start_seq, session_id)"
            )
            self._abort_session(
                victim,
                now,
                "E_DEADLOCK",
                "E_DEADLOCK",
                message,
                wait_ms_for_result=max(
                    0,
                    now - victim.current_wait.requested_virtual_ms,
                ) if victim.current_wait is not None else 0,
                extra_payload={
                    "cycle": cycle,
                    "cycle_sessions": cycle_sessions,
                    "arbitration_rule": "min(start_seq, session_id)",
                },
            )

    def _drain_grants(self, now: int) -> None:
        while True:
            notifications = self.lock_manager.drain_notifications()
            if not notifications:
                return
            for notification in notifications:
                runtime = self.by_session.get(notification.request.session_id)
                if runtime is None or runtime.status != "waiting_lock":
                    continue
                if runtime.current_wait is None or runtime.current_wait.request_seq != notification.request.request_seq:
                    continue
                runtime.current_grant_seq = notification.grant_seq
                runtime.current_grant_virtual_ms = notification.granted_virtual_ms
                self._write_wait_evidence(runtime, notification.granted_virtual_ms)
                runtime.wait_used_ms += runtime.current_operation.wait_ms if runtime.current_operation else 0
                runtime.status = "running"
                runtime.current_wait = None
                self._session_phase(
                    runtime,
                    "running",
                    now,
                    {
                        "grant_seq": notification.grant_seq,
                        "resumed_from_wait_ms": runtime.current_operation.wait_ms if runtime.current_operation else 0,
                    },
                )
                self._perform_operation(runtime, now)
            # A resumed operation may release a transient shared lock or commit and
            # unlock another queue; drain those grants at the same logical instant.
            continue

    def _activate_pending_sessions(self, now: int) -> None:
        while self._active_count() < self.params.global_concurrency_limit:
            pending = next(
                (runtime for runtime in self.runtimes if runtime.status == "pending"),
                None,
            )
            if pending is None:
                return
            self._start_session(pending, now)

    def _execute_round(self) -> None:
        self._drain_grants(self.now)
        self._check_deadlocks(self.now)
        self._check_timeouts(self.now)
        self._activate_pending_sessions(self.now)
        running = [runtime for runtime in self.runtimes if runtime.status == "running"]
        running.sort(key=lambda runtime: runtime.start_seq)
        for runtime in running:
            if runtime.status == "running":
                self._begin_operation(runtime, self.now)
        self._check_deadlocks(self.now)
        self._check_timeouts(self.now)
        self.lock_manager.pump(self.now)
        self._drain_grants(self.now)

    def _verify_isolation(self, result: RunResult) -> None:
        # Every read under repeatable_read must remain stable unless the same
        # session changed the row itself.
        for outcome in result.outcomes:
            if outcome.isolation_level != "repeatable_read":
                continue
            cached: dict[str, tuple[Any, int | None]] = {}
            for operation in outcome.operations:
                if operation.status != "success":
                    continue
                if operation.kind == "write":
                    cached.pop(operation.row_key, None)
                else:
                    current = (operation.value, operation.data_version)
                    previous = cached.get(operation.row_key)
                    if previous is not None and previous != current:
                        raise BusinessError(
                            "E_ISOLATION",
                            "repeatable_read returned two different values for the same row",
                            path=f"$.sessions.{outcome.session_id}.operations",
                            session_id=outcome.session_id,
                            op_no=operation.op_no,
                            details={
                                "row_key": operation.row_key,
                                "values": [previous, current],
                            },
                        )
                    cached[operation.row_key] = current

        # Replay commits in commit order and compare the complete final state.
        replay_values = dict(self.script.initial_rows if self.script else {})
        last_writer: dict[str, str] = {}
        for commit in result.commits:
            for write in commit.writes:
                replay_values[write["row_key"]] = write["value"]
                last_writer[write["row_key"]] = commit.session_id
        actual_values = {key: row["value"] for key, row in result.final_rows.items()}
        for row_key in sorted(set(actual_values) | set(replay_values)):
            if actual_values.get(row_key) != replay_values.get(row_key):
                raise BusinessError(
                    "E_ISOLATION",
                    "final database state differs from commit-order replay",
                    path=f"$.rows.{row_key}",
                    session_id=last_writer.get(row_key),
                    details={
                        "row_key": row_key,
                        "values": [actual_values.get(row_key), replay_values.get(row_key)],
                        "commit_order": [commit.to_dict() for commit in result.commits],
                    },
                )

    def run(self, script: EditScript) -> RunResult:
        self.script = script
        self.run_id = digest_json(
            {
                "script": script.semantic_dict(),
                "params": self.params.to_dict(),
            }
        )[:20]
        self.storage.reset_for_run(script, self.params, self.run_id)
        self.runtimes = [_Runtime(spec=spec, script_index=index) for index, spec in enumerate(script.sessions)]
        self.by_session = {runtime.spec.session_id: runtime for runtime in self.runtimes}
        started_monotonic = _monotonic_ms()
        self._emit(
            "run.start",
            None,
            None,
            {
                "script_id": script.script_id,
                "source_version": script.source_version,
                "normalized_version": script.normalized_version,
                "script_digest": script.semantic_digest,
                "params": self.params.to_dict(),
                "params_digest": self.params.digest,
                "run_id": self.run_id,
            },
            0,
        )
        max_rounds = 1_000_000
        rounds = 0
        while not all(runtime.terminal for runtime in self.runtimes):
            rounds += 1
            if rounds > max_rounds:
                raise BusinessError(
                    "E_ISOLATION",
                    "scheduler made no terminating progress",
                    path="$.scheduler",
                    details={"rounds": rounds},
                )
            self._execute_round()
            self.now += self.params.clock_quantum_ms

        if self.lock_manager.entry_count() != 0:
            raise BusinessError(
                "E_ISOLATION",
                "scheduler finished with residual locks or waiters",
                path="$.locks",
                details={"lock_state": self.lock_manager.state_dict()},
            )

        outcomes = sorted(
            (runtime.outcome for runtime in self.runtimes if runtime.outcome is not None),
            key=lambda outcome: (outcome.start_seq, outcome.session_id),
        )
        final_rows = self.storage.all_rows()
        self._emit(
            "run.end",
            None,
            None,
            {
                "global_data_version": self.storage.global_data_version(),
                "outcomes": [
                    {
                        "session_id": outcome.session_id,
                        "outcome": outcome.outcome,
                        "business_code": outcome.business_code,
                        "committed_data_version": outcome.committed_data_version,
                        "wait_used_ms": outcome.wait_used_ms,
                    }
                    for outcome in outcomes
                ],
                "final_rows": final_rows,
                "lock_entries": self.lock_manager.entry_count(),
            },
            self.now,
        )
        result = RunResult(
            script_id=script.script_id,
            source_version=script.source_version,
            normalized_version=script.normalized_version,
            script_digest=script.semantic_digest,
            params=self.params,
            run_id=self.run_id,
            initial_rows=dict(script.initial_rows),
            final_rows=final_rows,
            global_data_version=self.storage.global_data_version(),
            outcomes=outcomes,
            trace=self.trace,
            commits=self._commit_records,
            lock_state=self.lock_manager.state_dict(),
            started_monotonic_ms=started_monotonic,
            finished_monotonic_ms=_monotonic_ms(),
        )
        self._verify_isolation(result)
        self.storage.append_trace(run_id=self.run_id, script=script, trace=self.trace)
        self.storage.update_context_outcomes(
            script,
            self.params,
            self.run_id,
            [outcome.to_dict() for outcome in outcomes],
        )
        return result


def run_script(
    script: EditScript,
    *,
    db_path: str = ":memory:",
    params: SchedulerParams | None = None,
) -> tuple[RunResult, Storage]:
    storage = Storage(db_path)
    engine = Engine(storage, params or SchedulerParams())
    result = engine.run(script)
    return result, storage


def compare_run_outcomes(left: RunResult, right: RunResult) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if left.final_rows != right.final_rows:
        for row_key in sorted(set(left.final_rows) | set(right.final_rows)):
            if left.final_rows.get(row_key) != right.final_rows.get(row_key):
                differences.append(
                    {
                        "field": f"final_rows.{row_key}",
                        "left": left.final_rows.get(row_key),
                        "right": right.final_rows.get(row_key),
                    }
                )
    if left.global_data_version != right.global_data_version:
        differences.append(
            {
                "field": "global_data_version",
                "left": left.global_data_version,
                "right": right.global_data_version,
            }
        )
    left_outcomes = left.outcome_by_session()
    right_outcomes = right.outcome_by_session()
    for session_id in sorted(set(left_outcomes) | set(right_outcomes)):
        a = left_outcomes.get(session_id)
        b = right_outcomes.get(session_id)
        if a is None or b is None:
            differences.append({"field": f"sessions.{session_id}", "left": a, "right": b})
            continue
        for field in ("outcome", "business_code", "wait_used_ms", "committed_data_version"):
            if getattr(a, field) != getattr(b, field):
                differences.append(
                    {
                        "field": f"sessions.{session_id}.{field}",
                        "left": getattr(a, field),
                        "right": getattr(b, field),
                    }
                )
        for index, (left_op, right_op) in enumerate(
            zip_longest(a.operations, b.operations, fillvalue=None)
        ):
            if left_op is None or right_op is None:
                differences.append(
                    {
                        "field": f"sessions.{session_id}.operations[{index}]",
                        "left": left_op.to_dict() if left_op is not None else None,
                        "right": right_op.to_dict() if right_op is not None else None,
                    }
                )
                continue
            for field in (
                "op_no",
                "kind",
                "row_key",
                "status",
                "wait_ms",
                "grant_seq",
                "value",
                "data_version",
                "business_code",
            ):
                if getattr(left_op, field) != getattr(right_op, field):
                    differences.append(
                        {
                            "field": f"sessions.{session_id}.operations[{index}].{field}",
                            "left": getattr(left_op, field),
                            "right": getattr(right_op, field),
                        }
                    )
    return differences
