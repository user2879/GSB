"""Lease orchestration, pause/resume, and dependency-derived state."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from .declarations import DeclarationSet, RATIO_SCALE, SessionDefinition
from .errors import DependencyError, LeaseStateError
from .ledger import Ledger, RecoveryReport, RestoreReport, restore_from_snapshot
from .logging import StructuredLogger
from .probes import ProbeSupervisor
from .transport import ACK, UdpExchange, build_probe_batch


@dataclass
class RoundResult:
    session_id: str
    cycle: int
    attempt: int
    request_lease_version: int
    lease_version: int
    lease_renewed: bool
    status: str
    pause_reason: str | None
    ratio_bp: int | None
    acked: int
    total: int
    probes: tuple[dict[str, Any], ...]
    committed: bool = True
    dangling: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "cycle": self.cycle,
            "attempt": self.attempt,
            "request_lease_version": self.request_lease_version,
            "lease_version": self.lease_version,
            "lease_renewed": self.lease_renewed,
            "status": self.status,
            "pause_reason": self.pause_reason,
            "ratio_bp": self.ratio_bp,
            "acked": self.acked,
            "total": self.total,
            "probes": list(self.probes),
            "committed": self.committed,
            "dangling": self.dangling,
        }


class LeaseOrchestrator:
    def __init__(
        self,
        declarations: DeclarationSet,
        ledger_path: str | os.PathLike[str],
        *,
        logger: StructuredLogger | None = None,
        probe_work_dir: str | os.PathLike[str] | None = None,
        recover_incomplete: bool = True,
    ) -> None:
        self.declarations = declarations
        self.logger = logger or StructuredLogger()
        self.ledger = Ledger(
            ledger_path,
            declarations,
            logger=self.logger,
            recover_incomplete=recover_incomplete,
        )
        self.recovery_report: RecoveryReport = self.ledger.recovery_report
        self.dependency_changes_on_open = self.refresh_dependencies()
        self.probes = ProbeSupervisor(
            declarations, work_dir=probe_work_dir, logger=self.logger
        )

    def start_probes(self) -> "LeaseOrchestrator":
        self.probes.start()
        return self

    def stop_probes(self) -> None:
        self.probes.stop()

    def close(self) -> None:
        self.probes.close()
        self.ledger.close()

    def __enter__(self) -> "LeaseOrchestrator":
        self.start_probes()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _dependency_ready(self, session: SessionDefinition) -> bool:
        for dependency in session.depends_on:
            state = self.ledger.get_session(dependency)
            if state["status"] != "ACTIVE":
                return False
            last_ratio = state["last_ratio_bp"]
            dependency_session = self.declarations.session(dependency)
            if last_ratio is not None and last_ratio < dependency_session.ack_floor_bp:
                return False
        return True

    def refresh_dependencies(self) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        while True:
            changed_this_pass = False
            for session in self.declarations.sessions:
                state = self.ledger.get_session(session.session_id)
                if state["status"] == "TERMINATED":
                    continue
                ready = self._dependency_ready(session)
                if state["status"] == "ACTIVE" and not ready:
                    self.ledger.mark_paused_idle(
                        session_id=session.session_id,
                        reason="DEPENDENCY_LOST",
                        event_cycle=state["cycle_cursor"] + 1,
                        lease_version=state["lease_version"],
                        cursor=state["cycle_cursor"],
                        mono_ns=time.monotonic_ns(),
                        details={"depends_on": list(session.depends_on)},
                    )
                    changes.append(
                        {
                            "session_id": session.session_id,
                            "action": "paused",
                            "reason": "DEPENDENCY_LOST",
                        }
                    )
                    changed_this_pass = True
                elif (
                    state["status"] == "PAUSED"
                    and state["pause_reason"] == "DEPENDENCY_LOST"
                    and ready
                ):
                    self.ledger.mark_resumed(
                        session_id=session.session_id,
                        reason="DEPENDENCY_RESTORED",
                        event_cycle=state["cycle_cursor"] + 1,
                        lease_version=state["lease_version"],
                        cursor=state["cycle_cursor"],
                        mono_ns=time.monotonic_ns(),
                        reset_misses=False,
                        details={"depends_on": list(session.depends_on)},
                    )
                    changes.append(
                        {
                            "session_id": session.session_id,
                            "action": "resumed",
                            "reason": "DEPENDENCY_RESTORED",
                        }
                    )
                    changed_this_pass = True
            if not changed_this_pass:
                break
        return changes

    def execute_round(
        self,
        session_id: str,
        *,
        pause_after_send: bool = False,
        crash_before_commit: bool = False,
    ) -> RoundResult:
        session = self.declarations.session(session_id)
        state = self.ledger.get_session(session_id)
        if state["status"] == "PAUSED":
            raise LeaseStateError(
                "paused sessions cannot issue new leases",
                "sessions.status",
                {"session_id": session_id, "pause_reason": state["pause_reason"]},
            )
        if state["status"] == "TERMINATED":
            raise LeaseStateError(
                "terminated sessions cannot issue new leases",
                "sessions.status",
                {"session_id": session_id},
            )
        if not self._dependency_ready(session):
            self.refresh_dependencies()
            raise DependencyError(
                "session dependencies are not healthy",
                "sessions.depends_on",
                {"session_id": session_id, "depends_on": list(session.depends_on)},
            )

        cycle = int(state["cycle_cursor"]) + 1
        attempt = self.ledger.max_attempt(session_id, cycle) + 1
        start_ns = time.monotonic_ns()
        batch = build_probe_batch(
            session,
            self.declarations.probes,
            cycle=cycle,
            attempt=attempt,
            lease_version=int(state["lease_version"]),
            planned_mono_ns=start_ns,
        )
        self.ledger.stage_attempt(
            session_id=session_id,
            cycle=cycle,
            attempt=attempt,
            records=batch.records,
            started_mono_ns=start_ns,
        )
        self.logger.emit(
            "round.started",
            session_id=session_id,
            cycle=cycle,
            attempt=attempt,
            lease_version=int(state["lease_version"]),
            port=session.port,
        )
        exchange = UdpExchange(
            session,
            batch,
            cycle=cycle,
            lease_version=int(state["lease_version"]),
            attempt=attempt,
        )
        try:
            exchange.send_all()
            if pause_after_send:
                return self._pause_inflight(
                    session=session,
                    state=state,
                    cycle=cycle,
                    attempt=attempt,
                    records=exchange.records(),
                )
            exchange.collect()
            records = exchange.records()
        finally:
            exchange.close()
        if crash_before_commit:
            self.logger.emit(
                "process.forced_termination_injected",
                code="E_CRASH",
                session_id=session_id,
                cycle=cycle,
                attempt=attempt,
                no_exit_report=True,
            )
            os._exit(137)

        acked = sum(1 for record in records if record["outcome"] == ACK)
        total = len(records)
        ratio_bp = (acked * RATIO_SCALE + total // 2) // total
        renewed = ratio_bp >= session.ack_floor_bp
        request_lease_version = int(state["lease_version"])
        if renewed:
            lease_version = request_lease_version + 1
            consecutive_unrenewed = 0
            status_after = "ACTIVE"
            pause_reason_after = None
        else:
            lease_version = request_lease_version
            consecutive_unrenewed = int(state["consecutive_unrenewed"]) + 1
            if consecutive_unrenewed >= session.lease_miss_limit:
                status_after = "PAUSED"
                pause_reason_after = "LEASE_EXPIRED"
            else:
                status_after = "ACTIVE"
                pause_reason_after = None
        end_ns = time.monotonic_ns()
        pause_event = None
        if status_after == "PAUSED":
            pause_event = {
                "session_id": session_id,
                "kind": "PAUSE",
                "reason": "LEASE_EXPIRED",
                "cycle": cycle,
                "lease_version": lease_version,
                "cursor_before": int(state["cycle_cursor"]),
                "cursor_after": cycle,
                "mono_ns": end_ns,
                "details": {
                    "consecutive_unrenewed": consecutive_unrenewed,
                    "lease_miss_limit": session.lease_miss_limit,
                    "ratio_bp": ratio_bp,
                    "ack_floor_bp": session.ack_floor_bp,
                },
            }
        self.ledger.complete_round(
            session_id=session_id,
            cycle=cycle,
            attempt=attempt,
            start_mono_ns=start_ns,
            end_mono_ns=end_ns,
            acked=acked,
            total=total,
            ratio_bp=ratio_bp,
            request_lease_version=request_lease_version,
            lease_version=lease_version,
            lease_renewed=renewed,
            consecutive_unrenewed=consecutive_unrenewed,
            status_after=status_after,
            pause_reason_after=pause_reason_after,
            records=records,
            pause_event=pause_event,
        )
        self.refresh_dependencies()
        return RoundResult(
            session_id=session_id,
            cycle=cycle,
            attempt=attempt,
            request_lease_version=request_lease_version,
            lease_version=lease_version,
            lease_renewed=renewed,
            status=status_after,
            pause_reason=pause_reason_after,
            ratio_bp=ratio_bp,
            acked=acked,
            total=total,
            probes=tuple(records),
        )

    def _pause_inflight(
        self,
        *,
        session: SessionDefinition,
        state: dict[str, Any],
        cycle: int,
        attempt: int,
        records: Iterable[dict[str, Any]],
    ) -> RoundResult:
        record_list = list(records)
        now = time.monotonic_ns()
        dangling = [record["tx_id"] for record in record_list]
        event = {
            "session_id": session.session_id,
            "kind": "PAUSE",
            "reason": "USER_PAUSE_INFLIGHT",
            "cycle": cycle,
            "lease_version": int(state["lease_version"]),
            "cursor_before": int(state["cycle_cursor"]),
            "cursor_after": int(state["cycle_cursor"]),
            "mono_ns": now,
            "details": {
                "dangling_tx_ids": dangling,
                "dangling_probe_ids": [record["probe_id"] for record in record_list],
            },
        }
        self.ledger.commit_dangling_pause(
            session_id=session.session_id,
            cycle=cycle,
            attempt=attempt,
            events=[],
            records=record_list,
            pause_event=event,
            updated_mono_ns=now,
            lease_version=int(state["lease_version"]),
            cursor=int(state["cycle_cursor"]),
            pause_reason="USER_PAUSE_INFLIGHT",
        )
        self.refresh_dependencies()
        dangling_records = tuple(
            {
                **record,
                "outcome": "DANGLING_UNKNOWN",
                "reason": "session_paused_before_receive",
                "response_seq": None,
            }
            for record in record_list
        )
        return RoundResult(
            session_id=session.session_id,
            cycle=cycle,
            attempt=attempt,
            request_lease_version=int(state["lease_version"]),
            lease_version=int(state["lease_version"]),
            lease_renewed=False,
            status="PAUSED",
            pause_reason="USER_PAUSE_INFLIGHT",
            ratio_bp=None,
            acked=0,
            total=len(record_list),
            probes=dangling_records,
            committed=False,
            dangling=True,
        )

    def pause_session(self, session_id: str, reason: str = "USER_PAUSE") -> dict[str, Any]:
        session = self.declarations.session(session_id)
        state = self.ledger.get_session(session_id)
        if state["status"] == "PAUSED":
            return state
        if state["status"] == "TERMINATED":
            raise LeaseStateError(
                "terminated session cannot be paused", "sessions.status"
            )
        self.ledger.mark_paused_idle(
            session_id=session.session_id,
            reason=reason,
            event_cycle=int(state["cycle_cursor"]) + 1,
            lease_version=int(state["lease_version"]),
            cursor=int(state["cycle_cursor"]),
            mono_ns=time.monotonic_ns(),
        )
        self.refresh_dependencies()
        return self.ledger.get_session(session_id)

    def resume_session(self, session_id: str) -> dict[str, Any]:
        session = self.declarations.session(session_id)
        state = self.ledger.get_session(session_id)
        if state["status"] == "ACTIVE":
            return state
        if state["status"] == "TERMINATED":
            raise LeaseStateError(
                "terminated session cannot be resumed", "sessions.status"
            )
        if not self._dependency_ready(session):
            raise DependencyError(
                "session dependencies are not healthy",
                "sessions.depends_on",
                {"session_id": session_id, "depends_on": list(session.depends_on)},
            )
        self.ledger.mark_resumed(
            session_id=session_id,
            reason="USER_RESUME",
            event_cycle=int(state["cycle_cursor"]) + 1,
            lease_version=int(state["lease_version"]),
            cursor=int(state["cycle_cursor"]),
            mono_ns=time.monotonic_ns(),
            reset_misses=True,
        )
        self.refresh_dependencies()
        return self.ledger.get_session(session_id)

    def terminate_session(self, session_id: str, reason: str = "FORCE_TERMINATE") -> dict[str, Any]:
        session = self.declarations.session(session_id)
        state = self.ledger.get_session(session_id)
        if state["status"] == "TERMINATED":
            return state
        self.ledger.mark_terminated(
            session_id=session.session_id,
            reason=reason,
            event_cycle=int(state["cycle_cursor"]) + 1,
            lease_version=int(state["lease_version"]),
            cursor=int(state["cycle_cursor"]),
            mono_ns=time.monotonic_ns(),
        )
        self.refresh_dependencies()
        return self.ledger.get_session(session_id)

    def query_lease(self, session_id: str) -> dict[str, Any]:
        session = self.declarations.session(session_id)
        state = self.ledger.get_session(session_id)
        return {
            **state,
            "port": session.port,
            "probe_ids": list(session.probe_ids),
            "ack_floor_bp": session.ack_floor_bp,
            "lease_miss_limit": session.lease_miss_limit,
            "remaining_miss_budget": max(
                0, session.lease_miss_limit - int(state["consecutive_unrenewed"])
            ),
            "dependencies_ready": self._dependency_ready(session),
            "expires_by_round": (
                None
                if state["status"] != "ACTIVE"
                else int(state["cycle_cursor"])
                + max(1, session.lease_miss_limit - int(state["consecutive_unrenewed"]))
            ),
        }

    def query_leases(self) -> list[dict[str, Any]]:
        return [self.query_lease(session.session_id) for session in self.declarations.sessions]

    def topological_sessions(self) -> list[SessionDefinition]:
        order: list[SessionDefinition] = []
        visited: set[str] = set()

        def visit(session: SessionDefinition) -> None:
            if session.session_id in visited:
                return
            for dependency in session.depends_on:
                visit(self.declarations.session(dependency))
            visited.add(session.session_id)
            order.append(session)

        for session in self.declarations.sessions:
            visit(session)
        return order

    def run_cycles(
        self, session_id: str, count: int, *, tick: bool = False
    ) -> list[RoundResult]:
        if count < 1:
            return []
        session = self.declarations.session(session_id)
        results: list[RoundResult] = []
        previous_start: int | None = None
        for _ in range(count):
            if tick and previous_start is not None:
                target = previous_start + session.poll_period_ms * 1_000_000
                remaining = target - time.monotonic_ns()
                if remaining > 0:
                    time.sleep(remaining / 1_000_000_000)
            state = self.ledger.get_session(session_id)
            if state["status"] != "ACTIVE":
                break
            start = time.monotonic_ns()
            result = self.execute_round(session_id)
            previous_start = start
            results.append(result)
            if result.status != "ACTIVE":
                break
        return results

    def run_all(self, cycles: int, *, tick: bool = False) -> dict[str, list[RoundResult]]:
        output: dict[str, list[RoundResult]] = {
            session.session_id: [] for session in self.declarations.sessions
        }
        previous: dict[str, int] = {}
        for _ in range(cycles):
            for session in self.topological_sessions():
                if tick and session.session_id in previous:
                    target = previous[session.session_id] + session.poll_period_ms * 1_000_000
                    remaining = target - time.monotonic_ns()
                    if remaining > 0:
                        time.sleep(remaining / 1_000_000_000)
                state = self.ledger.get_session(session.session_id)
                if state["status"] != "ACTIVE" or not self._dependency_ready(session):
                    continue
                started = time.monotonic_ns()
                result = self.execute_round(session.session_id)
                previous[session.session_id] = started
                output[session.session_id].append(result)
        return output

    def save_snapshot(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        return self.ledger.save_snapshot(path)

    def restore_ledger(
        self, snapshot_path: str | os.PathLike[str]
    ) -> RestoreReport:
        self.probes.stop()
        self.ledger.close()
        report = restore_from_snapshot(
            snapshot_path=snapshot_path,
            db_path=self.ledger.path,
            declarations=self.declarations,
            logger=self.logger,
        )
        self.ledger = Ledger(
            self.ledger.path,
            self.declarations,
            logger=self.logger,
            recover_incomplete=True,
        )
        self.recovery_report = self.ledger.recovery_report
        return report
