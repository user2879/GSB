from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .errors import VersionMismatchError, issue_record
from .models import (
    BorrowLoan,
    BorrowRule,
    ConnectionState,
    ParameterSnapshot,
    PoolConfig,
    TraceEvent,
    WarningRecord,
)


ACTIVE_CONNECTION_STATES = {"established", "idle"}
TERMINAL_CONNECTION_STATES = {"released", "closed"}
ROUNDING_MODE = "floor_toward_negative_infinity"


@dataclass
class PoolRuntime:
    config: PoolConfig
    own_active: int = 0
    borrowed_in: int = 0
    lent_out: int = 0

    @property
    def active_connections(self) -> int:
        return self.own_active + self.borrowed_in

    @property
    def capacity_in_use(self) -> int:
        return self.own_active + self.lent_out

    @property
    def remaining_connections(self) -> int:
        return self.config.max_connections - self.capacity_in_use

    @property
    def active_with_lent(self) -> int:
        return self.own_active + self.borrowed_in + self.lent_out


@dataclass
class EventProcessingResult:
    event: TraceEvent
    issues: list[dict[str, Any]]
    warnings: list[WarningRecord]
    audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "issues": self.issues,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "audit": self.audit,
        }


class QuotaEngine:
    """Connection-level quota state machine and ownership ledger."""

    def __init__(self, snapshot: ParameterSnapshot, signal_broker: Any | None = None):
        self.snapshot = snapshot
        self.signal_broker = signal_broker
        self.pools: dict[str, PoolRuntime] = {
            pool_id: PoolRuntime(config=config) for pool_id, config in sorted(snapshot.pools.items())
        }
        self.connections: dict[str, ConnectionState] = {}
        self.loans: dict[str, BorrowLoan] = {}
        self.warnings: list[WarningRecord] = []
        self.parameter_history: list[ParameterSnapshot] = [snapshot]
        self.transitions: list[dict[str, Any]] = []
        self.unknown_connections: set[str] = set()
        self.last_event_seq = 0
        self.last_event_ms: int | None = None
        self.session_start_ms: int | None = None
        self._arrivals: dict[str, np.ndarray] = {
            pool_id: np.asarray([], dtype=np.int64) for pool_id in self.pools
        }
        self._loan_counter = 0
        self._warning_counter = 0
        self._current_warning_by_pool: dict[str, str] = {}
        self._event_issues: list[dict[str, Any]] = []
        self._event_warnings: list[WarningRecord] = []

    @classmethod
    def from_restored(
        cls,
        *,
        snapshot: ParameterSnapshot,
        connections: Iterable[ConnectionState],
        loans: Iterable[BorrowLoan],
        warnings: Iterable[WarningRecord],
        runtime: dict[str, Any],
        signal_broker: Any | None = None,
        parameter_history: Iterable[ParameterSnapshot] | None = None,
    ) -> "QuotaEngine":
        engine = cls(snapshot, signal_broker=signal_broker)
        engine.connections = {connection.connection_id: connection for connection in connections}
        engine.loans = {loan.loan_id: loan for loan in loans}
        engine.warnings = sorted(list(warnings), key=lambda item: (item.event_seq, item.warning_id))
        for pool in engine.pools.values():
            pool.own_active = 0
            pool.borrowed_in = 0
            pool.lent_out = 0
        for connection in sorted(engine.connections.values(), key=lambda item: item.connection_id):
            if connection.state not in ACTIVE_CONNECTION_STATES:
                continue
            logical_pool = engine.pools.get(connection.logical_pool)
            if logical_pool is None:
                raise ValueError(f"active connection {connection.connection_id} references unknown pool")
            if connection.loan_id:
                loan = engine.loans.get(connection.loan_id)
                if loan is None or not loan.outstanding:
                    raise ValueError(f"active connection {connection.connection_id} has no outstanding loan")
                logical_pool.borrowed_in += 1
                lender = engine.pools.get(loan.lender_pool)
                if lender is None:
                    raise ValueError(f"loan {loan.loan_id} references unknown lender")
                lender.lent_out += 1
            else:
                logical_pool.own_active += 1
        for loan in engine.loans.values():
            if loan.outstanding and not loan.stale:
                connection = engine.connections.get(loan.connection_id)
                if connection is None or connection.state not in ACTIVE_CONNECTION_STATES:
                    raise ValueError(f"outstanding loan {loan.loan_id} has no active connection")
        engine._arrivals = {
            pool_id: np.asarray(runtime.get("arrivals", {}).get(pool_id, []), dtype=np.int64)
            for pool_id in engine.pools
        }
        engine.last_event_seq = int(runtime.get("last_event_seq", 0))
        last_ms = runtime.get("last_event_ms")
        engine.last_event_ms = int(last_ms) if last_ms is not None else None
        start_ms = runtime.get("session_start_ms")
        engine.session_start_ms = int(start_ms) if start_ms is not None else None
        engine._loan_counter = int(runtime.get("loan_counter", 0))
        engine._warning_counter = int(runtime.get("warning_counter", 0))
        engine._current_warning_by_pool = dict(runtime.get("warning_current", {}))
        engine.unknown_connections = set(runtime.get("unknown_connections", []))
        if parameter_history:
            engine.parameter_history = sorted(list(parameter_history), key=lambda item: item.version)
        engine._assert_invariants()
        return engine

    @property
    def parameter_version(self) -> int:
        return self.snapshot.version

    def set_unknown_connections(self, connection_ids: Iterable[str]) -> None:
        self.unknown_connections = set(connection_ids)

    def process_event(self, event: TraceEvent) -> EventProcessingResult:
        if event.seq <= self.last_event_seq:
            raise ValueError(f"event seq {event.seq} is not greater than current seq {self.last_event_seq}")
        self._event_issues = []
        self._event_warnings = []
        if self.session_start_ms is None:
            self.session_start_ms = event.timestamp_ms

        self._check_overdue(event.timestamp_ms, event.seq)
        if event.event_type == "establish":
            self._handle_establish(event)
        elif event.event_type == "request":
            self._handle_request(event)
        elif event.event_type == "idle_release":
            self._handle_idle_release(event)
        elif event.event_type == "close":
            self._handle_close(event)
        else:
            raise ValueError(f"unsupported event type: {event.event_type}")

        self.last_event_seq = event.seq
        self.last_event_ms = event.timestamp_ms
        self._event_warnings = self._compute_warnings(
            event.seq,
            event.timestamp_ms,
            context="EVENT",
        )
        self._assert_invariants()
        audit = self._build_audit(event)
        return EventProcessingResult(
            event=event,
            issues=list(self._event_issues),
            warnings=list(self._event_warnings),
            audit=audit,
        )

    def _add_issue(
        self,
        code: str,
        message: str,
        *,
        event: TraceEvent | None = None,
        event_seq: int | None = None,
        pool_id: str | None = None,
        connection_id: str | None = None,
        severity: str = "ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        self._event_issues.append(
            issue_record(
                code,
                message,
                event_seq=event.seq if event is not None else event_seq,
                pool_id=pool_id,
                connection_id=connection_id,
                severity=severity,
                details=details,
            )
        )

    def _handle_establish(self, event: TraceEvent) -> None:
        existing = self.connections.get(event.connection)
        if existing is not None and existing.state != "unknown":
            self._add_issue(
                "E_LIFECYCLE",
                "connection is already known and cannot be established again",
                event=event,
                pool_id=event.pool,
                connection_id=event.connection,
                details={"state": existing.state},
            )
            return
        if existing is not None:
            self.connections.pop(event.connection, None)
            self.unknown_connections.discard(event.connection)

        pool = self.pools[event.pool]
        if pool.own_active < pool.config.max_connections and pool.capacity_in_use < pool.config.max_connections:
            connection = ConnectionState(
                connection_id=event.connection,
                logical_pool=event.pool,
                origin_pool=event.pool,
                current_owner_pool=event.pool,
                state="established",
                loan_id=None,
                established_seq=event.seq,
                established_ms=event.timestamp_ms,
                last_activity_ms=event.timestamp_ms,
            )
            pool.own_active += 1
            self.connections[event.connection] = connection
        else:
            allocation = self._allocate_borrow(event)
            if allocation is None:
                self._add_issue(
                    "E_POOLFULL",
                    "establish rejected: no borrow relation has usable lender capacity",
                    event=event,
                    pool_id=event.pool,
                    connection_id=event.connection,
                    details={
                        "resolution": "REJECTED_NO_USABLE_BORROW_CAPACITY",
                        "max_connections": pool.config.max_connections,
                        "own_active": pool.own_active,
                        "capacity_in_use": pool.capacity_in_use,
                    },
                )
                return
            loan = allocation
            connection = ConnectionState(
                connection_id=event.connection,
                logical_pool=event.pool,
                origin_pool=event.pool,
                current_owner_pool=loan.lender_pool,
                state="established",
                loan_id=loan.loan_id,
                established_seq=event.seq,
                established_ms=event.timestamp_ms,
                last_activity_ms=event.timestamp_ms,
            )
            pool.borrowed_in += 1
            self.pools[loan.lender_pool].lent_out += 1
            self.connections[event.connection] = connection
            self._add_issue(
                "E_POOLFULL",
                "base overflow resolved by borrowing one connection",
                event=event,
                pool_id=event.pool,
                connection_id=event.connection,
                severity="WARNING",
                details={
                    "resolution": "BORROW_ACCEPTED",
                    "max_connections": pool.config.max_connections,
                    "own_active": pool.own_active,
                    "capacity_in_use": pool.capacity_in_use,
                    "loan_id": loan.loan_id,
                    "lender_pool": loan.lender_pool,
                    "deadline_ms": loan.deadline_ms,
                },
            )

        self.unknown_connections.discard(event.connection)
        if self.signal_broker is not None:
            try:
                signal = self.signal_broker.establish(event.connection)
            except Exception as exc:  # loopback failures are expected business failures
                self._rollback_establish(event.connection)
                self._add_issue(
                    "E_LOOPBACK",
                    "loopback establish handshake failed; connection was not accepted",
                    event=event,
                    pool_id=event.pool,
                    connection_id=event.connection,
                    details={"error": str(exc)},
                )
                return
        self._record_arrival(event.pool, event.timestamp_ms)

    def _rollback_establish(self, connection_id: str) -> None:
        connection = self.connections.pop(connection_id, None)
        if connection is None:
            return
        pool = self.pools[connection.logical_pool]
        if connection.loan_id:
            loan = self.loans.pop(connection.loan_id, None)
            pool.borrowed_in = max(0, pool.borrowed_in - 1)
            if loan is not None:
                lender = self.pools.get(loan.lender_pool)
                if lender is not None:
                    lender.lent_out = max(0, lender.lent_out - 1)
        else:
            pool.own_active = max(0, pool.own_active - 1)

    def _allocate_borrow(self, event: TraceEvent) -> BorrowLoan | None:
        for rule in self.snapshot.sorted_rules():
            if rule.borrower_pool != event.pool:
                continue
            if self._outstanding_rule_loans(rule) >= rule.max_borrowed:
                continue
            lender = self.pools[rule.lender_pool]
            if lender.capacity_in_use >= lender.config.max_connections:
                continue
            self._loan_counter += 1
            loan = BorrowLoan(
                loan_id=f"loan-{self._loan_counter:06d}",
                lender_pool=rule.lender_pool,
                borrower_pool=event.pool,
                connection_id=event.connection,
                amount=1,
                borrowed_at_seq=event.seq,
                borrowed_at_ms=event.timestamp_ms,
                deadline_ms=event.timestamp_ms + rule.return_deadline_ms,
                status="active",
                parameter_version=self.parameter_version,
            )
            self.loans[loan.loan_id] = loan
            return loan
        return None

    def _outstanding_rule_loans(self, rule: BorrowRule) -> int:
        return sum(
            1
            for loan in self.loans.values()
            if loan.outstanding
            and not loan.stale
            and loan.parameter_version == self.parameter_version
            and loan.lender_pool == rule.lender_pool
            and loan.borrower_pool == rule.borrower_pool
        )

    def _handle_request(self, event: TraceEvent) -> None:
        connection = self.connections.get(event.connection)
        if connection is None or connection.state not in ACTIVE_CONNECTION_STATES:
            self._add_issue(
                "E_LIFECYCLE",
                "request rejected: connection is not established or idle",
                event=event,
                pool_id=event.pool,
                connection_id=event.connection,
                details={"state": connection.state if connection else "missing"},
            )
            return
        if connection.logical_pool != event.pool:
            self._add_issue(
                "E_LIFECYCLE",
                "request rejected: event pool differs from connection's logical pool",
                event=event,
                pool_id=event.pool,
                connection_id=event.connection,
                details={"connection_pool": connection.logical_pool},
            )
            return
        config = self.pools[connection.logical_pool].config
        if event.bytes_count > config.read_buffer_limit_bytes:
            self._add_issue(
                "E_BUFFER",
                "request rejected: byte count exceeds read buffer limit",
                event=event,
                pool_id=event.pool,
                connection_id=event.connection,
                details={
                    "bytes": event.bytes_count,
                    "read_buffer_limit_bytes": config.read_buffer_limit_bytes,
                },
            )
            return
        if connection.request_count >= config.per_connection_request_limit:
            self._add_issue(
                "E_CONNLIMIT",
                "request rejected: per-connection request limit exhausted",
                event=event,
                pool_id=event.pool,
                connection_id=event.connection,
                details={
                    "request_count": connection.request_count,
                    "per_connection_request_limit": config.per_connection_request_limit,
                },
            )
            return
        connection.request_count += 1
        connection.last_activity_ms = event.timestamp_ms
        connection.state = "idle"

    def _handle_idle_release(self, event: TraceEvent) -> None:
        connection = self.connections.get(event.connection)
        if connection is None or connection.state not in ACTIVE_CONNECTION_STATES:
            self._add_issue(
                "E_LIFECYCLE",
                "idle_release rejected: connection is not established or idle",
                event=event,
                pool_id=event.pool,
                connection_id=event.connection,
                details={"state": connection.state if connection else "missing"},
            )
            return
        if connection.logical_pool != event.pool:
            self._add_issue(
                "E_LIFECYCLE",
                "idle_release rejected: event pool differs from connection's logical pool",
                event=event,
                pool_id=event.pool,
                connection_id=event.connection,
                details={"connection_pool": connection.logical_pool},
            )
            return
        config = self.pools[connection.logical_pool].config
        idle_ms = event.timestamp_ms - connection.last_activity_ms
        if idle_ms < config.idle_release_ms:
            self._add_issue(
                "E_LIFECYCLE",
                "idle_release rejected: idle threshold has not elapsed",
                event=event,
                pool_id=event.pool,
                connection_id=event.connection,
                details={
                    "idle_ms": idle_ms,
                    "idle_release_ms": config.idle_release_ms,
                },
            )
            return
        self._release_connection(connection, event, new_state="released")

    def _handle_close(self, event: TraceEvent) -> None:
        connection = self.connections.get(event.connection)
        if connection is None or connection.state not in ACTIVE_CONNECTION_STATES:
            self._add_issue(
                "E_LIFECYCLE",
                "close rejected: connection is not established or idle",
                event=event,
                pool_id=event.pool,
                connection_id=event.connection,
                details={"state": connection.state if connection else "missing"},
            )
            return
        if connection.logical_pool != event.pool:
            self._add_issue(
                "E_LIFECYCLE",
                "close rejected: event pool differs from connection's logical pool",
                event=event,
                pool_id=event.pool,
                connection_id=event.connection,
                details={"connection_pool": connection.logical_pool},
            )
            return
        self._release_connection(connection, event, new_state="closed")

    def _release_connection(self, connection: ConnectionState, event: TraceEvent, *, new_state: str) -> None:
        pool = self.pools[connection.logical_pool]
        if connection.loan_id:
            loan = self.loans.get(connection.loan_id)
            pool.borrowed_in = max(0, pool.borrowed_in - 1)
            if loan is not None and loan.outstanding:
                lender = self.pools[loan.lender_pool]
                lender.lent_out = max(0, lender.lent_out - 1)
                if event.timestamp_ms >= loan.deadline_ms:
                    loan.status = "returned_late"
                else:
                    loan.status = "returned"
                loan.returned_at_seq = event.seq
                loan.returned_at_ms = event.timestamp_ms
        else:
            pool.own_active = max(0, pool.own_active - 1)
        connection.state = new_state
        connection.last_activity_ms = event.timestamp_ms
        if self.signal_broker is not None:
            try:
                self.signal_broker.close(event.connection)
            except Exception as exc:
                self._add_issue(
                    "E_LOOPBACK",
                    "loopback close handshake failed; state release still recorded",
                    event=event,
                    pool_id=event.pool,
                    connection_id=event.connection,
                    details={"error": str(exc)},
                )

    def _check_overdue(self, timestamp_ms: int, event_seq: int) -> None:
        for loan in sorted(self.loans.values(), key=lambda item: item.loan_id):
            if loan.status == "active" and timestamp_ms >= loan.deadline_ms:
                loan.status = "overdue"
                loan.overdue_at_seq = event_seq
                self._add_issue(
                    "E_OVERDUE",
                    "borrowed connection has not been returned before its deadline",
                    event_seq=event_seq,
                    pool_id=loan.borrower_pool,
                    connection_id=loan.connection_id,
                    details={
                        "loan_id": loan.loan_id,
                        "lender_pool": loan.lender_pool,
                        "deadline_ms": loan.deadline_ms,
                        "observed_ms": timestamp_ms,
                        "overdue_by_ms": timestamp_ms - loan.deadline_ms,
                    },
                )

    def _record_arrival(self, pool_id: str, timestamp_ms: int) -> None:
        existing = self._arrivals.get(pool_id)
        if existing is None:
            existing = np.asarray([], dtype=np.int64)
        self._arrivals[pool_id] = np.append(existing, np.int64(timestamp_ms))

    def _compute_warnings(self, event_seq: int, timestamp_ms: int, *, context: str) -> list[WarningRecord]:
        created: list[WarningRecord] = []
        for pool_id, pool in sorted(self.pools.items()):
            remaining = pool.remaining_connections
            if remaining > self.snapshot.alert_threshold:
                self._current_warning_by_pool.pop(pool_id, None)
                continue
            arrival_rate = self._arrival_diagnostic(pool_id, timestamp_ms)
            eta = self._eta_diagnostic(remaining, arrival_rate)
            self._warning_counter += 1
            warning = WarningRecord(
                warning_id=f"warn-{self._warning_counter:06d}",
                event_seq=event_seq,
                pool_id=pool_id,
                parameter_version=self.parameter_version,
                remaining_connections=remaining,
                capacity_in_use=pool.capacity_in_use,
                active_connections=pool.active_connections,
                own_active=pool.own_active,
                borrowed_in=pool.borrowed_in,
                lent_out=pool.lent_out,
                max_connections=pool.config.max_connections,
                alert_threshold=self.snapshot.alert_threshold,
                waterline_percentage_fixed=(
                    pool.capacity_in_use * self.snapshot.percentage_scale // pool.config.max_connections
                ),
                percentage_scale=self.snapshot.percentage_scale,
                rounding=ROUNDING_MODE,
                arrival_rate=arrival_rate,
                exhaustion_eta=eta,
                context=context,
            )
            self.warnings.append(warning)
            self._current_warning_by_pool[pool_id] = warning.warning_id
            created.append(warning)
        return created

    def _arrival_diagnostic(self, pool_id: str, timestamp_ms: int) -> dict[str, Any]:
        window_ms = self.snapshot.rate_window_ms
        observations = self._arrivals.get(pool_id, np.asarray([], dtype=np.int64))
        lower_exclusive = timestamp_ms - window_ms
        left = int(np.searchsorted(observations, lower_exclusive, side="right"))
        right = int(np.searchsorted(observations, timestamp_ms, side="right"))
        arrivals = max(0, right - left)
        warmup_complete = (
            self.session_start_ms is not None
            and timestamp_ms - self.session_start_ms >= window_ms
        )
        if not warmup_complete:
            reason = "WINDOW_WARMUP_INCOMPLETE"
            estimable = False
        elif arrivals == 0:
            reason = "NO_ESTABLISH_IN_WINDOW"
            estimable = False
        else:
            reason = "OK"
            estimable = True
        rate_fixed = None
        if estimable:
            rate_fixed = arrivals * self.snapshot.percentage_scale * 1000 // window_ms
        return {
            "estimable": estimable,
            "reason": reason,
            "window_start_exclusive_ms": lower_exclusive,
            "window_end_inclusive_ms": timestamp_ms,
            "window_ms": window_ms,
            "arrivals_in_window": arrivals,
            "rate_events_per_second_fixed": rate_fixed,
            "rate_scale": self.snapshot.percentage_scale,
            "rounding": ROUNDING_MODE,
        }

    @staticmethod
    def _eta_diagnostic(remaining: int, arrival_rate: dict[str, Any]) -> dict[str, Any]:
        window_ms = int(arrival_rate["window_ms"])
        if remaining <= 0:
            return {
                "estimable": True,
                "reason": "ALREADY_AT_THRESHOLD",
                "eta_ms": 0,
                "estimation_window_ms": window_ms,
                "error_bound_ms": 0,
                "formula": "remaining<=0 => 0",
            }
        if not arrival_rate["estimable"]:
            return {
                "estimable": False,
                "reason": arrival_rate["reason"],
                "eta_ms": None,
                "estimation_window_ms": window_ms,
                "error_bound_ms": None,
                "formula": "ceil(remaining * window_ms / arrivals_in_window)",
            }
        arrivals = int(arrival_rate["arrivals_in_window"])
        return {
            "estimable": True,
            "reason": "OK",
            "eta_ms": math.ceil(remaining * window_ms / arrivals),
            "estimation_window_ms": window_ms,
            "error_bound_ms": window_ms,
            "formula": "ceil(remaining * window_ms / arrivals_in_window)",
        }

    def apply_parameters(
        self,
        new_snapshot: ParameterSnapshot,
        *,
        after_event_seq: int,
        at_timestamp_ms: int | None = None,
    ) -> dict[str, Any]:
        if new_snapshot.version <= self.parameter_version:
            raise VersionMismatchError(
                "parameter version must increase monotonically",
                path="parameter.version",
                details={"current_version": self.parameter_version, "new_version": new_snapshot.version},
            )
        old_snapshot = self.snapshot
        old_current_warning_ids = dict(self._current_warning_by_pool)
        old_warnings_by_id = {warning.warning_id: warning for warning in self.warnings}
        old_active_connections_total = sum(pool.active_connections for pool in self.pools.values())
        at_ms = at_timestamp_ms if at_timestamp_ms is not None else (self.last_event_ms or 0)

        invalidated_borrows: list[dict[str, Any]] = []
        stale_legacy_borrows: list[dict[str, Any]] = []
        for loan in sorted(self.loans.values(), key=lambda item: item.loan_id):
            if loan.parameter_version < new_snapshot.version and not loan.stale:
                was_outstanding = loan.outstanding
                if was_outstanding:
                    loan.status = "stale"
                loan.stale = True
                loan.stale_reason = "PARAMETER_VERSION_REPLACED"
                if was_outstanding:
                    invalidated_borrows.append(loan.to_dict())
                else:
                    stale_legacy_borrows.append(loan.to_dict())

        for warning in self.warnings:
            if not warning.stale:
                warning.stale = True
                warning.stale_reason = "PARAMETER_VERSION_REPLACED"
                warning.stale_at_version = new_snapshot.version
        self._current_warning_by_pool.clear()

        active_connections = [
            connection
            for connection in self.connections.values()
            if connection.state in ACTIVE_CONNECTION_STATES
        ]
        active_connections.sort(key=lambda item: (item.established_seq, item.connection_id))
        self.snapshot = new_snapshot
        self.pools = {
            pool_id: PoolRuntime(config=config) for pool_id, config in sorted(new_snapshot.pools.items())
        }
        self._arrivals = {pool_id: np.asarray([], dtype=np.int64) for pool_id in self.pools}
        self.session_start_ms = at_ms
        self.last_event_ms = at_ms
        self.unknown_connections.clear()
        reconstituted_loans: list[dict[str, Any]] = []

        for connection in active_connections:
            pool = self.pools.get(connection.logical_pool)
            if pool is None:
                connection.state = "unknown"
                connection.current_owner_pool = ""
                connection.loan_id = None
                self.unknown_connections.add(connection.connection_id)
                continue
            connection.loan_id = None
            if pool.own_active < pool.config.max_connections and pool.capacity_in_use < pool.config.max_connections:
                pool.own_active += 1
                connection.current_owner_pool = connection.logical_pool
                continue
            synthetic_event = TraceEvent(
                seq=after_event_seq,
                pool=connection.logical_pool,
                connection=connection.connection_id,
                event_type="establish",
                timestamp_ms=at_ms,
                bytes_count=0,
                line=0,
                raw={},
            )
            loan = self._allocate_borrow(synthetic_event)
            if loan is None:
                connection.state = "unknown"
                connection.current_owner_pool = ""
                self.unknown_connections.add(connection.connection_id)
                continue
            pool.borrowed_in += 1
            self.pools[loan.lender_pool].lent_out += 1
            connection.current_owner_pool = loan.lender_pool
            connection.loan_id = loan.loan_id
            reconstituted_loans.append(loan.to_dict())

        transition_warnings = self._compute_warnings(
            after_event_seq,
            at_ms,
            context="PARAMETER_TRANSITION",
        )
        current_warning_by_pool = dict(self._current_warning_by_pool)
        cleared_alerts: list[dict[str, Any]] = []
        for pool_id, warning_id in sorted(old_current_warning_ids.items()):
            warning = old_warnings_by_id.get(warning_id)
            if warning is None:
                continue
            if pool_id in current_warning_by_pool:
                continue
            pool = self.pools.get(pool_id)
            new_remaining = pool.remaining_connections if pool else None
            cleared_alerts.append(
                {
                    "warning_id": warning.warning_id,
                    "pool_id": pool_id,
                    "event_seq": warning.event_seq,
                    "old_remaining_connections": warning.remaining_connections,
                    "new_remaining_connections": new_remaining,
                    "old_alert_threshold": old_snapshot.alert_threshold,
                    "new_alert_threshold": new_snapshot.alert_threshold,
                }
            )

        affected_pools = self._affected_pools(old_snapshot, new_snapshot)
        old_rule_map = {
            (rule.lender_pool, rule.borrower_pool): rule.to_dict()
            for rule in old_snapshot.sorted_rules()
        }
        new_rule_map = {
            (rule.lender_pool, rule.borrower_pool): rule.to_dict()
            for rule in new_snapshot.sorted_rules()
        }
        invalidated_borrow_relations = [
            old_rule_map[key]
            for key in sorted(old_rule_map)
            if old_rule_map.get(key) != new_rule_map.get(key)
        ]
        added_borrow_relations = [
            new_rule_map[key]
            for key in sorted(new_rule_map)
            if new_rule_map.get(key) != old_rule_map.get(key)
        ]
        transition = {
            "version": new_snapshot.version,
            "previous_version": old_snapshot.version,
            "after_event_seq": after_event_seq,
            "at_timestamp_ms": at_ms,
            "affected_pools": affected_pools,
            "cleared_alert_records": cleared_alerts,
            "invalidated_borrows": invalidated_borrows,
            "stale_legacy_borrows": stale_legacy_borrows,
            "invalidated_borrow_relations": invalidated_borrow_relations,
            "added_borrow_relations": added_borrow_relations,
            "reconstituted_loans": reconstituted_loans,
            "unknown_connections": sorted(self.unknown_connections),
            "current_warnings": [warning.to_dict() for warning in transition_warnings],
            "current_warning_ids_by_pool": dict(sorted(current_warning_by_pool.items())),
            "summary": {
                "old_parameter_version": old_snapshot.version,
                "new_parameter_version": new_snapshot.version,
                "old_active_connections_total": old_active_connections_total,
                "replaced_view": True,
                "stale_alert_count": len(old_current_warning_ids),
                "cleared_alert_count": len(cleared_alerts),
                "invalidated_borrow_count": len(invalidated_borrows),
                "stale_legacy_borrow_count": len(stale_legacy_borrows),
                "stale_parameter_versions": [old_snapshot.version],
            },
        }
        self.parameter_history.append(new_snapshot)
        self.transitions.append(transition)
        self._assert_invariants()
        return transition

    @staticmethod
    def _affected_pools(old: ParameterSnapshot, new: ParameterSnapshot) -> list[str]:
        affected: set[str] = set()
        all_pool_ids = set(old.pools) | set(new.pools)
        if old.alert_threshold != new.alert_threshold:
            affected.update(all_pool_ids)
        for pool_id in all_pool_ids:
            old_pool = old.pools.get(pool_id)
            new_pool = new.pools.get(pool_id)
            if old_pool != new_pool:
                affected.add(pool_id)
        old_rules = {(rule.lender_pool, rule.borrower_pool, rule.max_borrowed, rule.return_deadline_ms) for rule in old.borrow_rules}
        new_rules = {(rule.lender_pool, rule.borrower_pool, rule.max_borrowed, rule.return_deadline_ms) for rule in new.borrow_rules}
        if old_rules != new_rules:
            for lender, borrower, _, _ in old_rules | new_rules:
                affected.add(lender)
                affected.add(borrower)
        return sorted(affected)

    def _build_audit(self, event: TraceEvent) -> dict[str, Any]:
        ownership: dict[str, dict[str, Any]] = {}
        active_total = 0
        capacity_total = 0
        for pool_id, pool in sorted(self.pools.items()):
            active_total += pool.active_connections
            capacity_total += pool.capacity_in_use
        for connection in sorted(self.connections.values(), key=lambda item: item.connection_id):
            if connection.state not in ACTIVE_CONNECTION_STATES:
                continue
            ownership[connection.connection_id] = {
                "origin_pool": connection.origin_pool,
                "current_owner_pool": connection.current_owner_pool,
                "logical_pool": connection.logical_pool,
                "loan_id": connection.loan_id,
                "state": connection.state,
            }
        warning_by_pool = {pool_id: True for pool_id in self._current_warning_by_pool}
        for pool_id in self.pools:
            warning_by_pool.setdefault(pool_id, False)
        pools: dict[str, dict[str, Any]] = {}
        for pool_id, pool in sorted(self.pools.items()):
            pools[pool_id] = {
                "own_active": pool.own_active,
                "borrowed_in": pool.borrowed_in,
                "lent_out": pool.lent_out,
                "active_connections": pool.active_connections,
                "capacity_in_use": pool.capacity_in_use,
                "active_with_lent": pool.active_with_lent,
                "remaining_connections": pool.remaining_connections,
                "waterline_percentage_fixed": (
                    pool.capacity_in_use * self.snapshot.percentage_scale // pool.config.max_connections
                ),
                "warning": warning_by_pool[pool_id],
            }
        return {
            "seq": event.seq,
            "event_type": event.event_type,
            "pool_id": event.pool,
            "connection_id": event.connection,
            "parameter_version": self.parameter_version,
            "pools": pools,
            "ownership": ownership,
            "warning_conclusions": dict(sorted(warning_by_pool.items())),
            "totals": {
                "active_connections": active_total,
                "capacity_in_use": capacity_total,
                "assignment_count": len(ownership),
            },
        }

    def _assert_invariants(self) -> None:
        ownership_active = sum(
            1 for connection in self.connections.values() if connection.state in ACTIVE_CONNECTION_STATES
        )
        active_total = sum(pool.active_connections for pool in self.pools.values())
        if ownership_active != active_total:
            raise AssertionError(f"active ownership mismatch: {ownership_active} != {active_total}")
        for pool_id, pool in self.pools.items():
            if pool.own_active < 0 or pool.borrowed_in < 0 or pool.lent_out < 0:
                raise AssertionError(f"negative pool counter for {pool_id}")
            if pool.own_active > pool.config.max_connections:
                raise AssertionError(f"own quota exceeded for {pool_id}")
            if pool.capacity_in_use > pool.config.max_connections:
                raise AssertionError(f"capacity in use exceeded for {pool_id}")
            if pool.remaining_connections < 0:
                raise AssertionError(f"negative remaining quota for {pool_id}")

    def pool_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pool_id, pool in sorted(self.pools.items()):
            rows.append(
                {
                    "pool_id": pool_id,
                    "parameter_version": self.parameter_version,
                    "max_connections": pool.config.max_connections,
                    "own_active": pool.own_active,
                    "borrowed_in": pool.borrowed_in,
                    "lent_out": pool.lent_out,
                    "active_connections": pool.active_connections,
                    "active_with_lent": pool.active_with_lent,
                    "capacity_in_use": pool.capacity_in_use,
                    "remaining_connections": pool.remaining_connections,
                    "warning": pool_id in self._current_warning_by_pool,
                }
            )
        return rows

    def connection_rows(self) -> list[dict[str, Any]]:
        return [
            self.connections[connection_id].to_dict()
            for connection_id in sorted(self.connections)
        ]

    def loan_rows(self) -> list[dict[str, Any]]:
        return [self.loans[loan_id].to_dict() for loan_id in sorted(self.loans)]

    def warning_rows(self) -> list[dict[str, Any]]:
        return sorted(
            (warning.to_dict() for warning in self.warnings),
            key=lambda item: (item["event_seq"], item["warning_id"]),
        )

    def runtime_state(self) -> dict[str, Any]:
        return {
            "parameter_version": self.parameter_version,
            "last_event_seq": self.last_event_seq,
            "last_event_ms": self.last_event_ms,
            "session_start_ms": self.session_start_ms,
            "loan_counter": self._loan_counter,
            "warning_counter": self._warning_counter,
            "arrivals": {
                pool_id: [int(value) for value in observations.tolist()]
                for pool_id, observations in sorted(self._arrivals.items())
            },
            "warning_current": dict(sorted(self._current_warning_by_pool.items())),
            "unknown_connections": sorted(self.unknown_connections),
        }

    def describe_connection(self, connection_id: str) -> dict[str, Any]:
        connection = self.connections.get(connection_id)
        if connection is None:
            return {"connection_id": connection_id, "found": False}
        loans = [loan.to_dict() for loan in sorted(self.loans.values(), key=lambda item: item.loan_id) if loan.connection_id == connection_id]
        return {
            "connection_id": connection_id,
            "found": True,
            "origin_pool": connection.origin_pool,
            "logical_pool": connection.logical_pool,
            "current_owner_pool": connection.current_owner_pool,
            "state": connection.state,
            "loan_id": connection.loan_id,
            "loans": loans,
        }

    def current_warnings(self) -> list[dict[str, Any]]:
        current_ids = set(self._current_warning_by_pool.values())
        return [
            warning.to_dict()
            for warning in self.warnings
            if warning.warning_id in current_ids and not warning.stale
        ]
