from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PoolConfig:
    pool_id: str
    max_connections: int
    per_connection_request_limit: int
    idle_release_ms: int
    read_buffer_limit_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.pool_id,
            "max_connections": self.max_connections,
            "per_connection_request_limit": self.per_connection_request_limit,
            "idle_release_ms": self.idle_release_ms,
            "read_buffer_limit_bytes": self.read_buffer_limit_bytes,
        }


@dataclass(frozen=True)
class BorrowRule:
    lender_pool: str
    borrower_pool: str
    max_borrowed: int
    return_deadline_ms: int
    source_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lender_pool": self.lender_pool,
            "borrower_pool": self.borrower_pool,
            "max_borrowed": self.max_borrowed,
            "return_deadline_ms": self.return_deadline_ms,
        }


@dataclass(frozen=True)
class ParameterSnapshot:
    version: int
    alert_threshold: int
    rate_window_ms: int
    percentage_scale: int
    pools: dict[str, PoolConfig]
    borrow_rules: tuple[BorrowRule, ...]

    def sorted_pools(self) -> list[PoolConfig]:
        return [self.pools[key] for key in sorted(self.pools)]

    def sorted_rules(self) -> list[BorrowRule]:
        return sorted(
            self.borrow_rules,
            key=lambda rule: (rule.lender_pool, rule.borrower_pool, rule.source_index),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "alert_threshold": self.alert_threshold,
            "rate_window_ms": self.rate_window_ms,
            "percentage_scale": self.percentage_scale,
            "pools": [pool.to_dict() for pool in self.sorted_pools()],
            "borrows": [rule.to_dict() for rule in self.sorted_rules()],
        }


@dataclass(frozen=True)
class TraceEvent:
    seq: int
    pool: str
    connection: str
    event_type: str
    timestamp_ms: int
    bytes_count: int
    line: int
    raw: dict[str, Any] = field(compare=False, hash=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "pool": self.pool,
            "connection": self.connection,
            "type": self.event_type,
            "timestamp_ms": self.timestamp_ms,
            "bytes": self.bytes_count,
        }


@dataclass
class ConnectionState:
    connection_id: str
    logical_pool: str
    origin_pool: str
    current_owner_pool: str
    state: str
    loan_id: str | None
    established_seq: int
    established_ms: int
    last_activity_ms: int
    request_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "logical_pool": self.logical_pool,
            "origin_pool": self.origin_pool,
            "current_owner_pool": self.current_owner_pool,
            "state": self.state,
            "loan_id": self.loan_id,
            "established_seq": self.established_seq,
            "established_ms": self.established_ms,
            "last_activity_ms": self.last_activity_ms,
            "request_count": self.request_count,
        }


@dataclass
class BorrowLoan:
    loan_id: str
    lender_pool: str
    borrower_pool: str
    connection_id: str
    amount: int
    borrowed_at_seq: int
    borrowed_at_ms: int
    deadline_ms: int
    status: str
    parameter_version: int
    returned_at_seq: int | None = None
    returned_at_ms: int | None = None
    overdue_at_seq: int | None = None
    stale: bool = False
    stale_reason: str | None = None

    @property
    def outstanding(self) -> bool:
        return self.status in {"active", "overdue"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "loan_id": self.loan_id,
            "lender_pool": self.lender_pool,
            "borrower_pool": self.borrower_pool,
            "connection_id": self.connection_id,
            "amount": self.amount,
            "borrowed_at_seq": self.borrowed_at_seq,
            "borrowed_at_ms": self.borrowed_at_ms,
            "deadline_ms": self.deadline_ms,
            "status": self.status,
            "parameter_version": self.parameter_version,
            "returned_at_seq": self.returned_at_seq,
            "returned_at_ms": self.returned_at_ms,
            "overdue_at_seq": self.overdue_at_seq,
            "stale": self.stale,
            "stale_reason": self.stale_reason,
        }


@dataclass
class WarningRecord:
    warning_id: str
    event_seq: int
    pool_id: str
    parameter_version: int
    remaining_connections: int
    capacity_in_use: int
    active_connections: int
    own_active: int
    borrowed_in: int
    lent_out: int
    max_connections: int
    alert_threshold: int
    waterline_percentage_fixed: int
    percentage_scale: int
    rounding: str
    arrival_rate: dict[str, Any]
    exhaustion_eta: dict[str, Any]
    context: str = "EVENT"
    stale: bool = False
    stale_reason: str | None = None
    stale_at_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "warning_id": self.warning_id,
            "type": "W_APPROACHING",
            "event_seq": self.event_seq,
            "pool_id": self.pool_id,
            "parameter_version": self.parameter_version,
            "remaining_connections": self.remaining_connections,
            "capacity_in_use": self.capacity_in_use,
            "active_connections": self.active_connections,
            "own_active": self.own_active,
            "borrowed_in": self.borrowed_in,
            "lent_out": self.lent_out,
            "max_connections": self.max_connections,
            "alert_threshold": self.alert_threshold,
            "waterline_percentage_fixed": self.waterline_percentage_fixed,
            "percentage_scale": self.percentage_scale,
            "rounding": self.rounding,
            "arrival_rate": self.arrival_rate,
            "exhaustion_eta": self.exhaustion_eta,
            "context": self.context,
            "stale": self.stale,
            "stale_reason": self.stale_reason,
            "stale_at_version": self.stale_at_version,
        }
