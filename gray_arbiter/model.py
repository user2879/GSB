"""Immutable configuration and request models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PoolSpec:
    pool_id: str
    host: str
    port: int
    max_connections: int
    max_requests_per_connection: int
    idle_timeout_ms: int
    lendable_to: tuple[str, ...] = ()

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    prefix: str
    stable_pool: str
    gray_pool: str
    mirror_pool: str
    gray_weight_bp: int
    mirror_sample_bp: int


@dataclass(frozen=True)
class Config:
    version: int
    pools: dict[str, PoolSpec]
    rules: dict[str, RuleSpec]
    borrow_ttl_ms: int = 5000
    queue_timeout_ms: int = 30000
    main_timeout_ms: int = 2000
    mirror_timeout_ms: int = 500
    source: dict[str, Any] = field(default_factory=dict)

    def digest_material(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "pools": [
                {
                    "pool_id": pool.pool_id,
                    "host": pool.host,
                    "port": pool.port,
                    "max_connections": pool.max_connections,
                    "max_requests_per_connection": pool.max_requests_per_connection,
                    "idle_timeout_ms": pool.idle_timeout_ms,
                    "lendable_to": list(pool.lendable_to),
                }
                for pool in sorted(self.pools.values(), key=lambda item: item.pool_id)
            ],
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "prefix": rule.prefix,
                    "stable_pool": rule.stable_pool,
                    "gray_pool": rule.gray_pool,
                    "mirror_pool": rule.mirror_pool,
                    "gray_weight_bp": rule.gray_weight_bp,
                    "mirror_sample_bp": rule.mirror_sample_bp,
                }
                for rule in sorted(self.rules.values(), key=lambda item: item.rule_id)
            ],
            "borrow_ttl_ms": self.borrow_ttl_ms,
            "queue_timeout_ms": self.queue_timeout_ms,
            "main_timeout_ms": self.main_timeout_ms,
            "mirror_timeout_ms": self.mirror_timeout_ms,
        }


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    path: str
    client_key: str
    timestamp_ms: int
    ordinal: int = 0
    line_no: int = 0


@dataclass
class PoolTotals:
    """Mutable, lock-protected counters used by one pool runtime."""

    reuse_count: int = 0
    borrow_count: int = 0
    closed_count: int = 0
    created_count: int = 0


@dataclass
class ConnectionEvent:
    event_seq: int
    connection_id: str
    pool_id: str
    state: str
    reuse_sequence: int | None
    reason: str
    borrow_id: str | None = None
    lender_pool: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "connection_id": self.connection_id,
            "pool_id": self.pool_id,
            "state": self.state,
            "reuse_sequence": self.reuse_sequence,
            "reason": self.reason,
            "borrow_id": self.borrow_id,
            "lender_pool": self.lender_pool,
        }


@dataclass
class BorrowRecord:
    borrow_id: str
    version: int
    lender_pool: str
    borrower_pool: str
    amount: int
    request_id: str
    batch_id: str
    created_mono_ms: int
    deadline_mono_ms: int
    returned_mono_ms: int | None = None
    state: str = "active"
    disposition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "borrow_id": self.borrow_id,
            "version": self.version,
            "lender_pool": self.lender_pool,
            "borrower_pool": self.borrower_pool,
            "amount": self.amount,
            "request_id": self.request_id,
            "batch_id": self.batch_id,
            "created_mono_ms": self.created_mono_ms,
            "deadline_mono_ms": self.deadline_mono_ms,
            "returned_mono_ms": self.returned_mono_ms,
            "state": self.state,
            "disposition": self.disposition,
            "overdue": (
                self.returned_mono_ms is not None
                and self.returned_mono_ms > self.deadline_mono_ms
            ),
        }


@dataclass
class PoolBatchStats:
    pool_id: str
    max_connections: int
    active_connections: int
    in_use_connections: int
    idle_connections: int
    lent_out: int
    borrowed_in: int
    reuse_count: int
    borrow_count: int
    closed_count: int
    sample_count: int
    average_watermark_scaled: int
    peak_watermark: int

    def to_dict(self) -> dict[str, Any]:
        scale = 1000
        return {
            "pool_id": self.pool_id,
            "max_connections": self.max_connections,
            "active_connections_scaled": self.active_connections * scale,
            "in_use_connections_scaled": self.in_use_connections * scale,
            "idle_connections_scaled": self.idle_connections * scale,
            "lent_out_scaled": self.lent_out * scale,
            "borrowed_in_scaled": self.borrowed_in * scale,
            "reuse_count_scaled": self.reuse_count * scale,
            "borrow_count_scaled": self.borrow_count * scale,
            "closed_count_scaled": self.closed_count * scale,
            "average_watermark_scaled": self.average_watermark_scaled,
            "peak_watermark": self.peak_watermark,
            "watermark_samples": self.sample_count,
            "fixed_point": {
                "scale": scale,
                "truncation": "floor_toward_negative_infinity",
                "quantization_error_upper_bound_scaled": 1,
            },
        }


@dataclass
class WatermarkView:
    pool_id: str
    active_connections: int
    in_use_connections: int
    idle_connections: int
    lent_out: int
    borrowed_in: int
    watermark: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "active_connections": self.active_connections,
            "in_use_connections": self.in_use_connections,
            "idle_connections": self.idle_connections,
            "lent_out": self.lent_out,
            "borrowed_in": self.borrowed_in,
            "watermark": self.watermark,
            "watermark_scaled": self.watermark * 1000,
        }
