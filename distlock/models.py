"""Typed domain models and deterministic fixed-point water-level math."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ACTIONS = ("ACQUIRE", "RENEW", "RELEASE")
ACTION_CN = {"申请": "ACQUIRE", "续约": "RENEW", "释放": "RELEASE"}
ACTION_TO_CN = {v: k for k, v in ACTION_CN.items()}

# Lease lifecycle states (explicit permission table).
LEASE_STATES = ("pending", "granted", "renewed", "expired", "released")

# Verdict identifiers returned on every arbitration receipt.
VERDICTS = (
    "GRANTED",        # capacity available -> lease granted
    "QUEUED",         # capacity full / queue cap allows waiting
    "REJECTED_QUEUE_FULL",
    "RENEWED",        # renewal accepted before expiry
    "RENEW_EXPIRED",  # renewal arrived at/after expiry
    "RENEW_NOT_HELD",# renewal from a holder without an active lease
    "RELEASED",
    "RELEASE_NOT_HELD",
    "DUPLICATE",      # re-delivery of an already processed event seq
    "UNKNOWN",        # crash window: outcome not committed, replay required
)

LEDGER_FORMAT_VERSION = 1
SCALE_CALIBRE_VERSION = 1  # fixed-point water-level convention version


@dataclass(frozen=True)
class ScaleFactor:
    """Session fixed-point scale: water_ratio_fixed = floor(active*K/capacity).

    Truncation direction is *floor* and the worst-case absolute quantisation
    error is strictly less than ``1 / scale`` of the true ratio.
    """

    scale: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.scale, int) or self.scale <= 0:
            raise ValueError("scale must be a positive integer")

    def ratio_fixed(self, active: int, capacity: int) -> int:
        if capacity <= 0:
            return 0
        return int((active * self.scale) // capacity)

    def error_bound_fixed(self) -> int:
        # floor truncation: 0 <= true*scale - fixed < 1 (in fixed-point units).
        return 1


@dataclass(frozen=True)
class Domain:
    domain_id: str
    capacity: int
    lease_duration_ms: int
    renew_interval_ms: int
    queue_limit: int
    version: int = 1

    @property
    def domain_version(self) -> int:
        return self.version


@dataclass(frozen=True)
class Holder:
    holder_id: str
    priority: int
    target_domains: tuple[str, ...]


@dataclass(frozen=True)
class LeaseEvent:
    seq: int
    timestamp_ms: int
    holder_id: str
    domain_id: str
    action: str  # normalised ACTIONS value
    line_no: int = 0  # 1-based NDJSON line for diagnostics


@dataclass
class Watermark:
    domain_id: str
    seq: int
    active_leases: int
    capacity: int
    free_capacity: int
    queue_length: int
    ratio_fixed: int
    scale: int
    truncation: str
    error_bound_fixed: int

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.domain_id,
            self.seq,
            self.active_leases,
            self.capacity,
            self.free_capacity,
            self.queue_length,
            self.ratio_fixed,
            self.scale,
            self.truncation,
            self.error_bound_fixed,
        )


def vector_watermark(active: list[int], capacity: list[int], scale: int) -> np.ndarray:
    """Vectorised floor fixed-point water-level ratio (NumPy 1.x integer math)."""
    a = np.asarray(active, dtype=np.int64)
    c = np.asarray(capacity, dtype=np.int64)
    safe = np.where(c > 0, c, 1)
    return ((a * int(scale)) // safe).astype(np.int64)


@dataclass
class Transition:
    """One explicit lease permission-table transition."""

    seq: int
    domain_id: str
    holder_id: str
    state_from: str | None
    state_to: str
    reason: str
    started_ms: int | None
    expires_ms: int | None
    domain_version: int
    clock_ms: int

    def as_dict(self) -> dict[str, Any]:
        return field_dict(self)


@dataclass
class Receipt:
    seq: int
    domain_id: str
    holder_id: str
    verdict: str
    remaining_ms: int
    clock_ms: int
    state: str
    queue_length: int
    active_leases: int
    free_capacity: int
    ratio_fixed: int
    scale: int
    error_bound_fixed: int
    reason: str = ""
    unknown: bool = False

    def as_dict(self) -> dict[str, Any]:
        return field_dict(self)


def field_dict(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in getattr(obj, "__dataclass_fields__", {}):
        out[k] = getattr(obj, k)
    return out
