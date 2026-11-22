"""Immutable domain models and canonical byte serialisation helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def canonical_bytes(payload: Any) -> bytes:
    """Return deterministic bytes for a JSON-compatible payload.

    ``sort_keys`` removes any dependence on dict/insertion order and
    ``separators`` pins whitespace, so equal logical values are byte-equal.
    """

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def digest_of(payload: Any) -> str:
    """SHA-256 hex digest of a payload's canonical bytes."""

    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _is_int(value: Any) -> bool:
    # bool is a subclass of int; reject it explicitly for strict validation.
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class ServiceSpec:
    service_id: str
    requests: tuple[int, ...]
    concurrency_per_replica: int
    latency_ms: int
    scale_up_threshold: int
    scale_down_threshold: int
    min_replicas: int
    max_replicas: int
    concurrency_quota: int

    def declaration(self) -> dict:
        """JSON-compatible declaration used for hashing/replay."""

        return {
            "service_id": self.service_id,
            "requests": list(self.requests),
            "concurrency_per_replica": self.concurrency_per_replica,
            "latency_ms": self.latency_ms,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "concurrency_quota": self.concurrency_quota,
        }


@dataclass(frozen=True)
class ClusterSpec:
    services: tuple[ServiceSpec, ...]
    global_replica_budget: int
    fixed_point_scale: int
    latency_budget_ms: int
    incremental_threshold_minutes: int
    window_size: int

    @property
    def service_ids(self) -> tuple[str, ...]:
        # Explicit tuple order is the authoritative ordering for all algorithms.
        return tuple(service.service_id for service in self.services)

    @property
    def minutes(self) -> int:
        return len(self.services[0].requests) if self.services else 0

    def declaration(self) -> dict:
        return {
            "services": [service.declaration() for service in self.services],
            "global_replica_budget": self.global_replica_budget,
            "fixed_point_scale": self.fixed_point_scale,
            "latency_budget_ms": self.latency_budget_ms,
            "incremental_threshold_minutes": self.incremental_threshold_minutes,
            "window_size": self.window_size,
        }

    def digest(self) -> str:
        return digest_of(self.declaration())


@dataclass(frozen=True)
class MinuteEstimate:
    minute: int
    service_id: str
    requests: int
    per_replica_capacity: int
    raw_demand: int
    demand: int
    min_replicas: int
    max_replicas: int
    rule_version: str
    scale_action: str
    utilization_bp: int

    def as_tuple(self) -> tuple:
        # Tuple of pure scalars enables byte-stable cross-path comparisons.
        return (
            self.minute,
            self.service_id,
            self.requests,
            self.per_replica_capacity,
            self.raw_demand,
            self.demand,
            self.min_replicas,
            self.max_replicas,
            self.rule_version,
            self.scale_action,
            self.utilization_bp,
        )


@dataclass(frozen=True)
class AllocationResult:
    minute: int
    service_id: str
    demand: int
    effective_cap: int
    allocated: int
    trimmed: int
    trim_reason: str
    capacity_requests: int
    rejected: int

    def as_tuple(self) -> tuple:
        return (
            self.minute,
            self.service_id,
            self.demand,
            self.effective_cap,
            self.allocated,
            self.trimmed,
            self.trim_reason,
            self.capacity_requests,
            self.rejected,
        )


@dataclass(frozen=True)
class MinuteDecision:
    minute: int
    version: int
    estimates: tuple[MinuteEstimate, ...]
    allocations: tuple[AllocationResult, ...]
    total_allocated: int
    global_budget: int
    selection: str
    selection_reason: str
    threshold_boundary: int
    estimates_digest: str
    digest: str = field(default="")

    def ledger_payload(self) -> dict:
        return {
            "minute": self.minute,
            "version": self.version,
            "estimates": [estimate.__dict__ for estimate in self.estimates],
            "allocations": [allocation.__dict__ for allocation in self.allocations],
            "total_allocated": self.total_allocated,
            "global_budget": self.global_budget,
            "selection": self.selection,
            "selection_reason": self.selection_reason,
            "threshold_boundary": self.threshold_boundary,
            "estimates_digest": self.estimates_digest,
        }
