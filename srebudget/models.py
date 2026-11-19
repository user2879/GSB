"""Validated domain models and canonical serialization helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"

    @property
    def failed(self) -> bool:
        return self is not Outcome.SUCCESS


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class BudgetPolicy:
    window_hours: int
    ratio_bp: int
    min_samples: int

    @property
    def window_ms(self) -> int:
        return self.window_hours * 60 * 60 * 1000

    def as_dict(self) -> dict[str, int]:
        return {
            "window_hours": self.window_hours,
            "ratio_bp": self.ratio_bp,
            "min_samples": self.min_samples,
        }


@dataclass(frozen=True)
class CircuitPolicy:
    failure_threshold: int
    cooldown_ms: int
    half_open_probes: int

    def as_dict(self) -> dict[str, int]:
        return {
            "failure_threshold": self.failure_threshold,
            "cooldown_ms": self.cooldown_ms,
            "half_open_probes": self.half_open_probes,
        }


@dataclass(frozen=True)
class ServiceSpec:
    id: str
    upstream: tuple[str, ...]
    timeout_ms: int
    budget: BudgetPolicy
    circuit: CircuitPolicy

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "upstream": list(self.upstream),
            "timeout_ms": self.timeout_ms,
            "budget": self.budget.as_dict(),
            "circuit": self.circuit.as_dict(),
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChainSpec:
    version: str
    services: tuple[ServiceSpec, ...]

    def service_map(self) -> dict[str, ServiceSpec]:
        return {service.id: service for service in self.services}

    def get(self, service_id: str) -> ServiceSpec:
        for service in self.services:
            if service.id == service_id:
                return service
        raise KeyError(service_id)

    def canonical_dict(self) -> dict[str, Any]:
        services = sorted(
            (
                {
                    **service.as_dict(),
                    "upstream": sorted(service.upstream),
                }
                for service in self.services
            ),
            key=lambda item: item["id"],
        )
        return {"version": self.version, "services": services}

    @property
    def digest(self) -> str:
        return sha256_json(self.canonical_dict())

    @property
    def service_ids(self) -> tuple[str, ...]:
        return tuple(sorted(service.id for service in self.services))

    def budget_signature(self) -> dict[str, dict[str, int]]:
        return {
            service.id: service.budget.as_dict()
            for service in sorted(self.services, key=lambda item: item.id)
        }


@dataclass(frozen=True)
class TraceCall:
    index: int
    timestamp_ms: int
    path: tuple[str, ...]
    outcome: Outcome
    duration_ms: int
    source_line: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp_ms": self.timestamp_ms,
            "path": list(self.path),
            "outcome": self.outcome.value,
            "duration_ms": self.duration_ms,
            "source_line": self.source_line,
        }


@dataclass(frozen=True)
class TraceStream:
    calls: tuple[TraceCall, ...]
    content_digest: str

    def __len__(self) -> int:
        return len(self.calls)

    def __iter__(self) -> Iterable[TraceCall]:
        return iter(self.calls)

    def window(self, start_index: int, end_index: int) -> tuple[TraceCall, ...]:
        """Return 1-based inclusive/exclusive call-index slice."""
        return tuple(call for call in self.calls if start_index <= call.index < end_index)
