"""Validated domain models used by the simulator and ledger."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FunctionDecl:
    function_id: str
    memory_mib: int
    instance_concurrency: int
    lease_ms: int
    min_concurrency: int
    max_concurrency: int
    priority: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "function_id": self.function_id,
            "memory_mib": self.memory_mib,
            "instance_concurrency": self.instance_concurrency,
            "lease_ms": self.lease_ms,
            "min_concurrency": self.min_concurrency,
            "max_concurrency": self.max_concurrency,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class PreemptionRelation:
    higher: str
    lower: str
    return_deadline_ms: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "higher": self.higher,
            "lower": self.lower,
            "return_deadline_ms": self.return_deadline_ms,
        }


@dataclass(frozen=True)
class Declaration:
    declaration_version: int
    global_concurrency: int
    functions: Mapping[str, FunctionDecl]
    preemptions: tuple[PreemptionRelation, ...]
    preemption_checkpoint_ms: int
    source: Mapping[str, Any]

    @property
    def digest(self) -> str:
        return content_digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "declaration_version": self.declaration_version,
            "global_concurrency": self.global_concurrency,
            "preemption_checkpoint_ms": self.preemption_checkpoint_ms,
            "functions": [
                self.functions[key].public_dict() for key in sorted(self.functions)
            ],
            "preemptions": [
                relation.public_dict() for relation in self.preemptions
            ],
        }
        if include_digest:
            value["content_digest"] = self.digest
        return value

    def relation(self, higher: str, lower: str) -> PreemptionRelation | None:
        for item in self.preemptions:
            if item.higher == higher and item.lower == lower:
                return item
        return None


@dataclass(frozen=True)
class CallEvent:
    event_seq: int
    arrival_ms: int
    function_id: str
    expected_occupancy_ms: int
    concurrency_demand: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "arrival_ms": self.arrival_ms,
            "function_id": self.function_id,
            "expected_occupancy_ms": self.expected_occupancy_ms,
            "concurrency_demand": self.concurrency_demand,
        }


@dataclass(frozen=True)
class SessionConfig:
    watermark_scale: int
    borrow_return_error_ms: int = 50

    def __post_init__(self) -> None:
        scale = self.watermark_scale
        if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
            raise ValueError("watermark_scale must be a positive integer")
        if scale > 1_000_000_000:
            raise ValueError("watermark_scale is too large")
        if scale not in {10**power for power in range(0, 10)}:
            raise ValueError("watermark_scale must be an integer power of ten")
        if (
            isinstance(self.borrow_return_error_ms, bool)
            or not isinstance(self.borrow_return_error_ms, int)
            or self.borrow_return_error_ms < 0
        ):
            raise ValueError("borrow_return_error_ms must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "watermark_scale": self.watermark_scale,
            "borrow_return_error_ms": self.borrow_return_error_ms,
            "watermark_rounding": "FLOOR_TOWARD_ZERO",
        }
