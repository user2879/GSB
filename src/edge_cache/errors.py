"""Business error codes shared by every layer.

A failure record is always represented by :class:`Failure` so that the
rejectable ledger can be rebuilt deterministically, independent of which
thread or process observed the problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_LEASE_TAG_MISMATCH = "E_LEASE_TAG_MISMATCH"
E_LEASE_EXPIRED = "E_LEASE_EXPIRED"
E_LEASE_LIMIT = "E_LEASE_LIMIT"
E_HARD_WATERMARK = "E_HARD_WATERMARK"
E_BUDGET_EXHAUSTED = "E_BUDGET_EXHAUSTED"
E_NODE_UNAVAILABLE = "E_NODE_UNAVAILABLE"
E_PROTOCOL = "E_PROTOCOL"
E_SHARD_UNAVAILABLE = "E_SHARD_UNAVAILABLE"
E_SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"
E_INTERNAL = "E_INTERNAL"

ALL_CODES = (
    E_CONFIG_INVALID,
    E_LEASE_TAG_MISMATCH,
    E_LEASE_EXPIRED,
    E_LEASE_LIMIT,
    E_HARD_WATERMARK,
    E_BUDGET_EXHAUSTED,
    E_NODE_UNAVAILABLE,
    E_PROTOCOL,
    E_SHARD_UNAVAILABLE,
    E_SNAPSHOT_VERSION,
    E_INTERNAL,
)


@dataclass(frozen=True)
class Failure:
    """One rejectable business outcome, stable for ledger serialization."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    shard_id: str | None = None
    node_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "context": self.context,
            "shard_id": self.shard_id,
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Failure":
        return cls(
            code=data["code"],
            message=data["message"],
            context=dict(data.get("context", {})),
            shard_id=data.get("shard_id"),
            node_id=data.get("node_id"),
        )


class EdgeCacheError(Exception):
    """Exception carrying a structured :class:`Failure`."""

    def __init__(self, failure: Failure) -> None:
        super().__init__(failure.code + ": " + failure.message)
        self.failure = failure


class ConfigError(EdgeCacheError):
    """Raised by the loading/validation layer; the whole batch is rejected."""

    def __init__(self, failures: list[Failure]) -> None:
        if not failures:
            failures = [Failure(E_CONFIG_INVALID, "unknown configuration error")]
        super().__init__(failures[0])
        self.failures = failures
