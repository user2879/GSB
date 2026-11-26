"""Stable error codes used across the shard-metadata accounting component.

Every diagnostic error carries a machine readable ``code`` (E_*), a human
readable message and, where applicable, the offending node id / object key /
field and an optional line number so failures are reproducible and locatable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class ShardError(Exception):
    """Base class for all component errors."""

    code: str = "E_GENERIC"

    def __init__(
        self,
        message: str,
        *,
        node: Optional[str] = None,
        key: Optional[str] = None,
        field: Optional[str] = None,
        line: Optional[int] = None,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.node = node
        self.key = key
        self.field = field
        self.line = line
        self.detail = detail

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        for name in ("node", "key", "field", "line", "detail"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out

    def __str__(self) -> str:  # pragma: no cover - trivial
        loc = []
        if self.node is not None:
            loc.append(f"node={self.node}")
        if self.key is not None:
            loc.append(f"key={self.key}")
        if self.field is not None:
            loc.append(f"field={self.field}")
        if self.line is not None:
            loc.append(f"line={self.line}")
        suffix = (" [" + " ".join(loc) + "]") if loc else ""
        return f"{self.code}: {self.message}{suffix}"


class SpecError(ShardError):
    """Invalid local declaration / manifest / request stream (requirement 1)."""

    code = "E_SPEC"


class FormatError(ShardError):
    """Invalid c1/c2/c3 metadata exchange file (requirement 2)."""

    code = "E_FORMAT"


class LeaseError(ShardError):
    """Lease grant / renew / optimistic-version conflict (requirement 3)."""

    code = "E_LEASE"


class PortError(ShardError):
    """Loopback port cannot be bound or node cannot be reached (requirement 4)."""

    code = "E_PORT"


class TimeoutError(ShardError):  # noqa: A001 - intentional stable E_TIMEOUT
    """Handshake / reply timeout over loopback TCP (requirement 4)."""

    code = "E_TIMEOUT"


class ReplyError(ShardError):
    """Node reply missing required fields or malformed (requirement 4)."""

    code = "E_REPLY"


class DivergenceError(ShardError):
    """Cached view disagrees with authoritative ledger recompute (req. 6/8)."""

    code = "E_DIVERGE"


@dataclass(frozen=True)
class ErrorLocation:
    """First point of disagreement used by E_DIVERGE style diagnostics."""

    key: str
    shard_index: int
    field: str
    cached: Any = None
    authoritative: Any = None

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "shard_index": self.shard_index,
            "field": self.field,
            "cached": self.cached,
            "authoritative": self.authoritative,
        }


__all__ = [
    "ShardError",
    "SpecError",
    "FormatError",
    "LeaseError",
    "PortError",
    "TimeoutError",
    "ReplyError",
    "DivergenceError",
    "ErrorLocation",
]
