"""Structured error taxonomy for the distributed-lock / lease middleware.

Every error carries a stable machine code (``E_*``), a human message and a
``context`` mapping with the precise offending location (file, line, field,
domain, event sequence number, ...). Structured logs reuse the same shape.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


class DistLockError(Exception):
    code = "E_INTERNAL"

    def __init__(self, message: str, context: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}

    def __str__(self) -> str:  # pragma: no cover - trivial
        ctx = ", ".join(f"{k}={v}" for k, v in sorted(self.context.items()))
        return f"[{self.code}] {self.message}" + (f" ({ctx})" if ctx else "")


class SpecError(DistLockError):
    code = "E_SPEC"


class TimingSpecError(SpecError):
    """renew_interval_ms >= lease_duration_ms (specialised E_SPEC)."""

    code = "E_TIMING"


class ReferenceError_(SpecError):
    """holder target-domain reference does not exist (specialised E_SPEC)."""

    code = "E_REF"


class PortError(DistLockError):
    code = "E_PORT"


class HandshakeError(DistLockError):
    code = "E_HANDSHAKE"


class ReplyError(DistLockError):
    code = "E_REPLY"


class ExhaustError(DistLockError):
    code = "E_EXHAUST"


class SnapshotError(DistLockError):
    code = "E_SNAPSHOT"


class ReplayError(DistLockError):
    code = "E_REPLAY"


CODE_TO_CLASS = {
    "E_SPEC": SpecError,
    "E_TIMING": TimingSpecError,
    "E_REF": ReferenceError_,
    "E_PORT": PortError,
    "E_HANDSHAKE": HandshakeError,
    "E_REPLY": ReplyError,
    "E_EXHAUST": ExhaustError,
    "E_SNAPSHOT": SnapshotError,
    "E_REPLAY": ReplayError,
}
