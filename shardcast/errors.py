"""Stable error codes and exception hierarchy for shardcast."""

from __future__ import annotations


class ShardcastError(Exception):
    """Base error carrying a stable machine-readable code and context path."""

    code = "E_INTERNAL"

    def __init__(self, message: str, *, path: str | None = None, detail: dict | None = None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.detail = detail or {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        loc = f" at {self.path}" if self.path else ""
        return f"[{self.code}]{loc} {self.message}"


class SpecError(ShardcastError):
    code = "E_SPEC"


class PortError(ShardcastError):
    code = "E_PORT"


class TimeoutError(ShardcastError):
    code = "E_TIMEOUT"


class FieldError(ShardcastError):
    code = "E_FIELD"


class LeaseError(ShardcastError):
    code = "E_LEASE"


class DivergeError(ShardcastError):
    code = "E_DIVERGE"


class ProtocolError(ShardcastError):
    code = "E_PROTOCOL"


class LedgerError(ShardcastError):
    code = "E_LEDGER"
