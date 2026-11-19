"""Business error codes and structured exception hierarchy.

All business failures carry a stable code (E_SCENE, E_PORT, E_SNAPSHOT,
E_RETRY, E_REPLAY, ...) plus a machine-readable details payload so that the
structured logger and CLI can render deterministic reports.
"""
from __future__ import annotations

from typing import Any


class LedgerError(Exception):
    """Base class for all business-level errors."""

    code = "E_INTERNAL"

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class SceneError(LedgerError):
    """Rejected scene file (schema / semantic validation failure)."""

    code = "E_SCENE"


class PortError(LedgerError):
    """A loopback target port cannot be bound (already in use)."""

    code = "E_PORT"


class SnapshotError(LedgerError):
    """Snapshot digest mismatch or snapshot cannot be loaded atomically."""

    code = "E_SNAPSHOT"


class RetryExhaustedError(LedgerError):
    """A node consumed every allowed attempt without success."""

    code = "E_RETRY"


class ReplayError(LedgerError):
    """Deterministic replay diverged from the recorded snapshot/ledger."""

    code = "E_REPLAY"
