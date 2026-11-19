"""Structured error codes used across the disaster-recovery decision bench."""

from __future__ import annotations

from typing import Any


class BenchError(Exception):
    """Base error carrying a stable business code and a field path."""

    code = "E_UNKNOWN"

    def __init__(self, message: str, path: str = "", detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "detail": self.detail,
        }


class SceneError(BenchError):
    """Scene file or observation table failed strong validation."""

    code = "E_SCENE"


class InjectError(BenchError):
    """Injection parameters (e.g. ratio out of range) are invalid."""

    code = "E_INJECT"


class SnapshotError(BenchError):
    """Snapshot ledger is missing version, corrupt or incompatible."""

    code = "E_SNAPSHOT"


class DivergenceError(BenchError):
    """Two computation paths produced different conclusions."""

    code = "E_DIVERGE"


class ReplayError(BenchError):
    """Replay after recovery did not reproduce the committed ledger."""

    code = "E_REPLAY"
