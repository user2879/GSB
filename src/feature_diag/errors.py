"""Business error codes and deterministic failure records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_BOUNDARY_SENSITIVE = "E_BOUNDARY_SENSITIVE"
E_PRECISION_SENSITIVE = "E_PRECISION_SENSITIVE"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_RUNTIME = "E_RUNTIME"


@dataclass(frozen=True)
class Failure:
    code: str
    scope: str
    field: str
    message: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "scope": self.scope,
            "field": self.field,
            "reason": self.reason,
            "message": self.message,
            "details": self.details,
        }


class ConfigLoadError(ValueError):
    """A root-level configuration document could not be read or parsed."""

    def __init__(self, message: str, *, field: str = "config") -> None:
        super().__init__(message)
        self.field = field

    def as_failure(self) -> Failure:
        return Failure(
            code=E_CONFIG_INVALID,
            scope="__root__",
            field=self.field,
            message=str(self),
            reason="config_load_failed",
        )


class SnapshotStaleError(RuntimeError):
    """Raised internally when a snapshot manifest cannot be reused."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class StopAfterStage(RuntimeError):
    """Test/CLI hook for simulating an orderly interruption between stages."""

    def __init__(self, stage: str, recovery: dict[str, Any]) -> None:
        super().__init__(f"stopped after stage {stage}")
        self.stage = stage
        self.recovery = recovery
