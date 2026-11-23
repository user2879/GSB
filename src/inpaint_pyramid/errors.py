"""Typed errors and stable error codes used by the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_TILE_BUDGET_PENDING = "E_TILE_BUDGET_PENDING"
E_TILE_DEPENDENCY_PENDING = "E_TILE_DEPENDENCY_PENDING"
E_TILE_FAILED = "E_TILE_FAILED"
E_STATE_INVALID = "E_STATE_INVALID"
E_STATE_EXISTS = "E_STATE_EXISTS"
E_RESUME_MISMATCH = "E_RESUME_MISMATCH"


@dataclass
class WorkbenchError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigError(WorkbenchError):
    def __init__(self, errors: list[dict[str, Any]], message: str = "configuration is invalid"):
        super().__init__(E_CONFIG_INVALID, message, {"errors": errors})
        self.errors = errors


class CommandError(WorkbenchError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(E_COMMAND_INVALID, message, details or {})


class StateError(WorkbenchError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(code, message, details or {})

