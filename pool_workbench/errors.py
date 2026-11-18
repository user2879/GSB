"""Business error codes and the structured BusinessError type.

All validation / runtime failures in the workbench are reported with a stable
business code so that the CLI, the GUI and the replay engine can react to them
uniformly.
"""
from __future__ import annotations

from typing import Any, Iterable


class ErrorCode:
    E_CONFIG = "E_CONFIG"
    E_SCHEMA = "E_SCHEMA"
    E_GETTIMEOUT = "E_GETTIMEOUT"
    E_HOLDTIMEOUT = "E_HOLDTIMEOUT"
    E_EXEC = "E_EXEC"
    E_SNAPSHOT = "E_SNAPSHOT"
    E_REPLAY = "E_REPLAY"


class BusinessError(Exception):
    """An error carrying a stable business code and field paths."""

    def __init__(
        self,
        code: str,
        message: str,
        field_paths: Iterable[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field_paths = list(field_paths or [])
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field_paths": self.field_paths,
            "details": self.details,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        paths = (" [" + ", ".join(self.field_paths) + "]") if self.field_paths else ""
        return f"{self.code}: {self.message}{paths}"
