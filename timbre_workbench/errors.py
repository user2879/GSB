"""Stable error codes shared by the engine and diagnostics report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_FIRST_RESPONSE_PENDING = "E_FIRST_RESPONSE_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_SNAPSHOT_WRITE = "E_SNAPSHOT_WRITE"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_ASSIGNMENT_CONFLICT = "E_ASSIGNMENT_CONFLICT"


@dataclass
class WorkbenchError(Exception):
    """A deterministic, user-facing error with a stable code."""

    code: str
    message: str
    context: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def as_failure(
        self,
        *,
        file: str = "",
        stage: str = "",
        command_index: int | None = None,
        sort_key: str | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "error_code": self.code,
            "message": self.message,
            "file": file,
            "stage": stage,
            "sort_key": sort_key or f"{file}|{self.code}|{self.message}",
        }
        if self.context:
            record["context"] = self.context
        if command_index is not None:
            record["command_index"] = command_index
        return record

