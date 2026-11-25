"""Structured failures shared by all pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class DiagnosticFailure:
    code: str
    source: str
    line: int | None
    field: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiagnosticError(Exception):
    """Raised for a fatal, non-isolatable invocation error."""


class ConfigFailure(DiagnosticError):
    def __init__(self, source: str, line: int | None, field: str, message: str):
        super().__init__(f"{source}:{line or '-'}:{field}: {message}")
        self.failure = DiagnosticFailure("E_CONFIG", source, line, field, message)


class AudioFormatFailure(DiagnosticError):
    def __init__(self, source: str, message: str, code: str = "E_AUDIO_FORMAT"):
        super().__init__(f"{source}: {message}")
        self.failure = DiagnosticFailure(code, source, None, "data", message)
