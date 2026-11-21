from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    field_path: str | None = None
    line: int | None = None
    request_id: str | None = None
    injection_id: str | None = None
    severity: str = "ERROR"
    context: dict[str, Any] = field(default_factory=dict)

    def sort_key(self) -> tuple:
        return (
            self.code,
            self.request_id or "",
            self.field_path or "",
            self.line if self.line is not None else -1,
            self.injection_id or "",
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.field_path is not None:
            out["field_path"] = self.field_path
        if self.line is not None:
            out["line"] = self.line
        if self.request_id is not None:
            out["request_id"] = self.request_id
        if self.injection_id is not None:
            out["injection_id"] = self.injection_id
        if self.context:
            out["context"] = self.context
        return out


class ConfigAbort(Exception):
    """Raised after one or more fatal configuration errors."""

    def __init__(self, failures: list[Failure] | Failure):
        if isinstance(failures, Failure):
            failures = [failures]
        self.failures = list(failures)
        super().__init__(self.failures[0].message if self.failures else "invalid configuration")


def sorted_failures(failures: list[Failure]) -> list[Failure]:
    return sorted(failures, key=lambda item: item.sort_key())
