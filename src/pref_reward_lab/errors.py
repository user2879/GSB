from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LabError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ConfigError(LabError):
    def __init__(self, message: str, *, field: str | None = None) -> None:
        details = {"field": field} if field else {}
        super().__init__("E_CONFIG_INVALID", message, details)


class FormatError(LabError):
    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        pair_id: str | None = None,
        line: int | None = None,
        source_version: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if field is not None:
            details["field"] = field
        if pair_id is not None:
            details["pair_id"] = pair_id
        if line is not None:
            details["line"] = line
        if source_version is not None:
            details["source_version"] = source_version
        super().__init__("E_FORMAT_INVALID", message, details)


class SnapshotError(LabError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(code, message, details)


class SimulatedTermination(RuntimeError):
    """Raised by a crash test after an uncommitted temporary snapshot exists."""

