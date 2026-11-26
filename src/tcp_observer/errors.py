from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObserverError(Exception):
    """Structured error carrying one of the task's stable ``E_*`` codes."""

    code: str
    message: str
    field_name: str | None = None
    line: int | None = None
    event_seq: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        location = []
        if self.field_name is not None:
            location.append(f"field={self.field_name}")
        if self.line is not None:
            location.append(f"line={self.line}")
        if self.event_seq is not None:
            location.append(f"event_seq={self.event_seq}")
        suffix = f" ({', '.join(location)})" if location else ""
        return f"{self.code}: {self.message}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field_name is not None:
            payload["field"] = self.field_name
        if self.line is not None:
            payload["line"] = self.line
        if self.event_seq is not None:
            payload["event_seq"] = self.event_seq
        if self.details:
            payload["details"] = self.details
        return payload


class SpecError(ObserverError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_SPEC", message, **kwargs)


class FormatError(ObserverError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_FORMAT", message, **kwargs)


class PortError(ObserverError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_PORT", message, **kwargs)


class TimeoutError(ObserverError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_TIMEOUT", message, **kwargs)


class ScriptError(ObserverError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_SCRIPT", message, **kwargs)


class DupAckError(ObserverError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_DUPACK", message, **kwargs)


class ReplayError(ObserverError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_REPLAY", message, **kwargs)
