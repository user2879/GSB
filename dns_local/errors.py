"""Business errors shared by the DNS service, CLI and demo."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


ERROR_CODES = (
    "OK",
    "E_WIRE",
    "E_ZONE",
    "E_CNAME",
    "E_UPSTREAM",
    "E_CHAIN",
    "E_PORT",
    "E_SNAPSHOT",
    "E_STATE",
    "E_ARGUMENT",
)


@dataclass
class BusinessError(Exception):
    """An expected failure with a stable machine-readable business code."""

    code: str
    message: str
    offset: int | None = None
    record_index: int | None = None
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES or self.code == "OK":
            raise ValueError(f"unknown error code: {self.code}")
        Exception.__init__(self, self.render())

    def render(self) -> str:
        parts = [self.code, self.message]
        if self.offset is not None:
            parts.append(f"offset={self.offset}")
        if self.record_index is not None:
            parts.append(f"record_index={self.record_index}")
        if self.field is not None:
            parts.append(f"field={self.field}")
        return " | ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "offset": self.offset,
            "record_index": self.record_index,
            "field": self.field,
            "details": dict(self.details),
        }


def wire_error(message: str, offset: int, **details: Any) -> BusinessError:
    return BusinessError("E_WIRE", message, offset=offset, details=details)


def zone_error(message: str, record_index: int, field: str, **details: Any) -> BusinessError:
    return BusinessError(
        "E_ZONE",
        message,
        record_index=record_index,
        field=field,
        details=details,
    )
