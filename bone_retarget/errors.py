"""Business errors used across the animation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    E_SCHEMA_VERSION = "E_SCHEMA_VERSION"
    E_FIELD_MISSING = "E_FIELD_MISSING"
    E_INDEX_OVERFLOW = "E_INDEX_OVERFLOW"
    E_QUAT_INVALID = "E_QUAT_INVALID"
    E_TIME_NONMONOTONIC = "E_TIME_NONMONOTONIC"
    E_RETARGET_DIVERGE = "E_RETARGET_DIVERGE"
    E_FORMAT_ENDIAN = "E_FORMAT_ENDIAN"


@dataclass
class PipelineError(Exception):
    code: ErrorCode
    message: str
    joint: str | None = None
    frame: int | None = None
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        context: list[str] = [f"{self.code.value}: {self.message}"]
        if self.joint is not None:
            context.append(f"joint={self.joint}")
        if self.frame is not None:
            context.append(f"frame={self.frame}")
        if self.field is not None:
            context.append(f"field={self.field}")
        return " | ".join(context)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "joint": self.joint,
            "frame": self.frame,
            "field": self.field,
            "details": self.details,
        }
