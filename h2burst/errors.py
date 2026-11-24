"""Structured business errors used by the workbench.

Every error carries enough location information for the CLI and the Tkinter
view model to point at the first invalid field, event, or frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


ERROR_CODES = (
    "E_SPEC",
    "E_FRAME",
    "E_FLOW",
    "E_STATE",
    "E_REPLAY",
    "E_SNAPSHOT",
    "E_IO",
)


@dataclass
class WorkbenchError(Exception):
    """A deterministic, serializable workbench error."""

    code: str
    message: str
    field: str | None = None
    event_seq: int | None = None
    frame_seq: int | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unsupported error code: {self.code}")
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "event_seq": self.event_seq,
            "frame_seq": self.frame_seq,
        }
        if self.details:
            result["details"] = self.details
        return result

    def __str__(self) -> str:
        parts = [f"{self.code}: {self.message}"]
        if self.field is not None:
            parts.append(f"field={self.field}")
        if self.event_seq is not None:
            parts.append(f"event={self.event_seq}")
        if self.frame_seq is not None:
            parts.append(f"frame={self.frame_seq}")
        return " ".join(parts)
