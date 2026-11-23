"""Structured business errors used by the cacheproof package."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import time
from typing import Any


EXIT_CODES: dict[str, int] = {
    "OK": 0,
    "E_SPEC": 2,
    "E_CAPACITY": 3,
    "E_INTEGRITY": 4,
    "E_SNAPSHOT": 5,
    "E_REPLAY": 6,
    "E_TK_UNAVAILABLE": 7,
    "E_IO": 8,
    "E_NOT_FOUND": 9,
    "E_CONFLICT": 10,
    "E_UNKNOWN": 11,
}


@dataclass
class DiagnosticError(Exception):
    """A stable, machine-readable diagnostic failure.

    ``segment_id``, ``entry_key`` and ``field`` identify the first offending
    object. The dictionary representation deliberately contains no traceback
    or host-specific path so it can be written to the JSON diagnostic log.
    """

    code: str
    message: str
    segment_id: str | None = None
    entry_key: str | None = None
    field: str | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)
    monotonic_ns: int = dc_field(default_factory=time.monotonic_ns)

    def __post_init__(self) -> None:
        super().__init__(self.message)
        if self.code not in EXIT_CODES or self.code == "OK":
            raise ValueError(f"unknown diagnostic code: {self.code}")

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.code]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "monotonic_ns": self.monotonic_ns,
        }
        if self.segment_id is not None:
            payload["segment_id"] = self.segment_id
        if self.entry_key is not None:
            payload["entry_key"] = self.entry_key
        if self.field is not None:
            payload["field"] = self.field
        if self.details:
            payload["details"] = self.details
        return payload


def spec_error(
    message: str,
    *,
    segment_id: str | None = None,
    entry_key: str | None = None,
    field: str | None = None,
    details: dict[str, Any] | None = None,
) -> DiagnosticError:
    return DiagnosticError(
        "E_SPEC",
        message,
        segment_id=segment_id,
        entry_key=entry_key,
        field=field,
        details=details or {},
    )
