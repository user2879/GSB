from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ERROR_CODES = {
    "E_SCHEMA",
    "E_LEASE",
    "E_DATA",
    "E_SNAPSHOT",
    "E_PREVI",
    "E_REPLAY",
    "E_SCHEDULE",
    "E_STATE",
}


@dataclass
class BusinessError(Exception):
    """An expected, machine-readable business failure.

    ``location`` is deliberately JSON-compatible: callers can point at the
    first bad JSON path, NDJSON line, view, group key and field.
    """

    code: str
    message: str
    location: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unknown business error code: {self.code}")
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "details": self.details,
        }


def fail(
    code: str,
    message: str,
    *,
    location: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    raise BusinessError(code, message, location or {}, details or {})


def as_business_error(exc: BaseException) -> BusinessError:
    if isinstance(exc, BusinessError):
        return exc
    return BusinessError(
        "E_STATE",
        f"unexpected failure: {type(exc).__name__}: {exc}",
        {"type": type(exc).__name__},
    )
