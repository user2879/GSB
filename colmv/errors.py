"""Stable error envelope used by both the library and the command line."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, NoReturn


EXIT_CODES = {
    "E_USAGE": 2,
    "E_VIEW": 3,
    "E_BUDGET": 4,
    "E_INTEGRITY": 5,
    "E_REPLAY": 6,
    "E_DIVERGE": 7,
    "E_CHANGE": 8,
    "E_LEDGER": 9,
    "E_INTERNAL": 10,
}


@dataclass
class ColmvError(Exception):
    """A failure with a machine-readable code and optional first-error location."""

    code: str
    message: str
    view_id: str | None = None
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)
        if self.code not in EXIT_CODES:
            raise ValueError(f"unknown error code: {self.code}")

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.code]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.view_id is not None:
            payload["view_id"] = self.view_id
        if self.field is not None:
            payload["field"] = self.field
        if self.details:
            payload["details"] = self.details
        return payload


def fail(
    code: str,
    message: str,
    *,
    view_id: str | None = None,
    field: str | None = None,
    **details: Any,
) -> NoReturn:
    raise ColmvError(code, message, view_id=view_id, field=field, details=details)
