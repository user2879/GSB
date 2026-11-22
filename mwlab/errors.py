"""Business error codes and structured exception payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXIT_CODES = {
    "E_CONFIG": 2,
    "E_FORMAT": 3,
    "E_DUPKEY": 4,
    "E_SNAPSHOT": 5,
    "E_DIVERGE": 6,
    "E_REPLAY": 7,
    "E_STATE": 8,
    "E_IO": 9,
    "E_INTERNAL": 10,
}


@dataclass
class LabError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in EXIT_CODES:
            raise ValueError(f"unknown business code: {self.code}")
        super().__init__(self.message)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.code]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def format_error(line_no: int, field: str, reason: str, **details: Any) -> LabError:
    payload: dict[str, Any] = {"line": line_no, "field": field, "reason": reason}
    payload.update(details)
    return LabError("E_FORMAT", f"record line {line_no}, field {field}: {reason}", payload)


def state_error(message: str, **details: Any) -> LabError:
    return LabError("E_STATE", message, details)


def io_error(message: str, **details: Any) -> LabError:
    return LabError("E_IO", message, details)

