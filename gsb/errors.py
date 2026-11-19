from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXIT_CODES = {
    "E_SPEC": 2,
    "E_BUCKET": 3,
    "E_VERSION": 4,
    "E_SAMPLE": 5,
    "E_REPLAY": 6,
    "E_EST": 7,
}


@dataclass
class GSBError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, f"{self.code}: {self.message}")

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ERROR",
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def spec_error(message: str, errors: list[dict[str, Any]]) -> GSBError:
    return GSBError("E_SPEC", message, {"errors": errors})
