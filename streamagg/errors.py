from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


FORMAT_VERSION = 1

EXIT_CODES: dict[str, int] = {
    "OK": 0,
    "E_SPEC": 2,
    "E_ORDER": 3,
    "E_GRID": 4,
    "E_SNAPSHOT": 5,
    "E_REPLAY": 6,
    "E_INTERNAL": 70,
}


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    field: str | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            payload["field"] = self.field
        if self.details:
            payload["details"] = self.details
        return payload


def exit_code_for(code: str) -> int:
    return EXIT_CODES.get(code, EXIT_CODES["E_INTERNAL"])
