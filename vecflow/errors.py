"""Business errors shared by the CLI, engine, ledger, and GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXIT_CODES = {
    "E_DATA": 2,
    "E_RANGE": 2,
    "E_NULLMAP": 2,
    "E_BUDGET": 3,
    "E_REPLAY": 4,
    "E_LEDGER": 5,
    "E_GUI": 6,
    "E_INTERNAL": 10,
}


@dataclass
class AppError(Exception):
    """A stable business-code error with machine-readable context."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, EXIT_CODES["E_INTERNAL"])

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "code": self.code, "message": self.message, "details": self.details}


def as_app_error(exc: BaseException, default_code: str = "E_INTERNAL") -> AppError:
    if isinstance(exc, AppError):
        return exc
    return AppError(default_code, str(exc) or exc.__class__.__name__, {"type": exc.__class__.__name__})
