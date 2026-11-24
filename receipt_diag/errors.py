"""Stable business errors used by the diagnostic components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXIT_CODES = {
    "E_SPEC": 2,
    "E_FIELD": 3,
    "E_LEDGER": 4,
    "E_REPLAY": 5,
    "E_WORKER": 6,
    "E_IO": 7,
    "E_CONFIG": 8,
}


@dataclass(frozen=True)
class DiagnosticError(Exception):
    """A deterministic, machine-readable failure.

    ``path`` identifies the first invalid field/batch/request and ``detail`` is
    intentionally stable enough for tests and audit logs.
    """

    code: str
    path: str
    detail: str

    def __post_init__(self) -> None:
        Exception.__init__(self, f"{self.code}: {self.path}: {self.detail}")

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "path": self.path,
            "detail": self.detail,
        }


def fail(code: str, path: str, detail: str) -> "None":
    raise DiagnosticError(code=code, path=path, detail=detail)
