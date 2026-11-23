from __future__ import annotations

from typing import Any


class ArbiterError(Exception):
    """Base class for structured, user-facing failures."""

    code = "E_ARBITER"
    exit_code = 1

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class SpecError(ArbiterError):
    code = "E_SPEC"
    exit_code = 2


class RefError(ArbiterError):
    code = "E_REF"
    exit_code = 3


class SnapshotError(ArbiterError):
    code = "E_SNAPSHOT"
    exit_code = 4


class ReplayError(ArbiterError):
    code = "E_REPLAY"
    exit_code = 5


class LedgerError(ArbiterError):
    code = "E_LEDGER"
    exit_code = 6


class BindingError(ArbiterError):
    code = "E_BINDING"
    exit_code = 7
