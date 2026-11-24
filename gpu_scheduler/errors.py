from __future__ import annotations

from typing import Any


class AppError(Exception):
    code = "E_INTERNAL"
    exit_code = 1

    def __init__(self, message: str, *, locator: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.locator = locator or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "locator": self.locator,
        }


class SpecError(AppError):
    code = "E_SPEC"
    exit_code = 2


class PortInUseError(AppError):
    code = "E_PORT"
    exit_code = 3


class DependencyCycleError(AppError):
    code = "E_CYCLE"
    exit_code = 4


class SnapshotDigestError(AppError):
    code = "E_SNAPSHOT"
    exit_code = 5


class LedgerRecoveryError(AppError):
    code = "E_LEDGER"
    exit_code = 6
