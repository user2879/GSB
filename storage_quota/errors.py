from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXIT_CODES = {
    "E_SPEC": 2,
    "E_PORT": 3,
    "E_TIMEOUT": 4,
    "E_PROTO": 5,
    "E_LEASE": 6,
    "E_QUOTA": 7,
    "E_VERIFY": 8,
    "E_LEDGER": 9,
    "E_TK": 10,
    "E_IO": 11,
}


@dataclass
class AppError(Exception):
    code: str
    message: str
    path: str | None = None
    field: str | None = None
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        parts = [self.code, self.message]
        if self.path:
            parts.append(f"at {self.path}")
        if self.field:
            parts.append(f"field={self.field}")
        return ": ".join(parts)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 1)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            result["path"] = self.path
        if self.field is not None:
            result["field"] = self.field
        if self.details is not None:
            result["details"] = self.details
        return result


class SpecError(AppError):
    def __init__(self, path: str, field: str, message: str):
        super().__init__("E_SPEC", message, path=path, field=field)


class ProtocolError(AppError):
    def __init__(self, message: str, path: str | None = None, field: str | None = None):
        super().__init__("E_PROTO", message, path=path, field=field)


class TimeoutAppError(AppError):
    def __init__(self, message: str, path: str | None = None, field: str | None = None):
        super().__init__("E_TIMEOUT", message, path=path, field=field)


class PortInUseError(AppError):
    def __init__(self, message: str, path: str | None = None, field: str | None = None):
        super().__init__("E_PORT", message, path=path, field=field)


class LeaseError(AppError):
    def __init__(self, message: str, path: str | None = None, field: str | None = None,
                 details: dict[str, Any] | None = None):
        super().__init__("E_LEASE", message, path=path, field=field, details=details)


class QuotaError(AppError):
    def __init__(self, message: str, path: str | None = None, field: str | None = None):
        super().__init__("E_QUOTA", message, path=path, field=field)


class LedgerError(AppError):
    def __init__(self, message: str, path: str | None = None, field: str | None = None):
        super().__init__("E_LEDGER", message, path=path, field=field)
