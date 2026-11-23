"""Structured application errors and stable process exit codes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXIT_CODES = {
    "E_SPEC": 2,
    "E_PORT": 3,
    "E_HANDSHAKE": 4,
    "E_REPLY": 5,
    "E_VERIFY": 6,
    "E_CONVERGE": 7,
    "E_LEDGER": 8,
    "E_ARB": 9,
    "E_PROTOCOL": 10,
    "E_INTERNAL": 11,
}


@dataclass
class AppError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 1)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        payload.update(self.details)
        return payload


class SpecError(AppError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_SPEC", message, details)


class PortError(AppError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_PORT", message, details)


class HandshakeError(AppError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_HANDSHAKE", message, details)


class ReplyError(AppError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_REPLY", message, details)


class VerifyError(AppError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_VERIFY", message, details)


class ConvergeError(AppError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_CONVERGE", message, details)


class LedgerError(AppError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_LEDGER", message, details)


class ArbiterError(AppError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_ARB", message, details)


class ProtocolError(AppError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_PROTOCOL", message, details)
