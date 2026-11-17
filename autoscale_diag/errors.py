"""Business errors shared by the command line, ledger and replay layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_SPEC = "E_SPEC"
E_COOLDOWN = "E_COOLDOWN"
E_REPLAY = "E_REPLAY"
E_LEDGER = "E_LEDGER"
E_USAGE = "E_USAGE"


@dataclass
class BusinessError(Exception):
    """A stable, machine-readable business error.

    ``field_path`` is always present for specification failures.  Details are
    deliberately JSON-compatible so that CLI failures and GUI failures look the
    same to an operator.
    """

    code: str
    message: str
    field_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "business_code": self.code,
            "message": self.message,
        }
        if self.field_path is not None:
            payload["field_path"] = self.field_path
        if self.details:
            payload["details"] = self.details
        return payload


def spec_error(path: str, message: str, **details: Any) -> BusinessError:
    return BusinessError(E_SPEC, message, path, details)


def ledger_error(message: str, path: str | None = None, **details: Any) -> BusinessError:
    return BusinessError(E_LEDGER, message, path, details)

