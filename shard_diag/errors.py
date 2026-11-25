from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    SPEC = "E_SPEC"
    PORT = "E_PORT"
    TIMEOUT = "E_TIMEOUT"
    PROTO = "E_PROTO"
    LEASE = "E_LEASE"
    REPLAY = "E_REPLAY"
    ROUTE = "E_ROUTE"
    QUOTA = "E_QUOTA"
    STATE = "E_STATE"


@dataclass
class DiagnosticError(RuntimeError):
    """A machine-readable diagnostic failure.

    ``db_id``, ``key_column`` and ``field`` are populated whenever the
    specification or replay comparison can identify them.  They are also part
    of the JSON representation so callers do not have to parse messages.
    """

    code: str
    message: str
    db_id: str | None = None
    key_column: str | None = None
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.db_id is not None:
            result["db_id"] = self.db_id
        if self.key_column is not None:
            result["key_column"] = self.key_column
        if self.field is not None:
            result["field"] = self.field
        if self.details:
            result["details"] = self.details
        return result


def spec_error(
    message: str,
    *,
    db_id: str | None = None,
    key_column: str | None = None,
    field: str | None = None,
    details: dict[str, Any] | None = None,
) -> DiagnosticError:
    return DiagnosticError(
        ErrorCode.SPEC.value,
        message,
        db_id=db_id,
        key_column=key_column,
        field=field,
        details=details or {},
    )
