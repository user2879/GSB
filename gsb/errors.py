"""Error types shared by the CLI, workers and ledger."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


@dataclass
class DiagnosticError(Exception):
    code: str
    message: str
    field: str | None = None
    source: str | None = None
    line: int | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            payload["field"] = self.field
        if self.source is not None:
            payload["source"] = self.source
        if self.line is not None:
            payload["line"] = self.line
        if self.details:
            payload["details"] = self.details
        return payload


def spec_error(
    message: str,
    *,
    field: str | None = None,
    source: str | None = None,
    line: int | None = None,
) -> DiagnosticError:
    return DiagnosticError(
        "E_SPEC", message, field=field, source=source, line=line
    )


E_PORT = "E_PORT"
E_TIMEOUT = "E_TIMEOUT"
E_FIELD = "E_FIELD"
E_LEASE = "E_LEASE"
E_REPLAY = "E_REPLAY"
E_SPEC = "E_SPEC"
E_CHECKPOINT = "E_CHECKPOINT"
E_QUOTA = "E_QUOTA"
E_PROTOCOL = "E_PROTOCOL"
E_UI = "E_UI"
