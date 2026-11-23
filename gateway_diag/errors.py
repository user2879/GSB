from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


@dataclass
class GatewayError(Exception):
    """A machine-readable business failure.

    ``code`` is deliberately part of the public CLI contract.  Every raised
    instance is also written to the structured JSONL event log by the CLI.
    """

    code: str
    message: str
    rule_id: str | None = None
    field: str | None = None
    entry: int | str | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.rule_id is not None:
            payload["rule_id"] = self.rule_id
        if self.field is not None:
            payload["field"] = self.field
        if self.entry is not None:
            payload["entry"] = self.entry
        if self.details:
            payload["details"] = self.details
        return payload


E_SPEC = "E_SPEC"
E_REF = "E_REF"
E_CYCLE = "E_CYCLE"
E_STALE = "E_STALE"
E_DIVERGE = "E_DIVERGE"
E_SNAPSHOT = "E_SNAPSHOT"
E_REPLAY = "E_REPLAY"
E_VERIFY = "E_VERIFY"
E_BACKEND = "E_BACKEND"
E_IO = "E_IO"
