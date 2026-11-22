"""Business errors shared by the snapshot diagnostic component."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


E_GEOM = "E_GEOM"
E_FORMAT = "E_FORMAT"
E_CHAIN = "E_CHAIN"
E_INTEGRITY = "E_INTEGRITY"
E_VERSION = "E_VERSION"
E_STATE = "E_STATE"
E_EXISTS = "E_EXISTS"
E_UI = "E_UI"


@dataclass
class DiagnosticError(Exception):
    """A stable, machine-readable business failure.

    ``field`` always identifies the first rejected field.  ``snapshot_id`` is
    retained for format, chain, integrity and version failures so callers do
    not need to parse prose to locate the affected entry.
    """

    code: str
    message: str
    snapshot_id: str | None = None
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "snapshot_id": self.snapshot_id,
            "field": self.field,
        }
        result.update(self.details)
        if "first_violation" not in result:
            result["first_violation"] = {
                "field": self.field,
                "snapshot_id": self.snapshot_id,
                "message": self.message,
            }
        return result


def first_violation(
    code: str,
    field_name: str,
    message: str,
    *,
    snapshot_id: str | None = None,
    **details: Any,
) -> DiagnosticError:
    return DiagnosticError(
        code=code,
        message=message,
        snapshot_id=snapshot_id,
        field=field_name,
        details=details,
    )


EXIT_CODES = {
    E_GEOM: 2,
    E_FORMAT: 3,
    E_CHAIN: 4,
    E_INTEGRITY: 5,
    E_VERSION: 6,
    E_STATE: 7,
    E_EXISTS: 8,
    E_UI: 9,
}
