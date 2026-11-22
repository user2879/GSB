"""Domain errors with stable business codes and structured-log payloads."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


EXIT_CODES = {
    "E_SPEC": 10,
    "E_QUOTA": 11,
    "E_RESERVE": 12,
    "E_UNSATISFIED": 13,
    "E_SNAPSHOT": 14,
    "E_REPLAY": 15,
    "E_CONTEXT": 16,
    "E_LEDGER": 17,
    "E_IO": 18,
    "E_INTERNAL": 19,
}


@dataclass
class WorkbenchError(Exception):
    code: str
    message: str
    object_type: str | None = None
    object_id: str | None = None
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)
        if self.code not in EXIT_CODES:
            raise ValueError(f"unknown workbench error code: {self.code}")

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.code]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.object_type is not None:
            payload["object_type"] = self.object_type
        if self.object_id is not None:
            payload["object_id"] = self.object_id
        if self.field is not None:
            payload["field"] = self.field
        if self.details:
            payload["details"] = self.details
        return payload


def spec_error(
    message: str,
    *,
    object_type: str,
    object_id: str | None,
    field: str,
    details: dict[str, Any] | None = None,
) -> WorkbenchError:
    return WorkbenchError(
        "E_SPEC",
        message,
        object_type=object_type,
        object_id=object_id,
        field=field,
        details=details or {},
    )
