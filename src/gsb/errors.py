from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GSBError(Exception):
    code: str
    message: str
    field: str | None = None
    block_group_id: str | None = None
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        locator = []
        if self.block_group_id:
            locator.append(f"block_group={self.block_group_id}")
        if self.field:
            locator.append(f"field={self.field}")
        prefix = self.code
        if locator:
            prefix += "[" + ",".join(locator) + "]"
        return f"{prefix}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "block_group_id": self.block_group_id,
            "details": self.details or {},
        }


class SchemaError(GSBError):
    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        block_group_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "E_SCHEMA",
            message,
            field=field,
            block_group_id=block_group_id,
            details=details,
        )


class FormatError(GSBError):
    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("E_FORMAT", message, field=field, details=details)

