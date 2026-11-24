from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


@dataclass
class AppError(Exception):
    """Structured error used by every user-visible failure path."""

    code: str
    message: str
    block_id: str | None = None
    field: str | None = None
    block_ordinal: int | None = None
    basis: list[str] = dc_field(default_factory=list)
    details: dict[str, Any] = dc_field(default_factory=dict)

    def __str__(self) -> str:
        location = []
        if self.block_id is not None:
            location.append(f"block={self.block_id}")
        if self.field is not None:
            location.append(f"field={self.field}")
        if self.block_ordinal is not None:
            location.append(f"block_ordinal={self.block_ordinal}")
        suffix = f" ({', '.join(location)})" if location else ""
        return f"{self.code}: {self.message}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.block_id is not None:
            payload["block_id"] = self.block_id
        if self.field is not None:
            payload["field"] = self.field
        if self.block_ordinal is not None:
            payload["block_ordinal"] = self.block_ordinal
        if self.basis:
            payload["basis"] = list(self.basis)
        if self.details:
            payload["details"] = self.details
        return payload


def spec_error(
    message: str,
    *,
    block_id: str | None,
    field: str,
    basis: list[str] | None = None,
) -> AppError:
    return AppError(
        "E_SPEC",
        message,
        block_id=block_id,
        field=field,
        basis=basis or [],
    )
