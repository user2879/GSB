from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


@dataclass
class DiagnosticError(Exception):
    """A stable business error with optional object/shard/field coordinates."""

    code: str
    message: str
    object_key: str | None = None
    shard_index: int | None = None
    field: str | None = None
    detail: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "object_key": self.object_key,
            "shard_index": self.shard_index,
            "field": self.field,
        }
        if self.detail:
            value["detail"] = self.detail
        return value


class SpecError(DiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        object_key: str | None = None,
        field: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "E_SPEC",
            message,
            object_key=object_key,
            field=field,
            detail=detail or {},
        )


class LeaseError(DiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        object_key: str | None = None,
        shard_index: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "E_LEASE",
            message,
            object_key=object_key,
            shard_index=shard_index,
            detail=detail or {},
        )


class DigestError(DiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        object_key: str | None = None,
        shard_index: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "E_DIGEST",
            message,
            object_key=object_key,
            shard_index=shard_index,
            detail=detail or {},
        )


class SnapshotError(DiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        object_key: str | None = None,
        shard_index: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "E_SNAPSHOT",
            message,
            object_key=object_key,
            shard_index=shard_index,
            detail=detail or {},
        )


class BudgetError(DiagnosticError):
    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__("E_BUDGET", message, detail=detail or {})
