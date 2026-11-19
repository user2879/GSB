"""Business errors used by the public API and the command line."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShardRepairError(Exception):
    code: str
    message: str
    shard_id: str | None = None
    field_path: str | None = None
    version: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        location = f" shard={self.shard_id}" if self.shard_id else ""
        path = f" field={self.field_path}" if self.field_path else ""
        return f"{self.code}:{location}{path} {self.message}".strip()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "shard_id": self.shard_id,
            "field_path": self.field_path,
            "version": self.version,
            "details": self.details,
        }
        return result


class SpecValidationError(ShardRepairError):
    def __init__(
        self,
        message: str,
        *,
        shard_id: str | None = None,
        field_path: str | None = None,
        cause_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="E_SPEC",
            message=message,
            shard_id=shard_id,
            field_path=field_path,
            details={**(details or {}), **({"cause_code": cause_code} if cause_code else {})},
        )
        self.cause_code = cause_code


class IntegrityError(ShardRepairError):
    def __init__(
        self,
        message: str,
        *,
        shard_id: str | None = None,
        field_path: str | None = None,
        version: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("E_INTEGRITY", message, shard_id, field_path, version, details or {})


class CycleError(SpecValidationError):
    def __init__(self, message: str, *, shard_id: str | None = None) -> None:
        super().__init__(
            message,
            shard_id=shard_id,
            field_path="relations",
            cause_code="E_CYCLE",
        )


class IncrementalMismatch(ShardRepairError):
    def __init__(self, message: str, *, shard_id: str | None, field: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            "E_INCR",
            message,
            shard_id=shard_id,
            field_path=field,
            details=details or {},
        )


class SnapshotError(ShardRepairError):
    def __init__(
        self,
        message: str,
        *,
        shard_id: str | None = None,
        field_path: str | None = None,
        version: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("E_SNAPSHOT", message, shard_id, field_path, version, details or {})


class ReplayError(ShardRepairError):
    def __init__(self, message: str, *, shard_id: str | None = None, field_path: str | None = None, details: dict[str, Any] | None = None) -> None:
        super().__init__("E_REPLAY", message, shard_id, field_path, details=details or {})

