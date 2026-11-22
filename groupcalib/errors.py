"""Business error codes and structured validation failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_DATA_INVALID = "E_DATA_INVALID"
E_GROUP_UNKNOWN = "E_GROUP_UNKNOWN"
E_CALIB_BUDGET_PENDING = "E_CALIB_BUDGET_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"

CONFIG_CODES = {E_CONFIG_INVALID}
DATA_CODES = {E_DATA_INVALID, E_GROUP_UNKNOWN}


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    sort_key: tuple[Any, ...]
    details: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"code": self.code, "message": self.message}
        record.update(self.details)
        return record


@dataclass(frozen=True)
class BusinessEvent:
    code: str
    message: str
    sort_key: tuple[Any, ...]
    details: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"code": self.code, "message": self.message}
        record.update(self.details)
        return record


class GroupCalibError(Exception):
    """Unexpected pipeline error carrying a stable business code."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def failure_sort_key(
    *,
    source: str,
    batch_index: int = -1,
    action_index: int = -1,
    line: int = -1,
    code: str = "",
    row_id: str = "",
    field: str = "",
    message: str = "",
) -> tuple[Any, ...]:
    return (source, batch_index, action_index, line, code, row_id, field, message)

