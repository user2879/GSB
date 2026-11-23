"""Business error and failure-list primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E_TEXT_INVALID = "E_TEXT_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_PLAN_INVALID = "E_PLAN_INVALID"
E_SIGNATURE_BUDGET_PENDING = "E_SIGNATURE_BUDGET_PENDING"
E_SNAPSHOT_VERSION_MISMATCH = "E_SNAPSHOT_VERSION_MISMATCH"
E_CLUSTER_ID_CONFLICT = "E_CLUSTER_ID_CONFLICT"


@dataclass(frozen=True)
class Failure:
    code: str
    stage: str
    message: str
    item_id: str | None = None
    source: str | None = None
    line: int | None = None
    detail: Any = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
        }
        if self.item_id is not None:
            result["item_id"] = self.item_id
        if self.source is not None:
            result["source"] = self.source
        if self.line is not None:
            result["line"] = self.line
        if self.detail is not None:
            result["detail"] = self.detail
        return result


def failure_sort_key(value: Failure | dict[str, Any]) -> tuple[Any, ...]:
    item = value.to_dict() if isinstance(value, Failure) else value
    return (
        str(item.get("code", "")),
        str(item.get("item_id", "")),
        str(item.get("source", "")),
        int(item.get("line", -1) or -1),
        str(item.get("message", "")),
    )


def sort_failures(values: list[Failure | dict[str, Any]]) -> list[dict[str, Any]]:
    return [v.to_dict() if isinstance(v, Failure) else v for v in sorted(values, key=failure_sort_key)]


class BusinessError(Exception):
    """An error that must be surfaced as a business code, not a stack trace."""

    def __init__(self, code: str, message: str, *, detail: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_failure(self, *, stage: str = "runtime") -> Failure:
        return Failure(self.code, stage, self.message, detail=self.detail)

