"""Business failures and exception types used by the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class BusinessFailure:
    code: str
    field: str
    message: str
    stage: str = "configuration"
    tree_index: int | None = None
    exception_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "business_code": self.code,
            "field": self.field,
            "message": self.message,
            "stage": self.stage,
        }
        if self.tree_index is not None:
            value["tree_index"] = self.tree_index
        if self.exception_type is not None:
            value["exception_type"] = self.exception_type
        return value


def failure_sort_key(value: BusinessFailure | dict[str, Any]) -> tuple[Any, ...]:
    if isinstance(value, BusinessFailure):
        item = value.as_dict()
    else:
        item = value
    tree_index = item.get("tree_index")
    return (
        str(item.get("stage", "")),
        -1 if tree_index is None else int(tree_index),
        str(item.get("business_code", "")),
        str(item.get("field", "")),
        str(item.get("exception_type", "")),
        str(item.get("message", "")),
    )


def sorted_failures(values: Iterable[BusinessFailure | dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [value.as_dict() if isinstance(value, BusinessFailure) else dict(value) for value in values]
    return sorted(normalized, key=failure_sort_key)


class BusinessValidationError(Exception):
    def __init__(self, failures: Iterable[BusinessFailure | dict[str, Any]]):
        self.failures = sorted_failures(failures)
        super().__init__(f"{len(self.failures)} business validation failure(s)")


class SnapshotVersionChanged(RuntimeError):
    """Raised when CURRENT changed between snapshot load and commit."""


class InjectedTreeFailure(RuntimeError):
    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(f"injected tree failure at {stage}")

