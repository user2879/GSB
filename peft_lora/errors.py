from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E_BASE_INVALID = "E_BASE_INVALID"
E_TASK_INVALID = "E_TASK_INVALID"
E_ADAPTER_CONFIG = "E_ADAPTER_CONFIG"
E_BASE_VERSION_MISMATCH = "E_BASE_VERSION_MISMATCH"
E_MERGE_NOT_REVERSIBLE = "E_MERGE_NOT_REVERSIBLE"
E_MERGE_BUDGET_EXHAUSTED = "E_MERGE_BUDGET_EXHAUSTED"
E_TRAIN_FAILED = "E_TRAIN_FAILED"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    task_name: str | None = None
    target_layer: str | None = None
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "task_name": self.task_name,
            "target_layer": self.target_layer,
            "details": self.details or {},
        }


class BusinessError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        task_name: str | None = None,
        target_layer: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.task_name = task_name
        self.target_layer = target_layer
        self.details = details or {}

    def as_failure(self) -> Failure:
        return Failure(
            code=self.code,
            message=self.message,
            task_name=self.task_name,
            target_layer=self.target_layer,
            details=self.details,
        )


def failure_sort_key(failure: Failure | dict[str, Any]) -> tuple[str, str, str, str]:
    if isinstance(failure, Failure):
        task_name = failure.task_name or ""
        target_layer = failure.target_layer or ""
        code = failure.code
        message = failure.message
    else:
        task_name = failure.get("task_name") or ""
        target_layer = failure.get("target_layer") or ""
        code = failure.get("code") or ""
        message = failure.get("message") or ""
    return (task_name, target_layer, code, message)

