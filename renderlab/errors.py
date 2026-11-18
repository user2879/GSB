"""Business error codes used by the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_NOISE_TABLE_MISMATCH = "E_NOISE_TABLE_MISMATCH"
E_IMG2IMG_STEP_INVALID = "E_IMG2IMG_STEP_INVALID"
E_PREVIEW_BUDGET_PENDING = "E_PREVIEW_BUDGET_PENDING"
E_CACHE_MISS = "E_CACHE_MISS"


@dataclass(frozen=True)
class Failure:
    """A validation or execution failure that is safe to serialize."""

    code: str
    component: str
    item: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "component": self.component,
            "item": self.item,
            "message": self.message,
            "details": self.details,
        }


class WorkbenchError(Exception):
    """An exception carrying a stable business code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        component: str = "workbench",
        item: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.component = component
        self.item = item
        self.details = details or {}

    def to_failure(self) -> Failure:
        return Failure(
            code=self.code,
            component=self.component,
            item=self.item,
            message=self.message,
            details=self.details,
        )
