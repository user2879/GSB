"""Business errors used by the layout workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_IMAGE_INVALID = "E_IMAGE_INVALID"
E_BLOCK_BOUNDS = "E_BLOCK_BOUNDS"
E_COLUMN_GAP_INVALID = "E_COLUMN_GAP_INVALID"
E_ORDER_DUPLICATE = "E_ORDER_DUPLICATE"
E_ORDER_INVALID = "E_ORDER_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_INTERNAL = "E_INTERNAL"


@dataclass
class WorkbenchError(Exception):
    """A page-local, reportable error.

    Page failures are deliberately represented as values rather than raised
    across thread boundaries.  This keeps one bad page from affecting any
    other page in the same run.
    """

    code: str
    message: str
    stage: str
    injection_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        suffix = f" (injection={self.injection_id})" if self.injection_id else ""
        return f"{self.code}: {self.message}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
        }
        if self.injection_id is not None:
            result["injection_id"] = self.injection_id
        if self.details:
            result["details"] = self.details
        return result


class ConfigError(ValueError):
    """Raised for unusable configuration before page processing starts."""

