from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_MASK_OUT_OF_RANGE = "E_MASK_OUT_OF_RANGE"
E_IMAGE_DECODE = "E_IMAGE_DECODE"
E_RENDER_BUDGET_PENDING = "E_RENDER_BUDGET_PENDING"
E_RENDER_FAILED = "E_RENDER_FAILED"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_RESUME_INVALID = "E_RESUME_INVALID"


@dataclass
class StudioError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def as_failure(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


def failure_item(sample_id: str, error: StudioError) -> dict[str, Any]:
    item = {"sample_id": sample_id, **error.as_failure()}
    return item

