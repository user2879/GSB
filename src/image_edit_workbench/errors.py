from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


E_FIELD_TRUNCATED = "E_FIELD_TRUNCATED"
E_STRUCTURE_INVALID = "E_STRUCTURE_INVALID"
E_REGION_OUT_OF_RANGE = "E_REGION_OUT_OF_RANGE"
E_RECOMPUTE_BUDGET_PENDING = "E_RECOMPUTE_BUDGET_PENDING"
E_BOUNDARY_SENSITIVE = "E_BOUNDARY_SENSITIVE"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_IMAGE_IO = "E_IMAGE_IO"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"


@dataclass(frozen=True)
class BusinessFailure:
    code: str
    stage: str
    source: str
    message: str
    region_key: str | None = None
    injection_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorkbenchError(Exception):
    """A user-facing workbench error carrying a stable business code."""

    def __init__(self, code: str, message: str, *, stage: str = "workbench") -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.message = message

    def to_failure(self, source: str = "") -> BusinessFailure:
        return BusinessFailure(self.code, self.stage, source, self.message)

