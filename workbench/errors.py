"""Business error codes used by the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_SAMPLE_INVALID = "E_SAMPLE_INVALID"
E_IMAGE_MISSING = "E_IMAGE_MISSING"
E_TEXT_MISSING = "E_TEXT_MISSING"
E_PNG_HEADER = "E_PNG_HEADER"
E_PNG_TRUNCATED = "E_PNG_TRUNCATED"
E_DIMENSION_MISMATCH = "E_DIMENSION_MISMATCH"
E_REGION_INVALID = "E_REGION_INVALID"
E_TEXT_SCHEMA = "E_TEXT_SCHEMA"
E_DUPLICATE_SAMPLE = "E_DUPLICATE_SAMPLE"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_CONFIG_THRESHOLD = "E_CONFIG_THRESHOLD"
E_CHUNK_MISMATCH = "E_CHUNK_MISMATCH"
E_FEATURE_NONFINITE = "E_FEATURE_NONFINITE"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_BOUNDARY_SENSITIVE = "E_BOUNDARY_SENSITIVE"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_CACHE_CORRUPT = "E_CACHE_CORRUPT"
E_INJECTION_MISSED = "E_INJECTION_MISSED"


@dataclass
class WorkbenchError(Exception):
    """A stable, machine-readable error with optional context."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass
class Failure:
    """An isolated sample or stage failure."""

    sample: str
    code: str
    message: str
    stage: str
    parent_code: str | None = None
    injection_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        codes = [self.code]
        if self.parent_code and self.parent_code not in codes:
            codes.append(self.parent_code)
        return {
            "sample": self.sample,
            "code": self.code,
            "codes": codes,
            "parent_code": self.parent_code,
            "stage": self.stage,
            "message": self.message,
            "injection_id": self.injection_id,
            "details": self.details,
        }
