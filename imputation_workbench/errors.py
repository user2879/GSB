"""Business error codes and the small exception type used across the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_DATA_ROW_INVALID = "E_DATA_ROW_INVALID"
E_GROUP_MAP_INVALID = "E_GROUP_MAP_INVALID"
E_INSUFFICIENT_SAMPLES = "E_INSUFFICIENT_SAMPLES"
E_PATCH_BUDGET_PENDING = "E_PATCH_BUDGET_PENDING"
E_UPSTREAM_STALE = "E_UPSTREAM_STALE"
E_PLAN_INVALID = "E_PLAN_INVALID"
E_IMPUTATION_FAILED = "E_IMPUTATION_FAILED"
E_MODEL_FAILED = "E_MODEL_FAILED"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"


@dataclass
class WorkbenchError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {"business_code": self.code, "message": self.message, **self.details}


def diagnostic(
    code: str,
    message: str,
    *,
    field: str | None = None,
    row_id: str | None = None,
    column: str | None = None,
    line_no: int | None = None,
    stage: str | None = None,
    injection_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a JSON-serialisable diagnostic record with stable key names."""

    out: dict[str, Any] = {"business_code": code, "message": message}
    for key, value in (
        ("field", field),
        ("row_id", row_id),
        ("column", column),
        ("line_no", line_no),
        ("stage", stage),
        ("injection_id", injection_id),
    ):
        if value is not None:
            out[key] = value
    out.update(extra)
    return out

