"""Business error and failure-record helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


E_CONFIG = "E_CONFIG"
E_PARSE = "E_PARSE"
E_DUPLICATE_LAYER = "E_DUPLICATE_LAYER"
E_SHAPE = "E_SHAPE"
E_NONFINITE = "E_NONFINITE"
E_ROW_RANGE = "E_ROW_RANGE"
E_FIELD_TRUNCATED = "E_FIELD_TRUNCATED"
E_SCHEME_UNKNOWN = "E_SCHEME_UNKNOWN"
E_WIDTH_OUT_OF_RANGE = "E_WIDTH_OUT_OF_RANGE"
E_PRECISION_UNKNOWN = "E_PRECISION_UNKNOWN"
E_DUPLICATE_WEIGHT_ROW = "E_DUPLICATE_WEIGHT_ROW"
E_PARAM_MISSING = "E_PARAM_MISSING"
E_BUDGET_TOO_SMALL = "E_BUDGET_TOO_SMALL"
E_BUDGET_EXHAUSTED = "E_BUDGET_EXHAUSTED"
E_LAYER_QUOTA = "E_LAYER_QUOTA"
E_INTERNAL = "E_INTERNAL"


@dataclass
class LabError(Exception):
    code: str
    message: str
    stage: str
    layer: Optional[str] = None
    row: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    injection_id: Optional[str] = None

    def __str__(self) -> str:
        where = []
        if self.layer is not None:
            where.append(f"layer={self.layer}")
        if self.row is not None:
            where.append(f"row={self.row}")
        suffix = f" ({', '.join(where)})" if where else ""
        return f"{self.code}{suffix}: {self.message}"

    def failure(self) -> Dict[str, Any]:
        return make_failure(
            code=self.code,
            stage=self.stage,
            message=self.message,
            layer=self.layer,
            row=self.row,
            details=self.details,
            injection_id=self.injection_id,
        )


def make_failure(
    code: str,
    stage: str,
    message: str,
    *,
    layer: Optional[str] = None,
    row: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
    injection_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "code": code,
        "stage": stage,
        "layer": layer if layer is not None else "<unknown>",
        "row": row,
        "message": message,
        "details": details or {},
        "injection_id": injection_id,
    }


def failure_sort_key(item: Dict[str, Any]) -> tuple:
    layer = item.get("layer")
    row = item.get("row")
    return (
        "" if layer in (None, "<unknown>") else str(layer),
        -1 if row is None else int(row),
        str(item.get("stage", "")),
        str(item.get("code", "")),
        str(item.get("message", "")),
        str(item.get("injection_id") or ""),
    )


def sorted_failures(items: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    seen = set()
    unique = []
    for item in items:
        marker = (
            item.get("code"),
            item.get("stage"),
            item.get("layer"),
            item.get("row"),
            item.get("message"),
            item.get("injection_id"),
        )
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
    return sorted(unique, key=failure_sort_key)
