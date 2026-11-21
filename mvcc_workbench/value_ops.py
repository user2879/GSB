from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from .canonical import canonical_json, digest_json
from .model import Predicate


def normalize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    return value


def version_value_digest(value_summary: dict[str, Any]) -> str:
    normalized = normalize_value(value_summary)
    return digest_json({"value_summary": normalized})


_OPS = {
    "gt": np.greater,
    ">": np.greater,
    "gte": np.greater_equal,
    ">=": np.greater_equal,
    "lt": np.less,
    "<": np.less,
    "lte": np.less_equal,
    "<=": np.less_equal,
    "eq": np.equal,
    "==": np.equal,
    "ne": np.not_equal,
    "!=": np.not_equal,
}


def evaluate_predicate(predicate: Predicate, value_summary: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    left_values: list[float] = []
    right_values: list[float] = []
    missing: list[str] = []
    for condition in predicate.conditions:
        candidate = value_summary.get(condition.column)
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            missing.append(condition.column)
            left_values.append(float("nan"))
        else:
            left_values.append(float(candidate))
        right_values.append(float(condition.value))

    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    masks: list[bool] = []
    for index, condition in enumerate(predicate.conditions):
        operation = _OPS[condition.op]
        if math.isnan(float(left[index])):
            masks.append(False)
        else:
            masks.append(bool(operation(left[index], right[index])))

    matched = all(masks) if predicate.mode == "all" else any(masks)
    detail = {
        "mode": predicate.mode,
        "column_count": predicate.column_count,
        "condition_masks": masks,
        "matched": matched,
        "missing_columns": missing,
        "vector_sha256": "sha256:"
        + hashlib.sha256(
            np.ascontiguousarray(left, dtype=np.float64).tobytes()
            + np.ascontiguousarray(right, dtype=np.float64).tobytes()
        ).hexdigest(),
    }
    return matched, detail


def value_summary_bytes(value_summary: dict[str, Any]) -> bytes:
    return canonical_json(normalize_value(value_summary)).encode("utf-8")
