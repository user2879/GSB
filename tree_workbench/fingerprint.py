"""Canonical JSON hashing and precision helpers."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def rounded(value: float, decimals: int) -> float:
    result = round(float(value), decimals)
    return 0.0 if result == 0 else result


def rounded_tree(value: Any, decimals: int) -> Any:
    if isinstance(value, float):
        return rounded(value, decimals)
    if isinstance(value, list):
        return [rounded_tree(item, decimals) for item in value]
    if isinstance(value, tuple):
        return [rounded_tree(item, decimals) for item in value]
    if isinstance(value, dict):
        return {str(key): rounded_tree(item, decimals) for key, item in value.items()}
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value

