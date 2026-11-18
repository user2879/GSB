from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable


FORMAT_VERSION = 1
CLOCK_BASIS = "local_monotonic_ms"
TRUTH_SOURCE = "ledger.per_hit_records"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_digest(value: Any) -> str:
    return sha256_text(canonical_json(value))


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def utc_epoch_ms() -> int:
    return time.time_ns() // 1_000_000


def require_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def require_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def first_difference(left: Any, right: Any, path: str = "$") -> tuple[str, Any, Any]:
    """Return the first deterministic structural difference."""
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        left_keys = list(left.keys())
        right_keys = list(right.keys())
        if left_keys != right_keys:
            return f"{path}.__keys__", left_keys, right_keys
        for key in left_keys:
            diff = first_difference(left[key], right[key], f"{path}.{key}")
            if diff[0] != path:
                return diff
        return path, left, right
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.__length__", len(left), len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            diff = first_difference(left_item, right_item, f"{path}[{index}]")
            if diff[0] != path:
                return diff
        return path, left, right
    if left != right:
        return path, left, right
    return path, left, right


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

