from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def sha256_hex(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    else:
        data = canonical_bytes(value)
    return hashlib.sha256(data).hexdigest()


def diff_flat(expected: Any, actual: Any, path: str = "$") -> list[dict[str, Any]]:
    """Return a stable, flattened list of JSON differences."""
    differences: list[dict[str, Any]] = []
    if type(expected) is not type(actual) and not (
        isinstance(expected, (int, float))
        and isinstance(actual, (int, float))
        and not isinstance(expected, bool)
        and not isinstance(actual, bool)
    ):
        differences.append(
            {"path": path, "expected": expected, "actual": actual, "kind": "type"}
        )
        return differences
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual), key=str):
            child = f"{path}.{key}"
            if key not in expected:
                differences.append({"path": child, "kind": "unexpected", "actual": actual[key]})
            elif key not in actual:
                differences.append({"path": child, "kind": "missing", "expected": expected[key]})
            else:
                differences.extend(diff_flat(expected[key], actual[key], child))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append(
                {
                    "path": path,
                    "kind": "length",
                    "expected": len(expected),
                    "actual": len(actual),
                }
            )
        for index, (left, right) in enumerate(zip(expected, actual)):
            differences.extend(diff_flat(left, right, f"{path}[{index}]"))
    elif expected != actual:
        differences.append(
            {"path": path, "kind": "value", "expected": expected, "actual": actual}
        )
    return differences
