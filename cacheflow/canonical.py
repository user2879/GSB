from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


def canonical_bytes(value: Any) -> bytes:
    return canonical_text(value).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    digest = hashlib.sha256()
    if isinstance(data, str):
        digest.update(data.encode("utf-8"))
    else:
        digest.update(data)
    return digest.hexdigest()


def content_digest(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return sha256_hex(bytes(value))
    return sha256_hex(canonical_bytes(value))


def first_difference(expected: Any, actual: Any, path: str = "$") -> str | None:
    """Return a stable JSON path for the first field-level difference."""
    if type(expected) is not type(actual):
        if isinstance(expected, dict) and isinstance(actual, dict):
            pass
        elif isinstance(expected, list) and isinstance(actual, list):
            pass
        elif expected == actual:
            return None
        else:
            return f"{path} (type {type(expected).__name__}!={type(actual).__name__})"

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path} (expected object)"
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            if missing:
                return f"{path}.{missing[0]} (missing)"
            return f"{path}.{extra[0]} (unexpected)"
        for key in sorted(expected_keys, key=str):
            child = first_difference(expected[key], actual[key], f"{path}.{key}")
            if child:
                return child
        return None

    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return f"{path} (expected array)"
        if len(expected) != len(actual):
            return f"{path}.length ({len(expected)}!={len(actual)})"
        for index, (left, right) in enumerate(zip(expected, actual)):
            child = first_difference(left, right, f"{path}[{index}]")
            if child:
                return child
        return None

    if expected != actual:
        return f"{path} ({expected!r}!={actual!r})"
    return None


def assert_consistent(expected: Any, actual: Any, error_type: type[Exception]) -> None:
    diff = first_difference(expected, actual)
    if diff:
        raise error_type(f"replay divergence at {diff}", entry=diff, field=diff)
