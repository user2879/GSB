from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Return the unique compact byte encoding used for byte-for-byte comparisons."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(value: Any) -> str:
    return canonical_json(value).decode("utf-8")


def sha256_hex(payload: bytes | str) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()


def content_digest(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def first_deep_diff(expected: Any, actual: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    """Find the first stable field difference in sorted-object traversal order."""

    if type(expected) is not type(actual):
        # bool is a subclass of int; type equality keeps that distinction explicit.
        if not (isinstance(expected, (int, float)) and isinstance(actual, (int, float))):
            return path, expected, actual
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual), key=str):
            child = f"{path}.{key}"
            if key not in expected:
                return child, "<missing>", actual[key]
            if key not in actual:
                return child, expected[key], "<missing>"
            diff = first_deep_diff(expected[key], actual[key], child)
            if diff is not None:
                return diff
        return None
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        for index in range(max(len(expected), len(actual))):
            child = f"{path}[{index}]"
            if index >= len(expected):
                return child, "<missing>", actual[index]
            if index >= len(actual):
                return child, expected[index], "<missing>"
            diff = first_deep_diff(expected[index], actual[index], child)
            if diff is not None:
                return diff
        return None
    if expected != actual:
        return path, expected, actual
    return None
