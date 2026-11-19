"""Deterministic canonical JSON used for byte-for-byte cross-path comparison."""

from __future__ import annotations

import json
from typing import Any


def canonical_bytes(payload: Any) -> bytes:
    """Serialize payload with sorted keys and a fixed separator.

    Every path must build plain ``dict``/``list``/scalar structures so that the
    produced bytes depend only on content, never on insertion or hash order.
    """

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(payload: Any) -> str:
    return canonical_bytes(payload).decode("utf-8")


def first_diff(expected: Any, actual: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    """Locate the first differing field between two canonical structures."""

    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            sub_path = f"{path}.{key}"
            if key not in expected:
                return sub_path, None, actual[key]
            if key not in actual:
                return sub_path, expected[key], None
            diff = first_diff(expected[key], actual[key], sub_path)
            if diff is not None:
                return diff
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        for index in range(max(len(expected), len(actual))):
            sub_path = f"{path}[{index}]"
            if index >= len(expected):
                return sub_path, None, actual[index]
            if index >= len(actual):
                return sub_path, expected[index], None
            diff = first_diff(expected[index], actual[index], sub_path)
            if diff is not None:
                return diff
        return None
    if expected != actual:
        return path, expected, actual
    return None
