from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def normalize(value: Any, float_digits: int = 12) -> Any:
    """Return a JSON-friendly value with stable float representation."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be canonicalized")
        rounded = round(value, float_digits)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, dict):
        return {str(k): normalize(v, float_digits) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [normalize(item, float_digits) for item in value]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported canonical JSON type: {type(value)!r}")


def canonical_json(value: Any, float_digits: int = 12) -> str:
    return json.dumps(
        normalize(value, float_digits),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_loads_canonical(text: str) -> Any:
    return json.loads(text)


def deep_field_diff(expected: Any, actual: Any, prefix: str = "") -> tuple[str, Any, Any] | None:
    """Find the first deterministic mismatch in nested JSON-like values."""

    if type(expected) is not type(actual) and not (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        return prefix or "$", expected, actual
    if isinstance(expected, dict):
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            field = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                return field, None, actual[key]
            if key not in actual:
                return field, expected[key], None
            mismatch = deep_field_diff(expected[key], actual[key], field)
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return prefix or "$", expected, actual
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatch = deep_field_diff(left, right, f"{prefix}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if expected != actual:
        return prefix or "$", expected, actual
    return None
