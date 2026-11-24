from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def digest_hex(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def digest_text_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right}
    if isinstance(left, dict):
        left_keys = sorted(left)
        right_keys = sorted(right)
        if left_keys != right_keys:
            return {"path": path, "left": left_keys, "right": right_keys}
        for key in left_keys:
            result = first_difference(left[key], right[key], f"{path}.{key}")
            if result is not None:
                return result
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "left": len(left), "right": len(right)}
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            result = first_difference(left_value, right_value, f"{path}[{index}]")
            if result is not None:
                return result
        return None
    if left != right:
        return {"path": path, "left": left, "right": right}
    return None
