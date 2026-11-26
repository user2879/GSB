"""Canonical JSON and digest helpers.

All persisted conclusions and byte-for-byte comparisons use this module.  It
intentionally avoids floats in deterministic payloads.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def first_field_difference(left: Any, right: Any, prefix: str = "") -> str | None:
    """Return the first differing field path in canonical order.

    Lists are compared by index, mappings by sorted key, and leaves by value.
    Missing keys/elements produce a stable path such as ``route.database_id``
    or ``hits[2]``.
    """

    if type(left) is not type(right):
        return prefix or "$"
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys | right_keys):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                return path
            if key not in right:
                return path
            found = first_field_difference(left[key], right[key], path)
            if found is not None:
                return found
        return None
    if isinstance(left, list):
        for index in range(max(len(left), len(right))):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            if index >= len(left) or index >= len(right):
                return path
            found = first_field_difference(left[index], right[index], path)
            if found is not None:
                return found
        return None
    if left != right:
        return prefix or "$"
    return None


def iter_canonical_fields(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from iter_canonical_fields(value[key], f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_canonical_fields(item, f"{prefix}[{index}]")
    else:
        yield prefix or "$", value
