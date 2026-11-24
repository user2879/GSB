"""Canonical JSON and deterministic ordering helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence


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


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_digest(value: Any) -> str:
    return sha256_hex(canonical_bytes(value))


def typed_scalar(value: Any) -> list[Any]:
    """Return a type-tagged scalar so 1, 1.0, and "1" never collide."""

    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite value cannot be ordered")
        # repr is stable across supported Python versions and keeps -0.0 distinct.
        return ["float", repr(value)]
    if isinstance(value, str):
        return ["str", value]
    raise TypeError(f"unsupported scalar type: {type(value).__name__}")


def typed_group_key(values: Sequence[Any]) -> list[list[Any]]:
    return [typed_scalar(value) for value in values]


def group_sort_key(values: Sequence[Any]) -> str:
    return canonical_json(typed_group_key(values))


def page_sort_key(page_id: str) -> tuple[int, str]:
    # Base pages sort naturally; the virtual insertion page sorts last.
    if page_id == "delta":
        return (1, page_id)
    return (0, page_id)


def sort_pages(page_ids: Iterable[str]) -> list[str]:
    return sorted(set(page_ids), key=page_sort_key)
