"""Small deterministic helpers shared by all modules."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .errors import DeclarationError


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_int(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not is_int(value):
        raise DeclarationError("expected an integer", path)
    if minimum is not None and value < minimum:
        raise DeclarationError(f"integer must be >= {minimum}", path)
    if maximum is not None and value > maximum:
        raise DeclarationError(f"integer must be <= {maximum}", path)
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: str | os.PathLike[str]) -> Any:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise DeclarationError(f"cannot read JSON file: {exc}", "$") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeclarationError(
            f"invalid JSON: {exc.msg}",
            "$",
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc


def sorted_stream_ids(ids: Any) -> list[int]:
    return sorted(int(value) for value in ids)


def flatten_dict(value: Any, prefix: str = "$") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            flattened.update(flatten_dict(value[key], f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flattened.update(flatten_dict(item, f"{prefix}[{index}]"))
    else:
        flattened[prefix] = value
    return flattened


def declaration_diff(current: dict[str, Any], restored: dict[str, Any]) -> list[dict[str, Any]]:
    left = flatten_dict(current)
    right = flatten_dict(restored)
    changes: list[dict[str, Any]] = []
    for path in sorted(set(left) | set(right)):
        if left.get(path, "<missing>") != right.get(path, "<missing>"):
            changes.append(
                {
                    "path": path,
                    "current": left.get(path, "<missing>"),
                    "snapshot": right.get(path, "<missing>"),
                }
            )
    return changes
