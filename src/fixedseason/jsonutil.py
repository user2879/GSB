"""Canonical JSON, hashing and atomic file helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def loads_strict(text: str) -> Any:
    """Parse JSON while rejecting NaN/Infinity extension constants."""

    return json.loads(text, parse_constant=_reject_constant)


def to_jsonable(value: Any) -> Any:
    """Convert internal scalar values to JSON-safe values deterministically."""

    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite number cannot be serialized")
        if value == 0.0:
            return 0.0
        return value
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def pretty_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        sort_keys=False,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def write_text_atomic(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json_atomic(path: str | Path, value: Any) -> None:
    write_text_atomic(path, pretty_json(value))


def read_json(path: str | Path) -> Any:
    return loads_strict(Path(path).read_text(encoding="utf-8"))


def normalize_number(value: Any) -> Any:
    """Stable fingerprint representation for integral/float JSON numbers."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value == 0.0:
            return 0.0
        return repr(value)
    return value


def relative_path(path: str | Path, root: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path))
