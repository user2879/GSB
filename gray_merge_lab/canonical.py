from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_difference(expected: Any, actual: Any, prefix: str = "$") -> tuple[str, Any, Any]:
    """Return a deterministic first difference between two JSON-like values."""
    if type(expected) is not type(actual):
        return prefix, expected, actual
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}"
            if key not in expected:
                return path, "<missing>", actual[key]
            if key not in actual:
                return path, expected[key], "<missing>"
            if expected[key] != actual[key]:
                return first_difference(expected[key], actual[key], path)
        return prefix, expected, actual
    if isinstance(expected, list):
        for index, (left, right) in enumerate(zip(expected, actual)):
            if left != right:
                return first_difference(left, right, f"{prefix}[{index}]")
        if len(expected) != len(actual):
            return f"{prefix}.length", len(expected), len(actual)
        return prefix, expected, actual
    return prefix, expected, actual

