"""Deterministic serialization, hashing, file and diagnostic helpers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def sha256_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write_text(path: str | Path, text: str, *, readonly: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
        if readonly:
            os.chmod(target, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        return target
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_json_file(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "snapshot"


def unique_stable(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def first_deep_difference(expected: Any, actual: Any, path: str = "") -> dict[str, Any] | None:
    """Return the first deterministic structural difference.

    Dict keys are visited lexicographically. Lists are compared in index order.
    Missing values use a sentinel that is JSON-renderable in diagnostics.
    """
    missing = "<missing>"
    if type(expected) is not type(actual):
        return {"field": path or "<root>", "expected": expected, "actual": actual}
    if isinstance(expected, dict):
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                return {"field": child, "expected": missing, "actual": actual[key]}
            if key not in actual:
                return {"field": child, "expected": expected[key], "actual": missing}
            diff = first_deep_difference(expected[key], actual[key], child)
            if diff is not None:
                return diff
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return {"field": path or "<root>", "expected": len(expected), "actual": len(actual)}
        for index, (left, right) in enumerate(zip(expected, actual)):
            child = f"{path}[{index}]"
            diff = first_deep_difference(left, right, child)
            if diff is not None:
                return diff
        return None
    if expected != actual:
        return {"field": path or "<root>", "expected": expected, "actual": actual}
    return None


def short_value(value: Any, limit: int = 180) -> str:
    text = canonical_json(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
