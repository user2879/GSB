"""Canonical JSON, digests, and deterministic serialization helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


class CanonicalJSONError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalJSONError(str(exc)) from exc


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def event_content_fingerprint(event: dict[str, Any]) -> str:
    return sha256_value(
        {
            "event_id": event["event_id"],
            "user_id": event["user_id"],
            "item_id": event["item_id"],
            "event_type": event["event_type"],
            "timestamp": event["timestamp"],
            "context": event.get("context", {}),
        }
    )


def deep_get(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def deep_set(value: dict[str, Any], dotted_path: str, replacement: Any) -> None:
    parts = dotted_path.split(".")
    if len(parts) == 1:
        value[parts[0]] = replacement
        return
    current: Any = value
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise KeyError(dotted_path)
        if part not in current:
            current[part] = {}
        current = current[part]
    if not isinstance(current, dict):
        raise KeyError(dotted_path)
    current[parts[-1]] = replacement


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def sorted_failures(failures: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        failures,
        key=lambda item: (
            str(item.get("stage", "")),
            str(item.get("event_id", "")),
            str(item.get("code", "")),
            str(item.get("error_type", "")),
            str(item.get("message", "")),
            str(item.get("injection_id", "")),
        ),
    )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ordered:
        key = canonical_json(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
