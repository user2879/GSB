"""Canonical serialisation, hashing and durable file helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_value(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def digest_without_field(value: dict[str, Any], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    return digest_value(payload)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> str:
    data = canonical_json_bytes(value)
    atomic_write_bytes(path, data)
    return sha256_bytes(data)


def monotonic_ms() -> int:
    """Integer milliseconds from the process-wide monotonic clock."""

    return time.monotonic_ns() // 1_000_000


def wall_time_ms() -> int:
    return time.time_ns() // 1_000_000


class Clock:
    """Small clock protocol useful for deterministic unit tests."""

    def monotonic_ms(self) -> int:
        return monotonic_ms()

    def wall_time_ms(self) -> int:
        return wall_time_ms()


def first_mismatch(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any] | None:
    """Return the first lexicographically sorted field mismatch."""

    keys = sorted(set(left) | set(right))
    for key in keys:
        field = f"{prefix}.{key}" if prefix else key
        if key not in left:
            return {"field": field, "left": "<missing>", "right": right[key]}
        if key not in right:
            return {"field": field, "left": left[key], "right": "<missing>"}
        lv, rv = left[key], right[key]
        if isinstance(lv, dict) and isinstance(rv, dict):
            found = first_mismatch(lv, rv, prefix=field)
            if found is not None:
                return found
        elif lv != rv:
            return {"field": field, "left": lv, "right": rv}
    return None


def call_with_retry(operation: Callable[[], Any], attempts: int = 3) -> Any:
    """Retry short SQLite/file contention failures without hiding bad data."""

    last: Exception | None = None
    for _ in range(attempts):
        try:
            return operation()
        except (OSError, TimeoutError) as exc:
            last = exc
            time.sleep(0.01)
    assert last is not None
    raise last
