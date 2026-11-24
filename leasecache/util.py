from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


def mono_ms() -> int:
    """Local monotonic clock expressed as an integer number of milliseconds."""

    return time.monotonic_ns() // 1_000_000


def wall_ms() -> int:
    return time.time_ns() // 1_000_000


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_hex(canonical_json_bytes(value))


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
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


def atomic_write_json(path: str | Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


class StructuredLogger:
    """Thread-safe JSON-lines logger used by the broker and node processes."""

    def __init__(self, path: str | Path, component: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.component = component
        self._lock = threading.Lock()

    def write(self, level: str, message: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "wall_time_ms": wall_ms(),
            "mono_ms": mono_ms(),
            "component": self.component,
            "level": level,
            "message": message,
        }
        record.update(fields)
        line = canonical_json_bytes(record) + b"\n"
        with self._lock:
            with self.path.open("ab") as handle:
                handle.write(line)
                handle.flush()

    def info(self, message: str, **fields: Any) -> None:
        self.write("INFO", message, **fields)

    def error(self, code: str, message: str, **fields: Any) -> None:
        self.write("ERROR", message, code=code, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self.write("WARNING", message, **fields)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def stable_sorted(values: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    return sorted(values)
