"""Canonical serialization, clocks, stable identifiers and JSON-line helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import struct
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_hex(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def digest_json(value: Any) -> str:
    return sha256_hex(canonical_bytes(value))


def now_monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_int(text: str, *, bits: int = 63) -> int:
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") & ((1 << bits) - 1)


def scalar_to_u64(value: int | float | bool | str) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return int(value) & ((1 << 64) - 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        return int.from_bytes(struct.pack(">d", value), "big")
    if isinstance(value, str):
        return stable_int(value, bits=64)
    raise TypeError(f"unsupported scalar: {type(value)!r}")


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def terminate_process(process: Any, grace_seconds: float = 2.0) -> None:
    """Terminate a subprocess without relying on POSIX-only process groups."""

    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
    except Exception:
        pass
    try:
        process.wait(timeout=grace_seconds)
        return
    except Exception:
        pass
    try:
        if os.name == "nt":
            process.kill()
        else:
            os.kill(process.pid, signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
    try:
        process.wait(timeout=grace_seconds)
    except Exception:
        pass


class StructuredLog:
    """Append-only canonical JSON-lines logger safe for worker threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_directory(self.path.parent)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        record = {"event": event, "time_utc": utc_now(), **fields}
        line = canonical_json(record) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)


def write_json_line(handle: BinaryIO, value: Any) -> None:
    handle.write(canonical_bytes(value) + b"\n")
    handle.flush()
