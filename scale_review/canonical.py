from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def monotonic_ms() -> int:
    """Current local monotonic time in whole milliseconds."""

    return time.monotonic_ns() // 1_000_000


def canonical_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_dumps(value).encode("utf-8")


def sha256_hex(value: bytes | str | Any) -> str:
    if not isinstance(value, bytes):
        value = canonical_bytes(value)
    return hashlib.sha256(value).hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
