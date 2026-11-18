"""Small standard-library helpers shared by the engine and experiment tools."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def wall_ms() -> int:
    return int(time.time() * 1000)


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def normalize_ref(value: str, member_ids: set[str], token_ids: set[str]) -> str:
    """Normalize ``X``, ``member:X`` and ``token:X`` dependency references."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("dependency reference must be a non-empty string")
    value = value.strip()
    if value.startswith("member:") or value.startswith("token:"):
        kind, _, identifier = value.partition(":")
        if not identifier:
            raise ValueError(f"invalid dependency reference: {value!r}")
        if kind == "member" and identifier not in member_ids:
            raise ValueError(f"missing member dependency: {identifier}")
        if kind == "token" and identifier not in token_ids:
            raise ValueError(f"missing token dependency: {identifier}")
        return f"{kind}:{identifier}"
    if value in member_ids:
        return f"member:{value}"
    if value in token_ids:
        return f"token:{value}"
    raise ValueError(f"missing dependency reference: {value}")


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class JsonlLogger:
    """Thread-safe append-only structured log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {"event": event, **fields}
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(record) + "\n")
                handle.flush()
        return record


def json_line(value: Any) -> str:
    return canonical_json(value) + "\n"

