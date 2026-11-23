from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def wall_ms() -> int:
    return int(time.time() * 1000)


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_obj(value: Any) -> str:
    return digest_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clamp_int64(value: int) -> int:
    if value < INT64_MIN:
        return INT64_MIN
    if value > INT64_MAX:
        return INT64_MAX
    return value


def saturating_add(left: int, right: int) -> tuple[int, bool]:
    raw = left + right
    bounded = clamp_int64(raw)
    return bounded, bounded != raw


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".write-probe-{os.getpid()}-{threading.get_ident()}"
    try:
        with open(probe, "x", encoding="utf-8") as handle:
            handle.write("ok")
        probe.unlink()
    except OSError as exc:
        raise OSError(f"working directory is not writable: {directory}: {exc}") from exc
    return directory


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_json_file(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def format_ms(value: int | None) -> str:
    return "—" if value is None else f"{value} ms"
