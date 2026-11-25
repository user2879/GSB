from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def derive_int_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def round_half_away_from_zero(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("cannot quantize a non-finite value")
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def process_start_token(pid: int | None = None) -> str | None:
    target = pid if pid is not None else os.getpid()
    stat_path = Path(f"/proc/{target}/stat")
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
        return fields[21]
    except (OSError, IndexError):
        return None


def process_is_same_instance(pid: int, start_token: str | None) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    current_token = process_start_token(pid)
    if start_token is None or current_token is None:
        return True
    return current_token == start_token

