"""Canonical JSON and filesystem primitives.

Every deterministic artifact uses sorted keys, compact separators and UTF-8.
The trailing newline is intentional. Numeric values crossing this boundary are
rounded to 9 decimals by the domain layer so repeated runs serialize identically.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .constants import FLOAT_DIGITS
from .errors import BusinessError, ErrorCode


def clean_float(value: float, digits: int = FLOAT_DIGITS) -> float:
    value = round(float(value), digits)
    if value == 0.0:
        return 0.0
    return value


def clean_vector(values: Iterable[float], digits: int = FLOAT_DIGITS) -> list[float]:
    return [clean_float(v, digits) for v in values]


def canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".staging", dir=path.parent)
    tmp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> str:
    data = canonical_json_bytes(value)
    atomic_write_bytes(path, data)
    return sha256_bytes(data)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _fsync_directory(directory: Path) -> None:
    # Directory fsync is available on POSIX. Windows flushes the replacement via
    # its atomic MoveFileEx-backed os.replace implementation.
    if os.name == "nt":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def workspace_path(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BusinessError(
            ErrorCode.E_FRAME_MISSING,
            "path escapes the declared workspace",
            {"root": str(root), "relative": relative},
        ) from exc
    return candidate


def verify_sha256_file(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise BusinessError(
            ErrorCode.E_QUAT_INVALID,
            "artifact checksum mismatch",
            {"path": str(path), "expected": expected, "actual": actual},
        )


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

