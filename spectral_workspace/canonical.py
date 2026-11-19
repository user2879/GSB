"""Canonical serialisation, hashing and cross-platform atomic file writes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def qfloat(value: float, significant_digits: int = 15) -> float:
    """Round a number into a stable textual representation.

    Fifteen significant digits is enough to preserve float64 colour results
    while preventing platform-level last-bit noise from changing hashes and
    byte-for-byte reports.
    """

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite number cannot be serialised: {number!r}")
    if abs(number) < 5e-16:
        return 0.0
    return float(f"{number:.{significant_digits}g}")


def normalize_json(value: Any) -> Any:
    """Convert common numeric containers into canonical JSON-ready values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return qfloat(value)
    if isinstance(value, dict):
        return {str(key): normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]

    # NumPy scalars and arrays are deliberately handled without importing
    # NumPy here; the protocol is duck typed so this module stays lightweight.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        converted = item_method()
        if converted is not value:
            return normalize_json(converted)
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return normalize_json(to_list())
    raise TypeError(f"value is not JSON serialisable: {type(value)!r}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        normalize_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_bytes(value: Any, *, newline: bool = True) -> bytes:
    text = json.dumps(
        normalize_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    if newline:
        text += "\n"
    return text.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_obj(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def short_hash(value: Any, length: int = 12) -> str:
    return sha256_obj(value).split(":", 1)[1][:length]


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Write ``data`` and atomically replace ``path``.

    The temporary file is created in the destination directory so the final
    ``os.replace`` is on one filesystem.  This works on Windows and POSIX.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        fsync_directory(destination.parent)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: str | os.PathLike[str], value: Any, *, pretty: bool = False
) -> bytes:
    data = pretty_bytes(value) if pretty else canonical_bytes(value)
    atomic_write_bytes(path, data)
    return data


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def fsync_directory(path: str | os.PathLike[str]) -> None:
    """Best-effort directory fsync; not supported by Windows/Python."""

    if os.name == "nt":
        return
    try:
        descriptor = os.open(os.fspath(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
