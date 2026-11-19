"""Canonical serialization and byte-level hashes.

The project intentionally avoids pickle and timestamped archives.  Every
persisted or compared payload is canonical JSON, and geometric arrays are
hashed as little-endian IEEE-754 float64 bytes in C order.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import DomainError, ErrorCode


SCHEMA_VERSION = "cad-surface.v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def content_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def with_content_hash(payload: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in payload.items() if key != "content_hash"}
    result = dict(body)
    result["content_hash"] = content_hash(body)
    return result


def verify_content_hash(payload: dict[str, Any], expected: str | None = None) -> str:
    actual = content_hash({key: value for key, value in payload.items() if key != "content_hash"})
    recorded = payload.get("content_hash")
    if recorded != actual or (expected is not None and expected != actual):
        raise DomainError(
            ErrorCode.SNAPSHOT_STALE,
            "snapshot content hash does not match its payload",
            expected=expected or recorded,
            actual=actual,
        )
    return actual


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write and fsync a file, then atomically replace its destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os_getpid()}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        import os

        os.fsync(handle.fileno())
    import os

    os.replace(tmp, path)
    _fsync_directory(path.parent)


def atomic_write_json(path: Path, value: Any) -> str:
    data = canonical_json_bytes(value)
    atomic_write_bytes(path, data)
    return sha256_bytes(data)


def read_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DomainError(
            ErrorCode.SNAPSHOT_STALE,
            "invalid JSON snapshot",
            path=str(path),
            error=str(exc),
        ) from exc
    if not isinstance(value, dict):
        raise DomainError(
            ErrorCode.SNAPSHOT_STALE,
            "snapshot root must be a JSON object",
            path=str(path),
        )
    return value, data


def os_getpid() -> int:
    import os

    return os.getpid()


def _fsync_directory(path: Path) -> None:
    import os

    try:
        descriptor = os.open(path, getattr(os, "O_RDONLY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def hash_float_array(label: str, array: np.ndarray) -> str:
    array = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(struct.pack("<I", array.ndim))
    for size in array.shape:
        digest.update(struct.pack("<Q", int(size)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def hash_int_array(label: str, array: np.ndarray) -> str:
    array = np.ascontiguousarray(array, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(struct.pack("<I", array.ndim))
    for size in array.shape:
        digest.update(struct.pack("<Q", int(size)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def hash_arrays(items: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for label, array in items:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hash_float_array(label, array).encode("ascii"))
    return digest.hexdigest()


def floats3(value: Any) -> tuple[float, float, float]:
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def array_to_list(array: np.ndarray) -> list[Any]:
    return np.asarray(array, dtype="<f8").tolist()
