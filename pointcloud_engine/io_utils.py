"""Atomic filesystem helpers and stable digest utilities.

All durable state (tile state, images, reports) is written to a temporary
file in the *same directory* and then swapped with :func:`os.replace`, which
is atomic on POSIX and on Windows for files on the same volume. A crash can
therefore only leave either the old complete file or the new complete file,
never a half-written one. Temporary files carry a distinct suffix so they can
never be mistaken for a completed result.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Iterable

import numpy as np

TMP_SUFFIX = ".tmp"
TILE_PREFIX_GLOBAL = "tile_"
HASH_ALGO = "sha256"
CHUNK = 1 << 20


def new_hasher() -> "hashlib._Hash":
    return hashlib.sha256()


def hash_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def hash_file(path: str) -> str:
    hasher = new_hasher()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            hasher.update(block)
    return hasher.hexdigest()


def canonical_json(value: Any) -> bytes:
    """Deterministic JSON encoding used for version/digest canonicalisation."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def hash_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def hash_array(array: np.ndarray) -> str:
    """Content hash of a NumPy array independent of in-memory padding."""
    contiguous = np.ascontiguousarray(array)
    hasher = new_hasher()
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(b"|")
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(b"|")
    hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def atomic_write_bytes(path: str, data: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=TMP_SUFFIX, dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        _silent_unlink(tmp_path)
        raise


def atomic_write_text(path: str, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _silent_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def cleanup_temp_files(directory: str) -> list[str]:
    """Remove stale temp files left by a hard crash; return their names."""
    removed: list[str] = []
    if not os.path.isdir(directory):
        return removed
    for name in os.listdir(directory):
        is_generic_tmp = name.startswith(".tmp_") and name.endswith(TMP_SUFFIX)
        is_npz_tmp = name.startswith(TILE_PREFIX_GLOBAL) and name.endswith(".tmp.npz")
        if is_generic_tmp or is_npz_tmp:
            target = os.path.join(directory, name)
            try:
                os.unlink(target)
                removed.append(name)
            except OSError:
                pass
    return removed


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_with_index(items: Iterable[Any]) -> Iterable[tuple[int, Any]]:
    for index, item in enumerate(items):
        yield index, item
