"""Deterministic hashing, RNG, JSON and file-system helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np


def canonical_json(value: Any) -> str:
    """Serialise with sorted keys and stable separators for hashing."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rng_from_seed(seed: int) -> np.random.Generator:
    """Deterministic PCG64 generator; never relies on global NumPy state."""
    return np.random.default_rng(abs(int(seed)) % (2**32))


def derive_seed(*parts: object) -> int:
    """Fold any ordered set of hashable descriptors into a 32-bit seed."""
    digest = sha256_text("|".join(str(part) for part in parts))
    return int(digest[:8], 16)


def fingerprint_array(values: np.ndarray) -> str:
    """Content fingerprint of an array, robust to dtype/shape and NaN-free."""
    array = np.ascontiguousarray(values)
    header = f"{array.shape}:{array.dtype}:".encode("utf-8")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def fingerprint_arrays(arrays: Iterable[tuple[str, np.ndarray]]) -> str:
    """Fingerprint a named collection of arrays under a fixed ordering."""
    digest = hashlib.sha256()
    for name, array in sorted(arrays, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(fingerprint_array(array).encode("ascii"))
    return digest.hexdigest()


def write_json_atomic(path: str, value: Any) -> None:
    """Write to a temp file in the same directory then atomically replace."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp-", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def stage_directory(directory: str) -> str:
    """Create a fresh per-process staging directory inside ``directory``."""
    os.makedirs(directory, exist_ok=True)
    return tempfile.mkdtemp(prefix="stage-", dir=directory)


def commit_staging(staging: str, target: str) -> None:
    """Atomically replace ``target`` with the staged directory.

    ``os.replace`` is atomic on both POSIX and Windows when source and target
    live on the same volume, satisfying business rule 5 across platforms.
    """
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    if os.path.isdir(target):
        backup = target + ".old"
        if os.path.isdir(backup):
            shutil.rmtree(backup)
        os.replace(target, backup)
        try:
            os.replace(staging, target)
        except BaseException:
            os.replace(backup, target)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(staging, target)


def copy_tree_contents(source: str, target: str) -> None:
    """Copy a directory tree using plain file I/O (no symlinks)."""
    os.makedirs(target, exist_ok=True)
    for root, dirs, files in os.walk(source):
        rel = os.path.relpath(root, source)
        dest_dir = target if rel == "." else os.path.join(target, rel)
        os.makedirs(dest_dir, exist_ok=True)
        for name in files:
            shutil.copy2(
                os.path.join(root, name), os.path.join(dest_dir, name)
            )
