"""Small deterministic IO helpers used by every stage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Union

import numpy as np

PathLike = Union[str, os.PathLike[str]]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256_bytes(contiguous.tobytes(order="C"))


def key_for(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


def write_json(path: PathLike, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def read_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_bytes(path: PathLike, value: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, output)


def write_array(path: PathLike, array: np.ndarray, dtype: str = "<f4") -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    write_bytes(path, contiguous.tobytes(order="C"))
    return sha256_bytes(contiguous.tobytes(order="C"))


def read_array(path: PathLike, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = Path(path).read_bytes()
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(raw) != expected:
        raise ValueError(f"array size mismatch for {path}: have {len(raw)}, expected {expected}")
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()


@contextmanager
def atomic_directory(target: PathLike) -> Iterator[Path]:
    """Yield a temporary sibling directory and atomically replace target on success."""
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
    try:
        yield temporary
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def hash_tree(root: PathLike) -> str:
    digest = hashlib.sha256()
    base = Path(root)
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
