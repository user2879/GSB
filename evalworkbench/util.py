"""Hashing, canonical JSON, path safety, and atomic persistence helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator

from .errors import BusinessError


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def json_load(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise BusinessError(
            "E_SCHEMA",
            f"invalid JSON in {path.name}",
            {"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    except OSError as exc:
        raise BusinessError("E_IO", f"cannot read {path}", {"path": str(path)}) from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise BusinessError("E_IO", f"cannot hash {path}", {"path": str(path)}) from exc
    return digest.hexdigest()


def hash_object(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write a file via same-directory temp file, fsync, and os.replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{path.name}.tmp-{os.getpid()}-"
    descriptor, temp_name = tempfile.mkstemp(prefix=prefix, dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, pretty_json_bytes(value))


def atomic_write_canonical_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


@contextmanager
def atomic_binary_stream(path: Path) -> Iterator[BinaryIO]:
    """Stream to an atomic replacement target without materializing the file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{path.name}.tmp-{os.getpid()}-"
    descriptor, temp_name = tempfile.mkstemp(prefix=prefix, dir=str(path.parent))
    temp_path = Path(temp_name)
    handle = os.fdopen(descriptor, "wb")
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            handle.close()
        except OSError:
            pass
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def resolve_workspace_path(workspace: Path, declared: str) -> Path:
    root = workspace.resolve()
    candidate = Path(declared)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BusinessError(
            "E_SCHEMA",
            "carrier path escapes the workspace",
            {"declared": declared, "workspace": str(root)},
        ) from exc
    return candidate


def relative_posix(path: Path, workspace: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def numeric_equal(left: float | int | None, right: float | int | None, tolerance: float = 1e-7) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def chunks(iterable: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def remove_tree(path: Path) -> list[dict[str, Any]]:
    """Remove an uncommitted tree and return a deterministic orphan inventory."""

    inventory: list[dict[str, Any]] = []
    if not path.exists():
        return inventory
    if path.is_file():
        inventory.append({"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
        path.unlink(missing_ok=True)
        return inventory
    for child in sorted(path.rglob("*")):
        if child.is_file():
            inventory.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "size": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            child.rmdir()
    path.rmdir()
    return inventory
