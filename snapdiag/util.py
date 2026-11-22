"""Filesystem, hashing and canonical JSON helpers."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from .errors import DiagnosticError, E_FORMAT, E_STATE


CHUNK_SIZE = 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_digest(value: Any, *, snapshot_id: str | None, field_name: str) -> str:
    if not isinstance(value, str):
        raise DiagnosticError(
            E_FORMAT,
            f"{field_name} must be a hexadecimal SHA-256 string",
            snapshot_id=snapshot_id,
            field=field_name,
        )
    text = value.strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise DiagnosticError(
            E_FORMAT,
            f"{field_name} must contain exactly 64 hexadecimal characters",
            snapshot_id=snapshot_id,
            field=field_name,
        )
    return text


def display_digest(value: str) -> str:
    return f"sha256:{value.removeprefix('sha256:')}"


def fsync_directory(path: Path) -> None:
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


def atomic_write_bytes(
    path: str | os.PathLike[str],
    data: bytes,
    *,
    readonly: bool = False,
    sync: bool = True,
    sync_parent: bool = True,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for attempt in range(5):
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                if sync:
                    os.fsync(handle.fileno())
            if readonly:
                try:
                    os.chmod(temporary, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
                except OSError:
                    pass
            os.replace(temporary, target)
            if sync_parent:
                fsync_directory(target.parent)
            return
        except PermissionError as exc:
            last_error = exc
            try:
                os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
            try:
                os.unlink(temporary)
            except OSError:
                pass
        except OSError as exc:
            last_error = exc
            try:
                os.unlink(temporary)
            except OSError:
                pass
            break
    raise DiagnosticError(
        E_STATE,
        f"cannot atomically write {target}",
        field="filesystem",
        details={"path": str(target), "reason": str(last_error)},
    )


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    readonly: bool = False,
    sync: bool = True,
) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload, readonly=readonly, sync=sync)


def copy_file_atomic(
    source: Path,
    target: Path,
    *,
    readonly: bool = False,
    sync: bool = True,
    sync_parent: bool = True,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as destination, open(source, "rb") as origin:
            while True:
                chunk = origin.read(CHUNK_SIZE)
                if not chunk:
                    break
                destination.write(chunk)
            destination.flush()
            if sync:
                os.fsync(destination.fileno())
        if readonly:
            try:
                os.chmod(temporary, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            except OSError:
                pass
        os.replace(temporary, target)
        if sync_parent:
            fsync_directory(target.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def ensure_readonly(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def ensure_writable(path: Path) -> None:
    try:
        if path.is_dir():
            os.chmod(
                path,
                stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
            )
        else:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def make_tree_writable(path: Path) -> None:
    """Clear read-only attributes recursively (required by Windows cleanup)."""

    if not path.exists():
        return
    ensure_writable(path)
    for root, directories, files in os.walk(path):
        root_path = Path(root)
        ensure_writable(root_path)
        for directory in directories:
            ensure_writable(root_path / directory)
        for file_name in files:
            ensure_writable(root_path / file_name)


def unique_suffix() -> str:
    return uuid.uuid4().hex[:12]


def safe_relative_path(root: Path, value: str, *, field_name: str = "path") -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=False)
        root_resolved = root.resolve(strict=True)
    except OSError as exc:
        raise DiagnosticError(
            E_FORMAT,
            f"cannot resolve {field_name}",
            field=field_name,
            details={"value": value, "reason": str(exc)},
        )
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise DiagnosticError(
            E_FORMAT,
            f"{field_name} escapes the session directory",
            field=field_name,
            details={"value": value},
        )
    return resolved


def read_json(path: Path, *, snapshot_id: str | None = None, field_name: str = "json") -> Any:
    try:
        with open(path, "rb") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise DiagnosticError(
            E_FORMAT,
            "JSON file is missing",
            snapshot_id=snapshot_id,
            field=field_name,
            details={"path": str(path)},
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(
            E_FORMAT,
            "JSON file is truncated or malformed",
            snapshot_id=snapshot_id,
            field=field_name,
            details={"path": str(path), "reason": str(exc)},
        )


def file_handle_sha256(handle: BinaryIO) -> str:
    position = handle.tell()
    handle.seek(0)
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
    handle.seek(position)
    return digest.hexdigest()
