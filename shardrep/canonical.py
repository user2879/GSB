"""Canonical serialisation, digests and atomic filesystem helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_text(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        allow_nan=False,
    ) + "\n"


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_object(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        if mode is not None:
            os.chmod(path, mode)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def ensure_relative_path(value: Any, field_path: str, *, base: Path) -> Path:
    """Validate a workspace-relative path and return its resolved location."""

    from .errors import SpecValidationError

    if not isinstance(value, str) or not value.strip():
        raise SpecValidationError("path must be a non-empty string", field_path=field_path)
    candidate = Path(value)
    if candidate.is_absolute() or candidate.drive:
        raise SpecValidationError("absolute paths are not allowed", field_path=field_path)
    parts = candidate.parts
    if any(part == ".." for part in parts):
        raise SpecValidationError("parent traversal is not allowed", field_path=field_path)
    base_resolved = base.resolve()
    resolved = (base_resolved / candidate).resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise SpecValidationError("path escapes the workspace", field_path=field_path) from exc
    return resolved


def safe_artifact_name(logical_id: str) -> str:
    from urllib.parse import quote

    return quote(logical_id, safe="") + ".json"

