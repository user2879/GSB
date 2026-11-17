"""Small deterministic and crash-safe file helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, content: str, *, readonly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        if readonly:
            try:
                os.chmod(path, 0o444)
            except OSError:
                pass
        _fsync_directory(path.parent)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def atomic_write_json(
    path: Path, value: Any, *, readonly: bool = False, compact: bool = False
) -> None:
    if compact:
        content = canonical_json(value) + "\n"
    else:
        content = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ) + "\n"
    atomic_write_text(path, content, readonly=readonly)


def write_readonly_once(path: Path, value: Any) -> bool:
    """Write a deterministic backup once and mark it read-only.

    Returns True when the file was created by this call.
    """

    if path.exists():
        try:
            os.chmod(path, 0o444)
        except OSError:
            pass
        return False
    atomic_write_json(path, value, readonly=True)
    return True


def write_readonly_text_once(path: Path, content: str) -> bool:
    if path.exists():
        try:
            os.chmod(path, 0o444)
        except OSError:
            pass
        return False
    atomic_write_text(path, content, readonly=True)
    return True


def safe_component(value: str) -> str:
    result = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            result.append(char)
        else:
            result.append("_")
    cleaned = "".join(result).strip("._")
    return cleaned or "run"


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def flatten_mapping(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key in sorted(value):
        item = value[key]
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(flatten_mapping(item, path))
        else:
            flattened[path] = item
    return flattened


def workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def workspace_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return workspace / path
