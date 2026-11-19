from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_text(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=(",", ": ") if indent is not None else (",", ":"),
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_object(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
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


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], value: Any, *, indent: int | None = 2) -> None:
    text = canonical_text(value, indent=indent)
    if indent is not None:
        text += "\n"
    atomic_write_text(path, text)


def safe_component(value: str, *, limit: int = 48) -> str:
    safe = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in value
    ).strip("._")
    if not safe:
        safe = "item"
    return safe[:limit]


def stable_artifact_name(task_name: str, target_layer: str, suffix: str) -> str:
    digest = sha256_object({"task_name": task_name, "target_layer": target_layer})[:12]
    return f"{safe_component(task_name)}-{safe_component(target_layer)}-{digest}{suffix}"


def relative_path(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()

