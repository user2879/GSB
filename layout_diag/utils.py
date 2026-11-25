"""Stable JSON, hashing, and filesystem helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any, *, length: int = 64) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any, *, pretty: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    path.write_text(text + "\n", encoding="utf-8")


def atomic_write_json(path: str | Path, value: Any, *, pretty: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        write_json(temporary, value, pretty=pretty)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def copy_atomic(source: str | Path, destination: str | Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def remove_tree(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted(dict.fromkeys(values))


def require_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def as_relative(path: str | Path, root: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
