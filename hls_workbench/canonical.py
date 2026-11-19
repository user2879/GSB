"""Canonical, byte-stable serialization and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_text(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_object(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def write_canonical(path: str | Path, value: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(value)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    return sha256_bytes(data)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_posix(path: str | Path, root: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
