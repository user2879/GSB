from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_files(paths: Iterable[str | os.PathLike[str]], root: str | os.PathLike[str] | None = None) -> str:
    base = Path(root) if root is not None else None
    records: list[dict[str, str]] = []
    for item in sorted((Path(path) for path in paths), key=lambda value: str(value)):
        relative = str(item.relative_to(base)) if base is not None else item.name
        records.append({"path": relative, "sha256": sha256_file(item)})
    return sha256_json(records)


def hash_directory(root: str | os.PathLike[str], suffixes: tuple[str, ...] | None = None) -> str:
    base = Path(root)
    paths = [path for path in base.rglob("*") if path.is_file()]
    if suffixes is not None:
        paths = [path for path in paths if path.suffix.lower() in suffixes]
    return hash_files(paths, base)


def cache_key(*parts: Any, schema: str = "stemscope-cache-v1") -> str:
    return sha256_json({"schema": schema, "parts": list(parts)})
