from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_paths(paths: Iterable[str | Path]) -> str:
    """Digest a set of files by base name and bytes."""
    digest = hashlib.sha256()
    entries: list[tuple[str, Path]] = []
    for item in paths:
        path = Path(item)
        entries.append((path.name, path))
    for name, path in sorted(entries, key=lambda pair: pair[0]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(digest_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def digest_tree(path: str | Path, suffixes: tuple[str, ...] | None = None) -> str:
    path = Path(path)
    files = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        if suffixes and item.suffix.lower() not in suffixes:
            continue
        files.append(item)
    digest = hashlib.sha256()
    for item in files:
        rel = item.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(digest_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def state_key(stage: str, upstream: str, parameters: dict[str, Any], version: str) -> str:
    payload = {
        "stage": stage,
        "upstream_content_fingerprint": upstream,
        "parameters_fingerprint": digest_json(parameters),
        "format_algorithm_version": version,
    }
    return digest_json(payload)


def array_fingerprint(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any] | None = None
) -> str:
    digest = hashlib.sha256()
    if metadata is not None:
        digest.update(canonical_json(metadata).encode("utf-8"))
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()

