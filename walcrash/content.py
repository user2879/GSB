"""Deterministic, platform-independent page byte generation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: str | os.PathLike[str]) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def deterministic_page_bytes(page_id: str, size_bytes: int, version: int, lsn: int) -> bytes:
    """Generate exactly ``size_bytes`` bytes from stable logical coordinates.

    No random source, locale, clock, process identifier, or OS-specific order
    participates in this function.  Replaying the same LSN/version on Windows
    or Linux therefore produces byte-identical page files.
    """

    if size_bytes <= 0:
        raise ValueError("size_bytes must be positive")
    prefix = (
        f"WALPAGE\x1f{page_id}\x1fsize={size_bytes}\x1fversion={version}"
        f"\x1flsn={lsn}\x1f"
    ).encode("utf-8")
    output = bytearray()
    counter = 0
    while len(output) < size_bytes:
        block = hashlib.sha256(prefix + counter.to_bytes(8, "big")).digest()
        output.extend(block)
        counter += 1
    return bytes(output[:size_bytes])


def page_token(page_id: str) -> str:
    """Filesystem-safe stable token for arbitrary user page identifiers."""

    return hashlib.sha256(page_id.encode("utf-8")).hexdigest()[:20]

