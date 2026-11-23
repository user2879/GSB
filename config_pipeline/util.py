"""Small deterministic helpers shared by the pipeline and simulator."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON representation.

    Dict insertion and hash iteration order are deliberately erased.  The
    representation is also strict about non-finite floats because they cannot
    participate in a portable content digest.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))


def stable_int(text: str) -> int:
    """A process-independent non-negative 64-bit integer derived from text."""

    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Write JSON via a sibling temp file and os.replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(value) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
