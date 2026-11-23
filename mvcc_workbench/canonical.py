"""Canonical serialization and content digest helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Return the canonical UTF-8 JSON text used for byte comparisons."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def version_label(kind: str, value: Any) -> str:
    return f"v1:{kind}:{sha256_digest(value).split(':', 1)[1]}"


def write_json(path: str, value: Any, *, pretty: bool = True) -> None:
    """Write deterministic JSON while keeping a readable artifact form."""

    text = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ) if pretty else canonical_json(value) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
