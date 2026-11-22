"""Canonical serialization and digest helpers used by every persisted artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_value(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def chain_digest(previous: str, record_digest: str) -> str:
    return sha256_bytes(f"{previous}:{record_digest}".encode("utf-8"))


def write_canonical(path: Path | str, value: Any, *, readonly: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json(value) + "\n"
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
    if readonly:
        try:
            target.chmod(0o444)
        except OSError:
            pass
    return target


def make_readonly(path: Path | str) -> None:
    try:
        Path(path).chmod(0o444)
    except OSError:
        pass


def make_writable(path: Path | str) -> None:
    try:
        Path(path).chmod(0o644)
    except OSError:
        pass
