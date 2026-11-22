"""Canonical serialization, digests, atomic file writes, and JSONL logging."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .errors import io_error


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


def canonical_lines(values: Iterable[Any]) -> bytes:
    return b"".join(canonical_bytes(value) + b"\n" for value in values)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: str | Path, value: Any, *, pretty: bool = False) -> None:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    else:
        text = canonical_json(value) + "\n"
    atomic_write_text(path, text)


def write_jsonl(path: str | Path, values: Iterable[Any]) -> None:
    atomic_write_bytes(path, canonical_lines(values))


def read_json(path: str | Path, *, code: str = "E_IO") -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise io_error("file not found", path=str(path))
    except json.JSONDecodeError as exc:
        raise io_error(
            "invalid JSON",
            path=str(path),
            line=exc.lineno,
            column=exc.colno,
            reason=exc.msg,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(value))
            handle.write("\n")
            handle.flush()
    except OSError as exc:
        print(f"warning: could not append structured log {target}: {exc}", file=sys.stderr)


def make_read_only(path: str | Path) -> None:
    target = Path(path)
    if target.is_dir():
        for child in sorted(target.rglob("*"), reverse=True):
            make_read_only(child)
    if target.exists():
        mode = target.stat().st_mode
        os.chmod(target, mode & ~0o222)

