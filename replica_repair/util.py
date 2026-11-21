"""Deterministic serialization and crash-safe local file helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any


SIMPLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def canonical_json_dumps(value: Any, *, pretty: bool = False, newline: bool = False) -> str:
    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return text + ("\n" if newline else "")


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return canonical_json_dumps(value, pretty=pretty, newline=True).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes, *, mode: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value, pretty=True))


def make_readonly(path: Path) -> None:
    path = Path(path)
    writable = stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH
    current = path.stat().st_mode
    os.chmod(path, current & ~writable)


def is_readonly(path: Path) -> bool:
    return (Path(path).stat().st_mode & stat.S_IWRITE) == 0


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def validate_simple_id(value: str, path: str) -> str:
    from .errors import FormatError

    if not isinstance(value, str) or not SIMPLE_ID_RE.fullmatch(value):
        raise FormatError(
            "identifier must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}",
            path=path,
        )
    if value in {".", ".."}:
        raise FormatError("identifier must not be . or ..", path=path)
    return value


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def ensure_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
