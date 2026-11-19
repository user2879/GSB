"""Canonical JSON, digests and atomic local file output."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def dumps(value: Any) -> str:
    """Return canonical JSON with a stable key and array order.

    Array order is part of the data model.  Object keys are sorted so that the
    result does not depend on Python's insertion order or on JSON input order.
    Non-finite floats are rejected by ``allow_nan=False``.
    """

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def dumps_pretty(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def bytes_of(value: Any) -> bytes:
    return dumps(value).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(bytes_of(value)).hexdigest()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | os.PathLike[str], value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(dumps_pretty(value))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def digest_pack_rows(rows: Iterable[Any]) -> str:
    return digest(list(rows))
