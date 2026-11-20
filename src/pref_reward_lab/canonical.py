from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint(value: Any, *, prefix: str = "") -> str:
    payload = (prefix + canonical_json(value)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_fingerprint(text: str, *, prefix: str = "") -> str:
    return hashlib.sha256((prefix + text).encode("utf-8")).hexdigest()


def finite_float(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite float: {value!r}")
    return float(value)


def stable_float(value: float, digits: int = 12) -> float:
    finite_float(value)
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0 else rounded


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return target


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(path, canonical_json(value) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def failure_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item.get("code", "")),
        str(item.get("stage", "")),
        str(item.get("pair_id", "")),
        str(item.get("source_version", "")),
        int(item.get("line", 0) or 0),
        str(item.get("field", "")),
        str(item.get("message", "")),
        str(item.get("injection_id", "")),
    )


def sorted_failures(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        unique.setdefault(fingerprint(item, prefix="failure:"), dict(item))
    return sorted(unique.values(), key=failure_sort_key)

