"""Small deterministic and atomic output helpers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .hashing import canonical_json


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: str | os.PathLike[str], value: Any, *, pretty: bool = True) -> None:
    atomic_write_text(path, canonical_json(value, pretty=pretty) + "\n")


def atomic_write_jsonl(path: str | os.PathLike[str], records: Iterable[Any]) -> None:
    lines = [canonical_json(record) for record in records]
    text = "\n".join(lines)
    if text:
        text += "\n"
    atomic_write_text(path, text)


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl_strict(path: str | os.PathLike[str]) -> list[tuple[int, Any, str | None]]:
    """Read JSONL, retaining malformed lines for validation diagnostics."""

    records: list[tuple[int, Any, str | None]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.rstrip("\r\n")
            if not text.strip():
                records.append((line_number, None, "blank line"))
                continue
            try:
                records.append((line_number, json.loads(text), None))
            except json.JSONDecodeError as exc:
                records.append((line_number, None, f"invalid JSON: {exc.msg}"))
    return records


def replace_tree_from_staging(staging: Path, target: Path) -> None:
    """Atomically expose a fully written directory at target when possible."""

    target.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if target.exists():
        backup = target.with_name(f".{target.name}.old-{os.getpid()}")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(target, backup)
    os.replace(staging, target)
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)

