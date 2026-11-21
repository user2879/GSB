"""Deterministic JSON (de)serialization and content fingerprinting."""
from __future__ import annotations

import hashlib
import json
import os
from decimal import Decimal
from typing import Any


def _default(obj: Any):
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_default
    )


def dumps(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, indent=2, default=_default
    )


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle, parse_float=Decimal)


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            rows.append((line_no, json.loads(text, parse_float=Decimal)))
    return rows


def atomic_write(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_dir = os.path.join(directory, ".tmp-" + os.path.basename(path))
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, "part")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    os.rmdir(tmp_dir)


def write_json(path: str, obj: Any) -> None:
    atomic_write(path, dumps(obj) + "\n")


def write_jsonl(path: str, rows) -> None:
    text = "".join(canonical_json(row) + "\n" for row in rows)
    atomic_write(path, text)
