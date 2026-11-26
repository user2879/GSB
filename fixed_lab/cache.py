"""Content-addressed JSON cache and deterministic reuse ledger helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .hashing import canonical_dumps


class ContentCache:
    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root is not None else None
        if self.root is not None:
            (self.root / "objects").mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path | None:
        if self.root is None:
            return None
        safe = key.replace("/", "_").replace(":", "_")
        return self.root / "objects" / f"{safe}.json"

    def get(self, key: str) -> Any | None:
        path = self.path_for(key)
        if path is None or not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def has(self, key: str) -> bool:
        path = self.path_for(key)
        return path is not None and path.exists()

    def put(self, key: str, value: Any) -> None:
        path = self.path_for(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent, suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(canonical_dumps(value))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)


def write_json_atomic(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path
