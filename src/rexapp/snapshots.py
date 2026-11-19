"""Atomic stage snapshots written via temp files and replaced as a whole."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .fingerprint import fingerprint


class SnapshotStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "snapshot.json"
        self.version = 0
        self._last_content_fp: str | None = None
        if self.path.exists():
            self._last_content_fp = fingerprint(self.path.read_text(encoding="utf-8"))

    def commit(self, payload: dict[str, Any]) -> dict:
        payload = dict(payload)
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        content_fp = fingerprint(content)
        changed = content_fp != self._last_content_fp
        if not changed:
            return {"changed": False, "version": self.version, "path": str(self.path)}

        fd, tmp_name = tempfile.mkstemp(prefix="snapshot-", suffix=".tmp", dir=str(self.directory))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

        self.version += 1
        self._last_content_fp = content_fp
        return {"changed": True, "version": self.version, "path": str(self.path)}
