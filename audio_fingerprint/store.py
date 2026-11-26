from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class PersistentStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.cache_dir = self.root / "cache"
        self.commits_dir = self.root / "commits"
        self.snapshot_path = self.root / "snapshot.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.commits_dir.mkdir(parents=True, exist_ok=True)

    def atomic_write_json(
        self,
        path: Path,
        value: Any,
        *,
        fsync_file: bool = False,
        fsync_directory: bool = False,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.flush()
                if fsync_file:
                    os.fsync(handle.fileno())
            os.replace(temp_path, path)
            if fsync_directory:
                self._fsync_directory(path.parent)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _fsync_directory(self, directory: Path) -> None:
        try:
            fd = os.open(str(directory), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    def cache_path(self, cache_key: str) -> Path:
        safe = "".join(ch for ch in cache_key if ch.isalnum() or ch in {"_", "-"})
        return self.cache_dir / f"{safe}.json"

    def has(self, cache_key: str) -> bool:
        return self.cache_path(cache_key).exists()

    def get(self, cache_key: str) -> dict[str, Any]:
        with self.cache_path(cache_key).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def put(self, record: dict[str, Any]) -> None:
        cache_key = record["cache_key"]
        path = self.cache_path(cache_key)
        if not path.exists():
            self.atomic_write_json(path, record)

    def commit_snapshot(self, snapshot: dict[str, Any]) -> Path:
        version = int(snapshot["global_version"])
        commit_path = self.commits_dir / f"snapshot-{version:08d}.json"
        self.atomic_write_json(commit_path, snapshot, fsync_file=True, fsync_directory=True)
        self.atomic_write_json(self.snapshot_path, snapshot, fsync_file=True, fsync_directory=True)
        return commit_path

    def load_snapshot(self) -> dict[str, Any] | None:
        snapshot = self._load_json_if_valid(self.snapshot_path)
        if snapshot is not None:
            return snapshot
        commits = sorted(self.commits_dir.glob("snapshot-*.json"), reverse=True)
        for path in commits:
            snapshot = self._load_json_if_valid(path)
            if snapshot is not None:
                return snapshot
        return None

    def _load_json_if_valid(self, path: Path) -> dict[str, Any] | None:
        if not path.exists() or path.suffix != ".json":
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict) or "global_version" not in value:
                return None
            return value
        except (json.JSONDecodeError, OSError):
            return None

    def remove_stale_temps(self) -> list[str]:
        removed: list[str] = []
        for path in self.root.rglob("*.tmp"):
            try:
                path.unlink(missing_ok=True)
                removed.append(str(path))
            except OSError:
                pass
        return removed
