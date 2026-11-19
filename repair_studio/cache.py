from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

from .config import canonical_json, sha256_bytes


class CacheStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def stage_root(self, stage: str) -> Path:
        return self.root / "v1" / stage

    def artifact_path(self, stage: str, key: str) -> Path:
        return self.stage_root(stage) / key

    def load(self, stage: str, key: str) -> dict[str, Any] | None:
        path = self.artifact_path(stage, key)
        meta_path = path / "meta.json"
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if meta.get("stage") != stage or meta.get("cache_key") != key:
            return None
        files = meta.get("files")
        if not isinstance(files, dict):
            return None
        for filename, expected_hash in files.items():
            file_path = path / filename
            if not file_path.is_file():
                return None
            try:
                actual_hash = sha256_bytes(file_path.read_bytes())
            except OSError:
                return None
            if actual_hash != expected_hash:
                return None
        result = dict(meta)
        result["_path"] = str(path)
        return result

    def store(
        self,
        stage: str,
        key: str,
        meta: Mapping[str, Any],
        files: Mapping[str, bytes],
    ) -> dict[str, Any]:
        destination = self.artifact_path(stage, key)
        with self._lock:
            existing = self.load(stage, key)
            if existing is not None:
                return existing
            stage_root = self.stage_root(stage)
            stage_root.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".tmp-", dir=stage_root))
            try:
                file_hashes: dict[str, str] = {}
                for filename, payload in files.items():
                    if Path(filename).name != filename:
                        raise ValueError("artifact filenames must be flat")
                    target = temporary / filename
                    with target.open("wb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    file_hashes[filename] = sha256_bytes(payload)
                complete_meta = {
                    **dict(meta),
                    "stage": stage,
                    "cache_key": key,
                    "files": file_hashes,
                }
                meta_bytes = canonical_json(complete_meta)
                meta_path = temporary / "meta.json"
                with meta_path.open("wb") as handle:
                    handle.write(meta_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                _fsync_directory(temporary)
                if destination.exists():
                    shutil.rmtree(temporary)
                else:
                    os.replace(temporary, destination)
                    _fsync_directory(stage_root)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        loaded = self.load(stage, key)
        if loaded is None:
            raise RuntimeError(f"cache artifact failed verification after write: {stage}/{key}")
        return loaded


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

