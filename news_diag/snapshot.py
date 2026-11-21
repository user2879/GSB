"""Atomic generation-based snapshots.

A checkpoint is written into a unique temporary directory, fsynced, renamed to
an immutable generation directory, and finally made visible by atomically
replacing the CURRENT pointer. Orphan generations are never consulted.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .util import _fsync_dir, atomic_write_text


GENERATION_PATTERN = re.compile(r"^gen-(\d{12})$")


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_pointer = self.root / "CURRENT"
        self._last_loaded_version = 0
        self.last_load_error: str | None = None

    def load(self) -> dict[str, Any] | None:
        self.last_load_error = None
        if not self.current_pointer.exists():
            return None
        try:
            generation_name = self.current_pointer.read_text(encoding="utf-8").strip()
        except OSError as exc:
            self.last_load_error = f"CURRENT 读取失败: {exc}"
            return None
        if not GENERATION_PATTERN.match(generation_name):
            self.last_load_error = f"CURRENT 指向非法代名: {generation_name}"
            return None
        state_path = self.root / generation_name / "state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.last_load_error = f"快照状态读取失败: {exc}"
            return None
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            self.last_load_error = "快照 schema_version 不受支持"
            return None
        self._last_loaded_version = int(state.get("global_version", 0))
        return state

    def commit(self, state: dict[str, Any]) -> dict[str, Any]:
        committed = dict(state)
        version = max(
            self._last_loaded_version,
            self._current_disk_version(),
            int(committed.get("global_version", 0)),
        ) + 1
        committed["schema_version"] = 1
        committed["global_version"] = version
        temp_dir = Path(tempfile.mkdtemp(prefix=".tmp-", dir=self.root))
        try:
            state_path = temp_dir / "state.json"
            payload = json.dumps(committed, ensure_ascii=False, sort_keys=True, indent=2,
                                 separators=(",", ": "))
            with state_path.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_dir(temp_dir)
            generation_name = self._next_generation_name(version)
            generation_path = self.root / generation_name
            os.replace(temp_dir, generation_path)
            _fsync_dir(self.root)
            pointer_temp = self.root / f".CURRENT.{os.getpid()}.{id(committed)}.tmp"
            atomic_write_text(pointer_temp, generation_name + "\n")
            os.replace(pointer_temp, self.current_pointer)
            _fsync_dir(self.root)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        self._last_loaded_version = version
        self._cleanup_generations()
        return committed

    def _next_generation_name(self, version: int) -> str:
        candidate = version
        while True:
            name = f"gen-{candidate:012d}"
            if not (self.root / name).exists():
                return name
            candidate += 1

    def _cleanup_generations(self, keep: int = 3) -> None:
        current_name = ""
        try:
            current_name = self.current_pointer.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        generations = []
        for path in self.root.iterdir():
            match = GENERATION_PATTERN.match(path.name)
            if match and path.is_dir():
                generations.append((int(match.group(1)), path))
        generations.sort(reverse=True)
        retained: set[str] = {current_name}
        for _, path in generations:
            if len(retained) >= keep + 1:
                break
            retained.add(path.name)
        for _, path in generations:
            if path.name not in retained:
                shutil.rmtree(path, ignore_errors=True)
        for path in self.root.glob(".tmp-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    def current_generation(self) -> str | None:
        try:
            return self.current_pointer.read_text(encoding="utf-8").strip()
        except OSError:
            return None

    def _current_disk_version(self) -> int:
        try:
            generation_name = self.current_pointer.read_text(encoding="utf-8").strip()
        except OSError:
            return 0
        match = GENERATION_PATTERN.match(generation_name)
        return int(match.group(1)) if match else 0


def stage_versions(state: dict[str, Any]) -> dict[str, str]:
    cache = state.get("cache", {})
    versions: dict[str, str] = {}
    for stage in ("split", "scoring", "selection", "rewrite", "summary"):
        by_doc = cache.get(stage, {})
        keys = []
        for document_id in sorted(by_doc):
            entries = by_doc[document_id]
            if isinstance(entries, dict):
                keys.extend(f"{document_id}:{key}" for key in sorted(entries))
        versions[stage] = ",".join(keys)
    return {stage: _hash_short(value) for stage, value in versions.items()}


def _hash_short(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
