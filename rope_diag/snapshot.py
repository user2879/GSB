from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import ResultCache
from .errors import ErrorCode, Failure
from .util import atomic_write_json, read_json, sha256_json


@dataclass
class SnapshotState:
    manifest: dict[str, Any] = field(default_factory=dict)
    cache: ResultCache = field(default_factory=ResultCache)
    cache_failure: Failure | None = None

    @property
    def exists(self) -> bool:
        return bool(self.manifest)


class SnapshotManager:
    """Whole-directory snapshot commits below the workspace.

    A commit writes both the manifest and cache into a staging directory, then
    swaps that directory into ``.../current``.  The previous directory is kept
    as a recovery backup until the replacement has succeeded.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.root = self.workspace / ".rope_diag"
        self.snapshots_root = self.root / "snapshots"
        self.temporary_root = self.root / "tmp"
        self.current = self.snapshots_root / "current"
        self.backup = self.snapshots_root / "previous"
        self.snapshots_root.mkdir(parents=True, exist_ok=True)
        self.temporary_root.mkdir(parents=True, exist_ok=True)
        self._recover_backup()

    def _recover_backup(self) -> None:
        if self.backup.exists() and not self.current.exists():
            try:
                os.replace(self.backup, self.current)
            except OSError:
                pass
        elif self.backup.exists() and self.current.exists():
            shutil.rmtree(self.backup, ignore_errors=True)

    def load(self) -> SnapshotState:
        manifest_path = self.current / "manifest.json"
        cache_path = self.current / "cache.json"
        state = SnapshotState()
        if not manifest_path.exists():
            return state
        try:
            manifest = read_json(manifest_path)
            if not isinstance(manifest, dict):
                raise ValueError("manifest must be an object")
            state.manifest = manifest
        except (OSError, ValueError) as exc:
            state.cache_failure = Failure(
                code=ErrorCode.CACHE_CORRUPT,
                message=f"快照清单损坏: {exc}",
                stage="snapshot",
                details={"path": manifest_path.as_posix()},
            )
            return state
        if cache_path.exists():
            try:
                state.cache = ResultCache.from_dict(read_json(cache_path))
                if state.cache.corrupt_reason:
                    state.cache_failure = Failure(
                        code=ErrorCode.CACHE_CORRUPT,
                        message=state.cache.corrupt_reason,
                        stage="snapshot",
                        details={"path": cache_path.as_posix()},
                    )
            except (OSError, ValueError) as exc:
                state.cache = ResultCache()
                state.cache_failure = Failure(
                    code=ErrorCode.CACHE_CORRUPT,
                    message=f"缓存损坏，已按空缓存运行: {exc}",
                    stage="snapshot",
                    details={"path": cache_path.as_posix()},
                )
        return state

    def commit(
        self,
        manifest: dict[str, Any],
        cache: ResultCache,
    ) -> dict[str, Any]:
        staging = self.snapshots_root / f"staging-{os.getpid()}-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            cache_payload = cache.to_dict()
            commit_seed = {"manifest": manifest, "cache": cache_payload}
            commit_id = sha256_json(commit_seed)
            manifest_payload = {
                **manifest,
                "commit_id": commit_id,
                "cache_entry_count": len(cache.entries),
            }
            atomic_write_json(staging / "manifest.json", manifest_payload)
            atomic_write_json(staging / "cache.json", cache_payload)
            (staging / "COMMITTED").write_text(commit_id + "\n", encoding="utf-8")

            if self.backup.exists():
                shutil.rmtree(self.backup, ignore_errors=True)
            if self.current.exists():
                os.replace(self.current, self.backup)
            try:
                os.replace(staging, self.current)
            except OSError:
                if self.backup.exists() and not self.current.exists():
                    os.replace(self.backup, self.current)
                raise
            if self.backup.exists():
                shutil.rmtree(self.backup, ignore_errors=True)
            return manifest_payload
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

