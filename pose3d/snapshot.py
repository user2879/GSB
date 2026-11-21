"""Whole-stage snapshot commit with version-based replacement decisions."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .config import ENGINE_VERSION, SNAPSHOT_SCHEMA_VERSION
from .hashing import sha256_json


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


class SnapshotStore:
    """A versioned snapshot directory.

    Every commit writes a complete temporary tree and swaps the entire
    ``current`` directory.  Existing snapshots are never patched file by file.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.current_dir = self.root / "current"
        self.manifest_path = self.current_dir / "manifest.json"
        self.cache_path = self.current_dir / "cache.json"

    def load(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
        if not self.manifest_path.exists():
            return None, None, "no_snapshot"
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            cache = (
                json.loads(self.cache_path.read_text(encoding="utf-8"))
                if self.cache_path.exists()
                else None
            )
        except (OSError, json.JSONDecodeError):
            return None, None, "corrupt_snapshot"
        if not isinstance(manifest, dict):
            return None, None, "corrupt_snapshot"
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            return manifest, None, "schema_version_mismatch"
        if manifest.get("engine_version") != ENGINE_VERSION:
            return manifest, None, "engine_version_mismatch"
        return manifest, cache, "loaded"

    def _content_hash(
        self,
        *,
        input_version: str,
        completed_keypoints: list[str],
        consumed_budget: int,
        pending_queue: list[str],
        stage_payloads: dict[str, dict[str, Any]],
        cache_entries: dict[str, dict[str, Any]],
        config_hash: str,
        camera_hash: str,
        snapshot_commit_interval: int,
    ) -> tuple[str, dict[str, str]]:
        stage_hashes = {
            stage: sha256_json(stage_payloads.get(stage, {}))
            for stage in ("load", "triangulation", "rejection", "finalization")
        }
        cache_hash = sha256_json(cache_entries)
        content = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION,
            "input_version": input_version,
            "config_hash": config_hash,
            "camera_hash": camera_hash,
            "completed_keypoints": completed_keypoints,
            "consumed_budget": int(consumed_budget),
            "pending_queue": pending_queue,
            "stage_hashes": stage_hashes,
            "cache_hash": cache_hash,
            "snapshot_commit_interval": int(snapshot_commit_interval),
        }
        return sha256_json(content), stage_hashes

    def commit(
        self,
        *,
        input_version: str,
        config_hash: str,
        camera_hash: str,
        completed_keypoints: list[str],
        consumed_budget: int,
        pending_queue: list[str],
        stage_payloads: dict[str, dict[str, Any]],
        cache_entries: dict[str, dict[str, Any]],
        snapshot_commit_interval: int,
    ) -> dict[str, Any]:
        completed = sorted(completed_keypoints)
        pending = sorted(pending_queue)
        global_version, stage_hashes = self._content_hash(
            input_version=input_version,
            completed_keypoints=completed,
            consumed_budget=consumed_budget,
            pending_queue=pending,
            stage_payloads=stage_payloads,
            cache_entries=cache_entries,
            config_hash=config_hash,
            camera_hash=camera_hash,
            snapshot_commit_interval=snapshot_commit_interval,
        )
        existing_version = None
        if self.manifest_path.exists():
            try:
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    existing_version = existing.get("global_version")
            except (OSError, json.JSONDecodeError):
                existing_version = None
        if existing_version == global_version:
            return {
                "decision": "unchanged",
                "version": global_version,
                "input_version": input_version,
                "completed_keypoints": completed,
                "consumed_budget": int(consumed_budget),
                "path": str(self.current_dir),
            }

        self.root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=".snapshot-tmp-", dir=self.root))
        try:
            stages_dir = temp_dir / "stages"
            for stage in ("load", "triangulation", "rejection", "finalization"):
                _write_json(stages_dir / f"{stage}.json", stage_payloads.get(stage, {}))
            _write_json(temp_dir / "cache.json", {"schema": "pose3d-stage-cache/1", "entries": cache_entries})
            manifest = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "engine_version": ENGINE_VERSION,
                "global_version": global_version,
                "input_version": input_version,
                "config_hash": config_hash,
                "camera_hash": camera_hash,
                "completed_keypoints": completed,
                "consumed_budget": int(consumed_budget),
                "pending_queue": pending,
                "stage_hashes": stage_hashes,
                "cache_hash": sha256_json(cache_entries),
                "snapshot_commit_interval": int(snapshot_commit_interval),
            }
            _write_json(temp_dir / "manifest.json", manifest)
            self._swap_into_place(temp_dir)
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return {
            "decision": "replaced" if existing_version is not None else "created",
            "version": global_version,
            "previous_version": existing_version,
            "input_version": input_version,
            "completed_keypoints": completed,
            "consumed_budget": int(consumed_budget),
            "path": str(self.current_dir),
        }

    def _swap_into_place(self, temp_dir: Path) -> None:
        backup: Path | None = None
        if self.current_dir.exists():
            backup = self.root / f".snapshot-old-{os.getpid()}"
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            self.current_dir.rename(backup)
        try:
            temp_dir.rename(self.current_dir)
        except Exception:
            if backup is not None and backup.exists() and not self.current_dir.exists():
                backup.rename(self.current_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)


def read_snapshot_stage(
    root: str | Path, stage: str
) -> dict[str, Any] | None:
    path = Path(root) / "current" / "stages" / f"{stage}.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
