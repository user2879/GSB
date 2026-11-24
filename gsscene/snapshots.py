"""Stage-isolated snapshots and resume decisions."""

from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from . import codes
from .models import Failure
from .util import atomic_write_json, read_json, remove_tree

SNAPSHOT_SCHEMA_VERSION = 3
STAGES = ("load", "kernel", "render", "composite")


class SnapshotManager:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = Path(work_dir).resolve()
        self.root = self.work_dir / "snapshots"
        self.committed = self.root / "committed"
        self.staging = self.root / "staging"
        self.manifest_path = self.root / "manifest.json"
        self.committed.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        self._cleanup_staging()

    def _cleanup_staging(self) -> None:
        for path in sorted(self.staging.iterdir(), key=lambda item: item.name):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    def load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            value = read_json(self.manifest_path)
            if not isinstance(value, dict):
                return None
            return value
        except (OSError, ValueError):
            return None

    def resume_decision(self, config_family_hash: str) -> tuple[str, list[Failure]]:
        manifest = self.load_manifest()
        failures: list[Failure] = []
        if manifest is None:
            return "REBUILD", failures
        if int(manifest.get("snapshot_schema", -1)) != SNAPSHOT_SCHEMA_VERSION:
            failures.append(
                Failure(
                    code=codes.E_SNAPSHOT_MANIFEST,
                    message="snapshot schema changed; rebuilding cache",
                    source="snapshot",
                )
            )
            return "REBUILD", failures
        if manifest.get("config_family_hash") != config_family_hash:
            failures.append(
                Failure(
                    code=codes.E_SNAPSHOT_MANIFEST,
                    message="configuration family changed; rebuilding cache",
                    source="snapshot",
                )
            )
            return "REBUILD", failures
        return "RESUME", failures

    def reset(self) -> None:
        remove_tree(self.root)
        self.committed.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)

    def _stage_payload(self, stage: str, payload: dict[str, Any]) -> Path:
        run_id = uuid.uuid4().hex
        temp_dir = self.staging / f"{stage}-{time.time_ns()}-{run_id}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(temp_dir / "STAGE.json", payload)
        return temp_dir

    def _commit_stage(self, stage: str, payload: dict[str, Any]) -> int:
        temp_dir = self._stage_payload(stage, payload)
        target = self.committed / stage
        backup = self.committed / f".{stage}.backup-{uuid.uuid4().hex}"
        try:
            if target.exists():
                os.replace(target, backup)
            os.replace(temp_dir, target)
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        manifest = self.load_manifest() or {"stage_versions": {name: 0 for name in STAGES}}
        versions = dict(manifest.get("stage_versions", {}))
        versions[stage] = int(versions.get(stage, 0)) + 1
        manifest["stage_versions"] = versions
        atomic_write_json(self.manifest_path, manifest)
        return versions[stage]

    def commit_cycle(
        self,
        *,
        config_hash: str,
        config_family_hash: str,
        camera_hash: str,
        quant_gear: str,
        frac_bits: int,
        completed_views: list[int],
        completed_levels: list[int],
        edit_step: int,
        stage_payloads: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        old_manifest = self.load_manifest()
        next_version = int(old_manifest.get("version", 0)) + 1 if old_manifest else 1
        versions = dict(old_manifest.get("stage_versions", {})) if old_manifest else {name: 0 for name in STAGES}
        for stage in STAGES:
            payload = dict(stage_payloads.get(stage, {}))
            payload.update(
                {
                    "config_hash": config_hash,
                    "edit_step": int(edit_step),
                    "schema_version": SNAPSHOT_SCHEMA_VERSION,
                    "stage": stage,
                    "version": next_version,
                }
            )
            versions[stage] = int(versions.get(stage, 0)) + 1
            self._write_stage_atomically(stage, payload)
        manifest = {
            "camera_hash": camera_hash,
            "completed_levels": [int(item) for item in sorted(completed_levels)],
            "completed_views": [int(item) for item in sorted(completed_views)],
            "config_family_hash": config_family_hash,
            "config_hash": config_hash,
            "edit_step": int(edit_step),
            "frac_bits": int(frac_bits),
            "quant_gear": quant_gear,
            "snapshot_interval": payload.get("snapshot_interval"),
            "snapshot_schema": SNAPSHOT_SCHEMA_VERSION,
            "stage_versions": versions,
            "version": next_version,
        }
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def _write_stage_atomically(self, stage: str, payload: dict[str, Any]) -> None:
        temp_dir = self._stage_payload(stage, payload)
        target = self.committed / stage
        backup = self.committed / f".{stage}.backup-{uuid.uuid4().hex}"
        try:
            if target.exists():
                os.replace(target, backup)
            os.replace(temp_dir, target)
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
