"""Versioned staged snapshots with an atomically committed manifest."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, read_json

MANIFEST_SCHEMA = "lds-manifest-v1"


@dataclass
class StageContext:
    stage_dir: Path
    manifest: dict[str, Any]
    resumed: bool


class SnapshotManager:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.staging_root = self.root / "staging"
        self.manifest_path = self.root / "manifest.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict[str, Any] | None:
        return self._load_json(self.manifest_path)

    def current_version(self) -> int:
        manifest = self.load_manifest()
        if not manifest:
            return 0
        value = manifest.get("global_version", 0)
        return int(value) if isinstance(value, int) else 0

    def begin(
        self,
        *,
        config_hash: str,
        input_hash: str,
        random_source_digest: str,
    ) -> StageContext:
        staging_dir = self.staging_root / "current"
        existing = self._load_json(staging_dir / "manifest.json")
        if (
            existing
            and existing.get("config_hash") == config_hash
            and existing.get("input_hash") == input_hash
            and existing.get("status") in {"running", "prepared"}
        ):
            return StageContext(staging_dir, existing, True)

        if staging_dir.exists():
            abandoned = self.staging_root / f"abandoned-{self.current_version() + 1}"
            suffix = 1
            while abandoned.exists():
                suffix += 1
                abandoned = self.staging_root / (
                    f"abandoned-{self.current_version() + 1}-{suffix}"
                )
            staging_dir.replace(abandoned)

        target_version = self.current_version() + 1
        while (self.versions_dir / f"v{target_version:08d}").exists():
            target_version += 1
        staging_dir.mkdir(parents=True)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "running",
            "global_version": target_version,
            "config_hash": config_hash,
            "input_hash": input_hash,
            "random_source_digest": random_source_digest,
            "stage_versions": {},
            "pending_buckets": [],
            "completed_buckets": [],
            "completed_batch_cursor": 0,
            "consumed_budget": 0,
        }
        atomic_write_json(staging_dir / "manifest.json", manifest)
        return StageContext(staging_dir, manifest, False)

    def save(self, context: StageContext) -> None:
        atomic_write_json(context.stage_dir / "manifest.json", context.manifest)

    def commit(self, context: StageContext) -> dict[str, Any]:
        context.manifest["status"] = "prepared"
        context.manifest["pending_buckets"] = []
        self.save(context)
        version_name = f"v{int(context.manifest['global_version']):08d}"
        version_dir = self.versions_dir / version_name
        if version_dir.exists():
            raise RuntimeError(f"target version directory already exists: {version_dir}")
        os.replace(context.stage_dir, version_dir)
        global_manifest = {
            **context.manifest,
            "status": "committed",
            "version_dir": f"versions/{version_name}",
        }
        atomic_write_json(self.manifest_path, global_manifest)
        return global_manifest

    def read_stage_json(self, context: StageContext, name: str) -> Any | None:
        return self._load_json(context.stage_dir / name)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = read_json(path)
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def clear_all(self) -> None:
        """Test/demo helper; production callers should prefer versioned commits."""
        if self.staging_root.exists():
            shutil.rmtree(self.staging_root)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self.manifest_path.unlink()

