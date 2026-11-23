from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .cache import atomic_write_json, checksummed, payload_hash, verify_checksummed


STAGES = ("load", "inference", "merge", "evaluation")


class SnapshotStore:
    def __init__(self, state_dir: str | Path) -> None:
        self.root = Path(state_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.stages_root = self.root / "stages"
        self.stages_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    def load_manifest(self) -> tuple[dict[str, Any] | None, str | None]:
        if not self.manifest_path.exists():
            return None, None
        try:
            document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest = verify_checksummed(document)
            if not isinstance(manifest.get("stage_versions"), dict):
                raise ValueError("manifest has no stage_versions mapping")
            return manifest, None
        except Exception as exc:
            return None, str(exc)

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_json(self.manifest_path, checksummed(manifest))

    def publish_stage(
        self,
        stage: str,
        payload: dict[str, Any],
        manifest: dict[str, Any],
    ) -> tuple[bool, str, str]:
        if stage not in STAGES:
            raise ValueError(f"unknown snapshot stage {stage}")
        digest = payload_hash(payload)
        current_hash = manifest.setdefault("stage_hashes", {}).get(stage)
        if current_hash == digest:
            version = manifest.setdefault("stage_versions", {}).get(stage, "")
            return False, version, digest

        global_version = int(manifest.get("global_version", 0))
        version_name = f"v{global_version:08d}-{digest[:12]}"
        stage_dir = self.stages_root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        target = stage_dir / version_name
        if not target.exists():
            temp_dir = Path(
                tempfile.mkdtemp(prefix=f".{version_name}.", dir=stage_dir)
            )
            try:
                atomic_write_json(
                    temp_dir / "payload.json", checksummed(payload, schema=1)
                )
                os.replace(temp_dir, target)
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
        manifest.setdefault("stage_versions", {})[stage] = version_name
        manifest.setdefault("stage_hashes", {})[stage] = digest
        return True, version_name, digest

    def stage_payload(self, manifest: dict[str, Any], stage: str) -> dict[str, Any] | None:
        version = manifest.get("stage_versions", {}).get(stage)
        if not version:
            return None
        path = self.stages_root / stage / version / "payload.json"
        if not path.exists():
            return None
        try:
            return verify_checksummed(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None


def empty_manifest(config_hash: str, global_version: int = 0) -> dict[str, Any]:
    return {
        "batch_consumed": 0,
        "clean": False,
        "completed_images": [],
        "config_hash": config_hash,
        "cumulative_consumed": 0,
        "global_version": global_version,
        "pending": [],
        "schema": 1,
        "stage_hashes": {},
        "stage_versions": {},
    }
