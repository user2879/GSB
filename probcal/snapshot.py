from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .util import (
    atomic_write_json,
    atomic_write_text,
    fingerprint,
)


class SnapshotError(Exception):
    pass


class SnapshotStore:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir.resolve()
        self.manifest_path = self.state_dir / "manifest.json"
        self.cache_dir = self.state_dir / "cache"
        self.snapshots_dir = self.state_dir / "snapshots"
        self.reports_dir = self.state_dir / "reports"
        for directory in (
            self.state_dir,
            self.cache_dir,
            self.snapshots_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"cannot read manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise SnapshotError("manifest root must be an object")
        return manifest

    def artifact_dir(self, stage: str, cache_key: str) -> Path:
        return self.cache_dir / stage / cache_key

    def artifact_relative_path(self, stage: str, cache_key: str) -> str:
        return f"cache/{stage}/{cache_key}/artifact.json"

    def load_artifact(self, stage: str, cache_key: str) -> dict[str, Any] | None:
        artifact_path = self.artifact_dir(stage, cache_key) / "artifact.json"
        if not artifact_path.exists():
            return None
        try:
            with artifact_path.open("r", encoding="utf-8") as handle:
                wrapper = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if wrapper.get("stage") != stage or wrapper.get("cache_key") != cache_key:
            return None
        payload = wrapper.get("payload")
        if fingerprint(payload) != wrapper.get("payload_sha256"):
            return None
        return payload

    def commit_artifact(
        self, stage: str, cache_key: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        existing = self.load_artifact(stage, cache_key)
        if existing is not None:
            return existing
        stage_dir = self.cache_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".tmp-{stage}-", dir=str(stage_dir))
        )
        try:
            wrapper = {
                "stage": stage,
                "cache_key": cache_key,
                "payload_sha256": fingerprint(payload),
                "payload": payload,
            }
            artifact_path = temporary_dir / "artifact.json"
            atomic_write_text(
                artifact_path,
                json.dumps(wrapper, ensure_ascii=False, indent=2) + "\n",
            )
            destination = self.artifact_dir(stage, cache_key)
            if destination.exists():
                existing = self.load_artifact(stage, cache_key)
                if existing is not None:
                    return existing
                raise SnapshotError(f"invalid existing artifact directory: {destination}")
            os.replace(temporary_dir, destination)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)
        return payload

    def commit_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_json(self.manifest_path, manifest)

    def commit_snapshot(
        self, manifest: dict[str, Any], report: dict[str, Any]
    ) -> str:
        snapshot_id = f"v{int(manifest['global_version']):09d}"
        destination = self.snapshots_dir / snapshot_id
        if destination.exists():
            raise SnapshotError(f"snapshot already exists: {snapshot_id}")
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=".tmp-snapshot-", dir=str(self.snapshots_dir))
        )
        try:
            atomic_write_json(temporary_dir / "manifest.json", manifest)
            atomic_write_json(temporary_dir / "report.json", report)
            os.replace(temporary_dir, destination)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)
        atomic_write_json(self.reports_dir / f"{snapshot_id}.json", report)
        atomic_write_json(self.state_dir / "latest_report.json", report)
        return snapshot_id

    def verify_layout(self) -> dict[str, Any]:
        state_device = self.state_dir.stat().st_dev
        temp_device = self.cache_dir.stat().st_dev
        snapshot_device = self.snapshots_dir.stat().st_dev
        return {
            "state_dir": str(self.state_dir),
            "temporary_dir": str(self.cache_dir),
            "same_device": state_device == temp_device == snapshot_device,
            "state_device": state_device,
            "temporary_device": temp_device,
            "snapshot_device": snapshot_device,
            "system_temp_dir": tempfile.gettempdir(),
            "snapshot_temp_dir": str(self.snapshots_dir),
        }
