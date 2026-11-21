"""Atomic versioned stage snapshots with fail-closed resume validation."""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .utils import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    new_temp_dir,
    read_json,
    safe_rmtree,
    sha256_bytes,
)


class SnapshotStore:
    def __init__(
        self,
        output_root: Path,
        global_version: int,
        config_hash: str,
        injection_hash: str,
        enabled: bool = True,
    ):
        self.output_root = Path(output_root)
        self.global_version = int(global_version)
        self.config_hash = config_hash
        self.injection_hash = injection_hash
        self.enabled = bool(enabled)
        self.root = self.output_root / "snapshots"
        self.active = self.root / "active"
        self.archive = self.root / "invalidated"
        self._lock = threading.RLock()
        self.valid = False
        self.resume = False
        self.reason = "snapshots_disabled"
        self.invalidated_manifest: dict[str, Any] | None = None
        self.previous_manifest: dict[str, Any] | None = None

    def _initial_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "snapshot-v1",
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "injection_hash": self.injection_hash,
            "completed_candidates": [],
            "budget_consumed": 0,
            "iterations_executed": 0,
            "stage_versions": {},
            "status": "running",
        }

    def validate(self) -> dict[str, Any]:
        if not self.enabled:
            return {"valid": False, "resume": False, "reason": "snapshots_disabled"}
        manifest_path = self.active / "manifest.json"
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            self.root.mkdir(parents=True, exist_ok=True)
            self.active.mkdir(parents=True, exist_ok=True)
            atomic_write_json(manifest_path, self._initial_manifest())
            self.valid = True
            self.resume = False
            self.reason = "new_snapshot"
            return self.status()

        self.previous_manifest = manifest
        reasons = []
        if int(manifest.get("global_version", -1)) != self.global_version:
            reasons.append("global_version_mismatch")
        if manifest.get("config_hash") != self.config_hash:
            reasons.append("config_hash_mismatch")
        if manifest.get("injection_hash") != self.injection_hash:
            reasons.append("injection_hash_mismatch")
        if reasons:
            self.archive.mkdir(parents=True, exist_ok=True)
            archive_name = f"invalid-{self.config_hash.split('-')[-1][:10]}-{uuid.uuid4().hex[:8]}"
            archive_path = self.archive / archive_name
            try:
                os.replace(self.active, archive_path)
            except OSError:
                import shutil

                shutil.copytree(self.active, archive_path)
                safe_rmtree(self.active)
            self.active.mkdir(parents=True, exist_ok=True)
            self.invalidated_manifest = manifest
            atomic_write_json(manifest_path, self._initial_manifest())
            self.valid = False
            self.resume = False
            self.reason = "snapshot_version_invalid:" + ",".join(reasons)
            return self.status()

        self.valid = True
        self.resume = True
        self.reason = "resume_compatible"
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "valid": self.valid,
            "resume": self.resume,
            "reason": self.reason,
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "injection_hash": self.injection_hash,
            "active_path": str(self.active),
            "invalidated_manifest": self.invalidated_manifest,
        }

    def _manifest(self) -> dict[str, Any]:
        return read_json(self.active / "manifest.json", self._initial_manifest())

    def update_manifest(self, **updates: Any) -> dict[str, Any]:
        if not self.enabled:
            return {}
        with self._lock:
            manifest = self._manifest()
            manifest.update(updates)
            manifest["global_version"] = self.global_version
            manifest["config_hash"] = self.config_hash
            manifest["injection_hash"] = self.injection_hash
            atomic_write_json(self.active / "manifest.json", manifest)
            return manifest

    def commit_stage(
        self,
        stage: str,
        payload: dict[str, Any],
        *,
        fingerprint: str,
        binary_files: dict[str, bytes] | None = None,
        extra_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"stage": stage, "committed": False, "reason": "snapshots_disabled"}
        binary_files = binary_files or {}
        payload_text = canonical_json(payload) + "\n"
        digest = sha256_bytes(payload_text.encode("utf-8"))
        for name in sorted(binary_files):
            digest = sha256_bytes(digest.encode("ascii") + name.encode("utf-8") + binary_files[name])
        stage_version = f"v-{digest[:20]}"
        with self._lock:
            stage_root = self.active / "stages" / stage
            version_dir = stage_root / stage_version
            if not version_dir.exists():
                temp_dir = new_temp_dir(self.active / ".tmp", f"stage-{stage}")
                try:
                    atomic_write_bytes(temp_dir / "payload.json", payload_text.encode("utf-8"))
                    for name, data in sorted(binary_files.items()):
                        atomic_write_bytes(temp_dir / name, data)
                    stage_root.mkdir(parents=True, exist_ok=True)
                    os.replace(temp_dir, version_dir)
                finally:
                    if temp_dir.exists():
                        safe_rmtree(temp_dir)
            pointer = {
                "stage": stage,
                "version": stage_version,
                "fingerprint": fingerprint,
                "payload": payload_text,
                "binary_files": sorted(binary_files),
            }
            atomic_write_json(self.active / "stage-pointers" / f"{stage}.json", pointer)
            manifest = self._manifest()
            stage_versions = dict(manifest.get("stage_versions", {}))
            stage_versions[stage] = {
                "version": stage_version,
                "fingerprint": fingerprint,
                "committed_files": ["payload.json"] + sorted(binary_files),
            }
            manifest["stage_versions"] = stage_versions
            if extra_manifest:
                manifest.update(extra_manifest)
            manifest["global_version"] = self.global_version
            manifest["config_hash"] = self.config_hash
            manifest["injection_hash"] = self.injection_hash
            atomic_write_json(self.active / "manifest.json", manifest)
            return {
                "stage": stage,
                "committed": True,
                "version": stage_version,
                "fingerprint": fingerprint,
                "payload_hash": digest,
            }

    def load_stage(self, stage: str) -> dict[str, Any] | None:
        if not self.enabled or not self.valid:
            return None
        pointer = read_json(self.active / "stage-pointers" / f"{stage}.json")
        if not isinstance(pointer, dict):
            return None
        version = str(pointer.get("version", ""))
        version_dir = self.active / "stages" / stage / version
        payload = read_json(version_dir / "payload.json")
        if payload is None:
            return None
        return {
            "stage": stage,
            "version": version,
            "fingerprint": pointer.get("fingerprint"),
            "payload": payload,
            "directory": str(version_dir),
            "binary_files": pointer.get("binary_files", []),
        }

    def load_stage_npz(self, stage: str, filename: str) -> dict[str, np.ndarray] | None:
        stage_data = self.load_stage(stage)
        if stage_data is None or filename not in stage_data.get("binary_files", []):
            return None
        path = Path(stage_data["directory"]) / filename
        try:
            with np.load(path, allow_pickle=False) as loaded:
                return {name: loaded[name].copy() for name in loaded.files}
        except (FileNotFoundError, OSError, ValueError):
            return None

    def manifest(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        return self._manifest()
