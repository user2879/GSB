"""Atomic stage snapshots and resume handling.

Each stage is written to a temporary directory and promoted into the snapshot
directory with a single ``os.replace`` of its manifest pointer, so a hard kill can
never leave a half-written stage. The global manifest records the global version,
config fingerprint, caliber version, completed batches and per-stage artifact
versions. Resume is only allowed when the stored config fingerprint matches the
current configuration.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from .errors import E_CONFIG_MISMATCH, FailureRecord
from .fingerprint import fingerprint_json

MANIFEST_NAME = "manifest.json"


class SnapshotStore:
    def __init__(self, snapshot_dir: str):
        self.snapshot_dir = snapshot_dir
        self.tmp_dir = os.path.join(snapshot_dir, "tmp")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.snapshot_dir, MANIFEST_NAME)

    def has_manifest(self) -> bool:
        return os.path.exists(self.manifest_path)

    def load_manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_manifest_atomic(self, manifest: Dict[str, Any]) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=self.tmp_dir, suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.manifest_path)

    def write_stage_atomic(self, stage: str, payload: Any) -> str:
        stage_dir = os.path.join(self.snapshot_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)
        version = fingerprint_json(payload)
        fd, tmp_path = tempfile.mkstemp(dir=self.tmp_dir, prefix=f"{stage}-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, ensure_ascii=False, indent=2)
        final_path = os.path.join(stage_dir, "latest.json")
        os.replace(tmp_path, final_path)
        pointer = os.path.join(stage_dir, "version.txt")
        fd2, tmp_pointer = tempfile.mkstemp(dir=self.tmp_dir, prefix="ptr-")
        with os.fdopen(fd2, "w", encoding="utf-8") as handle:
            handle.write(version)
        os.replace(tmp_pointer, pointer)
        return version

    def read_stage(self, stage: str) -> Optional[Any]:
        path = os.path.join(self.snapshot_dir, stage, "latest.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def commit(
        self,
        global_version: int,
        config_fp: str,
        caliber: str,
        completed_batches: int,
        stage_payloads: Dict[str, Any],
    ) -> Dict[str, Any]:
        stage_versions: Dict[str, str] = {}
        for stage in sorted(stage_payloads):
            stage_versions[stage] = self.write_stage_atomic(
                stage, stage_payloads[stage]
            )
        manifest = {
            "global_version": global_version,
            "config_fingerprint": config_fp,
            "caliber_version": caliber,
            "completed_batches": completed_batches,
            "stage_versions": stage_versions,
        }
        self.write_manifest_atomic(manifest)
        return manifest

    def resume_check(self, config_fp: str, caliber: str) -> Optional[FailureRecord]:
        if not self.has_manifest():
            return None
        manifest = self.load_manifest()
        if manifest.get("config_fingerprint") != config_fp:
            return FailureRecord(
                stage="snapshot",
                code=E_CONFIG_MISMATCH,
                reason="config_fingerprint_changed",
                detail=(
                    f"stored={manifest.get('config_fingerprint')},"
                    f"current={config_fp}"
                ),
            )
        if manifest.get("caliber_version") != caliber:
            return FailureRecord(
                stage="snapshot",
                code=E_CONFIG_MISMATCH,
                reason="caliber_version_changed",
                detail=f"stored={manifest.get('caliber_version')},current={caliber}",
            )
        return None

    def reset(self):
        manifest = {
            "global_version": 0,
            "config_fingerprint": "",
            "caliber_version": "",
            "completed_batches": 0,
            "stage_versions": {},
        }
        self.write_manifest_atomic(manifest)
