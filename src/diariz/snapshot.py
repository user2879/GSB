"""Staged snapshots with temp-directory staging and atomic replacement."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .cache import STAGES, atomic_write_json
from .errors import SnapshotConflictError

MANIFEST_NAME = "manifest.json"


@dataclass
class SnapshotManager:
    root: str
    current_dir: str = ""
    temp_dir: str = ""

    def __post_init__(self):
        self.current_dir = os.path.join(self.root, "current")
        self.temp_dir = os.path.join(self.root, "tmp")
        os.makedirs(self.current_dir, exist_ok=True)

    def _manifest_path(self) -> str:
        return os.path.join(self.current_dir, MANIFEST_NAME)

    def read_manifest(self) -> Dict[str, Any]:
        path = self._manifest_path()
        if not os.path.exists(path):
            return {"global_version": 0, "config_hash": "",
                    "audio_cursor": [], "artifact_versions": {}}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def begin(self) -> str:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        for stage in STAGES:
            os.makedirs(os.path.join(self.temp_dir, stage), exist_ok=True)
        return self.temp_dir

    def stage_artifact(self, stage: str, name: str, payload: Any) -> str:
        path = os.path.join(self.temp_dir, stage, name)
        atomic_write_json(path, payload)
        return path

    def commit(self, config_hash: str, audio_cursor: List[str],
               artifact_versions: Dict[str, int],
               expected_version: Optional[int] = None) -> Dict[str, Any]:
        prior = self.read_manifest()
        if expected_version is not None and prior["global_version"] != expected_version:
            raise SnapshotConflictError(
                "manifest version changed before commit",
                {"expected": expected_version,
                 "observed": prior["global_version"]})
        next_version = int(prior["global_version"]) + 1
        prior_versions = dict(prior.get("artifact_versions", {}))
        for stage, version in artifact_versions.items():
            prior_versions[stage] = version
        manifest = {
            "global_version": next_version,
            "config_hash": config_hash,
            "audio_cursor": sorted(audio_cursor),
            "artifact_versions": prior_versions,
        }
        atomic_write_json(os.path.join(self.temp_dir, MANIFEST_NAME), manifest)
        # Whole-directory replacement: swap temp into current atomically via
        # a backup/rename dance that works on both Linux and Windows.
        backup = os.path.join(self.root, "backup")
        if os.path.exists(backup):
            shutil.rmtree(backup)
        os.replace(self.current_dir, backup)
        try:
            os.replace(self.temp_dir, self.current_dir)
        except OSError:
            os.replace(backup, self.current_dir)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return manifest
