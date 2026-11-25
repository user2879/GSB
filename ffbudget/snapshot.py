"""Staged snapshot commits with whole-directory atomic replacement.

Every stage artifact is first written under a temp directory and then moved
into place with os.replace, which is atomic on both Windows NTFS and POSIX.
A manifest tracks the global version, config hash, batch cursor, consumed
points and per-stage artifact versions. Startup identifies changes through
the manifest's format version and config hash rather than per-file merges.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from . import errors
from .fingerprint import content_fingerprint

STAGE_LOAD = "01_load"
STAGE_AUTHORITY = "02_authority"
STAGE_VIEW = "03_view"
STAGE_SCORE = "04_score"
STAGES = [STAGE_LOAD, STAGE_AUTHORITY, STAGE_VIEW, STAGE_SCORE]
MANIFEST_NAME = "manifest.json"


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.tmp_root = self.root / "tmp"
        self.root.mkdir(parents=True, exist_ok=True)

    def config_hash(self, config: dict) -> str:
        minimal = {key: value for key, value in config.items()
                   if not key.startswith("_")}
        return content_fingerprint(minimal)

    def read_manifest(self) -> dict | None:
        path = self.root / MANIFEST_NAME
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None

    def recovery_decision(self, config: dict) -> dict:
        manifest = self.read_manifest()
        config_hash = self.config_hash(config)
        if manifest is None:
            return {"mode": "full_rebuild", "reason": "no manifest"}
        if manifest.get("format_version") != config.get("format_version"):
            return {"mode": "full_rebuild",
                    "reason": errors.E_CONFIG_MISMATCH,
                    "detail": "format_version changed"}
        if manifest.get("config_hash") != config_hash:
            return {"mode": "full_rebuild",
                    "reason": errors.E_CONFIG_MISMATCH,
                    "detail": "config hash changed"}
        if manifest.get("committed") is not True:
            return {"mode": "full_rebuild",
                    "reason": errors.E_SNAPSHOT_CORRUPT,
                    "detail": "previous commit incomplete"}
        return {"mode": "incremental", "reason": "manifest matched",
                "manifest": manifest, "config_hash": config_hash}

    def write_stage(self, stage: str, payload: dict) -> str:
        stage_tmp = self.root / f".{stage}.tmp"
        if stage_tmp.exists():
            shutil.rmtree(stage_tmp)
        stage_tmp.mkdir(parents=True)
        artifact_path = stage_tmp / "artifact.json"
        self._atomic_json(artifact_path, payload)
        final_dir = self.root / stage
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(stage_tmp, final_dir)
        return str(final_dir / "artifact.json")

    def commit_manifest(self, manifest: dict) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix=".manifest-", dir=self.root))
        tmp_manifest = tmp_dir / MANIFEST_NAME
        manifest = dict(manifest, committed=True)
        self._atomic_json(tmp_manifest, manifest)
        os.replace(tmp_manifest, self.root / MANIFEST_NAME)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def cleanup_temp(self) -> None:
        for child in self.root.iterdir():
            if child.name.startswith(".") and child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _atomic_json(self, path: Path, payload: dict) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, ensure_ascii=False,
                      indent=2)
            handle.flush()
            os.fsync(handle.fileno())

    def artifact_versions(self, stage_paths: dict[str, str]) -> dict:
        versions = {}
        for stage, path in stage_paths.items():
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            versions[stage] = content_fingerprint(payload)
        return versions
