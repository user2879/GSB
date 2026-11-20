"""Atomic stage snapshots via temp-directory build + directory swap.

Each commit writes a brand-new versioned directory under <root>/tmp and then
atomically replaces the active snapshot directory. A version change is always
a directory replacement, never an in-place file overwrite, so a hard kill can
only leave the previous complete snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

STAGE_FEATURES = "features"
STAGE_DECODE = "decode"
STAGE_MERGE = "merge"
STAGE_PREFIX = "prefix"
STAGES = (STAGE_FEATURES, STAGE_DECODE, STAGE_MERGE, STAGE_PREFIX)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SnapshotStore:
    root: str
    active_name: str = "active"

    def __post_init__(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        self.tmp_root = os.path.join(self.root, "tmp")
        os.makedirs(self.tmp_root, exist_ok=True)
        self.active_path = os.path.join(self.root, self.active_name)

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.active_path, "manifest.json")

    def load_manifest(self) -> Optional[Dict[str, Any]]:
        path = self.manifest_path
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def commit(self, version: int, config_hash_value: str, audio_fingerprint: str,
               stage_artifacts: Dict[str, List[str]], recovery_chunk: int,
               extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a complete snapshot in a temp dir and atomically swap it in."""
        build_dir = tempfile.mkdtemp(prefix=f"snap-v{version}-", dir=self.tmp_root)
        stage_versions: Dict[str, Dict[str, Any]] = {}
        for stage in STAGES:
            files = stage_artifacts.get(stage, [])
            stage_dir = os.path.join(build_dir, stage)
            os.makedirs(stage_dir, exist_ok=True)
            file_entries: List[Dict[str, Any]] = []
            for source_path in files:
                destination = os.path.join(stage_dir, os.path.basename(source_path))
                shutil.copy2(source_path, destination)
                file_entries.append({
                    "name": os.path.basename(source_path),
                    "sha256": _sha256_file(destination),
                })
            stage_payload = json.dumps(sorted(file_entries, key=lambda item: item["name"]),
                                       sort_keys=True).encode("utf-8")
            stage_versions[stage] = {
                "version": version,
                "hash": hashlib.sha256(stage_payload).hexdigest()[:16],
                "files": file_entries,
            }
        manifest = {
            "global_version": version,
            "config_hash": config_hash_value,
            "audio_fingerprint": audio_fingerprint,
            "recovery_chunk_index": recovery_chunk,
            "stages": stage_versions,
            "extra": extra or {},
        }
        with open(os.path.join(build_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2)
        previous = self.active_path
        backup = os.path.join(self.root, f"_old_v{version - 1}")
        had_previous = os.path.exists(previous)
        if had_previous:
            if os.path.exists(backup):
                shutil.rmtree(backup)
            os.rename(previous, backup)
        os.rename(build_dir, previous)
        if had_previous:
            shutil.rmtree(backup, ignore_errors=True)
        self._cleanup_tmp()
        return manifest

    def _cleanup_tmp(self) -> None:
        for name in os.listdir(self.tmp_root):
            path = os.path.join(self.tmp_root, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    def recovery_point(self) -> Optional[Dict[str, Any]]:
        manifest = self.load_manifest()
        if manifest is None:
            return None
        return {
            "global_version": manifest["global_version"],
            "recovery_chunk_index": manifest["recovery_chunk_index"],
            "config_hash": manifest["config_hash"],
            "audio_fingerprint": manifest["audio_fingerprint"],
            "stages": sorted(manifest["stages"].keys()),
        }

    def stage_file(self, stage: str, name: str) -> Optional[str]:
        path = os.path.join(self.active_path, stage, name)
        return path if os.path.exists(path) else None
