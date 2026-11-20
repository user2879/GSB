"""Staged snapshots with whole-directory replacement and resume checks.

Each stage (preprocess / cluster / evaluate / diagnostics) is first written
into a temporary sibling directory and then swapped into place with
os.replace-style renames. The manifest records the global version, config
hash, injection-manifest hash, completed candidates, consumed iteration
budget and per-stage artifact versions. On startup a version/config
mismatch yields E_SNAPSHOT_MISMATCH and old products are not reused.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .fingerprint import _normalize

STAGES = ("preprocess", "cluster", "evaluate", "diagnostics")


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_normalize(obj), fh, sort_keys=True, ensure_ascii=True, indent=1)
    os.replace(tmp, path)


class SnapshotManager:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    # --- manifest -----------------------------------------------------------
    def load_manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def check_resume(self, config_hash: str,
                     injection_hash: str) -> tuple[bool, dict | None]:
        """Returns (reuse_ok, old_manifest)."""
        old = self.load_manifest()
        if old is None:
            return True, None
        if old.get("config_hash") != config_hash or \
                old.get("injection_hash") != injection_hash:
            return False, old
        return True, old

    def new_manifest(self, config_hash: str, injection_hash: str) -> dict:
        return {
            "version": 0,
            "config_hash": config_hash,
            "injection_hash": injection_hash,
            "completed_candidates": [],
            "consumed_budget": 0,
            "stages": {},
        }

    def commit_manifest(self, manifest: dict) -> None:
        manifest["version"] = int(manifest.get("version", 0)) + 1
        _write_json(self.manifest_path, manifest)

    # --- stage commit --------------------------------------------------------
    def commit_stage(self, stage: str, artifacts: dict[str, Any],
                     manifest: dict) -> None:
        """Write artifacts to a temp dir, then wholesale-replace the stage dir."""
        assert stage in STAGES, f"unknown stage {stage}"
        final_dir = self.root / stage
        tmp_dir = self.root / f".{stage}.tmp"
        old_dir = self.root / f".{stage}.old"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        for name, obj in artifacts.items():
            _write_json(tmp_dir / f"{name}.json", obj)
        if old_dir.exists():
            shutil.rmtree(old_dir)
        if final_dir.exists():
            os.replace(final_dir, old_dir)  # move aside first (Windows-safe)
        os.replace(tmp_dir, final_dir)
        if old_dir.exists():
            shutil.rmtree(old_dir)
        stages = manifest.setdefault("stages", {})
        prev = stages.get(stage, {}).get("version", 0)
        stages[stage] = {"version": prev + 1,
                         "artifacts": sorted(artifacts.keys())}
        self.commit_manifest(manifest)
