"""Atomic versioned global snapshots and crash recovery.

A snapshot is only considered a recovery point when its directory has been
fully renamed into place *and* its manifest is complete. Partial temporary
directories are ignored (and reaped) so a hard kill can never expose a
half-written state.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BusinessError, ErrorCode

MANIFEST_NAME = "manifest.json"
GLOBAL_GRID_NAME = "global_grid.npy"
POSTPROCESS_NAME = "postprocess.json"
CURRENT_LINK = "current"
TMP_PREFIX = "snapshot.tmp."


@dataclass(frozen=True)
class SnapshotInfo:
    version: int
    path: Path
    manifest: dict[str, Any]

    @property
    def complete(self) -> bool:
        return (self.path / MANIFEST_NAME).exists() and (self.path / GLOBAL_GRID_NAME).exists()


class SnapshotManager:
    def __init__(self, work_root: str | Path):
        self.root = Path(work_root)
        self.snapshots_dir = self.root / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_temp(self) -> None:
        for entry in self.snapshots_dir.glob(f"{TMP_PREFIX}*"):
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)

    def latest(self) -> SnapshotInfo | None:
        self.cleanup_temp()
        candidates = []
        for entry in self.snapshots_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("v"):
                continue
            try:
                version = int(entry.name[1:])
            except ValueError:
                continue
            manifest_path = entry / MANIFEST_NAME
            if not manifest_path.exists():
                continue
            try:
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if not (entry / GLOBAL_GRID_NAME).exists():
                continue
            candidates.append(SnapshotInfo(version, entry, manifest))
        if not candidates:
            return None
        return max(candidates, key=lambda info: info.version)

    def commit(
        self,
        *,
        version: int,
        config_hash: str,
        index_version: str,
        tile_versions: dict[str, str],
        fusion_versions: dict[str, str],
        global_grid: np.ndarray,
        postprocess_records: list[dict[str, Any]],
        pending: list[list[int]],
        failures: list[dict[str, Any]],
    ) -> SnapshotInfo:
        target = self.snapshots_dir / f"v{version:06d}"
        if target.exists():
            shutil.rmtree(target)
        tmp = self.snapshots_dir / f"{TMP_PREFIX}{os.getpid()}_{version:06d}"
        tmp.mkdir(parents=True)
        try:
            np.save(tmp / GLOBAL_GRID_NAME, global_grid)
            with open(tmp / POSTPROCESS_NAME, "w", encoding="utf-8") as handle:
                json.dump(postprocess_records, handle, sort_keys=True, indent=2)
            manifest = {
                "version": version,
                "config_hash": config_hash,
                "index_version": index_version,
                "tile_versions": dict(sorted(tile_versions.items())),
                "fusion_versions": dict(sorted(fusion_versions.items())),
                "pending": pending,
                "failures": failures,
                "grid_shape": list(global_grid.shape),
            }
            with open(tmp / MANIFEST_NAME, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, indent=2)
            os.replace(tmp, target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        current = self.snapshots_dir / CURRENT_LINK
        # Symlinks are nice-to-have; replace with a plain pointer file for portability.
        pointer = self.snapshots_dir / "current.txt"
        pointer.write_text(target.name, encoding="utf-8")
        try:
            if current.is_symlink() or current.exists():
                current.unlink()
            current.symlink_to(target.name)
        except OSError:
            pass
        return SnapshotInfo(version, target, manifest)

    def load_grid(self, info: SnapshotInfo) -> np.ndarray:
        path = info.path / GLOBAL_GRID_NAME
        if not path.exists():
            raise BusinessError(ErrorCode.E_SNAPSHOT_INVALID,
                                f"global grid missing in snapshot v{info.version}")
        return np.load(path)

    def versions_changed(self, previous: SnapshotInfo | None,
                         current: SnapshotInfo) -> dict[str, list[str]]:
        if previous is None:
            return {
                "tiles": list(current.manifest["tile_versions"]),
                "fusions": list(current.manifest["fusion_versions"]),
            }
        prev_tiles = previous.manifest["tile_versions"]
        prev_fusions = previous.manifest["fusion_versions"]
        cur_tiles = current.manifest["tile_versions"]
        cur_fusions = current.manifest["fusion_versions"]
        return {
            "tiles": sorted(k for k, v in cur_tiles.items() if prev_tiles.get(k) != v),
            "fusions": sorted(k for k, v in cur_fusions.items() if prev_fusions.get(k) != v),
        }
