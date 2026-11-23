"""Two-level cache and staged, atomically-replaced snapshots.

Layout under the snapshot directory::

    snapshot/
      manifest.json          # global version, config hash, stage status, quotas
      tiles/<cache_key>.json # tile stage artefacts keyed by content/weights
      fusions/<fp>.json      # fusion stage artefacts keyed by dependency fp
      global.json            # post-process stage artefact
      tmp.<pid>/             # in-progress stage output, renamed into place

A stage is always written to a temp directory and then *whole directory*
replaced via ``os.replace``; we never overwrite individual committed files.
Resumption compares the global version and config hash, emitting
``E_SNAPSHOT_STALE`` when they no longer match the requested configuration and
rebuilding only the affected stages.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .classifier import TileResult
from .config import Config
from .errors import Code, FailureList
from .fusion import FusionResult
from .postprocess import GlobalRaster

STAGES = ("load", "classify", "fuse", "postprocess")


def config_hash(config: Config) -> str:
    payload = json.dumps(config.fingerprint_inputs(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class SnapshotStore:
    def __init__(self, root: str | Path, config: Config):
        self.root = Path(root)
        self.config = config
        self.tiles_dir = self.root / "tiles"
        self.fusions_dir = self.root / "fusions"
        self.manifest_path = self.root / "manifest.json"
        self.global_path = self.root / "global.json"
        self.tmp_root = self.root / f"tmp.{os.getpid()}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tiles_dir.mkdir(exist_ok=True)
        self.fusions_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ low io
    def _atomic_write_json(self, target: Path, doc: Any) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = self.tmp_root / target.parent.name
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_dir / (target.name + ".tmp")
        tmp_file.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
        os.replace(tmp_file, target)

    def _replace_stage_dir(self, stage: str, files: Dict[str, Any]) -> None:
        """Whole-directory replacement of a stage output."""
        stage_dir = self.root / stage
        fresh = self.tmp_root / f"{stage}.fresh"
        if fresh.exists():
            shutil.rmtree(fresh)
        fresh.mkdir(parents=True)
        for name, doc in files.items():
            (fresh / name).write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
        if stage_dir.exists():
            old = self.tmp_root / f"{stage}.old"
            if old.exists():
                shutil.rmtree(old)
            os.replace(stage_dir, old)
        os.replace(fresh, stage_dir)

    # ------------------------------------------------------------------ tile
    def tile_cache_path(self, cache_key: str) -> Path:
        return self.tiles_dir / f"{cache_key}.json"

    def load_tile(self, cache_key: str) -> Optional[TileResult]:
        path = self.tile_cache_path(cache_key)
        if not path.exists():
            return None
        return TileResult.from_cache_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_tile(self, result: TileResult) -> None:
        self._atomic_write_json(self.tile_cache_path(result.cache_key), result.to_cache_dict())

    # ------------------------------------------------------------------ fusion
    def fusion_cache_path(self, fingerprint: str) -> Path:
        return self.fusions_dir / f"{fingerprint}.json"

    def load_fusion(self, fingerprint: str) -> Optional[FusionResult]:
        path = self.fusion_cache_path(fingerprint)
        if not path.exists():
            return None
        return FusionResult.from_cache_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_fusion(self, result: FusionResult) -> None:
        self._atomic_write_json(self.fusion_cache_path(result.fingerprint), result.to_cache_dict())

    # ------------------------------------------------------------------ global
    def save_global(self, global_ras: GlobalRaster, meta: Dict[str, Any]) -> None:
        doc = {
            "raster": global_ras.raster.tolist(),
            "height": global_ras.height,
            "width": global_ras.width,
            "overlap_writes": global_ras.overlap_writes,
            "removed": [r.__dict__ for r in global_ras.removed],
            "meta": meta,
        }
        self._atomic_write_json(self.global_path, doc)

    def load_global(self) -> Optional[dict]:
        if not self.global_path.exists():
            return None
        return json.loads(self.global_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ manifest
    def read_manifest(self) -> Optional[dict]:
        if not self.manifest_path.exists():
            return None
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def write_manifest(self, doc: Dict[str, Any]) -> None:
        self._atomic_write_json(self.manifest_path, doc)

    def check_resume(self, expected_version: int, failures: FailureList) -> Dict[str, Any]:
        """Decide resume vs stale rebuild. Returns the existing manifest or {}."""
        existing = self.read_manifest()
        if existing is None:
            return {}
        chash = config_hash(self.config)
        stale_reasons: List[str] = []
        if existing.get("global_version") != expected_version:
            stale_reasons.append(
                f"version {existing.get('global_version')} != {expected_version}"
            )
        if existing.get("config_hash") != chash:
            stale_reasons.append("config hash mismatch")
        if stale_reasons:
            failures.add(
                Code.E_SNAPSHOT_STALE,
                str(self.manifest_path),
                "snapshot stale; rebuilding affected stages",
                detail={"reasons": stale_reasons},
            )
            return {}
        return existing

    def reset_stages_from(self, stage: str) -> None:
        """Invalidate a stage and every later stage (version change rebuild)."""
        try:
            idx = STAGES.index(stage)
        except ValueError:
            idx = 0
        for name in STAGES[idx:]:
            stage_dir = self.root / name
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
        if stage in ("classify",):
            if self.global_path.exists():
                self.global_path.unlink()
        if self.manifest_path.exists():
            self.manifest_path.unlink()
        self.tiles_dir.mkdir(exist_ok=True)
        self.fusions_dir.mkdir(exist_ok=True)

    def cleanup_tmp(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
