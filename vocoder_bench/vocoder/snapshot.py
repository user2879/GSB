"""Stage snapshots: temp-dir writes followed by atomic directory replacement."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

STAGE_ORDER = ("load", "phase", "harmonic", "overlap_add", "metric")


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.tmp_root = self.root / "tmp"
        self.stage_root = self.root / "stages"
        self.manifest_path = self.root / "manifest.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.stage_root.mkdir(parents=True, exist_ok=True)

    def _stage_dir(self, stage: str, version: int) -> Path:
        return self.stage_root / f"v{version:06d}_{stage}"

    def commit_stage(self, stage: str, version: int, config_hash: str,
                     phase_bits: int, phase_quant_step: int,
                     completed_frames: list[str], command_cursor: int,
                     artifacts: dict) -> Path:
        """Write to a unique temp dir, then atomically replace the stage dir."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{stage}_", dir=self.tmp_root))
        try:
            for name, content in artifacts.items():
                (temp_dir / name).write_text(
                    json.dumps(content, indent=2, sort_keys=True), encoding="utf-8"
                )
            manifest = {
                "version": version,
                "stage": stage,
                "config_hash": config_hash,
                "phase_bits": phase_bits,
                "phase_quant_step": phase_quant_step,
                "completed_frames": list(completed_frames),
                "command_cursor": command_cursor,
            }
            (temp_dir / "stage_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            target = self._stage_dir(stage, version)
            if target.exists():
                shutil.rmtree(target)
            os.replace(temp_dir, target)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        self._write_global_manifest(version, config_hash, phase_bits,
                                    phase_quant_step, completed_frames,
                                    command_cursor, stage)
        return target

    def _write_global_manifest(self, version: int, config_hash: str,
                               phase_bits: int, phase_quant_step: int,
                               completed_frames: list[str],
                               command_cursor: int, last_stage: str) -> None:
        previous = self.read_manifest()
        if previous is not None and version < previous["version"]:
            raise RuntimeError(
                f"snapshot version regression: {version} < {previous['version']}"
            )
        manifest = {
            "version": version,
            "config_hash": config_hash,
            "phase_bits": phase_bits,
            "phase_quant_step": phase_quant_step,
            "completed_frames": list(completed_frames),
            "command_cursor": command_cursor,
            "last_stage": last_stage,
            "committed_stages": sorted(
                p.name for p in self.stage_root.glob(f"v{version:06d}_*")
            ),
        }
        fd, tmp_name = tempfile.mkstemp(prefix="manifest_", dir=self.tmp_root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            os.replace(tmp_name, self.manifest_path)
        finally:
            if Path(tmp_name).exists():
                Path(tmp_name).unlink()

    def read_manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def latest_complete_version(self) -> int:
        """Only fully replaced stage dirs count; partial tmp dirs are ignored."""
        manifest = self.read_manifest()
        return 0 if manifest is None else int(manifest["version"])

    def recover(self) -> dict | None:
        manifest = self.read_manifest()
        if manifest is None:
            return None
        version = manifest["version"]
        available = {
            p.name.split("_", 1)[1]: p
            for p in self.stage_root.glob(f"v{version:06d}_*")
            if p.is_dir() and (p / "stage_manifest.json").exists()
        }
        manifest["available_stages"] = sorted(available)
        manifest["stage_paths"] = {k: str(v) for k, v in available.items()}
        return manifest

    def cleanup_older(self, keep_versions: int = 3) -> None:
        versions = sorted({
            int(p.name.split("_")[0][1:])
            for p in self.stage_root.glob("v*_*")
        })
        for version in versions[:-keep_versions]:
            for path in self.stage_root.glob(f"v{version:06d}_*"):
                shutil.rmtree(path, ignore_errors=True)
