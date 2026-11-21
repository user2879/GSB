"""Versioned, atomically committed stage snapshots."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import FusionConfig
from .io_utils import (
    fsync_directory,
    json_bytes,
    sha256_file,
    sha256_json,
    write_bytes_fsynced,
    write_json,
)


STAGES = ("load", "normalize", "fuse", "truncate")
VERSION_PATTERN = re.compile(r"^v(\d{6,})-[0-9a-f]{8}$")


class SimulatedCrash(RuntimeError):
    """Raised by tests to model termination before the pointer swap."""


@dataclass
class SnapshotResult:
    exists: bool = False
    valid: bool = False
    stale: bool = False
    reason: str | None = None
    manifest_version: int | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)
    ignored_temp_snapshots: int = 0


class SnapshotStore:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.state_root = workspace / "state"
        self.snapshots_root = self.state_root / "snapshots"
        self.current_path = self.state_root / "CURRENT"

    def reset(self) -> None:
        shutil.rmtree(self.state_root, ignore_errors=True)

    def _temp_snapshot_paths(self) -> list[Path]:
        if not self.snapshots_root.is_dir():
            return []
        return sorted(
            [path for path in self.snapshots_root.iterdir() if path.is_dir() and path.name.startswith(".tmp-")],
            key=lambda item: item.name,
        )

    def next_version_hint(self) -> int:
        maximum = 0
        if self.snapshots_root.is_dir():
            for child in self.snapshots_root.iterdir():
                match = VERSION_PATTERN.match(child.name)
                if match:
                    maximum = max(maximum, int(match.group(1)))
        return maximum + 1

    def load(self, expected_version: int | None = None) -> SnapshotResult:
        temp_paths = self._temp_snapshot_paths()
        result = SnapshotResult(ignored_temp_snapshots=len(temp_paths))
        if not self.current_path.exists():
            return result
        result.exists = True
        try:
            current_name = self.current_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            result.stale = True
            result.reason = f"cannot read CURRENT pointer: {exc}"
            return result
        if not VERSION_PATTERN.match(current_name):
            result.stale = True
            result.reason = f"invalid CURRENT pointer content: {current_name!r}"
            return result

        snapshot_dir = self.snapshots_root / current_name
        manifest_path = snapshot_dir / "manifest.json"
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            result.stale = True
            result.reason = f"cannot read committed manifest: {exc}"
            return result
        if not isinstance(manifest, dict):
            result.stale = True
            result.reason = "committed manifest is not an object"
            return result
        manifest_version = manifest.get("run_version")
        if isinstance(manifest_version, int) and not isinstance(manifest_version, bool):
            result.manifest_version = manifest_version
        result.manifest = manifest
        if manifest.get("schema_version") != 1:
            result.stale = True
            result.reason = f"unsupported snapshot schema: {manifest.get('schema_version')!r}"
            return result
        if not isinstance(manifest_version, int) or isinstance(manifest_version, bool) or manifest_version < 1:
            result.stale = True
            result.reason = "manifest run_version is invalid"
            return result
        if expected_version is not None and manifest_version != expected_version:
            result.stale = True
            result.reason = f"expected run_version {expected_version}, committed version is {manifest_version}"
            return result

        config_snapshot_path = snapshot_dir / "config_snapshot.json"
        try:
            import json

            config_snapshot = json.loads(config_snapshot_path.read_text(encoding="utf-8"))
        except Exception as exc:
            result.stale = True
            result.reason = f"cannot read committed config snapshot: {exc}"
            return result
        if sha256_json(config_snapshot) != manifest.get("config_hash"):
            result.stale = True
            result.reason = "config snapshot hash does not match manifest config_hash"
            return result

        stage_manifest = manifest.get("stages")
        if not isinstance(stage_manifest, dict) or set(stage_manifest) != set(STAGES):
            result.stale = True
            result.reason = "manifest must describe exactly load/normalize/fuse/truncate stages"
            return result
        loaded_stages: dict[str, Any] = {}
        for stage in STAGES:
            entry = stage_manifest.get(stage)
            if not isinstance(entry, dict):
                result.stale = True
                result.reason = f"missing stage manifest for {stage}"
                return result
            if entry.get("version") != manifest_version:
                result.stale = True
                result.reason = f"stage {stage} version differs from run_version"
                return result
            file_name = entry.get("file")
            if not isinstance(file_name, str) or "/" in file_name or "\\" in file_name:
                result.stale = True
                result.reason = f"invalid stage file name for {stage}"
                return result
            stage_path = snapshot_dir / file_name
            if not stage_path.is_file():
                result.stale = True
                result.reason = f"missing stage file for {stage}: {file_name}"
                return result
            if sha256_file(stage_path) != entry.get("sha256"):
                result.stale = True
                result.reason = f"stage file hash mismatch for {stage}"
                return result
            try:
                import json

                loaded_stages[stage] = json.loads(stage_path.read_text(encoding="utf-8"))
            except Exception as exc:
                result.stale = True
                result.reason = f"cannot parse stage {stage}: {exc}"
                return result
        result.stages = loaded_stages
        result.valid = True
        return result

    def _stage_entry(self, file_name: str, run_version: int, path: Path) -> dict[str, Any]:
        return {
            "file": file_name,
            "sha256": sha256_file(path),
            "version": run_version,
        }

    def commit(
        self,
        *,
        run_version: int,
        config: FusionConfig,
        stage_payloads: dict[str, Any],
        completed_channels: list[str],
        simulate_crash_after: str | None = None,
    ) -> dict[str, Any]:
        if set(stage_payloads) != set(STAGES):
            raise ValueError("stage_payloads must contain load, normalize, fuse and truncate")
        if simulate_crash_after is not None and simulate_crash_after not in STAGES:
            raise ValueError(f"simulate_crash_after must be one of {STAGES}")
        self.snapshots_root.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:8]
        snapshot_dir = self.snapshots_root / f".tmp-v{run_version:06d}-{token}"
        snapshot_dir.mkdir(parents=False, exist_ok=False)

        config_path = snapshot_dir / "config_snapshot.json"
        write_json(config_path, config.raw)
        stage_files: dict[str, tuple[str, Path]] = {}
        for stage in STAGES:
            file_name = f"{stage}.json"
            stage_path = snapshot_dir / file_name
            write_json(stage_path, stage_payloads[stage])
            stage_files[stage] = (file_name, stage_path)
            if simulate_crash_after == stage:
                write_json(
                    snapshot_dir / "CRASH_MARKER.json",
                    {"crash_after": stage, "run_version": run_version},
                )
                fsync_directory(snapshot_dir)
                raise SimulatedCrash(f"simulated forced termination after stage {stage}")

        load_stage = stage_payloads["load"]
        manifest = {
            "completed_channels": sorted(completed_channels),
            "config_hash": config.config_hash(),
            "fragment_set_hashes": {
                channel: load_stage["channels"][channel]["fragment_set_hash"]
                for channel in config.channel_names
            },
            "normalization": config.normalization,
            "quant_bits": config.quant_bits,
            "rounding": config.rounding,
            "run_version": run_version,
            "schema_version": 1,
            "stages": {
                stage: self._stage_entry(file_name, run_version, path)
                for stage, (file_name, path) in stage_files.items()
            },
            "struct_hash": config.structural_hash(),
            "trunc_k": config.trunc_k,
            "weights": config.weights(),
        }
        write_json(snapshot_dir / "manifest.json", manifest)
        fsync_directory(snapshot_dir)

        final_dir = self.snapshots_root / f"v{run_version:06d}-{token}"
        if final_dir.exists():
            raise FileExistsError(f"snapshot target already exists: {final_dir}")
        os.rename(snapshot_dir, final_dir)
        fsync_directory(self.snapshots_root)

        pointer_temp = self.state_root / f".CURRENT.{uuid.uuid4().hex}.tmp"
        write_bytes_fsynced(pointer_temp, (final_dir.name + "\n").encode("utf-8"))
        os.replace(pointer_temp, self.current_path)
        fsync_directory(self.state_root)

        for temp_path in self._temp_snapshot_paths():
            shutil.rmtree(temp_path, ignore_errors=True)
        return manifest
