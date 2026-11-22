from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .util import atomic_write_json, read_json, sha256_bytes, sha256_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class SnapshotInspection:
    available: bool
    info: dict[str, Any] | None
    manifest: dict[str, Any] | None
    report: dict[str, Any] | None
    stale_codes: list[str]
    details: dict[str, Any]


@dataclass
class CheckpointLoad:
    available: bool
    state: dict[str, Any] | None
    manifest: dict[str, Any] | None
    reason: str


class SnapshotStore:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir.resolve()
        self.snapshots_dir = self.state_dir / "snapshots"
        self.staging_dir = self.state_dir / "staging"
        self.checkpoints_dir = self.state_dir / "checkpoints"
        self.reports_dir = self.state_dir / "reports"
        self.cache_dir = self.state_dir / "cache"
        for path in (
            self.state_dir,
            self.snapshots_dir,
            self.staging_dir,
            self.checkpoints_dir,
            self.reports_dir,
            self.cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.current_path = self.state_dir / "current.json"
        self.checkpoint_current_path = self.checkpoints_dir / "current.json"

    def next_version(self) -> int:
        versions: list[int] = []
        if self.current_path.exists():
            try:
                current = read_json(self.current_path)
                versions.append(int(current.get("global_version", 0)))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        for path in self.snapshots_dir.glob("v*"):
            try:
                versions.append(int(path.name[1:]))
            except ValueError:
                continue
        return max(versions, default=0) + 1

    def inspect_current(self) -> SnapshotInspection:
        stale_codes: list[str] = []
        details: dict[str, Any] = {}
        if not self.current_path.exists():
            return SnapshotInspection(False, None, None, None, [], {"reason": "cold_start"})
        try:
            info = read_json(self.current_path)
        except (OSError, json.JSONDecodeError) as exc:
            return SnapshotInspection(
                False,
                None,
                None,
                None,
                ["E_SNAPSHOT_STALE"],
                {"reason": "current_pointer_unreadable", "error": str(exc)},
            )
        if not isinstance(info, dict):
            return SnapshotInspection(
                False,
                None,
                None,
                None,
                ["E_SNAPSHOT_STALE"],
                {"reason": "current_pointer_not_object"},
            )
        relative = info.get("snapshot_dir")
        if not isinstance(relative, str):
            stale_codes.append("E_SNAPSHOT_STALE")
            return SnapshotInspection(
                False, info, None, None, stale_codes, {"reason": "snapshot_path_missing"}
            )
        snapshot_dir = (self.state_dir / relative).resolve()
        try:
            snapshot_dir.relative_to(self.snapshots_dir.resolve())
        except ValueError:
            stale_codes.append("E_SNAPSHOT_STALE")
            return SnapshotInspection(
                False, info, None, None, stale_codes, {"reason": "snapshot_path_outside_root"}
            )
        manifest_path = snapshot_dir / "manifest.json"
        report_path = snapshot_dir / "report.json"
        if not manifest_path.exists():
            stale_codes.append("E_SNAPSHOT_STALE")
            return SnapshotInspection(
                False,
                info,
                None,
                None,
                stale_codes,
                {"reason": "manifest_missing", "snapshot_dir": relative},
            )
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            stale_codes.append("E_SNAPSHOT_STALE")
            return SnapshotInspection(
                False,
                info,
                None,
                None,
                stale_codes,
                {"reason": "manifest_unreadable", "error": str(exc), "snapshot_dir": relative},
            )
        report: dict[str, Any] | None = None
        if report_path.exists():
            try:
                report_value = read_json(report_path)
                if isinstance(report_value, dict):
                    report = report_value
            except (OSError, json.JSONDecodeError):
                report = None
        if not isinstance(manifest, dict):
            stale_codes.append("E_SNAPSHOT_STALE")
            return SnapshotInspection(
                False, info, None, report, stale_codes, {"reason": "manifest_not_object"}
            )
        if info.get("manifest_sha256") != sha256_json(manifest):
            stale_codes.append("E_SNAPSHOT_STALE")
            details["reason"] = "manifest_hash_mismatch"
        if manifest.get("status") != "complete":
            stale_codes.append("E_SNAPSHOT_STALE")
            details["reason"] = "manifest_not_complete"
        if int(info.get("global_version", -1)) != int(manifest.get("global_version", -2)):
            stale_codes.append("E_SNAPSHOT_STALE")
            details["reason"] = "pointer_version_mismatch"
        directory_version = snapshot_dir.name
        if directory_version != f"v{int(manifest.get('global_version', -1)):08d}":
            stale_codes.append("E_SNAPSHOT_STALE")
            details["reason"] = "directory_version_mismatch"
        details.setdefault("snapshot_dir", relative)
        return SnapshotInspection(True, info, manifest, report, sorted(set(stale_codes)), details)

    def previous_report(self) -> dict[str, Any] | None:
        inspection = self.inspect_current()
        return inspection.report if inspection.available else None

    def allocate_staging(self, run_id: str) -> Path:
        path = self.staging_dir / f"{run_id}-{uuid.uuid4().hex[:12]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _write_file(self, root: Path, relative: str, content: bytes | str) -> str:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8") if isinstance(content, str) else content
        path.write_bytes(data)
        return sha256_bytes(data)

    def write_stage(
        self,
        staging: Path,
        stage_name: str,
        files: dict[str, bytes | str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        temp_stage = staging / f".{stage_name}.{uuid.uuid4().hex}.tmp"
        temp_stage.mkdir(parents=True, exist_ok=False)
        file_hashes: dict[str, str] = {}
        try:
            for relative, content in sorted(files.items()):
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError(f"unsafe stage file path: {relative}")
                file_hashes[relative] = self._write_file(temp_stage, relative, content)
            artifact_version = sha256_json({"files": file_hashes, "metadata": metadata})
            manifest = {
                "schema": 1,
                "stage": stage_name,
                "status": "complete",
                "artifact_version": artifact_version,
                "files": file_hashes,
                "metadata": metadata,
                "committed_at": utc_now(),
            }
            atomic_write_json(temp_stage / "stage-manifest.json", manifest)
            destination = staging / stage_name
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(temp_stage, destination)
            return manifest
        finally:
            if temp_stage.exists():
                shutil.rmtree(temp_stage, ignore_errors=True)

    def commit_snapshot(
        self,
        staging: Path,
        *,
        global_version: int,
        config_hash: str,
        source_fingerprint: str,
        run_id: str,
        completed_shards: list[str],
        consumed_token_budget: int,
        stage_manifests: dict[str, dict[str, Any]],
        report: dict[str, Any],
        extra_manifest: dict[str, Any] | None = None,
    ) -> Path:
        stage_versions = {
            stage: manifest["artifact_version"] for stage, manifest in sorted(stage_manifests.items())
        }
        manifest: dict[str, Any] = {
            "schema": 1,
            "status": "complete",
            "global_version": global_version,
            "run_id": run_id,
            "config_hash": config_hash,
            "source_fingerprint": source_fingerprint,
            "completed_shards": sorted(completed_shards),
            "consumed_token_budget": consumed_token_budget,
            "stage_versions": stage_versions,
            "stages": {stage: "complete" for stage in sorted(stage_manifests)},
            "committed_at": utc_now(),
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        atomic_write_json(staging / "report.json", report)
        atomic_write_json(staging / "manifest.json", manifest)
        destination = self.snapshots_dir / f"v{global_version:08d}"
        if destination.exists():
            raise FileExistsError(f"snapshot version already exists: {destination.name}")
        os.replace(staging, destination)
        pointer = {
            "schema": 1,
            "global_version": global_version,
            "config_hash": config_hash,
            "source_fingerprint": source_fingerprint,
            "snapshot_dir": str(destination.relative_to(self.state_dir)),
            "manifest_sha256": sha256_json(manifest),
            "committed_at": utc_now(),
        }
        atomic_write_json(self.current_path, pointer)
        return destination

    def write_latest_report(self, report: dict[str, Any]) -> Path:
        path = self.reports_dir / "latest.json"
        atomic_write_json(path, report)
        return path

    def commit_checkpoint(
        self,
        *,
        run_id: str,
        global_version: int,
        config_hash: str,
        source_fingerprint: str,
        state: dict[str, Any],
        reason: str,
        stage_loading: dict[str, Any] | None = None,
    ) -> Path:
        generation = uuid.uuid4().hex
        temp_dir = self.checkpoints_dir / f".{generation}.tmp"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            atomic_write_json(temp_dir / "state.json", state)
            if stage_loading is not None:
                atomic_write_json(temp_dir / "stage-loading.json", stage_loading)
            manifest = {
                "schema": 1,
                "status": "partial",
                "run_id": run_id,
                "global_version": global_version,
                "config_hash": config_hash,
                "source_fingerprint": source_fingerprint,
                "reason": reason,
                "state_file": "state.json",
                "committed_at": utc_now(),
            }
            atomic_write_json(temp_dir / "manifest.json", manifest)
            destination = self.checkpoints_dir / f"g-{generation}"
            os.replace(temp_dir, destination)
            pointer = {
                "schema": 1,
                "status": "partial",
                "global_version": global_version,
                "config_hash": config_hash,
                "source_fingerprint": source_fingerprint,
                "checkpoint_dir": str(destination.relative_to(self.state_dir)),
                "manifest_sha256": sha256_json(manifest),
                "committed_at": utc_now(),
            }
            atomic_write_json(self.checkpoint_current_path, pointer)
            return destination
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def load_checkpoint(
        self, config_hash: str, source_fingerprint: str, expected_version: int
    ) -> CheckpointLoad:
        if not self.checkpoint_current_path.exists():
            return CheckpointLoad(False, None, None, "no_checkpoint")
        try:
            pointer = read_json(self.checkpoint_current_path)
        except (OSError, json.JSONDecodeError) as exc:
            return CheckpointLoad(False, None, None, f"checkpoint_pointer_unreadable:{exc}")
        if not isinstance(pointer, dict) or pointer.get("status") != "partial":
            return CheckpointLoad(False, None, None, "checkpoint_completed")
        if pointer.get("global_version") != expected_version:
            return CheckpointLoad(False, None, pointer, "checkpoint_version_mismatch")
        if pointer.get("config_hash") != config_hash:
            return CheckpointLoad(False, None, pointer, "config_hash_mismatch")
        if pointer.get("source_fingerprint") != source_fingerprint:
            return CheckpointLoad(False, None, pointer, "source_fingerprint_mismatch")
        relative = pointer.get("checkpoint_dir")
        if not isinstance(relative, str):
            return CheckpointLoad(False, None, pointer, "checkpoint_path_missing")
        checkpoint_dir = (self.state_dir / relative).resolve()
        try:
            checkpoint_dir.relative_to(self.checkpoints_dir.resolve())
        except ValueError:
            return CheckpointLoad(False, None, pointer, "checkpoint_path_outside_root")
        manifest_path = checkpoint_dir / "manifest.json"
        state_path = checkpoint_dir / "state.json"
        if not manifest_path.exists() or not state_path.exists():
            return CheckpointLoad(False, None, pointer, "checkpoint_incomplete")
        try:
            manifest = read_json(manifest_path)
            state = read_json(state_path)
        except (OSError, json.JSONDecodeError) as exc:
            return CheckpointLoad(False, None, pointer, f"checkpoint_unreadable:{exc}")
        if not isinstance(manifest, dict) or not isinstance(state, dict):
            return CheckpointLoad(False, None, pointer, "checkpoint_shape_invalid")
        if manifest.get("config_hash") != config_hash or manifest.get("source_fingerprint") != source_fingerprint:
            return CheckpointLoad(False, None, manifest, "checkpoint_signature_mismatch")
        if manifest.get("global_version") != expected_version:
            return CheckpointLoad(False, None, manifest, "checkpoint_version_mismatch")
        if pointer.get("manifest_sha256") != sha256_json(manifest):
            return CheckpointLoad(False, None, manifest, "checkpoint_manifest_hash_mismatch")
        return CheckpointLoad(True, state, manifest, "resumed")

    def clear_checkpoint(self, *, run_id: str, config_hash: str, source_fingerprint: str) -> None:
        pointer = {
            "schema": 1,
            "status": "completed",
            "run_id": run_id,
            "config_hash": config_hash,
            "source_fingerprint": source_fingerprint,
            "completed_at": utc_now(),
        }
        atomic_write_json(self.checkpoint_current_path, pointer)

    def list_snapshot_dirs(self) -> list[Path]:
        return sorted(path for path in self.snapshots_dir.glob("v*") if path.is_dir())
