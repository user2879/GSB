"""Atomic whole-directory snapshots and resume/staleness decisions."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import STAGES, StageCache
from .errors import ErrorCode
from .util import atomic_write_json, canonical_json, fsync_directory, jsonable

SNAPSHOT_FORMAT_VERSION = 1


@dataclass
class SnapshotLoadResult:
    manifest: dict[str, Any] | None
    data: dict[str, Any] | None
    cache: StageCache | None
    status: str
    message: str = ""
    business_code: str | None = None


class SnapshotManager:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.snapshot_dir = self.state_dir / "snapshot"
        self._recover_directory_swap()

    def _recover_directory_swap(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        backups = sorted(path for path in self.state_dir.glob("snapshot.backup-*") if path.is_dir())
        temps = sorted(path for path in self.state_dir.glob("snapshot.tmp-*") if path.is_dir())
        if not self.snapshot_dir.exists() and backups:
            latest = max(backups, key=lambda path: path.stat().st_mtime_ns)
            os.replace(latest, self.snapshot_dir)
        for temp in temps:
            shutil.rmtree(temp, ignore_errors=True)
        # Retain one most recent backup for forensic recovery, remove older generations.
        remaining = sorted(
            (path for path in self.state_dir.glob("snapshot.backup-*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
        )
        for old in remaining[:-1]:
            shutil.rmtree(old, ignore_errors=True)

    def load(self) -> SnapshotLoadResult:
        manifest_path = self.snapshot_dir / "manifest.json"
        data_path = self.snapshot_dir / "state.json"
        if not manifest_path.exists():
            return SnapshotLoadResult(None, None, None, "empty")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            data = json.loads(data_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return SnapshotLoadResult(
                None,
                None,
                None,
                "corrupt",
                f"snapshot manifest/state cannot be read: {exc}",
                ErrorCode.SNAPSHOT_STALE.value,
            )
        if manifest.get("format_version") != SNAPSHOT_FORMAT_VERSION:
            return SnapshotLoadResult(
                manifest,
                data,
                None,
                "stale",
                "snapshot format version mismatch",
                ErrorCode.SNAPSHOT_STALE.value,
            )
        stage_payload: dict[str, Any] = {}
        for stage in STAGES:
            stage_path = self.snapshot_dir / "stages" / f"{stage}.json"
            try:
                stage_payload[stage] = json.loads(stage_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return SnapshotLoadResult(
                    manifest,
                    data,
                    None,
                    "corrupt",
                    f"snapshot stage {stage} cannot be read: {exc}",
                    ErrorCode.SNAPSHOT_STALE.value,
                )
        cache = StageCache.from_stage_payload(stage_payload)
        return SnapshotLoadResult(manifest, data, cache, "ok")

    def save(
        self,
        *,
        data: dict[str, Any],
        cache: StageCache,
        manifest_updates: dict[str, Any],
    ) -> dict[str, Any]:
        token = uuid.uuid4().hex
        temp_dir = self.state_dir / f"snapshot.tmp-{token}"
        backup_dir = self.state_dir / f"snapshot.backup-{token}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            previous_manifest: dict[str, Any] = {}
            previous_path = self.snapshot_dir / "manifest.json"
            if previous_path.exists():
                try:
                    previous_manifest = json.loads(previous_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    previous_manifest = {}
            previous_versions = previous_manifest.get("stage_versions", {})
            global_version = int(previous_manifest.get("global_version", 0)) + 1
            manifest: dict[str, Any] = {
                "format_version": SNAPSHOT_FORMAT_VERSION,
                "global_version": global_version,
                "stage_versions": cache.stage_versions(previous_versions),
            }
            manifest.update(jsonable(manifest_updates))
            stages_dir = temp_dir / "stages"
            stages_dir.mkdir(parents=True, exist_ok=True)
            for stage in STAGES:
                _write_json_fsync(stages_dir / f"{stage}.json", cache.to_stage_payload()[stage])
            _write_json_fsync(temp_dir / "state.json", data)
            # Manifest is written last inside the new generation.
            _write_json_fsync(temp_dir / "manifest.json", manifest)
            fsync_directory(temp_dir)
            had_current = self.snapshot_dir.exists()
            if had_current:
                os.replace(self.snapshot_dir, backup_dir)
            os.replace(temp_dir, self.snapshot_dir)
            fsync_directory(self.state_dir)
            if had_current:
                # Keep the immediately previous generation; older backups are pruned next startup.
                for old in self.state_dir.glob("snapshot.backup-*"):
                    if old != backup_dir and old.is_dir():
                        shutil.rmtree(old, ignore_errors=True)
            return manifest
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            # If a crash happened after moving the previous generation, restore it.
            if not self.snapshot_dir.exists() and backup_dir.exists():
                os.replace(backup_dir, self.snapshot_dir)
            raise


def _write_json_fsync(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(jsonable(value)))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def decide_resume(
    load_result: SnapshotLoadResult,
    *,
    expected_config_hash: str,
    base_input_fingerprint: str,
    command_stream_fingerprint: str,
    expected_global_version: int | None = None,
) -> tuple[bool, str | None, str]:
    if load_result.status == "empty":
        return False, None, "no snapshot"
    if load_result.status != "ok" or load_result.manifest is None:
        return False, load_result.business_code or ErrorCode.SNAPSHOT_STALE.value, load_result.message or "snapshot is unusable"
    manifest = load_result.manifest
    if expected_global_version is not None and int(manifest.get("global_version", -1)) != int(expected_global_version):
        return (
            False,
            ErrorCode.SNAPSHOT_STALE.value,
            f"expected global version {expected_global_version}, found {manifest.get('global_version')}",
        )
    if manifest.get("config_hash") != expected_config_hash:
        return False, ErrorCode.SNAPSHOT_STALE.value, "configuration hash mismatch"
    if manifest.get("base_input_fingerprint") != base_input_fingerprint:
        return False, ErrorCode.SNAPSHOT_STALE.value, "base input fingerprint mismatch"
    if manifest.get("command_stream_fingerprint") != command_stream_fingerprint:
        return False, ErrorCode.SNAPSHOT_STALE.value, "command stream fingerprint mismatch"
    return True, None, "snapshot version and configuration hash match"

