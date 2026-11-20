"""Stage snapshot commits, stale recovery checks, and final manifest writing."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from .util import (
    clean_temporary_directories,
    read_json,
    replace_directory,
    write_json,
    write_json_atomic,
)


class SnapshotStore:
    def __init__(self, state_dir: str | Path, commit_interval: int):
        self.state_dir = Path(state_dir)
        self.snapshot_root = self.state_dir / "snapshots"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.commit_interval = max(1, int(commit_interval))
        clean_temporary_directories(self.snapshot_root)

    @property
    def manifest_path(self) -> Path:
        return self.state_dir / "manifest.json"

    def recovery_status(self, global_version: str, config_hash_value: str) -> dict[str, Any]:
        if not self.manifest_path.exists():
            stage_manifests: list[dict[str, Any]] = []
            for path in sorted(self.snapshot_root.glob("*/manifest.json")):
                try:
                    stage_manifests.append(read_json(path))
                except (OSError, ValueError):
                    continue
            if stage_manifests:
                reasons = [
                    reason
                    for manifest in stage_manifests
                    for reason in (
                        ["global_version changed"]
                        if manifest.get("global_version") != global_version
                        else []
                    )
                    + (
                        ["config_hash changed"]
                        if manifest.get("config_hash") != config_hash_value
                        else []
                    )
                ]
                reasons = list(dict.fromkeys(reasons))
                completed: list[str] = []
                seen: set[str] = set()
                for manifest in stage_manifests:
                    for packet_id in manifest.get("completed_packet_ids", []):
                        if packet_id not in seen:
                            seen.add(packet_id)
                            completed.append(packet_id)
                return {
                    "status": "stale" if reasons else "partial_resume",
                    "stale": bool(reasons),
                    "error_code": "E_SNAPSHOT_STALE" if reasons else None,
                    "reasons": reasons
                    if reasons
                    else ["final manifest missing; stage snapshots are reusable"],
                    "previous_manifest": {
                        "global_version": stage_manifests[-1].get("global_version"),
                        "config_hash": stage_manifests[-1].get("config_hash"),
                        "calibration_fingerprint": stage_manifests[-1].get(
                            "calibration_fingerprint"
                        ),
                        "completed_packet_ids": completed,
                    },
                }
            return {
                "status": "empty",
                "stale": False,
                "error_code": None,
                "reasons": [],
                "previous_manifest": None,
            }
        try:
            previous = read_json(self.manifest_path)
        except (OSError, ValueError):
            return {
                "status": "stale",
                "stale": True,
                "error_code": "E_SNAPSHOT_STALE",
                "reasons": ["snapshot manifest is unreadable"],
                "previous_manifest": None,
            }
        reasons: list[str] = []
        if previous.get("global_version") != global_version:
            reasons.append("global_version changed")
        if previous.get("config_hash") != config_hash_value:
            reasons.append("config_hash changed")
        return {
            "status": "stale" if reasons else "resumed",
            "stale": bool(reasons),
            "error_code": "E_SNAPSHOT_STALE" if reasons else None,
            "reasons": reasons,
            "previous_manifest": {
                "global_version": previous.get("global_version"),
                "config_hash": previous.get("config_hash"),
                "calibration_fingerprint": previous.get("calibration_fingerprint"),
                "completed_packet_ids": previous.get("completed_packet_ids", []),
            },
        }

    def commit_stage(
        self,
        stage: str,
        records: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Write chunked records to a temporary directory, then replace the stage dir."""

        temporary = self.snapshot_root / f".{stage}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        chunk_files: list[str] = []
        try:
            for chunk_index, offset in enumerate(range(0, len(records), self.commit_interval)):
                chunk_name = f"chunk-{chunk_index:05d}.json"
                write_json(temporary / chunk_name, records[offset : offset + self.commit_interval])
                chunk_files.append(chunk_name)
            manifest = {
                "schema_version": "aed-stage-snapshot/1",
                "stage": stage,
                **metadata,
                "record_count": len(records),
                "cache_keys": [
                    record.get("cache_key")
                    for record in records
                    if isinstance(record, dict) and record.get("cache_key")
                ],
                "chunk_interval": self.commit_interval,
                "chunk_files": chunk_files,
            }
            write_json(temporary / "manifest.json", manifest)
            target = self.snapshot_root / stage
            replace_directory(temporary, target)
            return manifest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    def write_final_manifest(self, manifest: dict[str, Any]) -> None:
        write_json_atomic(
            self.manifest_path,
            {"schema_version": "aed-run-manifest/1", **manifest},
        )
