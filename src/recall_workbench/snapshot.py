"""Cross-platform atomic stage snapshots backed by os.replace."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ConfigMismatch, SnapshotInvalid
from .models import AppConfig


SNAPSHOT_SCHEMA_VERSION = 1
STAGE_PRODUCT_VERSIONS = {"recall": 1, "fusion": 1, "evaluation": 1}
STAGES = ("recall", "fusion", "evaluation")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def atomic_write_json(path: Path, value: Any, *, temp_parent: Path | None = None) -> None:
    """Write a complete JSON artifact then atomically replace its destination.

    The temporary file lives in a temporary directory on the same filesystem as
    the destination, which is required for a platform-correct ``os.replace``.
    """

    path = path.resolve()
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    base_parent = (temp_parent or parent).resolve()
    base_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".atomic-", dir=str(base_parent)))
    temp_path = temp_dir / path.name
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def atomic_write_text(path: Path, value: str, *, temp_parent: Path | None = None) -> None:
    path = path.resolve()
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    base_parent = (temp_parent or parent).resolve()
    base_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".atomic-", dir=str(base_parent)))
    temp_path = temp_dir / path.name
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class SnapshotStore:
    def __init__(self, run_dir: Path, config: AppConfig, run_version: int) -> None:
        if not isinstance(run_version, int) or run_version < 1:
            raise ConfigMismatch("run version must be a positive integer", details={"run_version": run_version})
        self.run_dir = run_dir.resolve()
        self.config = config
        self.run_version = run_version
        self.manifest_path = self.run_dir / "manifest.json"
        self.stage_dir = self.run_dir / "stages"
        self.temp_dir = self.run_dir / ".snapshot-tmp"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_temp_dirs()
        self.manifest = self._load_or_initialize_manifest()

    def _cleanup_stale_temp_dirs(self) -> None:
        for path in self.temp_dir.glob(".atomic-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    def _initial_manifest(self) -> dict[str, Any]:
        now = utc_now()
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "global_version": self.run_version,
            "run_version": self.run_version,
            "config_hash": self.config.config_hash,
            "fixed_point": {
                "primary_bits": self.config.primary_bits,
                "reference_bits": self.config.reference_bits,
                "rounding": self.config.rounding,
            },
            "created_at_utc": now,
            "updated_at_utc": now,
            "status": "running",
            "stages": {
                stage: {
                    "product_version": STAGE_PRODUCT_VERSIONS[stage],
                    "artifact_version": 0,
                    "completed_users": [],
                    "cursor": None,
                    "complete": False,
                    "artifact": f"stages/{stage}.json",
                }
                for stage in STAGES
            },
            "failures": {"count": 0, "artifact": "failures.json"},
        }

    def _load_or_initialize_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            manifest = self._initial_manifest()
            atomic_write_json(self.manifest_path, manifest, temp_parent=self.temp_dir)
            return manifest
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SnapshotInvalid(f"cannot read snapshot manifest: {exc}", details={"path": str(self.manifest_path)}) from exc
        if not isinstance(manifest, dict):
            raise SnapshotInvalid("snapshot manifest must be an object")
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotInvalid("unsupported snapshot schema version")
        if manifest.get("run_version") != self.run_version or manifest.get("global_version") != self.run_version:
            raise ConfigMismatch(
                "snapshot run version differs from requested version",
                details={"snapshot_version": manifest.get("run_version"), "requested_version": self.run_version},
            )
        if manifest.get("config_hash") != self.config.config_hash:
            raise ConfigMismatch(
                "snapshot configuration hash differs from current configuration",
                details={"snapshot_hash": manifest.get("config_hash"), "requested_hash": self.config.config_hash},
            )
        stages = manifest.get("stages")
        if not isinstance(stages, dict) or any(stage not in stages for stage in STAGES):
            raise SnapshotInvalid("snapshot manifest is missing a required stage entry")
        for stage in STAGES:
            if stages[stage].get("product_version") != STAGE_PRODUCT_VERSIONS[stage]:
                raise ConfigMismatch(
                    f"snapshot product version differs for stage {stage}",
                    details={
                        "stage": stage,
                        "snapshot_version": stages[stage].get("product_version"),
                        "requested_version": STAGE_PRODUCT_VERSIONS[stage],
                    },
                )
        return manifest

    def _save_manifest(self) -> None:
        self.manifest["updated_at_utc"] = utc_now()
        atomic_write_json(self.manifest_path, self.manifest, temp_parent=self.temp_dir)

    def commit_failures(self, failures: list[dict[str, Any]]) -> None:
        payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "run_version": self.run_version,
            "config_hash": self.config.config_hash,
            "failures": failures,
        }
        atomic_write_json(self.run_dir / "failures.json", payload, temp_parent=self.temp_dir)
        self.manifest["failures"] = {"count": len(failures), "artifact": "failures.json"}
        self._save_manifest()

    def stage_path(self, stage: str) -> Path:
        return self.stage_dir / f"{stage}.json"

    def load_stage(self, stage: str) -> dict[str, Any] | None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        path = self.stage_path(stage)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SnapshotInvalid(f"cannot read {stage} snapshot: {exc}", details={"path": str(path)}) from exc
        if not isinstance(payload, dict):
            raise SnapshotInvalid(f"{stage} snapshot must be an object")
        if payload.get("run_version") != self.run_version or payload.get("config_hash") != self.config.config_hash:
            raise ConfigMismatch(
                f"{stage} snapshot does not match current run version or configuration",
                details={"stage": stage},
            )
        if payload.get("product_version") != STAGE_PRODUCT_VERSIONS[stage]:
            raise ConfigMismatch(f"{stage} snapshot product version differs", details={"stage": stage})
        manifest_stage = self.manifest["stages"][stage]
        artifact_version = int(payload.get("artifact_version", 0))
        if artifact_version < int(manifest_stage.get("artifact_version", 0)):
            raise SnapshotInvalid(f"{stage} artifact is older than its manifest entry")
        started = set(payload.get("completed_users", []))
        manifest_version = int(manifest_stage.get("artifact_version", 0))
        if artifact_version > manifest_version:
            # A crash may occur after the artifact replace but before the manifest
            # replace. The artifact is authoritative and self-consistent.
            manifest_stage["completed_users"] = sorted(started)
            manifest_stage["cursor"] = payload.get("cursor")
            manifest_stage["complete"] = bool(payload.get("complete", False))
            manifest_stage["artifact_version"] = artifact_version
            self._save_manifest()
        return payload

    def commit_stage(self, stage: str, payload: dict[str, Any], completed_users: list[str], complete: bool) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        manifest_stage = self.manifest["stages"][stage]
        artifact_version = int(manifest_stage.get("artifact_version", 0)) + 1
        completed_sorted = sorted(set(completed_users))
        artifact = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "stage": stage,
            "product_version": STAGE_PRODUCT_VERSIONS[stage],
            "artifact_version": artifact_version,
            "run_version": self.run_version,
            "config_hash": self.config.config_hash,
            "primary_bits": self.config.primary_bits,
            "reference_bits": self.config.reference_bits,
            "rounding": self.config.rounding,
            "completed_users": completed_sorted,
            "cursor": completed_sorted[-1] if completed_sorted else None,
            "complete": bool(complete),
            "committed_at_utc": utc_now(),
            "payload": payload,
        }
        atomic_write_json(self.stage_path(stage), artifact, temp_parent=self.temp_dir)
        manifest_stage.update(
            {
                "artifact_version": artifact_version,
                "completed_users": completed_sorted,
                "cursor": artifact["cursor"],
                "complete": bool(complete),
            }
        )
        self._save_manifest()
        return artifact

    def set_status(self, status: str) -> None:
        self.manifest["status"] = status
        self._save_manifest()

    def set_progress(self, stage: str, completed_users: list[str], complete: bool) -> None:
        target = self.manifest["stages"][stage]
        target["completed_users"] = sorted(completed_users)
        target["cursor"] = target["completed_users"][-1] if target["completed_users"] else None
        target["complete"] = complete
        self._save_manifest()

    def load_all_stages(self) -> dict[str, dict[str, Any] | None]:
        return {stage: self.load_stage(stage) for stage in STAGES}
