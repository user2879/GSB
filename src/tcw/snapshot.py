"""Versioned, atomically committed stage snapshots."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import (
    E_CONFIG_MISMATCH,
    E_SNAPSHOT_CORRUPT,
    E_SNAPSHOT_MISSING,
    BusinessError,
)
from .util import atomic_write_json, sha256_value


STAGES = ("preprocess", "clustering", "evaluation")


@dataclass(frozen=True)
class StageCommit:
    stage: str
    version: str
    payload_hash: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "payload_hash": self.payload_hash,
            "path": self.path,
        }


class SnapshotStore:
    def __init__(
        self,
        root: Path,
        snapshot_version: int,
        config_hash: str,
        injection_hash: str,
        data_hash: str,
    ) -> None:
        self.root = Path(root).resolve()
        self.snapshot_version = int(snapshot_version)
        self.config_hash = config_hash
        self.injection_hash = injection_hash
        self.data_hash = data_hash
        self.manifest_path = self.root / "manifest.json"

    def initialize(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        self._cleanup_temps()
        manifest = self._new_manifest()
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def _new_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "snapshot_version": self.snapshot_version,
            "config_hash": self.config_hash,
            "injection_hash": self.injection_hash,
            "data_hash": self.data_hash,
            "stages": {},
            "checkpoints": [],
            "commit_counter": 0,
        }

    def _cleanup_temps(self) -> None:
        if not self.root.exists():
            return
        for directory in self.root.rglob(".tmp-*"):
            if directory.is_dir():
                shutil.rmtree(directory, ignore_errors=True)
            else:
                try:
                    directory.unlink()
                except OSError:
                    pass

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            raise BusinessError(
                E_SNAPSHOT_MISSING,
                f"snapshot manifest does not exist: {self.manifest_path}",
                stage="snapshot",
                path=str(self.manifest_path),
                fatal=True,
            )
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise BusinessError(
                E_SNAPSHOT_CORRUPT,
                f"cannot read snapshot manifest: {exc}",
                stage="snapshot",
                path=str(self.manifest_path),
                fatal=True,
            ) from exc
        if not isinstance(manifest, dict):
            raise BusinessError(
                E_SNAPSHOT_CORRUPT,
                "snapshot manifest root must be an object",
                stage="snapshot",
                path=str(self.manifest_path),
                fatal=True,
            )
        return manifest

    def verify_manifest(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        mismatches: list[str] = []
        if manifest.get("snapshot_version") != self.snapshot_version:
            mismatches.append(
                f"snapshot_version: expected {self.snapshot_version}, found {manifest.get('snapshot_version')}"
            )
        if manifest.get("config_hash") != self.config_hash:
            mismatches.append("config_hash differs")
        if manifest.get("injection_hash") != self.injection_hash:
            mismatches.append("injection_hash differs")
        if manifest.get("data_hash") != self.data_hash:
            mismatches.append("data_hash differs")
        if mismatches:
            raise BusinessError(
                E_CONFIG_MISMATCH,
                "; ".join(mismatches),
                stage="snapshot",
                path=str(self.manifest_path),
                fatal=True,
                extra={"mismatches": mismatches},
            )
        for stage in STAGES:
            entry = manifest.get("stages", {}).get(stage)
            if entry is not None:
                self._load_verified_stage(entry)
        return manifest

    def _load_verified_stage(self, entry: dict[str, Any]) -> dict[str, Any]:
        relative = entry.get("path")
        if not isinstance(relative, str):
            raise BusinessError(
                E_SNAPSHOT_CORRUPT,
                "stage manifest entry has no path",
                stage="snapshot",
                path=str(self.manifest_path),
                fatal=True,
            )
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise BusinessError(
                E_SNAPSHOT_CORRUPT,
                "stage snapshot path escapes snapshot root",
                stage="snapshot",
                path=relative,
                fatal=True,
            ) from exc
        artifact_path = candidate / "artifact.json"
        try:
            with artifact_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise BusinessError(
                E_SNAPSHOT_CORRUPT,
                f"cannot read stage snapshot: {exc}",
                stage="snapshot",
                path=str(artifact_path),
                fatal=True,
            ) from exc
        payload_hash = sha256_value(payload)
        if payload_hash != entry.get("payload_hash"):
            raise BusinessError(
                E_SNAPSHOT_CORRUPT,
                "stage snapshot payload hash does not match manifest",
                stage="snapshot",
                path=str(artifact_path),
                fatal=True,
            )
        return payload

    def restore_stage(self, stage: str) -> dict[str, Any] | None:
        manifest = self.load_manifest()
        entry = manifest.get("stages", {}).get(stage)
        if entry is None:
            return None
        return self._load_verified_stage(entry)

    def _write_unit(self, path: Path, payload: dict[str, Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            artifact = path / "artifact.json"
            try:
                with artifact.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise BusinessError(
                    E_SNAPSHOT_CORRUPT,
                    f"existing snapshot is unreadable: {exc}",
                    stage="snapshot",
                    path=str(path),
                    fatal=True,
                ) from exc
            payload_hash = sha256_value(payload)
            if sha256_value(existing) != payload_hash:
                raise BusinessError(
                    E_SNAPSHOT_CORRUPT,
                    "snapshot version exists with different content",
                    stage="snapshot",
                    path=str(path),
                    fatal=True,
                )
            return payload_hash
        temp_path = Path(tempfile.mkdtemp(prefix=".tmp-", dir=path.parent))
        try:
            atomic_write_json(temp_path / "artifact.json", payload)
            payload_hash = sha256_value(payload)
            (temp_path / "artifact.sha256").write_text(payload_hash + "\n", encoding="utf-8")
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
        return payload_hash

    def commit_stage(self, stage: str, payload: dict[str, Any]) -> StageCommit:
        if stage not in STAGES:
            raise ValueError(f"unknown snapshot stage: {stage}")
        payload_hash = sha256_value(payload)
        version = f"{stage}-{payload_hash[:16]}"
        relative = Path("units") / stage / version
        path = self.root / relative
        self._write_unit(path, payload)
        manifest = self.load_manifest()
        manifest["commit_counter"] = int(manifest.get("commit_counter", 0)) + 1
        manifest["stages"][stage] = {
            "version": version,
            "payload_hash": payload_hash,
            "path": str(relative),
            "commit_index": manifest["commit_counter"],
        }
        atomic_write_json(self.manifest_path, manifest)
        return StageCommit(stage=stage, version=version, payload_hash=payload_hash, path=str(relative))

    def commit_checkpoint(
        self,
        stage: str,
        sequence: int,
        payload: dict[str, Any],
    ) -> StageCommit:
        if stage not in STAGES:
            raise ValueError(f"unknown snapshot stage: {stage}")
        payload_hash = sha256_value(payload)
        version = f"{stage}-checkpoint-{sequence:04d}-{payload_hash[:12]}"
        relative = Path("checkpoints") / stage / version
        path = self.root / relative
        self._write_unit(path, payload)
        manifest = self.load_manifest()
        entries = [
            entry
            for entry in manifest.get("checkpoints", [])
            if not (entry.get("stage") == stage and entry.get("sequence") == sequence)
        ]
        entries.append(
            {
                "stage": stage,
                "sequence": int(sequence),
                "version": version,
                "payload_hash": payload_hash,
                "path": str(relative),
            }
        )
        entries.sort(key=lambda entry: (entry["stage"], entry["sequence"], entry["version"]))
        manifest["checkpoints"] = entries
        atomic_write_json(self.manifest_path, manifest)
        return StageCommit(stage=stage, version=version, payload_hash=payload_hash, path=str(relative))

    def manifest_summary(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        return {
            "snapshot_version": manifest.get("snapshot_version"),
            "config_hash": manifest.get("config_hash"),
            "injection_hash": manifest.get("injection_hash"),
            "data_hash": manifest.get("data_hash"),
            "stage_versions": {
                stage: entry.get("version")
                for stage, entry in sorted(manifest.get("stages", {}).items())
            },
            "checkpoint_versions": [
                entry.get("version") for entry in manifest.get("checkpoints", [])
            ],
        }
