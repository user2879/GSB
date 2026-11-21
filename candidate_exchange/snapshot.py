from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from .errors import E_SNAPSHOT_INVALID, BusinessError
from .util import (
    atomic_write_json,
    file_sha256,
    read_json,
    remove_tree,
    stable_hash,
)


STAGE_ORDER = ("detection", "matching", "serialization")
MANIFEST_VERSION = 1


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.stage_payloads: dict[str, dict[str, Any]] = {}
        self.latest_manifest: dict[str, Any] | None = None
        self.recovery: dict[str, Any] = {
            "recovered": False,
            "recovery_point_version": None,
            "available_stages": [],
        }

    def _valid_manifest(self, directory: Path) -> dict[str, Any] | None:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = read_json(manifest_path)
        except Exception:
            return None
        if not isinstance(manifest, dict) or manifest.get("manifest_version") != MANIFEST_VERSION:
            return None
        stages = manifest.get("stages")
        if not isinstance(stages, dict):
            return None
        for stage, metadata in stages.items():
            if not isinstance(metadata, dict):
                return None
            relative = metadata.get("path")
            expected_hash = metadata.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                return None
            stage_path = directory / relative
            if not stage_path.exists() or file_sha256(stage_path) != expected_hash:
                return None
        return manifest

    def scan(self) -> list[tuple[Path, dict[str, Any]]]:
        valid: list[tuple[Path, dict[str, Any]]] = []
        for directory in self.versions_dir.iterdir():
            if not directory.is_dir() or directory.name.startswith(".tmp-"):
                continue
            manifest = self._valid_manifest(directory)
            if manifest is not None:
                valid.append((directory, manifest))
        valid.sort(key=lambda item: int(item[1].get("global_version", 0)))
        return valid

    def recover(self) -> dict[str, Any]:
        candidates = self.scan()
        if not candidates:
            self.recovery = {
                "recovered": False,
                "recovery_point_version": None,
                "available_stages": [],
                "ignored_temporary_directories": self._temporary_directories(),
            }
            return self.recovery
        directory, manifest = candidates[-1]
        self.latest_manifest = manifest
        self.stage_payloads = {}
        for stage, metadata in manifest.get("stages", {}).items():
            self.stage_payloads[stage] = read_json(directory / metadata["path"])
        self.recovery = {
            "recovered": True,
            "recovery_point_version": manifest.get("global_version"),
            "version_id": directory.name,
            "available_stages": [
                stage for stage in STAGE_ORDER if stage in self.stage_payloads
            ],
            "config_hash": manifest.get("config_hash"),
            "format_fingerprints": manifest.get("format_fingerprints"),
            "sidecar_metadata_hash": manifest.get("sidecar_metadata_hash"),
            "ignored_temporary_directories": self._temporary_directories(),
        }
        return self.recovery

    def _temporary_directories(self) -> list[str]:
        return sorted(
            directory.name
            for directory in self.versions_dir.iterdir()
            if directory.is_dir() and directory.name.startswith(".tmp-")
        )

    def drop_stages_from(self, stage_name: str) -> None:
        if stage_name not in STAGE_ORDER:
            return
        index = STAGE_ORDER.index(stage_name)
        for stage in STAGE_ORDER[index:]:
            self.stage_payloads.pop(stage, None)

    def commit(
        self,
        *,
        stage_name: str,
        payload: dict[str, Any],
        config_hash: str,
        format_fingerprints: dict[str, str],
        sidecar_metadata_hash: str,
        input_fingerprint: str,
        snapshot_commit_interval: int,
        partial: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stage_name not in STAGE_ORDER:
            raise BusinessError(
                E_SNAPSHOT_INVALID, f"unknown snapshot stage: {stage_name}", "snapshot"
            )
        self.drop_stages_from(stage_name)
        self.stage_payloads[stage_name] = payload
        previous_version = (
            int(self.latest_manifest.get("global_version", 0))
            if self.latest_manifest is not None
            else 0
        )
        version = previous_version + 1
        version_id = f"{version:012d}"
        temporary = self.versions_dir / f".tmp-{version_id}-{uuid.uuid4().hex}"
        final = self.versions_dir / version_id
        try:
            temporary.mkdir(parents=True, exist_ok=False)
            stage_manifest: dict[str, Any] = {}
            for stage in STAGE_ORDER:
                if stage not in self.stage_payloads:
                    continue
                relative = f"stages/{stage}.json"
                stage_path = temporary / relative
                stage_path.parent.mkdir(parents=True, exist_ok=True)
                stage_value = self.stage_payloads[stage]
                text = json.dumps(
                    stage_value,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ) + "\n"
                stage_path.write_text(text, encoding="utf-8")
                stage_manifest[stage] = {
                    "stage_version": version,
                    "path": relative,
                    "sha256": file_sha256(stage_path),
                    "partial": bool(partial and stage == stage_name),
                    "bytes": stage_path.stat().st_size,
                }
            manifest = {
                "manifest_version": MANIFEST_VERSION,
                "global_version": version,
                "version_id": version_id,
                "parent_version": previous_version or None,
                "config_hash": config_hash,
                "format_fingerprints": dict(sorted(format_fingerprints.items())),
                "sidecar_metadata_hash": sidecar_metadata_hash,
                "input_fingerprint": input_fingerprint,
                "snapshot_commit_interval": snapshot_commit_interval,
                "stages": stage_manifest,
                "complete_stages": [
                    stage for stage in STAGE_ORDER if stage in stage_manifest
                ],
                "partial": partial,
                "created_unix_ms": int(time.time() * 1000),
                "extra": extra or {},
            }
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            self._fsync_tree_files(temporary)
            os.replace(temporary, final)
            self._fsync_directory(self.versions_dir)
            atomic_write_json(
                self.root / "CURRENT.json",
                {
                    "global_version": version,
                    "version_id": version_id,
                    "manifest_sha256": stable_hash(manifest),
                },
            )
            self.latest_manifest = manifest
            return manifest
        except Exception as exc:
            remove_tree(temporary)
            if isinstance(exc, BusinessError):
                raise
            raise BusinessError(
                E_SNAPSHOT_INVALID,
                f"snapshot commit failed: {exc}",
                "snapshot",
                details={"stage": stage_name, "version": version},
            ) from exc

    def _fsync_tree_files(self, root: Path) -> None:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        self._fsync_directory(root)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def latest_stage(self, stage_name: str) -> dict[str, Any] | None:
        return self.stage_payloads.get(stage_name)
