from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from .canonical import (
    atomic_write_json,
    read_json,
    sha256_file,
    sha256_object,
)
from .compatibility import ExistingAdapter
from .errors import E_SNAPSHOT_INVALID, BusinessError


SNAPSHOT_FORMAT = "peft-lora-snapshot-v1"
STAGE_NAMES = ("load", "training", "merge", "export")


class SnapshotStore:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / "snapshot"
        self.manifests_directory = self.root / "manifests"
        self.stages_directory = self.root / "stages"
        self.temp_directory = self.root / ".tmp"
        self.current_path = self.root / "CURRENT.json"
        self.manifests_directory.mkdir(parents=True, exist_ok=True)
        self.stages_directory.mkdir(parents=True, exist_ok=True)
        self.temp_directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.current: dict[str, Any] | None
        if self.current_path.is_file():
            try:
                self.current = read_json(self.current_path)
                self._validate_manifest(self.current)
            except (OSError, ValueError, TypeError) as exc:
                raise BusinessError(
                    E_SNAPSHOT_INVALID,
                    f"snapshot CURRENT.json is invalid: {exc}",
                ) from exc
        else:
            self.current = None
        self.started_from_version = int(self.current.get("global_version", 0)) if self.current else 0

    @staticmethod
    def _validate_manifest(manifest: Any) -> None:
        if not isinstance(manifest, dict) or manifest.get("format") != SNAPSHOT_FORMAT:
            raise ValueError("unsupported snapshot manifest")
        version = manifest.get("global_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("invalid global version")
        if not isinstance(manifest.get("stages"), dict):
            raise ValueError("invalid stage map")

    def _all_versions(self) -> list[int]:
        versions: list[int] = []
        for path in self.manifests_directory.glob("v*.json"):
            try:
                versions.append(int(path.stem[1:]))
            except ValueError:
                continue
        for stage in STAGE_NAMES:
            for path in (self.stages_directory / stage).glob("v*"):
                try:
                    versions.append(int(path.name[1:]))
                except ValueError:
                    continue
        if self.current is not None:
            versions.append(int(self.current["global_version"]))
        return versions

    def next_version(self) -> int:
        versions = self._all_versions()
        return (max(versions) if versions else 0) + 1

    def peek_next_version(self) -> int:
        with self._lock:
            return self.next_version()

    def stage_path(self, stage: str, version: int) -> Path:
        if stage not in STAGE_NAMES:
            raise ValueError(f"unknown snapshot stage: {stage}")
        return self.stages_directory / stage / f"v{version:08d}"

    @staticmethod
    def _stage_reference(stage: str, version: int) -> str:
        return f"stages/{stage}/v{version:08d}"

    def _resolve_stage_reference(self, reference: str) -> Path:
        base = (self.root / reference).resolve()
        try:
            base.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError("snapshot stage path escapes the snapshot root") from exc
        return base

    def _verify_stage(self, stage: str, reference: Mapping[str, Any]) -> Path:
        path = self._resolve_stage_reference(str(reference.get("path")))
        if not path.is_dir():
            raise ValueError(f"snapshot stage directory is missing: {path}")
        marker_path = path / "_stage.json"
        marker = read_json(marker_path)
        if (
            not isinstance(marker, dict)
            or marker.get("stage") != stage
            or marker.get("version") != reference.get("version")
            or marker.get("complete") is not True
        ):
            raise ValueError(f"snapshot stage marker is invalid: {path}")
        expected_marker_hash = reference.get("marker_sha256")
        if isinstance(expected_marker_hash, str) and sha256_object(marker) != expected_marker_hash:
            raise ValueError(f"snapshot stage marker checksum mismatch: {path}")
        for item in marker.get("files", []):
            relative = item.get("path")
            if not isinstance(relative, str):
                raise ValueError("invalid stage file entry")
            file_path = (path / relative).resolve()
            try:
                file_path.relative_to(path.resolve())
            except ValueError as exc:
                raise ValueError("stage file path escapes the stage") from exc
            if not file_path.is_file() or sha256_file(file_path) != item.get("sha256"):
                raise ValueError(f"stage file checksum mismatch: {file_path}")
        return path

    def current_stage_path(self, stage: str) -> Path | None:
        if self.current is None:
            return None
        reference = self.current.get("stages", {}).get(stage)
        if not isinstance(reference, dict):
            return None
        try:
            return self._verify_stage(stage, reference)
        except (OSError, ValueError, TypeError):
            return None

    def identity_matches(self, identity: Mapping[str, Any]) -> bool:
        if self.current is None:
            return False
        return all(self.current.get(key) == value for key, value in identity.items())

    def commit_stage(
        self,
        stage: str,
        identity: Mapping[str, Any],
        populate: Callable[[Path, int], dict[str, Any] | None],
        *,
        compatibility_hash: str | None = None,
        snapshot_commit_interval: int | None = None,
        replace_stages: bool = False,
        stage_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stage not in STAGE_NAMES:
            raise ValueError(f"unknown snapshot stage: {stage}")
        with self._lock:
            version = self.next_version()
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f"stage-{stage}-{uuid.uuid4().hex[:8]}-",
                    dir=self.temp_directory,
                )
            )
            try:
                callback_result = populate(temporary, version)
                files: list[dict[str, Any]] = []
                for file_path in sorted(temporary.rglob("*")):
                    if not file_path.is_file():
                        continue
                    relative = file_path.relative_to(temporary).as_posix()
                    files.append(
                        {
                            "path": relative,
                            "size": file_path.stat().st_size,
                            "sha256": sha256_file(file_path),
                        }
                    )
                marker = {
                    "format": "peft-lora-stage-v1",
                    "stage": stage,
                    "version": version,
                    "complete": True,
                    "files": files,
                    "metadata": dict(stage_metadata or {}),
                }
                atomic_write_json(temporary / "_stage.json", marker)
                destination = self.stage_path(stage, version)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise BusinessError(
                        E_SNAPSHOT_INVALID,
                        f"snapshot stage destination already exists: {destination}",
                    )
                os.replace(temporary, destination)
                reference = {
                    "version": version,
                    "path": self._stage_reference(stage, version),
                    "marker_sha256": sha256_object(marker),
                    "metadata": dict(stage_metadata or {}),
                }
                stages: dict[str, Any] = {}
                if self.current is not None and not replace_stages:
                    stages.update(self.current.get("stages", {}))
                stages[stage] = reference
                manifest = {
                    "format": SNAPSHOT_FORMAT,
                    "global_version": version,
                    "stages": stages,
                    "snapshot_commit_interval": snapshot_commit_interval,
                    **dict(identity),
                }
                if compatibility_hash is not None:
                    manifest["compatibility_hash"] = compatibility_hash
                # CURRENT is the commit point. Historical manifests are written only
                # after the commit so an interrupted pre-commit manifest can never be
                # discovered as if it represented a complete snapshot.
                atomic_write_json(self.current_path, manifest)
                self.current = manifest
                historical = self.manifests_directory / f"v{version:08d}.json"
                try:
                    atomic_write_json(historical, manifest)
                except OSError:
                    pass
                if callback_result is not None:
                    return {**manifest, "_callback_result": callback_result}
                return manifest
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

    def read_stage_json(self, stage: str, relative_file: str) -> Any:
        path = self.current_stage_path(stage)
        if path is None:
            raise FileNotFoundError(f"stage {stage} is not current or complete")
        return read_json(path / relative_file)

    def _manifest_history(self) -> list[dict[str, Any]]:
        manifests: list[dict[str, Any]] = []
        seen: set[int] = set()
        current_version = int(self.current["global_version"]) if self.current else -1
        for path in sorted(self.manifests_directory.glob("v*.json"), reverse=True):
            try:
                manifest = read_json(path)
                self._validate_manifest(manifest)
                version = int(manifest["global_version"])
            except (OSError, ValueError, TypeError):
                continue
            if version in seen:
                continue
            seen.add(version)
            manifests.append(manifest)
        if self.current is not None and current_version not in seen:
            manifests.append(self.current)
        manifests.sort(key=lambda item: int(item["global_version"]), reverse=True)
        return manifests

    def discover_adapters(self) -> dict[str, ExistingAdapter]:
        discovered: dict[str, ExistingAdapter] = {}
        for manifest in self._manifest_history():
            training_reference = manifest.get("stages", {}).get("training")
            if not isinstance(training_reference, dict):
                continue
            stage = manifest.get("stages", {}).get("training")
            try:
                stage_path = self._verify_stage("training", stage)
                index = read_json(stage_path / "index.json")
            except (OSError, ValueError, TypeError):
                continue
            if not isinstance(index, dict) or index.get("complete") is not True:
                continue
            for entry in index.get("adapters", []):
                if not isinstance(entry, dict):
                    continue
                task_name = entry.get("task_name")
                relative = entry.get("file")
                if not isinstance(task_name, str) or not isinstance(relative, str):
                    continue
                if task_name in discovered:
                    continue
                adapter_path = (stage_path / relative).resolve()
                try:
                    adapter_path.relative_to(stage_path.resolve())
                    payload = read_json(adapter_path)
                except (OSError, ValueError, TypeError):
                    continue
                discovered[task_name] = ExistingAdapter(
                    payload=payload,
                    artifact_file=str(adapter_path.relative_to(self.workspace).as_posix()),
                    snapshot_version=int(manifest["global_version"]),
                )
        return discovered

    def current_training_adapters(self, identity: Mapping[str, Any]) -> dict[str, Any]:
        if not self.identity_matches(identity):
            return {"complete": False, "adapters": []}
        try:
            index = self.read_stage_json("training", "index.json")
        except (OSError, TypeError, ValueError):
            return {"complete": False, "adapters": []}
        return index if isinstance(index, dict) else {"complete": False, "adapters": []}

    def recovery_info(self, identity: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "started_from_version": self.started_from_version,
            "resumed": self.started_from_version > 0,
            "identity_matches": self.identity_matches(identity),
            "current_version": int(self.current["global_version"]) if self.current else 0,
        }
