from __future__ import annotations

import copy
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .errors import SnapshotError
from .fingerprint import canonical_json, sha256_value


SNAPSHOT_FORMAT_VERSION = 1
STAGE_ORDER = ["load", "binning", "woe", "monotonic", "evaluation"]


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


class SnapshotStore:
    def __init__(self, root: str | Path, algorithm_version: str) -> None:
        self.root = Path(root)
        self.algorithm_version = algorithm_version
        self.stages_root = self.root / "stages"
        self.temp_root = self.root / "tmp"
        self.manifest_path = self.root / "manifest.json"
        self.progress_path = self.root / "progress.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.stages_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def blank_manifest(self, config_fingerprint: str, data_fingerprint: str) -> dict[str, Any]:
        return {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "algorithm_version": self.algorithm_version,
            "global_version": 0,
            "config_fingerprint": config_fingerprint,
            "data_fingerprint": data_fingerprint,
            "last_completed_stage": None,
            "completed_columns": {stage: [] for stage in STAGE_ORDER},
            "stage_versions": {},
            "failures": [],
            "revision_of": None,
        }

    def load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"cannot read committed snapshot manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise SnapshotError("committed snapshot manifest is not a mapping")
        if manifest.get("format_version") != SNAPSHOT_FORMAT_VERSION:
            raise SnapshotError(
                f"snapshot format mismatch: expected {SNAPSHOT_FORMAT_VERSION}, got {manifest.get('format_version')!r}"
            )
        if manifest.get("algorithm_version") != self.algorithm_version:
            raise SnapshotError(
                "snapshot algorithm version mismatch: "
                f"expected {self.algorithm_version}, got {manifest.get('algorithm_version')!r}"
            )
        return manifest

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_json(self.manifest_path, manifest)

    def begin_revision(
        self,
        previous: dict[str, Any],
        config_fingerprint: str,
        data_fingerprint: str,
    ) -> dict[str, Any]:
        manifest = self.blank_manifest(config_fingerprint, data_fingerprint)
        manifest["global_version"] = int(previous.get("global_version", 0)) + 1
        manifest["revision_of"] = int(previous.get("global_version", 0))
        self.write_manifest(manifest)
        self.clear_progress()
        return manifest

    def _stage_directory(self, stage: str, version: int) -> tuple[str, Path]:
        relative = Path("stages") / stage / f"v{version:08d}"
        return relative.as_posix(), self.root / relative

    def commit_stage(
        self,
        base_manifest: dict[str, Any],
        stage: str,
        payload: dict[str, Any],
        completed_columns: list[str],
        keys: dict[str, str],
        failures: list[dict[str, Any]],
        upstream_versions: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        if stage not in STAGE_ORDER:
            raise ValueError(f"unknown stage: {stage}")
        version = int(base_manifest.get("global_version", 0)) + 1
        relative, target = self._stage_directory(stage, version)
        temporary = self.temp_root / f"stage-{stage}-{uuid.uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)

        artifact_document = {
            "stage": stage,
            "version": version,
            "payload": payload,
            "completed_columns": sorted(completed_columns),
            "keys": {name: keys[name] for name in sorted(keys)},
            "failures": failures,
            "upstream_versions": upstream_versions or {},
        }
        artifact_version = sha256_value(artifact_document)
        stage_document = {
            **artifact_document,
            "artifact_version": artifact_version,
            "config_fingerprint": base_manifest["config_fingerprint"],
            "data_fingerprint": base_manifest["data_fingerprint"],
        }

        try:
            stage_path = temporary / "stage.json"
            with stage_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(stage_document, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(temporary)
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, target)
            _fsync_directory(target.parent)
        except OSError as exc:
            raise SnapshotError(f"cannot commit stage {stage}: {exc}") from exc
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

        manifest = copy.deepcopy(base_manifest)
        stage_index = STAGE_ORDER.index(stage)
        for later_stage in STAGE_ORDER[stage_index:]:
            manifest["stage_versions"].pop(later_stage, None)
            manifest["completed_columns"][later_stage] = []
        manifest["global_version"] = version
        manifest["last_completed_stage"] = stage
        manifest["completed_columns"][stage] = sorted(completed_columns)
        manifest["stage_versions"][stage] = {
            "version": version,
            "artifact_version": artifact_version,
            "directory": relative,
            "completed_columns": sorted(completed_columns),
            "key_count": len(keys),
        }
        manifest["failures"] = failures
        self.write_manifest(manifest)
        self.clear_progress()
        return manifest

    def load_stage(self, manifest: dict[str, Any], stage: str) -> dict[str, Any] | None:
        stage_meta = manifest.get("stage_versions", {}).get(stage)
        if not stage_meta:
            return None
        path = self.root / str(stage_meta["directory"]) / "stage.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"cannot read committed stage {stage}: {exc}") from exc
        artifact_version = document.pop("artifact_version", None)
        document.pop("config_fingerprint", None)
        document.pop("data_fingerprint", None)
        if artifact_version != sha256_value(document):
            raise SnapshotError(f"stage {stage} failed artifact checksum verification")
        if document.get("stage") != stage:
            raise SnapshotError(f"stage pointer mismatch for {stage}")
        return document

    def write_progress(
        self,
        *,
        stage: str,
        config_fingerprint: str,
        data_fingerprint: str,
        base_global_version: int,
        completed_columns: list[str],
        keys: dict[str, str],
        failures: list[dict[str, Any]],
    ) -> None:
        progress = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "algorithm_version": self.algorithm_version,
            "stage": stage,
            "config_fingerprint": config_fingerprint,
            "data_fingerprint": data_fingerprint,
            "base_global_version": int(base_global_version),
            "completed_columns": sorted(completed_columns),
            "keys": {name: keys[name] for name in sorted(keys)},
            "failures": failures,
        }
        atomic_write_json(self.progress_path, progress)

    def read_progress(self) -> dict[str, Any] | None:
        if not self.progress_path.exists():
            return None
        try:
            with self.progress_path.open("r", encoding="utf-8") as handle:
                progress = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(progress, dict):
            return None
        if progress.get("format_version") != SNAPSHOT_FORMAT_VERSION:
            return None
        if progress.get("algorithm_version") != self.algorithm_version:
            return None
        return progress

    def clear_progress(self) -> None:
        if self.progress_path.exists():
            self.progress_path.unlink(missing_ok=True)
            _fsync_directory(self.root)


def canonical_manifest_digest(manifest: dict[str, Any]) -> str:
    return sha256_value(manifest)
