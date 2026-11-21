"""Atomic stage snapshots with content-addressed artifact reuse."""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from .util import canonical_json


STAGES = ("visual", "text", "fusion")


class SnapshotStore:
    """A run-scoped staging directory that replaces the previous snapshot atomically.

    Artifacts are keyed by SHA-256 values. Old artifacts are loaded into a small
    index, then reused by content key. A reused artifact is copied forward into
    the new staging tree so the next run remains incremental.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.parent = self.root.parent
        self.parent.mkdir(parents=True, exist_ok=True)
        token = f"{os.getpid()}-{uuid.uuid4().hex}"
        self.staging_root = self.parent / f".{self.root.name}.staging-{token}"
        self.backup_root = self.parent / f".{self.root.name}.previous-{token}"
        self.old_manifest: dict[str, Any] | None = None
        self._old_artifacts: dict[str, dict[str, Any]] = {stage: {} for stage in STAGES}
        self._old_paths: dict[str, dict[str, Path]] = {stage: {} for stage in STAGES}
        self._new_keys: dict[str, set[str]] = {stage: set() for stage in STAGES}
        self._lock = threading.RLock()
        self._write_counter = 0
        self._closed = False

        self._load_old_snapshot()
        for stage in STAGES:
            (self.staging_root / stage).mkdir(parents=True, exist_ok=True)

    def _load_old_snapshot(self) -> None:
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_file():
            try:
                self.old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.old_manifest = None
        for stage in STAGES:
            stage_dir = self.root / stage
            if not stage_dir.is_dir():
                continue
            for artifact_path in sorted(stage_dir.glob("*.json")):
                key = artifact_path.stem
                try:
                    value = json.loads(artifact_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                self._old_artifacts[stage][key] = value
                self._old_paths[stage][key] = artifact_path

    @property
    def old_artifact_counts(self) -> dict[str, int]:
        return {stage: len(self._old_artifacts[stage]) for stage in STAGES}

    def get(self, stage: str, key: str) -> dict[str, Any] | None:
        if stage not in STAGES:
            raise KeyError(stage)
        with self._lock:
            if key in self._new_keys[stage]:
                path = self.staging_root / stage / f"{key}.json"
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
            value = self._old_artifacts[stage].get(key)
            if value is None:
                return None
            self._copy_old_artifact(stage, key)
            return value

    def put(self, stage: str, key: str, value: dict[str, Any]) -> None:
        if stage not in STAGES:
            raise KeyError(stage)
        with self._lock:
            self._write_json(self.staging_root / stage / f"{key}.json", value)
            self._new_keys[stage].add(key)

    def contains_new(self, stage: str, key: str) -> bool:
        with self._lock:
            return key in self._new_keys[stage]

    def _copy_old_artifact(self, stage: str, key: str) -> None:
        source = self._old_paths[stage].get(key)
        if source is None:
            if key in self._old_artifacts[stage]:
                self._write_json(self.staging_root / stage / f"{key}.json", self._old_artifacts[stage][key])
                self._new_keys[stage].add(key)
            return
        destination = self.staging_root / stage / f"{key}.json"
        if not destination.exists():
            self._write_bytes(destination, source.read_bytes())
        self._new_keys[stage].add(key)

    def flush(self) -> None:
        """Flush current staging files to disk at a configured checkpoint."""

        with self._lock:
            for path in self.staging_root.glob("*/*.json"):
                try:
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
                except OSError:
                    pass
            try:
                descriptor = os.open(self.staging_root, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                pass

    def artifact_counts(self) -> dict[str, int]:
        with self._lock:
            return {stage: len(self._new_keys[stage]) for stage in STAGES}

    def commit(self, manifest: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("snapshot store is closed")
            self._write_json(self.staging_root / "manifest.json", manifest)
            had_previous = self.root.exists()
            if had_previous:
                if self.backup_root.exists():
                    shutil.rmtree(self.backup_root)
                os.replace(self.root, self.backup_root)
            try:
                os.replace(self.staging_root, self.root)
            except BaseException:
                if had_previous and self.backup_root.exists() and not self.root.exists():
                    os.replace(self.backup_root, self.root)
                raise
            if had_previous and self.backup_root.exists():
                shutil.rmtree(self.backup_root)
            self._closed = True

    def abort(self) -> None:
        with self._lock:
            if not self._closed and self.staging_root.exists():
                shutil.rmtree(self.staging_root, ignore_errors=True)
            self._closed = True

    def _write_json(self, path: Path, value: Any) -> None:
        payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        self._write_bytes(path, payload)

    def _write_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_counter += 1
        temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}-{self._write_counter}")
        with temp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)


def restore_decision(
    store: SnapshotStore,
    *,
    current_global_version: str,
    config_hash: str,
    tier: int,
    reused: dict[str, int],
) -> dict[str, Any]:
    old = store.old_manifest
    if old is None:
        status = "fresh"
        previous_global_version = None
        config_changed = None
        quantization_changed = None
    else:
        previous_global_version = old.get("global_version")
        if previous_global_version == current_global_version and old.get("config_hash") == config_hash:
            status = "resumed_same_version"
        else:
            status = "changed_version_rebuild"
        config_changed = old.get("config_hash") != config_hash
        quantization_changed = old.get("quantization_tier") != tier
    return {
        "status": status,
        "previous_global_version": previous_global_version,
        "current_global_version": current_global_version,
        "config_changed": config_changed,
        "quantization_changed": quantization_changed,
        "old_artifact_counts": store.old_artifact_counts,
        "reused_artifacts": dict(sorted(reused.items())),
    }


def manifest_payload(
    *,
    global_version: str,
    config_hash: str,
    tier: int,
    fractional_bits: int,
    word_bits: int,
    stage_versions: dict[str, str],
    artifact_counts: dict[str, int],
    snapshot_interval: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "complete": True,
        "global_version": global_version,
        "config_hash": config_hash,
        "quantization_tier": tier,
        "fractional_bits": fractional_bits,
        "word_bits": word_bits,
        "stage_versions": stage_versions,
        "artifact_counts": artifact_counts,
        "snapshot_interval": snapshot_interval,
        "canonical_manifest_hash": canonical_json(stage_versions),
    }
