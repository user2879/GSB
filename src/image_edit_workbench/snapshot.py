from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from .artifact_io import read_artifact, write_artifact
from .errors import E_SNAPSHOT_CORRUPT, WorkbenchError
from .util import atomic_write_json, link_or_copy, read_json, sha256_json


class SnapshotStore:
    """Immutable snapshot directories committed through an atomic CURRENT pointer."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.snapshots_dir = state_dir / "snapshots"
        self.work_dir = state_dir / ".work"
        self.current_version = 0
        self.current_snapshot: Path | None = None
        self.current_manifest: dict[str, Any] = {}
        self.artifact_index: dict[str, dict[str, Any]] = {}
        self.recovery: dict[str, Any] = {"status": "fresh", "snapshot": None}
        self._started_work = False
        self._lock = threading.RLock()

    def load(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        pointer = self.state_dir / "CURRENT.json"
        candidate: Path | None = None
        if pointer.exists():
            try:
                data = read_json(pointer)
                snapshot_name = data.get("snapshot")
                if isinstance(snapshot_name, str):
                    candidate = self.snapshots_dir / snapshot_name
            except (OSError, ValueError, json.JSONDecodeError):
                candidate = None

        manifest: dict[str, Any] | None = None
        if candidate is not None and (candidate / "manifest.json").exists():
            try:
                manifest = read_json(candidate / "manifest.json")
                self.recovery = {"status": "resumed", "snapshot": candidate.name}
            except (OSError, ValueError, json.JSONDecodeError):
                manifest = None

        if manifest is None:
            manifests = sorted(self.snapshots_dir.glob("*/manifest.json"), reverse=True) if self.snapshots_dir.exists() else []
            for path in manifests:
                try:
                    manifest = read_json(path)
                    candidate = path.parent
                    self.recovery = {
                        "status": "recovered",
                        "snapshot": candidate.name,
                        "message": "CURRENT pointer was absent/corrupt; newest valid manifest used",
                    }
                    break
                except (OSError, ValueError, json.JSONDecodeError):
                    continue

        if manifest is None:
            self.current_version = 0
            self.current_snapshot = None
            self.current_manifest = {}
            self.artifact_index = {}
            if pointer.exists():
                self.recovery = {"status": "fresh", "snapshot": None, "message": "no valid snapshot found"}
            return

        if manifest.get("schema_version") != 1:
            raise WorkbenchError(E_SNAPSHOT_CORRUPT, "unsupported snapshot schema", stage="snapshot")
        self.current_version = int(manifest.get("global_version", 0))
        self.current_snapshot = candidate
        self.current_manifest = manifest
        index = manifest.get("artifact_index", {})
        self.artifact_index = index if isinstance(index, dict) else {}

    def begin_work(self) -> None:
        if self._started_work:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        (self.work_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        if self.current_snapshot is not None:
            source_artifacts = self.current_snapshot / "artifacts"
            if source_artifacts.exists():
                for source in source_artifacts.rglob("*"):
                    if source.is_file():
                        target = self.work_dir / "artifacts" / source.relative_to(source_artifacts)
                        link_or_copy(source, target)
        self._started_work = True

    def get_record(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            return self.artifact_index.get(cache_key)

    def artifact_path(self, record: dict[str, Any]) -> Path | None:
        relative = record.get("artifact")
        if not relative:
            return None
        path = self.work_dir / str(relative)
        return path if path.exists() else None

    def load_arrays(self, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        path = self.artifact_path(record)
        if path is None:
            return None
        try:
            return read_artifact(path)
        except (OSError, WorkbenchError):
            return None

    def save_artifact(
        self,
        *,
        cache_key: str,
        stage: str,
        region_key: str,
        arrays: dict[str, Any] | None,
        metadata: dict[str, Any],
        upstream: list[str],
        image_block: list[int] | None,
    ) -> dict[str, Any]:
        with self._lock:
            self.begin_work()
            relative = f"artifacts/{cache_key[:2]}/{cache_key}.bin"
            if arrays is None:
                record = {
                    "cache_key": cache_key,
                    "stage": stage,
                    "region_key": region_key,
                    "artifact": None,
                    "sha256": sha256_json(metadata),
                    "size": 0,
                    "metadata": metadata,
                    "upstream": upstream,
                    "image_block": image_block,
                }
            else:
                path = self.work_dir / relative
                digest, size = write_artifact(path, arrays, metadata)
                record = {
                    "cache_key": cache_key,
                    "stage": stage,
                    "region_key": region_key,
                    "artifact": relative,
                    "sha256": digest,
                    "size": size,
                    "metadata": metadata,
                    "upstream": upstream,
                    "image_block": image_block,
                }
            self.artifact_index[cache_key] = record
            return record

    def commit(self, manifest: dict[str, Any]) -> dict[str, Any]:
        if not self._started_work:
            self.begin_work()
        assert self.work_dir.exists()
        next_version = self.current_version + 1
        snapshot_name = f"snap-{next_version:06d}"
        target = self.snapshots_dir / snapshot_name
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # A repeated run number can only happen after manual state edits. Keep history intact.
            snapshot_name = f"snap-{next_version:06d}-{uuid.uuid4().hex[:8]}"
            target = self.snapshots_dir / snapshot_name
        manifest = dict(manifest)
        manifest["global_version"] = next_version
        manifest["snapshot_id"] = snapshot_name
        atomic_write_json(self.work_dir / "manifest.json", manifest)
        self.work_dir.rename(target)
        pointer_tmp = self.state_dir / "CURRENT.json.tmp"
        atomic_write_json(pointer_tmp, {"snapshot": snapshot_name})
        pointer_tmp.replace(self.state_dir / "CURRENT.json")
        self.current_version = next_version
        self.current_snapshot = target
        self.current_manifest = manifest
        self._started_work = False
        self._cleanup_old_snapshots(keep=5)
        return {"global_version": next_version, "snapshot_id": snapshot_name, "path": str(target)}

    def _cleanup_old_snapshots(self, keep: int) -> None:
        snapshots = sorted(self.snapshots_dir.glob("snap-*"), key=lambda path: path.name)
        for path in snapshots[:-keep]:
            if self.current_snapshot is not None and path.resolve() == self.current_snapshot.resolve():
                continue
            shutil.rmtree(path, ignore_errors=True)

    def discard_work(self) -> None:
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)
        self._started_work = False


def resume_decision(
    previous: dict[str, Any],
    *,
    config_hash: str,
    exchange_version: int,
    exchange_fingerprint: str,
    sidecar_hash: str,
    image_fingerprints: dict[str, str],
) -> dict[str, Any]:
    if not previous:
        return {"status": "cold_start", "reasons": ["no_previous_snapshot"]}
    reasons: list[str] = []
    if previous.get("config_hash") != config_hash:
        reasons.append("config_changed")
    if int(previous.get("exchange_version", 0)) != exchange_version:
        reasons.append("exchange_version_changed")
    if previous.get("exchange_fingerprint") != exchange_fingerprint:
        reasons.append("exchange_fingerprint_changed")
    if previous.get("sidecar_hash", "") != sidecar_hash:
        reasons.append("sidecar_hash_changed")
    if previous.get("image_fingerprints", {}) != image_fingerprints:
        reasons.append("source_image_changed")
    pending = previous.get("pending_queue", [])
    if pending:
        reasons.append("pending_queue_present")
    return {
        "status": "resume_cache_checked" if reasons else "resume_exact",
        "reasons": reasons,
        "previous_global_version": previous.get("global_version"),
        "previous_pending": pending,
    }
