from __future__ import annotations

import os
import shutil
import signal
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SnapshotRecoveryError
from .io_utils import atomic_write_json, read_json, sha256_file
from .pipeline import PipelineSession


SNAPSHOT_SCHEMA_VERSION = 1


@dataclass
class RecoveredSnapshot:
    generation: str
    manifest: dict[str, Any]
    category_grid: np.ndarray
    instance_masks: dict[str, np.ndarray]


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.generations_dir = self.root / "generations"
        self.current_link = self.root / "current"
        self.generations_dir.mkdir(parents=True, exist_ok=True)

    def _current_generation(self) -> Path | None:
        if not self.current_link.exists():
            return None
        try:
            resolved = self.current_link.resolve(strict=True)
        except OSError:
            return None
        if resolved.parent != self.generations_dir.resolve():
            raise SnapshotRecoveryError("current snapshot points outside generations")
        return resolved

    def _previous_manifest(self) -> dict[str, Any] | None:
        generation = self._current_generation()
        if generation is None:
            return None
        manifest_path = generation / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = read_json(manifest_path)
        except Exception:
            return None
        return manifest if isinstance(manifest, dict) else None

    @staticmethod
    def _safe_instance_name(key: str) -> str:
        return key.replace(":", "_").replace("/", "_")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _version_changes(
        self,
        previous: dict[str, Any] | None,
        *,
        instance_versions: dict[str, str],
        fusion_version: str,
        global_version: int,
    ) -> dict[str, Any]:
        previous_versions = (
            {} if previous is None else previous.get("instance_result_versions", {})
        )
        previous_keys = set(previous_versions)
        current_keys = set(instance_versions)
        changed = sorted(
            key
            for key in previous_keys.intersection(current_keys)
            if previous_versions[key] != instance_versions[key]
        )
        changes = {
            "previous_global_version": (
                None if previous is None else previous.get("global_version")
            ),
            "global_version": global_version,
            "global_version_incremented": (
                previous is None
                or global_version == int(previous.get("global_version", -1)) + 1
            ),
            "added_instances": sorted(current_keys - previous_keys),
            "removed_instances": sorted(previous_keys - current_keys),
            "changed_instances": changed,
            "unchanged_instance_count": len(
                current_keys.intersection(previous_keys) - set(changed)
            ),
            "fusion_changed": (
                previous is None
                or previous.get("fusion_result_version") != fusion_version
            ),
        }
        if not changes["global_version_incremented"]:
            raise SnapshotRecoveryError("global snapshot version did not increment")
        return changes

    def _validate_generation(self, generation: Path, manifest: dict[str, Any]) -> None:
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotRecoveryError("unsupported snapshot manifest schema")
        if manifest.get("committed") is not True:
            raise SnapshotRecoveryError("snapshot manifest is not committed")
        instance_versions = manifest.get("instance_result_versions")
        files = manifest.get("files")
        if not isinstance(instance_versions, dict) or not isinstance(files, dict):
            raise SnapshotRecoveryError("snapshot manifest is incomplete")
        for relative, expected_hash in files.items():
            path = generation / relative
            if not path.is_file():
                raise SnapshotRecoveryError(f"snapshot file is missing: {relative}")
            if sha256_file(path) != expected_hash:
                raise SnapshotRecoveryError(f"snapshot file hash mismatch: {relative}")
        for key, expected_version in instance_versions.items():
            relative = f"instances/{self._safe_instance_name(key)}.npz"
            if relative not in files:
                raise SnapshotRecoveryError(
                    f"instance result is absent from manifest files: {key}"
                )
            with np.load(generation / relative, allow_pickle=False) as payload:
                actual_version = str(payload["result_version"].item())
                stored_key = str(payload["instance_key"].item())
            if stored_key != key or actual_version != expected_version:
                raise SnapshotRecoveryError(
                    f"instance result version mismatch for {key}"
                )
        fusion_relative = "fusion/category_grid.npz"
        if fusion_relative not in files:
            raise SnapshotRecoveryError("fusion result is absent from manifest files")
        with np.load(generation / fusion_relative, allow_pickle=False) as payload:
            fusion_version = str(payload["result_version"].item())
        if fusion_version != manifest.get("fusion_result_version"):
            raise SnapshotRecoveryError("fusion result version mismatch")

    def commit(
        self,
        session: PipelineSession,
        *,
        report_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if session.fusion_result is None:
            raise SnapshotRecoveryError("cannot snapshot a session without fusion")
        previous = self._previous_manifest()
        global_version = (
            1 if previous is None else int(previous.get("global_version", 0)) + 1
        )
        temporary = Path(
            tempfile.mkdtemp(prefix=".tmp-generation-", dir=self.root)
        )
        try:
            instances_dir = temporary / "instances"
            fusion_dir = temporary / "fusion"
            instances_dir.mkdir()
            fusion_dir.mkdir()
            file_hashes: dict[str, str] = {}
            instance_versions: dict[str, str] = {}
            for item in session.sorted_instances:
                relative = f"instances/{self._safe_instance_name(item.key)}.npz"
                destination = temporary / relative
                np.savez_compressed(
                    destination,
                    mask=np.ascontiguousarray(item.mask, dtype=np.uint8),
                    result_version=np.array(item.cache_key),
                    instance_key=np.array(item.key),
                    class_id=np.array(item.class_id, dtype=np.int32),
                )
                file_hashes[relative] = sha256_file(destination)
                instance_versions[item.key] = item.cache_key
            fusion_relative = "fusion/category_grid.npz"
            fusion_destination = temporary / fusion_relative
            np.savez_compressed(
                fusion_destination,
                category_grid=np.ascontiguousarray(
                    session.fusion_result.category_grid,
                    dtype=np.int32,
                ),
                result_version=np.array(
                    session.fusion_result.fusion_result_version
                ),
            )
            file_hashes[fusion_relative] = sha256_file(fusion_destination)
            if report_payload is not None:
                report_relative = "state_summary.json"
                atomic_write_json(temporary / report_relative, report_payload)
                file_hashes[report_relative] = sha256_file(temporary / report_relative)
            generation_name = f"gen-{global_version:08d}"
            manifest = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "committed": True,
                "generation": generation_name,
                "global_version": global_version,
                "format_version": "+".join(session.formats),
                "formats": list(session.formats),
                "config_hash": session.config.config_hash,
                "instance_result_versions": instance_versions,
                "fusion_result_version": session.fusion_result.fusion_result_version,
                "files": file_hashes,
                "version_changes": self._version_changes(
                    previous,
                    instance_versions=instance_versions,
                    fusion_version=session.fusion_result.fusion_result_version,
                    global_version=global_version,
                ),
            }
            atomic_write_json(temporary / "manifest.json", manifest)
            self._fsync_directory(temporary)
            self._validate_generation(temporary, manifest)
            destination = self.generations_dir / generation_name
            if destination.exists():
                current = self._current_generation()
                if current is not None and current == destination:
                    raise SnapshotRecoveryError(
                        f"snapshot generation is already published: {destination}"
                    )
                shutil.rmtree(destination)
            os.replace(temporary, destination)
            self._fsync_directory(self.generations_dir)
            temporary_link = self.root / f".current-{uuid.uuid4().hex}"
            os.symlink(
                Path("generations") / generation_name,
                temporary_link,
                target_is_directory=True,
            )
            os.replace(temporary_link, self.current_link)
            self._fsync_directory(self.root)
            return manifest
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    def recover(self) -> RecoveredSnapshot:
        generation = self._current_generation()
        if generation is None:
            raise SnapshotRecoveryError("no current snapshot generation exists")
        manifest_path = generation / "manifest.json"
        if not manifest_path.is_file():
            raise SnapshotRecoveryError("current snapshot has no manifest")
        try:
            manifest = read_json(manifest_path)
        except Exception as exc:
            raise SnapshotRecoveryError(f"cannot read snapshot manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise SnapshotRecoveryError("snapshot manifest root must be an object")
        if manifest.get("generation") != generation.name:
            raise SnapshotRecoveryError("manifest generation does not match current link")
        self._validate_generation(generation, manifest)
        with np.load(
            generation / "fusion/category_grid.npz",
            allow_pickle=False,
        ) as payload:
            category_grid = payload["category_grid"].copy()
        instance_masks: dict[str, np.ndarray] = {}
        for key, version in manifest["instance_result_versions"].items():
            relative = f"instances/{self._safe_instance_name(key)}.npz"
            with np.load(generation / relative, allow_pickle=False) as payload:
                stored_version = str(payload["result_version"].item())
                if stored_version != version:
                    raise SnapshotRecoveryError(
                        f"recovery version mismatch for instance {key}"
                    )
                instance_masks[key] = payload["mask"].astype(bool, copy=True)
        return RecoveredSnapshot(
            generation=generation.name,
            manifest=manifest,
            category_grid=category_grid,
            instance_masks=instance_masks,
        )


class GracefulShutdown:
    def __init__(self) -> None:
        self.event = threading.Event()
        self._previous_handlers: dict[int, Any] = {}

    def request(self, signum: int | None = None, frame: Any = None) -> None:
        self.event.set()

    def install(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self.request)

    def restore(self) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    @property
    def requested(self) -> bool:
        return self.event.is_set()
