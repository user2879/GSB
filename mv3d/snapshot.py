"""Atomic stage snapshots, workspace locking, and crash recovery."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .constants import ENGINE_VERSION, SNAPSHOT_KEEP
from .errors import WorkspaceLockedError
from .fixed import bytes_hash, canonical_json
from .io import write_json_atomic


@dataclass
class Snapshot:
    generation: int
    path: Path
    manifest: dict[str, Any]
    stages: dict[str, Any]

    @property
    def engine_version(self) -> str:
        return str(self.manifest.get("engine_version", ""))

    @property
    def config_hash(self) -> str:
        return str(self.manifest.get("config_hash", ""))

    @property
    def completed_stages(self) -> list[str]:
        return list(self.manifest.get("completed_stages", []))

    @property
    def run_complete(self) -> bool:
        return bool(self.manifest.get("run_complete", False))


class WorkspaceLock:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock_path = root / ".mv3d" / "run.lock"
        self.handle: Any = None

    def __enter__(self) -> "WorkspaceLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.lock_path.open("a+", encoding="ascii")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            self.handle = None
            raise WorkspaceLockedError(str(self.lock_path)) from error
        self.handle.seek(0)
        self.handle.write(f"{os.getpid()}\n")
        self.handle.flush()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
                self.handle = None


class SnapshotStore:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.base = workspace_root / ".mv3d"
        self.snapshots = self.base / "snapshots"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.latest_loaded: Snapshot | None = None

    def _read_snapshot(self, path: Path) -> Snapshot | None:
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if manifest.get("engine_version") != ENGINE_VERSION:
            return None
        stages: dict[str, Any] = {}
        stage_files = manifest.get("stage_files", {})
        stage_hashes = manifest.get("stage_hashes", {})
        for stage, relative in sorted(stage_files.items()):
            stage_path = path / relative
            if not stage_path.is_file():
                return None
            data = stage_path.read_bytes()
            if bytes_hash(data) != stage_hashes.get(stage):
                return None
            try:
                stages[stage] = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                return None
        return Snapshot(
            generation=int(manifest.get("generation", 0)),
            path=path,
            manifest=manifest,
            stages=stages,
        )

    def load_latest(self, engine_version: str = ENGINE_VERSION) -> Snapshot | None:
        candidates = sorted(
            (
                path
                for path in self.snapshots.iterdir()
                if path.is_dir() and not path.name.startswith(".") and path.name.isdigit()
            ),
            reverse=True,
        )
        for path in candidates:
            snapshot = self._read_snapshot(path)
            if snapshot is not None and snapshot.engine_version == engine_version:
                self.latest_loaded = snapshot
                return snapshot
        self.latest_loaded = None
        return None

    def _next_generation(self) -> int:
        values = [
            int(path.name)
            for path in self.snapshots.iterdir()
            if path.is_dir() and path.name.isdigit()
        ]
        return max(values, default=0) + 1

    def _write_stage(
        self,
        staging: Path,
        stage: str,
        payload: Any,
        previous: Snapshot | None,
    ) -> tuple[str, str, bool]:
        relative = f"stages/{stage}.json"
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data_hash = bytes_hash(canonical_json(payload).encode("utf-8"))
        linked = False
        if previous is not None:
            previous_hash = previous.manifest.get("stage_hashes", {}).get(stage)
            previous_file = previous.manifest.get("stage_files", {}).get(stage)
            if previous_hash == data_hash and previous_file is not None:
                source = previous.path / previous_file
                if source.is_file():
                    try:
                        os.link(source, destination)
                        linked = True
                    except OSError:
                        linked = False
        if not linked:
            data = canonical_json(payload).encode("utf-8")
            with destination.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        return relative, data_hash, linked

    def commit(
        self,
        stages: dict[str, Any],
        pending_queue: list[dict[str, Any]],
        config_hash: str,
        project_fingerprint: str,
        recipe_version: str,
        completed_stages: list[str],
        run_complete: bool,
        previous: Snapshot | None = None,
    ) -> Snapshot:
        previous = previous or self.latest_loaded
        generation = self._next_generation()
        staging = Path(
            tempfile.mkdtemp(prefix=f".staging-{os.getpid()}-{generation}-", dir=self.snapshots)
        )
        try:
            stage_files: dict[str, str] = {}
            stage_hashes: dict[str, str] = {}
            for stage, payload in sorted(stages.items()):
                relative, data_hash, _ = self._write_stage(
                    staging, stage, payload, previous
                )
                stage_files[stage] = relative
                stage_hashes[stage] = data_hash
            pending_relative = "pending.json"
            pending_data = canonical_json(pending_queue).encode("utf-8")
            with (staging / pending_relative).open("wb") as handle:
                handle.write(pending_data)
                handle.flush()
                os.fsync(handle.fileno())
            manifest = {
                "schema_version": 1,
                "engine_version": ENGINE_VERSION,
                "recipe_version": recipe_version,
                "generation": generation,
                "config_hash": config_hash,
                "project_fingerprint": project_fingerprint,
                "completed_stages": sorted(completed_stages, key=_stage_sort),
                "run_complete": run_complete,
                "stage_files": stage_files,
                "stage_hashes": stage_hashes,
                "pending_file": pending_relative,
                "pending_hash": bytes_hash(pending_data),
            }
            manifest_data = canonical_json(manifest).encode("utf-8")
            with (staging / "manifest.json").open("wb") as handle:
                handle.write(manifest_data)
                handle.flush()
                os.fsync(handle.fileno())
            directory = os.open(staging, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            destination = self.snapshots / f"{generation:012d}"
            os.rename(staging, destination)
            directory = os.open(self.snapshots, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            write_json_atomic(self.base / "current.json", {"generation": generation})
            snapshot = Snapshot(
                generation=generation,
                path=destination,
                manifest=manifest,
                stages=stages,
            )
            self.latest_loaded = snapshot
            self._prune()
            return snapshot
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _prune(self) -> None:
        candidates = sorted(
            (
                path
                for path in self.snapshots.iterdir()
                if path.is_dir() and not path.name.startswith(".") and path.name.isdigit()
            ),
            reverse=True,
        )
        for path in candidates[SNAPSHOT_KEEP:]:
            shutil.rmtree(path, ignore_errors=True)


def _stage_sort(stage: str) -> int:
    order = ("load", "features", "matches", "triangulation", "depth", "voxels")
    return order.index(stage) if stage in order else len(order)


def read_pending(snapshot: Snapshot) -> list[dict[str, Any]]:
    relative = snapshot.manifest.get("pending_file", "pending.json")
    path = snapshot.path / relative
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))

