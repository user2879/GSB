"""Atomic, generation-based snapshot storage."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .errors import SnapshotVersionChanged


SNAPSHOT_PATTERN = re.compile(r"^snapshot-(\d{6,})$")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )


@dataclass(frozen=True)
class Snapshot:
    version: int
    path: Path
    manifest: dict[str, Any]
    stages: dict[str, Any]
    trees: dict[int, dict[str, Any]]

    def recovery_info(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "config_hash": self.manifest.get("config_hash"),
            "data_fingerprint": self.manifest.get("data_fingerprint"),
            "completed_trees": list(self.manifest.get("completed_trees", [])),
            "stage_versions": self.manifest.get("stage_versions", {}),
        }


class SnapshotManager:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshots_root = self.root / "snapshots"
        self.current_file = self.root / "CURRENT"
        self.lock_file = self.root / ".lock"

    @contextlib.contextmanager
    def _lock(self, exclusive: bool = True) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _current_name(self) -> str | None:
        try:
            value = self.current_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if not value:
            return None
        return value

    def current_version(self) -> int:
        name = self._current_name()
        match = SNAPSHOT_PATTERN.match(name or "")
        return int(match.group(1)) if match else 0

    def load_current(self) -> Snapshot | None:
        with self._lock(exclusive=False):
            name = self._current_name()
            candidates: list[Path] = []
            if name and SNAPSHOT_PATTERN.match(name):
                candidates.append(self.snapshots_root / name)
            candidates.extend(self._complete_snapshot_directories(exclude={path.name for path in candidates}))
            for candidate in candidates:
                snapshot = self._load_candidate(candidate)
                if snapshot is not None:
                    return snapshot
        return None

    def _complete_snapshot_directories(self, exclude: set[str] | None = None) -> list[Path]:
        if not self.snapshots_root.exists():
            return []
        excluded = exclude or set()
        paths: list[Path] = []
        for path in self.snapshots_root.iterdir():
            if path.is_dir() and SNAPSHOT_PATTERN.match(path.name) and path.name not in excluded:
                paths.append(path)
        return sorted(paths, key=lambda item: int(SNAPSHOT_PATTERN.match(item.name).group(1)), reverse=True)

    def _load_candidate(self, path: Path) -> Snapshot | None:
        manifest_path = path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if manifest.get("complete") is not True:
            return None
        match = SNAPSHOT_PATTERN.match(path.name)
        if not match or int(manifest.get("global_version", -1)) != int(match.group(1)):
            return None
        stages: dict[str, Any] = {}
        try:
            for stage_name in ("binning", "ensemble", "contributions"):
                stage_path = path / "stages" / f"{stage_name}.json"
                if stage_path.exists():
                    stages[stage_name] = json.loads(stage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        trees: dict[int, dict[str, Any]] = {}
        trees_dir = path / "stages" / "trees"
        if trees_dir.exists():
            try:
                for tree_path in sorted(trees_dir.glob("tree-*.json")):
                    tree = json.loads(tree_path.read_text(encoding="utf-8"))
                    tree_index = int(tree["tree_index"])
                    trees[tree_index] = tree
            except (json.JSONDecodeError, OSError, KeyError, ValueError):
                return None
        stage_versions = manifest.get("stage_versions", {})
        binning_version = stage_versions.get("binning")
        if binning_version:
            if "binning" not in stages or stages["binning"].get("fingerprint") != binning_version.get("fingerprint"):
                return None
        completed = [int(index) for index in manifest.get("completed_trees", [])]
        if sorted(trees) != sorted(completed):
            return None
        tree_versions = stage_versions.get("trees", {}).get("by_index", {})
        for tree_index, tree in trees.items():
            version = tree_versions.get(str(tree_index))
            if version and version.get("fingerprint") != tree.get("artifact_fingerprint"):
                return None
        if "ensemble" in stages and stage_versions.get("ensemble"):
            if stages["ensemble"].get("fingerprint") != stage_versions["ensemble"].get("fingerprint"):
                return None
        if "contributions" in stages and stage_versions.get("contributions"):
            if stages["contributions"].get("fingerprint") != stage_versions["contributions"].get("fingerprint"):
                return None
        return Snapshot(
            version=int(manifest["global_version"]),
            path=path,
            manifest=manifest,
            stages=stages,
            trees=trees,
        )

    def commit(
        self,
        *,
        expected_version: int,
        config_hash: str,
        data_fingerprint: str,
        binning: dict[str, Any],
        trees: dict[int, dict[str, Any]],
        ensemble: dict[str, Any] | None = None,
        contributions: dict[str, Any] | None = None,
    ) -> Snapshot:
        with self._lock(exclusive=True):
            actual_version = self.current_version()
            if actual_version != expected_version:
                raise SnapshotVersionChanged(
                    f"snapshot version changed before commit: expected {expected_version}, found {actual_version}"
                )
            version = actual_version + 1
            self.snapshots_root.mkdir(parents=True, exist_ok=True)
            while (self.snapshots_root / f"snapshot-{version:06d}").exists():
                version += 1
            final_path = self.snapshots_root / f"snapshot-{version:06d}"
            temporary = self.snapshots_root / f".tmp-snapshot-{version:06d}-{os.getpid()}-{uuid.uuid4().hex}"
            temporary.mkdir(parents=False, exist_ok=False)
            try:
                _write_json(temporary / "stages" / "binning.json", binning)
                trees_dir = temporary / "stages" / "trees"
                trees_dir.mkdir(parents=True, exist_ok=False)
                for tree_index in sorted(trees):
                    _write_json(trees_dir / f"tree-{tree_index:06d}.json", trees[tree_index])
                if ensemble is not None:
                    _write_json(temporary / "stages" / "ensemble.json", ensemble)
                if contributions is not None:
                    _write_json(temporary / "stages" / "contributions.json", contributions)

                completed_trees = sorted(int(index) for index in trees)
                stage_versions: dict[str, Any] = {
                    "binning": {
                        "key": binning.get("key"),
                        "fingerprint": binning.get("fingerprint"),
                        "snapshot_version": version,
                    },
                    "trees": {
                        "completed": completed_trees,
                        "snapshot_version": version,
                        "by_index": {
                            str(tree_index): {
                                "key": trees[tree_index].get("key"),
                                "fingerprint": trees[tree_index].get("artifact_fingerprint"),
                                "snapshot_version": version,
                            }
                            for tree_index in completed_trees
                        },
                    },
                }
                if ensemble is not None:
                    stage_versions["ensemble"] = {
                        "key": ensemble.get("key"),
                        "fingerprint": ensemble.get("fingerprint"),
                        "snapshot_version": version,
                    }
                if contributions is not None:
                    stage_versions["contributions"] = {
                        "key": contributions.get("key"),
                        "fingerprint": contributions.get("fingerprint"),
                        "snapshot_version": version,
                    }
                manifest = {
                    "schema": "snapshot/v1",
                    "complete": True,
                    "global_version": version,
                    "config_hash": config_hash,
                    "data_fingerprint": data_fingerprint,
                    "completed_trees": completed_trees,
                    "stage_versions": stage_versions,
                }
                _write_json(temporary / "manifest.json", manifest)
                try:
                    os.replace(temporary, final_path)
                    _fsync_directory(self.snapshots_root)
                except OSError:
                    if final_path.exists():
                        raise SnapshotVersionChanged(f"snapshot target already exists: {final_path.name}")
                    raise
                _atomic_write_text(self.current_file, f"{final_path.name}\n")
            except Exception:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)
                raise
        loaded = self._load_candidate(final_path)
        if loaded is None:
            raise RuntimeError("committed snapshot failed validation")
        return loaded
