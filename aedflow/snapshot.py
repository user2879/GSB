"""Stage snapshots with whole-directory atomic replacement and recovery."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .errors import AedError

STAGE_ORDER = ("load_align", "boundary_decision", "trajectory_build", "match_evaluation")
ARTIFACT_VERSIONS = {
    "load_align": "artifact_v1",
    "boundary_decision": "artifact_v1",
    "trajectory_build": "artifact_v1",
    "match_evaluation": "artifact_v1",
}
MANIFEST_NAME = "snapshot_manifest.json"


def _atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _replace_dir(target: Path, staging: Path) -> None:
    backup = target.with_name(target.name + ".old")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    os.replace(staging, target)
    if backup.exists():
        shutil.rmtree(backup)


class SnapshotStore:
    def __init__(self, root: Path, identity: dict[str, Any]):
        self.root = root
        self.identity = identity
        self.root.mkdir(parents=True, exist_ok=True)

    def _write_manifest(
        self,
        stage_dir: Path,
        stage: str,
        completed_shards: list[str],
        progress: dict[str, Any],
    ) -> None:
        manifest = {
            "global_version": progress.get("global_version", 1),
            "config_fingerprint": self.identity["config_fingerprint"],
            "header_version_set": list(self.identity["header_version_set"]),
            "completed_shard_ids": list(completed_shards),
            "stage": stage,
            "stage_artifact_versions": {
                key: value for key, value in ARTIFACT_VERSIONS.items() if key <= stage
            },
        }
        _atomic_json(stage_dir / MANIFEST_NAME, manifest)

    def commit_stage(
        self,
        stage: str,
        completed_shards: list[str],
        artifacts: dict[str, Any],
        progress: dict[str, Any],
    ) -> Path:
        if stage not in STAGE_ORDER:
            raise ValueError(f"unknown stage {stage}")
        staging = self.root.with_name(self.root.name + ".staging")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        # Carry every previously committed artifact forward, then replace the
        # current stage's files. Recovery therefore sees one complete snapshot
        # while publication remains a whole-directory atomic replacement.
        if self.root.exists():
            for child in self.root.iterdir():
                if child.name == MANIFEST_NAME:
                    continue
                shutil.copy2(child, staging / child.name)
        for name, value in artifacts.items():
            if isinstance(value, np.ndarray):
                np.save(staging / f"{name}.npy", value)
            else:
                _atomic_json(staging / f"{name}.json", value)
        self._write_manifest(staging, stage, completed_shards, progress)
        _replace_dir(self.root, staging)
        return self.root

    def is_valid(self) -> bool:
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.exists():
            return False
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return (
            manifest.get("config_fingerprint") == self.identity["config_fingerprint"]
            and sorted(manifest.get("header_version_set", []))
            == sorted(self.identity["header_version_set"])
            and manifest.get("global_version") == self.identity["global_version"]
        )

    def load_manifest(self) -> dict[str, Any]:
        if not self.is_valid():
            raise AedError(
                "snapshot identity mismatch; whole-snapshot replacement required",
                code="E_SNAPSHOT_STALE",
            )
        with (self.root / MANIFEST_NAME).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_json(self, name: str) -> Any:
        with (self.root / f"{name}.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_array(self, name: str) -> np.ndarray:
        return np.load(self.root / f"{name}.npy")

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)


def assert_or_reset(store: SnapshotStore) -> dict[str, Any] | None:
    if not store.root.exists() or not (store.root / MANIFEST_NAME).exists():
        return None
    if not store.is_valid():
        raise AedError(
            "existing snapshot is stale for the current config/header set",
            code="E_SNAPSHOT_STALE",
        )
    return store.load_manifest()
