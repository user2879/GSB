"""Staged, atomically replaced execution snapshots."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .artifacts import Artifact
from .util import atomic_replace_directory, atomic_write_json, to_jsonable


class SnapshotManager:
    def __init__(self, workspace: str | os.PathLike[str]):
        self.workspace = Path(workspace)
        self.snapshots = self.workspace / "snapshots"
        self.current = self.snapshots / "current"
        self.previous = self.snapshots / "current.previous"
        self.tmp_root = self.workspace / "tmp"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self) -> Path | None:
        for directory in (self.current, self.previous):
            candidate = directory / "manifest.json"
            if candidate.is_file():
                return candidate
        return None

    def inspect(self) -> dict[str, Any] | None:
        manifest_path = self._manifest_path()
        if manifest_path is None:
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["_snapshot_dir"] = str(manifest_path.parent)
            return manifest
        except (OSError, ValueError, TypeError):
            return None

    def check_resume(self, *, global_version: int, config_hash: str) -> dict[str, Any]:
        manifest = self.inspect()
        if manifest is None:
            if self.current.exists() or self.previous.exists():
                return {
                    "status": "stale",
                    "resumable": False,
                    "manifest": None,
                    "expected_global_version": global_version,
                    "actual_global_version": None,
                    "expected_config_hash": config_hash,
                    "actual_config_hash": None,
                    "reason": "snapshot_manifest_unreadable",
                }
            return {"status": "absent", "resumable": False, "manifest": None}
        version_matches = manifest.get("global_version") == global_version
        config_matches = manifest.get("config_hash") == config_hash
        if not version_matches or not config_matches:
            return {
                "status": "stale",
                "resumable": False,
                "manifest": manifest,
                "expected_global_version": global_version,
                "actual_global_version": manifest.get("global_version"),
                "expected_config_hash": config_hash,
                "actual_config_hash": manifest.get("config_hash"),
            }
        resumable = manifest.get("status") != "complete"
        return {
            "status": "resumable" if resumable else "complete",
            "resumable": resumable,
            "manifest": manifest,
        }

    def commit(
        self,
        *,
        global_version: int,
        config_hash: str,
        chain_hash: str,
        input_hash: str,
        completed_nodes: Iterable[str],
        pending_nodes: Iterable[str],
        failures: Iterable[dict[str, Any]],
        artifacts: Iterable[Artifact],
        status: str,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        staging = Path(tempfile.mkdtemp(prefix="snapshot-", dir=self.tmp_root))
        try:
            (staging / "validation").mkdir()
            (staging / "execution" / "nodes").mkdir(parents=True)
            (staging / "summary").mkdir()
            atomic_write_json(staging / "validation" / "validation.json", validation)

            artifact_by_id = {artifact.node_id: artifact for artifact in artifacts}
            completed = list(dict.fromkeys(completed_nodes))
            artifact_versions: dict[str, str] = {}
            for node_id in completed:
                artifact = artifact_by_id.get(node_id)
                if artifact is None:
                    continue
                node_dir = staging / "execution" / "nodes" / node_id.replace("/", "_")
                node_dir.mkdir(parents=True, exist_ok=True)
                np.save(node_dir / "values.npy", artifact.values, allow_pickle=False)
                if artifact.raw_values is not None:
                    np.save(node_dir / "raw.npy", artifact.raw_values, allow_pickle=False)
                atomic_write_json(
                    node_dir / "metadata.json",
                    {
                        "node_id": artifact.node_id,
                        "op": artifact.op,
                        "cache_key": artifact.cache_key,
                        "values_hash": artifact.values_hash,
                        "has_raw_values": artifact.raw_values is not None,
                        "raw_values_hash": artifact.raw_values_hash,
                        "metadata": artifact.metadata,
                    },
                )
                artifact_versions[node_id] = artifact.cache_key

            pending = list(dict.fromkeys(pending_nodes))
            failure_list = list(failures)
            manifest = {
                "global_version": global_version,
                "config_hash": config_hash,
                "chain_hash": chain_hash,
                "input_hash": input_hash,
                "status": status,
                "completed_nodes": completed,
                "pending_nodes": pending,
                "artifact_versions": artifact_versions,
                "failure_count": len(failure_list),
            }
            summary = {
                "status": status,
                "completed_nodes": completed,
                "pending_nodes": pending,
                "failures": to_jsonable(failure_list),
            }
            atomic_write_json(staging / "summary" / "summary.json", summary)
            atomic_write_json(staging / "manifest.json", manifest)
            atomic_replace_directory(staging, self.current)
            return manifest
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise
