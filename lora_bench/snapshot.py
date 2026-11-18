"""Staged snapshot management with atomic directory replacement."""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

from .errors import E_CONFIG_MISMATCH, E_SNAPSHOT_CORRUPT, WorkbenchError
from .utils import (
    commit_staging,
    copy_tree_contents,
    read_json,
    stage_directory,
    write_json_atomic,
)

MANIFEST_NAME = "manifest.json"


class SnapshotStore:
    """Owns the snapshot workspace layout and all atomic write transitions.

    Layout::

        workspace/
          current/manifest.json        latest committed run
          current/forward/             base representation stage
          current/train/               per-task adapter artifacts
          current/validate/            validation summary stage
          versions/v000001/...         immutable committed copies
          staging/...                  in-flight temp directories
          cache/tasks/<cache_key>.npz  content-addressed task artifacts
    """

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.current_dir = os.path.join(self.root, "current")
        self.versions_dir = os.path.join(self.root, "versions")
        self.staging_dir = os.path.join(self.root, "staging")
        self.cache_dir = os.path.join(self.root, "cache", "tasks")
        self.summary_dir = os.path.join(self.root, "cache", "task_summaries")
        for directory in (
            self.root,
            self.current_dir,
            self.versions_dir,
            self.staging_dir,
            self.cache_dir,
            self.summary_dir,
        ):
            os.makedirs(directory, exist_ok=True)

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.current_dir, MANIFEST_NAME)

    def load_manifest(self) -> dict | None:
        if not os.path.exists(self.manifest_path):
            return None
        try:
            manifest = read_json(self.manifest_path)
        except (OSError, ValueError) as exc:
            raise WorkbenchError(
                E_SNAPSHOT_CORRUPT, f"manifest unreadable: {exc}", "snapshot"
            ) from exc
        required = ("version", "config_hash", "base_fingerprint", "stages")
        missing = [key for key in required if key not in manifest]
        if missing:
            raise WorkbenchError(
                E_SNAPSHOT_CORRUPT,
                f"manifest missing fields {sorted(missing)}",
                "snapshot",
            )
        return manifest

    def reconcile(
        self,
        config_hash: str,
        base_fingerprint: str,
        strict: bool,
    ) -> dict | None:
        """Resume check based on version number + config hash (rule 5)."""
        manifest = self.load_manifest()
        if manifest is None:
            return None
        mismatch_fields: list[str] = []
        if manifest["config_hash"] != config_hash:
            mismatch_fields.append("config_hash")
        if manifest["base_fingerprint"] != base_fingerprint:
            mismatch_fields.append("base_fingerprint")
        if mismatch_fields:
            if strict:
                raise WorkbenchError(
                    E_CONFIG_MISMATCH,
                    "snapshot cannot be reused: "
                    + ", ".join(mismatch_fields)
                    + " differs",
                    "snapshot",
                )
            return {"manifest": manifest, "mismatch_fields": mismatch_fields}
        return {"manifest": manifest, "mismatch_fields": []}

    def new_staging(self) -> str:
        return stage_directory(self.staging_dir)

    def commit(self, staging: str, manifest: dict) -> str:
        """Atomically publish a staged run and freeze a versioned copy."""
        write_json_atomic(os.path.join(staging, MANIFEST_NAME), manifest)
        commit_staging(staging, self.current_dir)
        version = int(manifest["version"])
        version_dir = os.path.join(self.versions_dir, f"v{version:06d}")
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
        copy_tree_contents(self.current_dir, version_dir)
        return version_dir

    def list_versions(self) -> list[str]:
        if not os.path.exists(self.versions_dir):
            return []
        return sorted(
            name
            for name in os.listdir(self.versions_dir)
            if name.startswith("v")
        )

    def next_version_number(self) -> int:
        """Highest frozen version number, so a config-change run never
        overwrites snapshots created under the previous config lineage."""
        versions = self.list_versions()
        if not versions:
            return 0
        return max(int(name[1:]) for name in versions)

    def cached_task_path(self, cache_key: str) -> str | None:
        path = os.path.join(self.cache_dir, f"{cache_key}.npz")
        return path if os.path.exists(path) else None

    def put_task_cache(self, cache_key: str, arrays: dict) -> None:
        target = os.path.join(self.cache_dir, f"{cache_key}.npz")
        if os.path.exists(target):
            return
        fd, tmp = tempfile.mkstemp(
            prefix=".tmp-", suffix=".npz", dir=self.cache_dir
        )
        os.close(fd)
        try:
            np.savez(tmp, **arrays)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def put_task_summary(self, task: str, summary: dict) -> None:
        write_json_atomic(os.path.join(self.summary_dir, f"{task}.json"), summary)

    def load_task_summary(self, task: str) -> dict:
        return read_json(os.path.join(self.summary_dir, f"{task}.json"))

    def load_task_cache(self, cache_key: str) -> dict[str, np.ndarray]:
        path = self.cached_task_path(cache_key)
        if path is None:
            raise FileNotFoundError(cache_key)
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def stage_subdir(self, staging: str, name: str) -> str:
        path = os.path.join(staging, name)
        os.makedirs(path, exist_ok=True)
        return path

    def previous_stage_dir(self, name: str) -> str | None:
        path = os.path.join(self.current_dir, name)
        return path if os.path.isdir(path) else None
