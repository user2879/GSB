"""Whole-generation atomic snapshots and recovery verification."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .errors import E_SNAPSHOT_INVALID, WorkbenchError
from .util import fsync_directory, read_json, sha256_file, write_json_atomic


STAGE_FILES = {
    "stats": "stats.json",
    "imputation": "imputations.json",
    "model": "model.json",
    "state": "state.json",
    "report": "report.json",
}


class SnapshotStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.generations = self.root / "generations"
        self.user_report_path = self.root / "report.json"
        self.generations.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            manifest = read_json(self.manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkbenchError(E_SNAPSHOT_INVALID, f"manifest is not readable: {exc}") from exc
        if not isinstance(manifest, dict) or not isinstance(manifest.get("global_version"), int):
            raise WorkbenchError(E_SNAPSHOT_INVALID, "manifest has no valid global_version")
        return manifest

    def load_state(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        manifest = self.load_manifest()
        if manifest is None:
            return None
        generation = self.generations / manifest["generation"]
        if not generation.is_dir():
            raise WorkbenchError(
                E_SNAPSHOT_INVALID,
                "snapshot generation directory is missing",
                {"generation": manifest["generation"]},
            )
        versions = manifest.get("stage_versions", {})
        for stage, filename in STAGE_FILES.items():
            path = generation / filename
            if not path.exists():
                raise WorkbenchError(
                    E_SNAPSHOT_INVALID,
                    f"snapshot stage artifact {filename} is missing",
                    {"stage": stage},
                )
            actual = sha256_file(path)
            if versions.get(stage) != actual:
                raise WorkbenchError(
                    E_SNAPSHOT_INVALID,
                    f"snapshot stage artifact {filename} failed its content hash check",
                    {"stage": stage, "expected": versions.get(stage), "actual": actual},
                )
        try:
            state = read_json(generation / STAGE_FILES["state"])
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkbenchError(E_SNAPSHOT_INVALID, f"state artifact is unreadable: {exc}") from exc
        return state, manifest

    def commit(
        self,
        state: dict[str, Any],
        report: dict[str, Any],
        *,
        expected_parent_version: int,
        stage_fingerprints: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        current = self.load_manifest()
        current_version = int(current["global_version"]) if current else 0
        if current_version != expected_parent_version:
            raise WorkbenchError(
                E_SNAPSHOT_INVALID,
                "manifest version changed while preparing a snapshot",
                {"expected": expected_parent_version, "actual": current_version},
            )
        version = current_version + 1
        generation_name = f"v{version:08d}"
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.generations))
        try:
            stage_payloads = {
                "stats": state.get("stats", {}),
                "imputation": state.get("imputations", {}),
                "model": state.get("model"),
                "state": state,
            }
            for stage, payload in stage_payloads.items():
                write_json_atomic(staging / STAGE_FILES[stage], payload)
            report_copy = deepcopy(report)
            report_copy["snapshot_version"] = version
            report_copy["snapshot_generation"] = generation_name
            write_json_atomic(staging / STAGE_FILES["report"], report_copy)
            fsync_directory(staging)

            stage_versions = {
                stage: sha256_file(staging / filename)
                for stage, filename in STAGE_FILES.items()
            }
            final_generation = self.generations / generation_name
            if final_generation.exists():
                raise WorkbenchError(
                    E_SNAPSHOT_INVALID,
                    "target snapshot generation already exists",
                    {"generation": generation_name},
                )
            os.replace(staging, final_generation)
            fsync_directory(self.generations)

            # Re-read immediately before atomic manifest replacement.  This is the
            # compare step in the compare-and-swap commit protocol.
            latest = self.load_manifest()
            latest_version = int(latest["global_version"]) if latest else 0
            if latest_version != current_version:
                raise WorkbenchError(
                    E_SNAPSHOT_INVALID,
                    "manifest changed during snapshot generation",
                    {"expected": current_version, "actual": latest_version},
                )
            manifest = {
                "format": 1,
                "global_version": version,
                "parent_version": current_version,
                "generation": generation_name,
                "config_hash": state.get("config_hash"),
                "base_config_hash": state.get("base_config_hash"),
                "base_executable_config_hash": state.get(
                    "base_executable_config_hash"
                ),
                "executable_config_hash": state.get("executable_config_hash"),
                "data_fingerprint": state.get("data_fingerprint"),
                "base_data_fingerprint": state.get("base_data_fingerprint"),
                "group_map_hash": state.get("group_map_hash"),
                "plan_hash": state.get("plan_hash"),
                "snapshot_interval_batches": (
                    state.get("config", {}).get("snapshot_interval_batches")
                ),
                "batch_cell_budget": state.get("config", {}).get("batch_cell_budget"),
                "thread_pool_workers": state.get("config", {}).get("thread_pool_workers"),
                "completed_batches": list(state.get("completed_batches", [])),
                "in_progress_batch": state.get("in_progress_batch"),
                "completed_batch_count": len(state.get("completed_batches", [])),
                "pending_count": len(state.get("pending_jobs", [])),
                "stage_versions": stage_versions,
                "stage_files": dict(STAGE_FILES),
                "stage_fingerprints": stage_fingerprints,
                "result_fingerprint": state.get("result_fingerprint"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json_atomic(self.manifest_path, manifest)
            fsync_directory(self.root)
            write_json_atomic(self.user_report_path, report_copy)
            self._cleanup_old_generations(keep=3)
            return manifest, report_copy
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def _cleanup_old_generations(self, keep: int) -> None:
        manifests = sorted(
            (path for path in self.generations.iterdir() if path.is_dir() and path.name.startswith("v")),
            key=lambda path: path.name,
            reverse=True,
        )
        for path in manifests[keep:]:
            shutil.rmtree(path, ignore_errors=True)

    def read_report(self) -> dict[str, Any] | None:
        if not self.user_report_path.exists():
            return None
        try:
            return read_json(self.user_report_path)
        except (OSError, json.JSONDecodeError):
            return None


def clear_state(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
