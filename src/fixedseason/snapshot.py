"""Staged run snapshots with whole-directory replacement."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .jsonutil import read_json, write_json_atomic

SNAPSHOT_STAGES = (
    "loading",
    "interpolation",
    "basis",
    "normal_equations",
    "elimination",
    "back_substitution",
)


@dataclass(frozen=True)
class SnapshotView:
    status: str
    manifest: dict[str, Any] | None
    artifacts: dict[str, dict[str, Any]]
    final_report: dict[str, Any] | None
    archive_path: str | None = None


class SnapshotManager:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.root = self.workspace / "snapshot"
        self.current = self.root / "current"
        self.archive_root = self.root / "archive"
        self.root.mkdir(parents=True, exist_ok=True)

    def _recover_interrupted_swap(self) -> None:
        """Recover the tiny window between backup rename and new-dir replace."""

        if not self.current.exists():
            backups = sorted(
                self.root.glob(".backup-*.tmp"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if backups:
                os.replace(backups[0], self.current)
        if self.current.exists():
            for backup in self.root.glob(".backup-*.tmp"):
                shutil.rmtree(backup, ignore_errors=True)
        for staging in self.root.glob(".current-*.tmp"):
            shutil.rmtree(staging, ignore_errors=True)

    def _archive_current(self) -> str | None:
        if not self.current.exists():
            return None
        self.archive_root.mkdir(parents=True, exist_ok=True)
        name = f"invalidated-{uuid.uuid4().hex[:12]}"
        target = self.archive_root / name
        os.replace(self.current, target)
        return str(target)

    def begin(
        self,
        *,
        global_version: int,
        config_hash: str,
        compute_config_hash: str,
        series_data_fingerprint: Mapping[str, str],
        series_library_fingerprint: str,
        bit_width_tier: str,
    ) -> SnapshotView:
        self._recover_interrupted_swap()
        if not self.current.exists():
            return SnapshotView("fresh", None, {}, None)
        try:
            manifest = read_json(self.current / "manifest.json")
        except (OSError, ValueError, TypeError):
            archive = self._archive_current()
            return SnapshotView("invalidated_corrupt", None, {}, None, archive)
        expected = {
            "global_version": global_version,
            "config_hash": config_hash,
            "series_library_fingerprint": series_library_fingerprint,
            "bit_width_tier": bit_width_tier,
        }
        mismatch_reason = None
        for key, value in expected.items():
            if manifest.get(key) != value:
                mismatch_reason = key
                break
        if mismatch_reason is not None:
            archive = self._archive_current()
            return SnapshotView(
                f"invalidated_{mismatch_reason}", manifest, {}, None, archive
            )

        artifacts: dict[str, dict[str, Any]] = {}
        for stage in SNAPSHOT_STAGES:
            path = self.current / "stages" / f"{stage}.json"
            try:
                value = read_json(path)
                artifacts[stage] = value if isinstance(value, dict) else {}
            except (OSError, ValueError, TypeError):
                artifacts[stage] = {}
        final_report = None
        report_path = self.current / "report.json"
        if report_path.exists():
            try:
                candidate = read_json(report_path)
                if isinstance(candidate, dict):
                    final_report = candidate
            except (OSError, ValueError, TypeError):
                final_report = None
        status = "already_complete" if manifest.get("complete") else "resume"
        if status == "already_complete" and final_report is None:
            status = "resume"
        return SnapshotView(status, manifest, artifacts, final_report)

    def commit(
        self,
        manifest: Mapping[str, Any],
        artifacts: Mapping[str, Mapping[str, Any]],
        *,
        partial_report: Mapping[str, Any] | None = None,
        final_report: Mapping[str, Any] | None = None,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._recover_interrupted_swap()
        temp_dir = Path(
            tempfile.mkdtemp(prefix=".current-", suffix=".tmp", dir=str(self.root))
        )
        try:
            (temp_dir / "stages").mkdir(parents=True, exist_ok=True)
            complete_manifest = dict(manifest)
            complete_manifest["stage_files"] = {
                stage: f"stages/{stage}.json" for stage in SNAPSHOT_STAGES
            }
            write_json_atomic(temp_dir / "manifest.json", complete_manifest)
            for stage in SNAPSHOT_STAGES:
                write_json_atomic(
                    temp_dir / "stages" / f"{stage}.json",
                    dict(artifacts.get(stage, {})),
                )
            if partial_report is not None:
                write_json_atomic(temp_dir / "partial_report.json", dict(partial_report))
            if final_report is not None:
                write_json_atomic(temp_dir / "report.json", dict(final_report))

            backup = None
            if self.current.exists():
                backup = self.root / f".backup-{uuid.uuid4().hex}.tmp"
                os.replace(self.current, backup)
            try:
                os.replace(temp_dir, self.current)
            except Exception:
                if backup is not None and backup.exists() and not self.current.exists():
                    os.replace(backup, self.current)
                raise
            if backup is not None and backup.exists():
                shutil.rmtree(backup)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def clear_current(self) -> None:
        if self.current.exists():
            shutil.rmtree(self.current)
