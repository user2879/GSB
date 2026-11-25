"""Stage snapshot staging and atomic pointer replacement."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .fingerprint import sha256_json, read_json, write_json_atomic

SNAPSHOT_SCHEMA = 1


class SnapshotStore:
    def __init__(self, root: Path):
        self.root = root
        self.snapshots_dir = root / "snapshots"
        self.current_path = root / "current.json"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def current(self) -> dict[str, Any] | None:
        if not self.current_path.is_file():
            return None
        try:
            payload = read_json(self.current_path)
        except (OSError, ValueError):
            return None
        if payload.get("snapshot_schema") != SNAPSHOT_SCHEMA:
            return None
        return payload

    def commit(
        self,
        *,
        result: dict[str, Any],
        run_report: dict[str, Any],
        stages: dict[str, Any],
        pending_queue: list[dict[str, Any]],
        completed_cursors: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        version_material = {
            "version": metadata.get("config_version"),
            "structural_config_hash": metadata.get("structural_config_hash"),
            "runtime_config_hash": metadata.get("runtime_config_hash"),
            "input_digest": metadata.get("input_digest"),
            "result_cache_keys": {
                key: value
                for key, value in result.items()
                if key.endswith("cache_key") and isinstance(value, str)
            },
        }
        state_version = sha256_json(version_material)
        final_dir = self.snapshots_dir / state_version
        staging_dir = self.snapshots_dir / f".staging-{os.getpid()}-{state_version[:12]}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        write_json_atomic(staging_dir / "result.json", result, indent=None)
        write_json_atomic(staging_dir / "run_report.json", run_report)
        for name in sorted(stages):
            write_json_atomic(staging_dir / f"stage-{name}.json", stages[name], indent=None)
        write_json_atomic(staging_dir / "pending-queue.json", pending_queue)
        write_json_atomic(staging_dir / "completed-cursors.json", completed_cursors)
        manifest = {
            **metadata,
            "snapshot_schema": SNAPSHOT_SCHEMA,
            "state_version": state_version,
            "created_from": "temporary_stage_atomic_pointer_swap",
        }
        write_json_atomic(staging_dir / "manifest.json", manifest)
        if final_dir.exists():
            shutil.rmtree(staging_dir)
            write_json_atomic(final_dir / "run_report.json", run_report)
        else:
            os.replace(staging_dir, final_dir)
        pointer = {
            **manifest,
            "snapshot_dir": str(final_dir.relative_to(self.root)),
            "result_file": str((final_dir / "result.json").relative_to(self.root)),
            "report_file": str((final_dir / "run_report.json").relative_to(self.root)),
        }
        pointer_temp = self.current_path.with_name(f".current.json.tmp-{os.getpid()}")
        write_json_atomic(pointer_temp, pointer)
        os.replace(pointer_temp, self.current_path)
        self._prune()
        return pointer

    def _prune(self, keep: int = 5) -> None:
        dirs = sorted(
            (path for path in self.snapshots_dir.iterdir() if path.is_dir() and not path.name.startswith(".staging-")),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for stale in dirs[keep:]:
            shutil.rmtree(stale, ignore_errors=True)
