"""Atomic run manifest with strict resume compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_MISMATCH, WorkbenchError
from .util import atomic_write_json


SNAPSHOT_SCHEMA_VERSION = 1
GLOBAL_VERSION = 1


class SnapshotManager:
    def __init__(
        self,
        state_dir: str | Path,
        *,
        config_hash: str,
        injection_hash: str,
        samples_hash: str,
        resume_requested: bool = False,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.state_dir / "manifest.json"
        self.config_hash = config_hash
        self.injection_hash = injection_hash
        self.samples_hash = samples_hash
        self.resume_requested = resume_requested
        self.resumed = False
        self.completed_stages: list[str] = []
        self.completed_tasks: list[str] = []
        self.manifest: dict[str, Any] = {}

    def _load_existing(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if self.resume_requested:
                raise WorkbenchError(
                    E_CONFIG_MISMATCH,
                    "snapshot manifest is unreadable; refusing to resume",
                    stage="snapshot",
                )
            return None
        return loaded if isinstance(loaded, dict) else None

    def _assert_compatible(self, existing: dict[str, Any]) -> None:
        expected = {
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "global_version": GLOBAL_VERSION,
            "config_hash": self.config_hash,
            "injection_hash": self.injection_hash,
            "samples_hash": self.samples_hash,
        }
        mismatches: dict[str, dict[str, Any]] = {}
        for key, value in expected.items():
            if existing.get(key) != value:
                mismatches[key] = {"existing": existing.get(key), "requested": value}
        if mismatches:
            raise WorkbenchError(
                E_CONFIG_MISMATCH,
                "snapshot version/config/injection/sample fingerprint mismatch; refusing to resume",
                stage="snapshot",
                details={"mismatches": mismatches},
            )

    def start(self) -> None:
        existing = self._load_existing()
        if existing is not None:
            status = existing.get("status")
            if status != "complete":
                self._assert_compatible(existing)
                self.resumed = True
                self.completed_stages = list(existing.get("completed_stages", []))
                self.completed_tasks = list(existing.get("completed_tasks", []))
            elif self.resume_requested:
                self._assert_compatible(existing)
                self.resumed = True
                self.completed_stages = list(existing.get("completed_stages", []))
                self.completed_tasks = list(existing.get("completed_tasks", []))
        self.manifest = self._base_manifest("running")
        atomic_write_json(self.manifest_path, self.manifest)

    def _base_manifest(self, status: str) -> dict[str, Any]:
        return {
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "global_version": GLOBAL_VERSION,
            "status": status,
            "config_hash": self.config_hash,
            "injection_hash": self.injection_hash,
            "samples_hash": self.samples_hash,
            "completed_stages": list(dict.fromkeys(self.completed_stages)),
            "completed_schemes": [],
            "completed_samples": [],
            "completed_tasks": sorted(set(self.completed_tasks)),
        }

    def mark_stage(self, stage: str, completed_task_keys: list[str]) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.completed_tasks.extend(completed_task_keys)
        self.manifest = self._base_manifest("running")
        atomic_write_json(self.manifest_path, self.manifest)

    def complete(self, schemes: list[str], samples: list[str]) -> None:
        self.manifest = self._base_manifest("complete")
        self.manifest["completed_schemes"] = sorted(set(schemes))
        self.manifest["completed_samples"] = sorted(set(samples))
        atomic_write_json(self.manifest_path, self.manifest)

    def interrupt(self, schemes: list[str] | None = None, samples: list[str] | None = None) -> None:
        self.manifest = self._base_manifest("interrupted")
        self.manifest["completed_schemes"] = sorted(set(schemes or []))
        self.manifest["completed_samples"] = sorted(set(samples or []))
        atomic_write_json(self.manifest_path, self.manifest)
