from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .util import atomic_write_json, atomic_write_text, stable_hash


class SnapshotStore:
    def __init__(self, root: Path):
        self.root = root
        self.snapshots_dir = root / "snapshots"
        self.pointer_path = root / "CURRENT"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _completed_manifests(self) -> list[tuple[int, Path, dict[str, Any]]]:
        completed: list[tuple[int, Path, dict[str, Any]]] = []
        for manifest_path in self.snapshots_dir.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not manifest.get("complete"):
                continue
            version = manifest.get("global_version")
            if not isinstance(version, int):
                continue
            if not manifest_path.parent.name.startswith("v"):
                continue
            required = [
                manifest_path.parent / "load" / "events.json",
                manifest_path.parent / "load" / "state_machine.json",
                manifest_path.parent / "load" / "config.json",
                manifest_path.parent / "state" / "session_states.json",
                manifest_path.parent / "state" / "run_metadata.json",
                manifest_path.parent / "summary" / "diagnostic.json",
            ]
            if all(path.exists() for path in required):
                completed.append((version, manifest_path.parent, manifest))
        completed.sort(key=lambda item: item[0])
        return completed

    def latest(self) -> dict[str, Any] | None:
        completed = self._completed_manifests()
        if not completed:
            return None
        version, directory, manifest = completed[-1]
        events = json.loads((directory / "load" / "events.json").read_text(encoding="utf-8"))
        machine = json.loads((directory / "load" / "state_machine.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "load" / "config.json").read_text(encoding="utf-8"))
        session_states = json.loads(
            (directory / "state" / "session_states.json").read_text(encoding="utf-8")
        )
        run_metadata = json.loads(
            (directory / "state" / "run_metadata.json").read_text(encoding="utf-8")
        )
        diagnostic = json.loads(
            (directory / "summary" / "diagnostic.json").read_text(encoding="utf-8")
        )
        return {
            "version": version,
            "path": str(directory),
            "manifest": manifest,
            "events": events.get("events", []),
            "machine": machine,
            "config": config,
            "sessions": session_states.get("sessions", {}),
            "global_state_table": session_states.get("global_state_table", {}),
            "budget": session_states.get("budget", {}),
            "execution": session_states.get("execution", {}),
            "failures": session_states.get("failures", []),
            "run_metadata": run_metadata,
            "diagnostic": diagnostic,
        }

    def commit(
        self,
        *,
        events: list[dict[str, Any]],
        machine: dict[str, Any],
        config: dict[str, Any],
        sessions: dict[str, Any],
        global_state_table: dict[str, Any],
        budget: dict[str, Any],
        execution: dict[str, Any],
        failures: list[dict[str, Any]],
        run_metadata: dict[str, Any],
        diagnostic: dict[str, Any],
        completed_batch: int,
        config_hash: str,
        machine_hash: str,
    ) -> dict[str, Any]:
        existing = self._completed_manifests()
        version = (existing[-1][0] + 1) if existing else 1
        version_label = f"v{version:08d}"
        final_directory = self.snapshots_dir / version_label
        staging_directory = self.snapshots_dir / f".tmp-{version_label}-{uuid.uuid4().hex}"
        staging_directory.mkdir(parents=True, exist_ok=False)
        file_hashes: dict[str, str] = {}
        try:
            artifacts = {
                "load/events.json": {"events": events},
                "load/state_machine.json": machine,
                "load/config.json": config,
                "state/session_states.json": {
                    "sessions": sessions,
                    "global_state_table": global_state_table,
                    "budget": budget,
                    "execution": execution,
                    "failures": failures,
                },
                "state/run_metadata.json": run_metadata,
                "summary/diagnostic.json": diagnostic,
            }
            for relative, payload in artifacts.items():
                target = staging_directory / relative
                atomic_write_json(target, payload)
                file_hashes[relative] = stable_hash(payload)

            manifest = {
                "manifest_version": 1,
                "global_version": version,
                "snapshot_version": version_label,
                "config_hash": config_hash,
                "machine_hash": machine_hash,
                "completed_batch": completed_batch,
                "completed_batch_count": completed_batch,
                "stages": {
                    "loading": {
                        "version": version_label,
                        "complete": True,
                        "artifacts": [
                            "load/events.json",
                            "load/state_machine.json",
                            "load/config.json",
                        ],
                    },
                    "advance": {
                        "version": version_label,
                        "complete": True,
                        "artifacts": [
                            "state/session_states.json",
                            "state/run_metadata.json",
                        ],
                    },
                    "summary": {
                        "version": version_label,
                        "complete": True,
                        "artifacts": ["summary/diagnostic.json"],
                    },
                },
                "file_hashes": file_hashes,
                "complete": True,
            }
            atomic_write_text(
                staging_directory / "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            os.replace(staging_directory, final_directory)
            atomic_write_text(
                self.pointer_path,
                json.dumps({"global_version": version, "snapshot": version_label}, sort_keys=True) + "\n",
            )
        finally:
            if staging_directory.exists():
                shutil.rmtree(staging_directory, ignore_errors=True)
        return {
            "global_version": version,
            "snapshot_version": version_label,
            "path": str(final_directory),
            "completed_batch": completed_batch,
        }

