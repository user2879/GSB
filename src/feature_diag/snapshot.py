"""Immutable stage snapshots with atomic manifest commits and stale detection."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .fingerprint import sha256_text
from .serialization import atomic_write_text, dump_json, load_json, to_wire

SNAPSHOT_STAGES = ("crossing", "reductions", "metrics", "diagnostics")


class SnapshotManager:
    def __init__(
        self,
        root: str | Path,
        *,
        global_version: str,
        config_hash: str,
        precision_fingerprints: dict[str, str],
        schema_version: int = 1,
    ) -> None:
        self.root = Path(root)
        self.global_version = str(global_version)
        self.config_hash = str(config_hash)
        self.precision_fingerprints = dict(sorted(precision_fingerprints.items()))
        self.schema_version = int(schema_version)
        self.manifest_path = self.root / "manifest.json"
        self.stages_root = self.root / "stages"
        self.temp_root = self.root / ".tmp"
        self.root.mkdir(parents=True, exist_ok=True)
        self.stages_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.manifest: dict[str, Any] | None = None
        self.reusable = False
        self.stale_event: dict[str, Any] | None = None
        self.resumed_stages: list[str] = []
        self.computed_stages: list[str] = []
        self.stage_commits: dict[str, dict[str, Any]] = {}
        self.ignored_incomplete_temp_entries: list[str] = []
        self._bootstrap()

    def _base_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "precision_fingerprints": self.precision_fingerprints,
            "stages": {},
        }

    def _bootstrap(self) -> None:
        self.ignored_incomplete_temp_entries = sorted(path.name for path in self.temp_root.iterdir())
        if not self.manifest_path.exists():
            return
        try:
            manifest = load_json(self.manifest_path, decode_wire=False)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.stale_event = {
                "code": "E_SNAPSHOT_STALE",
                "reason": "manifest_unreadable",
                "message": str(exc),
            }
            return
        if not isinstance(manifest, dict):
            self.stale_event = {
                "code": "E_SNAPSHOT_STALE",
                "reason": "manifest_not_mapping",
                "message": "snapshot manifest root is not a mapping",
            }
            return
        mismatches: list[dict[str, Any]] = []
        if manifest.get("schema_version") != self.schema_version:
            mismatches.append(
                {
                    "field": "schema_version",
                    "manifest": manifest.get("schema_version"),
                    "current": self.schema_version,
                }
            )
        if manifest.get("global_version") != self.global_version:
            mismatches.append(
                {
                    "field": "global_version",
                    "manifest": manifest.get("global_version"),
                    "current": self.global_version,
                }
            )
        if manifest.get("config_hash") != self.config_hash:
            mismatches.append(
                {
                    "field": "config_hash",
                    "manifest": manifest.get("config_hash"),
                    "current": self.config_hash,
                }
            )
        if manifest.get("precision_fingerprints") != self.precision_fingerprints:
            mismatches.append(
                {
                    "field": "precision_fingerprints",
                    "manifest": manifest.get("precision_fingerprints"),
                    "current": self.precision_fingerprints,
                }
            )
        if mismatches:
            self.stale_event = {
                "code": "E_SNAPSHOT_STALE",
                "reason": "version_or_configuration_mismatch",
                "message": "existing snapshot was refused because version/config/precision fingerprint changed",
                "mismatches": mismatches,
            }
            return
        self.manifest = manifest
        self.reusable = True

    def _stage_state_path(self, stage: str, version: str) -> Path:
        return self.stages_root / stage / version / "state.json"

    def load_stage(self, stage: str, dependencies: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if stage not in SNAPSHOT_STAGES:
            raise ValueError(f"unknown snapshot stage: {stage}")
        if not self.reusable or self.manifest is None:
            return None, {"stage": stage, "status": "not_reused"}
        pointer = (self.manifest.get("stages") or {}).get(stage)
        if not isinstance(pointer, dict) or pointer.get("status") != "complete":
            return None, {"stage": stage, "status": "missing_or_incomplete"}
        if pointer.get("dependencies") != dependencies:
            return None, {
                "stage": stage,
                "status": "dependency_mismatch",
                "expected": dependencies,
                "stored": pointer.get("dependencies"),
            }
        version = pointer.get("version")
        path = self._stage_state_path(stage, str(version))
        if not path.exists():
            return None, {"stage": stage, "status": "artifact_missing"}
        try:
            state = load_json(path, decode_wire=True)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return None, {"stage": stage, "status": "artifact_unreadable", "message": str(exc)}
        if not isinstance(state, dict):
            return None, {"stage": stage, "status": "artifact_not_mapping"}
        if state.get("global_version") != self.global_version or state.get("config_hash") != self.config_hash:
            return None, {"stage": stage, "status": "artifact_header_mismatch"}
        if state.get("dependencies") != dependencies:
            return None, {"stage": stage, "status": "artifact_dependency_mismatch"}
        payload = state.get("payload")
        expected_hash = state.get("payload_fingerprint")
        actual_hash = sha256_text(dump_json(payload))
        if expected_hash != actual_hash:
            return None, {"stage": stage, "status": "artifact_fingerprint_mismatch"}
        self.resumed_stages.append(stage)
        return payload, {
            "stage": stage,
            "status": "reused",
            "version": version,
            "payload_fingerprint": actual_hash,
        }

    def commit_stage(self, stage: str, payload: dict[str, Any], dependencies: dict[str, Any]) -> dict[str, Any]:
        if stage not in SNAPSHOT_STAGES:
            raise ValueError(f"unknown snapshot stage: {stage}")
        payload_hash = sha256_text(dump_json(payload))
        version = f"{stage}-{payload_hash[:20]}"
        final_dir = self.stages_root / stage / version
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": self.schema_version,
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "precision_fingerprints": self.precision_fingerprints,
            "stage": stage,
            "dependencies": dependencies,
            "payload_fingerprint": payload_hash,
            "payload": to_wire(payload),
        }
        if not final_dir.exists():
            temp_dir = Path(tempfile.mkdtemp(prefix=f"{stage}-", dir=self.temp_root))
            try:
                atomic_write_text(temp_dir / "state.json", json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                os.replace(temp_dir, final_dir)
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
        pointer = {
            "status": "complete",
            "version": version,
            "path": str((Path("stages") / stage / version / "state.json")).replace("\\", "/"),
            "payload_fingerprint": payload_hash,
            "dependencies": dependencies,
        }
        if self.manifest is None or not self.reusable:
            self.manifest = self._base_manifest()
            self.reusable = True
        self.manifest.setdefault("stages", {})[stage] = pointer
        atomic_write_text(
            self.manifest_path,
            json.dumps(self.manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        self.stage_commits[stage] = pointer
        if stage not in self.computed_stages:
            self.computed_stages.append(stage)
        return pointer

    def recovery_info(self) -> dict[str, Any]:
        recovery_point = self.resumed_stages[-1] if self.resumed_stages else None
        return {
            "snapshot_root": str(self.root),
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "precision_fingerprints": self.precision_fingerprints,
            "stale_event": self.stale_event,
            "resumed_stages": list(self.resumed_stages),
            "computed_stages": list(self.computed_stages),
            "recovery_point": recovery_point,
            "ignored_incomplete_temp_entries": list(self.ignored_incomplete_temp_entries),
            "stage_versions": {
                stage: (self.manifest or {}).get("stages", {}).get(stage, {}).get("version")
                for stage in SNAPSHOT_STAGES
                if stage in (self.manifest or {}).get("stages", {})
            },
        }
