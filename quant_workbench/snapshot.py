"""Whole-snapshot replacement, version checks and resumable stage artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from .errors import (
    E_SNAPSHOT_CORRUPT,
    E_SNAPSHOT_VERSION_MISMATCH,
    WorkbenchError,
)
from .util import atomic_write_text, canonical_json, fingerprint


SNAPSHOT_FORMAT_VERSION = 1
STAGES = ("ingest", "calibration", "quantization", "evaluation", "deployment")


def _encoded_name(layer: str) -> str:
    return quote(layer, safe="") or "_"


def _decoded_name(filename: str) -> str:
    if filename.endswith(".json"):
        filename = filename[:-5]
    return unquote(filename)


class SnapshotStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.manifest_path = self.path / "manifest.json"
        self._manifest: dict[str, Any] | None = None
        self._artifact_cache: dict[str, dict[str, dict[str, Any]]] = {}

    @property
    def manifest(self) -> dict[str, Any] | None:
        return self._manifest

    def open(
        self,
        *,
        global_version: int,
        config_hash: str,
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        """Open and validate a snapshot without modifying any file.

        A mismatch is terminal: callers must not fall back to per-file patching.
        """

        if not self.manifest_path.exists():
            self._manifest = None
            return None
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkbenchError(
                E_SNAPSHOT_CORRUPT,
                f"cannot read snapshot manifest {self.manifest_path}: {exc}",
            ) from exc
        if not isinstance(manifest, dict):
            raise WorkbenchError(E_SNAPSHOT_CORRUPT, "snapshot manifest root must be an object")
        stored_version = manifest.get("global_version")
        stored_hash = manifest.get("config_hash")
        requested = global_version if expected_version is None else expected_version
        version_mismatch = stored_version != global_version or (
            expected_version is not None and stored_version != expected_version
        )
        if version_mismatch or stored_hash != config_hash:
            raise WorkbenchError(
                E_SNAPSHOT_VERSION_MISMATCH,
                "snapshot version or config hash does not match the requested run; snapshot was left untouched",
                details={
                    "stored_global_version": stored_version,
                    "requested_global_version": requested,
                    "stored_config_hash": stored_hash,
                    "requested_config_hash": config_hash,
                    "snapshot_path": str(self.path),
                },
            )
        missing = [stage for stage in STAGES if stage not in manifest.get("artifact_versions", {})]
        if missing:
            raise WorkbenchError(
                E_SNAPSHOT_CORRUPT,
                f"snapshot manifest is missing stage versions: {missing}",
            )
        self._manifest = manifest
        self._artifact_cache = {}
        return manifest

    def _load_stage(self, stage: str) -> dict[str, dict[str, Any]]:
        if stage in self._artifact_cache:
            return self._artifact_cache[stage]
        directory = self.path / "stages" / stage
        result: dict[str, dict[str, Any]] = {}
        if directory.exists():
            for path in sorted(directory.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise WorkbenchError(
                        E_SNAPSHOT_CORRUPT,
                        f"cannot read snapshot artifact {path}: {exc}",
                    ) from exc
                if not isinstance(payload, dict) or "layer" not in payload:
                    raise WorkbenchError(
                        E_SNAPSHOT_CORRUPT,
                        f"snapshot artifact {path} is not a layer payload",
                    )
                result[str(payload["layer"])] = payload
        expected_version = (self._manifest or {}).get("artifact_versions", {}).get(stage)
        actual_version = fingerprint({"stage": stage, "layers": result})
        if expected_version is not None and actual_version != expected_version:
            raise WorkbenchError(
                E_SNAPSHOT_CORRUPT,
                f"snapshot stage {stage!r} content does not match its manifest version",
            )
        self._artifact_cache[stage] = result
        return result

    def load_stage(self, stage: str) -> dict[str, dict[str, Any]]:
        if self._manifest is None:
            return {}
        return self._load_stage(stage)

    def load_artifacts(self) -> dict[str, dict[str, dict[str, Any]]]:
        if self._manifest is None:
            return {}
        return {
            stage: self._load_stage(stage)
            for stage in STAGES
            if stage not in {"ingest"}
        }

    def load_runtime_state(self) -> dict[str, Any] | None:
        if self._manifest is None:
            return None
        path = self.path / "runtime_state.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkbenchError(
                E_SNAPSHOT_CORRUPT,
                f"cannot read snapshot runtime state {path}: {exc}",
            ) from exc
        if not isinstance(payload, dict):
            raise WorkbenchError(E_SNAPSHOT_CORRUPT, "snapshot runtime state must be an object")
        return payload

    def commit(
        self,
        *,
        global_version: int,
        config_hash: str,
        calibration_hash: str,
        completed_layers: list[str],
        generation: int,
        command_count: int,
        artifacts: dict[str, dict[str, dict[str, Any]]],
        ingest: dict[str, Any],
        runtime_state: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a complete replacement snapshot and swap it in as one unit."""

        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{self.path.name}.tmp-", dir=str(parent))
        )
        backup: Path | None = None
        try:
            stage_versions: dict[str, str] = {}
            for stage in STAGES:
                stage_dir = temporary / "stages" / stage
                stage_dir.mkdir(parents=True, exist_ok=True)
                if stage == "ingest":
                    atomic_write_text(
                        stage_dir / "ingest.json",
                        canonical_json(ingest) + "\n",
                    )
                    stage_versions[stage] = fingerprint(ingest)
                    continue
                stage_payloads = artifacts.get(stage, {})
                for layer in sorted(stage_payloads):
                    path = stage_dir / f"{_encoded_name(layer)}.json"
                    atomic_write_text(path, canonical_json(stage_payloads[layer]) + "\n")
                stage_versions[stage] = fingerprint(
                    {
                        "stage": stage,
                        "layers": {
                            layer: stage_payloads[layer]
                            for layer in sorted(stage_payloads)
                        },
                    }
                )

            previous_revision = int((self._manifest or {}).get("revision", 0))
            manifest = {
                "format_version": SNAPSHOT_FORMAT_VERSION,
                "global_version": int(global_version),
                "revision": previous_revision + 1,
                "config_hash": config_hash,
                "calibration_hash": calibration_hash,
                "completed_layers": sorted(completed_layers),
                "generation": int(generation),
                "command_count": int(command_count),
                "artifact_versions": stage_versions,
                "stages": list(STAGES),
            }
            atomic_write_text(
                temporary / "manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
            if runtime_state is not None:
                atomic_write_text(
                    temporary / "runtime_state.json",
                    canonical_json(runtime_state) + "\n",
                )
            if diagnostics is not None:
                atomic_write_text(
                    temporary / "diagnostics.json",
                    json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
                )

            if self.path.exists():
                backup = parent / f".{self.path.name}.backup-{uuid.uuid4().hex}"
                os.replace(self.path, backup)
            try:
                os.replace(temporary, self.path)
            except BaseException:
                if backup is not None and backup.exists() and not self.path.exists():
                    os.replace(backup, self.path)
                raise
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
            self._manifest = manifest
            self._artifact_cache = {}
            return manifest
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
