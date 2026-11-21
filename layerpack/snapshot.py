"""Per-phase atomic snapshots, complete-snapshot recovery and version checks."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .fingerprint import canonical_json, sha256_json, to_builtin


SNAPSHOT_FORMAT = "layerpack-snapshot-v1"
PHASES = ("loading", "quantization", "calibration", "deployment")
_SNAPSHOT_PATTERN = re.compile(r"^v(\d{8,})\.json$")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(to_builtin(payload)))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


class SnapshotStore:
    """Store immutable phase artifacts and atomically-published manifests."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.snapshot_dir = self.root / "snapshots"
        self.stage_dir = self.root / "stages"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_versions(self) -> list[int]:
        versions: list[int] = []
        for path in self.snapshot_dir.iterdir():
            match = _SNAPSHOT_PATTERN.match(path.name)
            if match:
                versions.append(int(match.group(1)))
        return sorted(versions)

    def next_version(self) -> int:
        versions = self._snapshot_versions()
        return (versions[-1] + 1) if versions else 1

    def _load_manifest_file(self, path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("format") != SNAPSHOT_FORMAT:
            return None
        if not isinstance(payload.get("phases"), dict):
            return None
        for phase in payload["phases"].values():
            if not isinstance(phase, dict):
                return None
            artifact = self.root / str(phase.get("artifact", ""))
            marker = artifact / "COMMITTED"
            payload_file = artifact / "payload.json"
            if not marker.is_file() or not payload_file.is_file():
                return None
            try:
                marker_data = json.loads(marker.read_text(encoding="utf-8"))
                payload_data = json.loads(payload_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if marker_data.get("payload_sha256") != phase.get("payload_sha256"):
                return None
            if sha256_json(payload_data) != phase.get("payload_sha256"):
                return None
        return payload

    def latest(self) -> dict[str, Any] | None:
        for version in reversed(self._snapshot_versions()):
            path = self.snapshot_dir / f"v{version:08d}.json"
            manifest = self._load_manifest_file(path)
            if manifest is not None:
                return manifest
        return None

    def restore_point(self) -> dict[str, Any] | None:
        manifest = self.latest()
        if manifest is None:
            return None
        return {
            "snapshot_version": int(manifest.get("global_version", 0)),
            "status": manifest.get("status", "unknown"),
            "completed_layers": list(manifest.get("completed_layers", [])),
            "config_hash": manifest.get("config_hash", ""),
            "calibration_hash": manifest.get("calibration_hash", ""),
            "phases": {
                name: {
                    "version": int(phase.get("version", 0)),
                    "payload_sha256": phase.get("payload_sha256", ""),
                }
                for name, phase in manifest.get("phases", {}).items()
            },
        }

    def _commit_artifact(self, phase: str, version: int, payload: dict[str, Any]) -> tuple[str, str]:
        payload_hash = sha256_json(to_builtin(payload))
        artifact_name = f"v{version:08d}-{payload_hash[:12]}"
        relative = Path("stages") / phase / artifact_name
        target = self.root / relative
        if target.is_dir() and (target / "COMMITTED").is_file():
            return relative.as_posix(), payload_hash

        parent = self.stage_dir / phase
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".v{version:08d}-", suffix=".tmp", dir=parent)
        )
        try:
            payload_path = temporary / "payload.json"
            with payload_path.open("w", encoding="utf-8") as stream:
                stream.write(canonical_json(to_builtin(payload)))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            marker_path = temporary / "COMMITTED"
            with marker_path.open("w", encoding="utf-8") as stream:
                stream.write(canonical_json({"payload_sha256": payload_hash}))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(temporary)
            if target.exists() and not (target / "COMMITTED").is_file():
                shutil.rmtree(target)
            if target.exists():
                # A complete artifact with the same version/hash is already
                # present. The manifest still decides whether it is visible.
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, target)
                _fsync_directory(parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
        return relative.as_posix(), payload_hash

    def commit_phase(
        self,
        *,
        phase: str,
        payload: dict[str, Any],
        config_hash: str,
        calibration_hash: str,
        completed_layers: Iterable[str],
        status: str,
        pending_queue: list[dict[str, Any]] | None = None,
        base: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if phase not in PHASES:
            raise ValueError(f"unknown snapshot phase: {phase}")
        payload_hash = sha256_json(to_builtin(payload))
        previous = base if base is not None else self.latest()
        previous_phase = (
            previous.get("phases", {}).get(phase) if previous is not None else None
        )
        if (
            previous is not None
            and previous_phase is not None
            and previous_phase.get("payload_sha256") == payload_hash
            and previous.get("config_hash") == config_hash
            and previous.get("calibration_hash") == calibration_hash
            and list(previous.get("completed_layers", [])) == sorted(completed_layers)
            and previous.get("pending_queue", []) == (pending_queue or [])
            and (
                status == "in_progress"
                or previous.get("status") == status
            )
        ):
            # Version comparison replaces the old "write every file again"
            # behavior. No snapshot file is touched when nothing changed.
            return previous

        version = self.next_version()
        artifact, payload_hash = self._commit_artifact(phase, version, payload)
        phases: dict[str, Any] = {}
        if previous is not None:
            phase_index = PHASES.index(phase)
            for name, value in previous.get("phases", {}).items():
                if name in PHASES and PHASES.index(name) < phase_index:
                    phases[name] = value
        phases[phase] = {
            "version": version,
            "artifact": artifact,
            "payload_sha256": payload_hash,
            "completed_layers": sorted(completed_layers),
        }
        manifest = {
            "format": SNAPSHOT_FORMAT,
            "global_version": version,
            "status": status,
            "current_phase": phase,
            "config_hash": config_hash,
            "calibration_hash": calibration_hash,
            "completed_layers": sorted(completed_layers),
            "pending_queue": pending_queue or [],
            "phases": phases,
        }
        path = self.snapshot_dir / f"v{version:08d}.json"
        _write_json(path, manifest)
        return manifest

    def artifact_payload(self, manifest: dict[str, Any], phase: str) -> dict[str, Any] | None:
        phase_value = manifest.get("phases", {}).get(phase)
        if not isinstance(phase_value, dict):
            return None
        path = self.root / str(phase_value.get("artifact", "")) / "payload.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            if sha256_json(payload) != phase_value.get("payload_sha256"):
                return None
            return payload
        except (OSError, json.JSONDecodeError):
            return None
