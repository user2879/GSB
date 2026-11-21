from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import read_wav_strict, write_wav
from .config import AnalysisConfig, canonical_json, digest_json
from .errors import E_SNAPSHOT_CORRUPT, E_SNAPSHOT_STALE, BusinessError
from .model import AudioDocument


SNAPSHOT_SCHEMA_VERSION = 1


@dataclass
class RestoredSnapshot:
    state: dict[str, Any]
    document: AudioDocument
    feature_cache: dict[str, dict[str, Any]]
    description_cache: dict[str, dict[str, Any]]
    assignment_cache: dict[str, dict[str, Any]]
    speaker_cache: dict[str, dict[str, Any]]
    manifest: dict[str, Any]
    path: Path

    @property
    def command_cursor(self) -> int:
        return int(self.state["command_cursor"])

    @property
    def overrides(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in self.state.get("overrides", {}).items()}

    @property
    def failures(self) -> list[dict[str, Any]]:
        return list(self.state.get("failures", []))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def commands_hash(commands: list[dict[str, Any]]) -> str:
    return digest_json(commands)


def _write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_bytes(payload)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class SnapshotManager:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.snapshot_root = self.output_dir / "snapshots"
        self.current_path = self.snapshot_root / "CURRENT"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

    def load(self, config: AnalysisConfig, commands: list[dict[str, Any]]) -> RestoredSnapshot | None:
        if not self.current_path.exists():
            return None
        try:
            pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
            snapshot_name = str(pointer["snapshot"])
            expected_manifest_hash = str(pointer["manifest_sha256"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise BusinessError(E_SNAPSHOT_STALE, f"snapshot pointer is not usable: {exc}") from exc
        snapshot_path = self.snapshot_root / snapshot_name
        manifest_path = snapshot_path / "manifest.json"
        try:
            manifest_bytes = manifest_path.read_bytes()
        except OSError as exc:
            raise BusinessError(E_SNAPSHOT_STALE, f"snapshot manifest is missing: {exc}", snapshot=snapshot_name) from exc
        if sha256_bytes(manifest_bytes) != expected_manifest_hash:
            raise BusinessError(E_SNAPSHOT_CORRUPT, "snapshot manifest hash does not match CURRENT", snapshot=snapshot_name)
        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as exc:
            raise BusinessError(E_SNAPSHOT_CORRUPT, f"snapshot manifest is invalid JSON: {exc}", snapshot=snapshot_name) from exc
        if int(manifest.get("schema_version", -1)) != SNAPSHOT_SCHEMA_VERSION:
            raise BusinessError(
                E_SNAPSHOT_STALE,
                "snapshot schema version is incompatible",
                snapshot_schema=manifest.get("schema_version"),
                expected_schema=SNAPSHOT_SCHEMA_VERSION,
            )
        if manifest.get("config_hash") != config.config_hash:
            raise BusinessError(
                E_SNAPSHOT_STALE,
                "configuration hash differs from the completed snapshot",
                snapshot_config_hash=manifest.get("config_hash"),
                current_config_hash=config.config_hash,
                snapshot=snapshot_name,
            )
        incoming_commands_hash = commands_hash(commands)
        if manifest.get("commands_sha256") != incoming_commands_hash:
            raise BusinessError(
                E_SNAPSHOT_STALE,
                "command stream differs from the completed snapshot",
                snapshot_commands_hash=manifest.get("commands_sha256"),
                current_commands_hash=incoming_commands_hash,
                snapshot=snapshot_name,
            )
        artifact_versions = manifest.get("artifact_versions", {})
        for artifact, expected_hash in artifact_versions.items():
            artifact_path = snapshot_path / "artifacts" / f"{artifact}.json"
            if not artifact_path.exists() or sha256_file(artifact_path) != expected_hash:
                raise BusinessError(
                    E_SNAPSHOT_CORRUPT,
                    f"snapshot artifact failed integrity verification: {artifact}",
                    snapshot=snapshot_name,
                )
        try:
            state = json.loads((snapshot_path / "state.json").read_text(encoding="utf-8"))
            feature_cache = json.loads((snapshot_path / "artifacts" / "features.json").read_text(encoding="utf-8"))
            description_cache = json.loads((snapshot_path / "artifacts" / "descriptions.json").read_text(encoding="utf-8"))
            assignment_cache = json.loads((snapshot_path / "artifacts" / "assignments.json").read_text(encoding="utf-8"))
            speaker_cache = json.loads((snapshot_path / "artifacts" / "speakers.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BusinessError(E_SNAPSHOT_CORRUPT, f"snapshot payload is unreadable: {exc}", snapshot=snapshot_name) from exc
        files = {}
        for name in state["document"]["files"]:
            audio_path = snapshot_path / "audio" / name
            if not audio_path.exists():
                raise BusinessError(E_SNAPSHOT_CORRUPT, f"snapshot audio file is missing: {name}", snapshot=snapshot_name)
            audio = read_wav_strict(audio_path)
            files[name] = audio
        document = AudioDocument.from_state(state["document"], files)
        document.refresh_frame_indices(config)
        return RestoredSnapshot(
            state=state,
            document=document,
            feature_cache=feature_cache,
            description_cache=description_cache,
            assignment_cache=assignment_cache,
            speaker_cache=speaker_cache,
            manifest=manifest,
            path=snapshot_path,
        )

    def commit(
        self,
        *,
        config: AnalysisConfig,
        document: AudioDocument,
        overrides: dict[str, int],
        failures: list[dict[str, Any]],
        command_cursor: int,
        commands: list[dict[str, Any]],
        global_version: int,
        feature_cache: dict[str, dict[str, Any]],
        description_cache: dict[str, dict[str, Any]],
        assignment_cache: dict[str, dict[str, Any]],
        speaker_cache: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.snapshot_root))
        try:
            audio_dir = staging / "audio"
            artifacts_dir = staging / "artifacts"
            audio_dir.mkdir()
            artifacts_dir.mkdir()
            for name, audio in sorted(document.files.items()):
                write_wav(audio_dir / name, audio.samples)
            artifact_hashes = {
                "features": _write_json(artifacts_dir / "features.json", feature_cache),
                "descriptions": _write_json(artifacts_dir / "descriptions.json", description_cache),
                "assignments": _write_json(artifacts_dir / "assignments.json", assignment_cache),
                "speakers": _write_json(artifacts_dir / "speakers.json", speaker_cache),
            }
            state = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "command_cursor": command_cursor,
                "global_version": global_version,
                "config_hash": config.config_hash,
                "overrides": dict(sorted(overrides.items())),
                "document": document.state_dict(),
                "failures": failures,
            }
            _write_json(staging / "state.json", state)
            completed_segments = [segment.segment_id for segment in document.ordered_segments()]
            manifest = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "engine_version": "timbre-lab/0.1.0",
                "global_version": global_version,
                "config_hash": config.config_hash,
                "command_cursor": command_cursor,
                "commands_sha256": commands_hash(commands),
                "completed_segments": completed_segments,
                "artifact_versions": artifact_hashes,
            }
            _write_json(staging / "manifest.json", manifest)
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_file():
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
            _fsync_directory(staging)
            final_name = f"v{global_version:08d}-{config.config_hash[:12]}"
            final_path = self.snapshot_root / final_name
            suffix = 0
            while final_path.exists():
                suffix += 1
                final_path = self.snapshot_root / f"{final_name}-{suffix:03d}"
            os.replace(staging, final_path)
            _fsync_directory(self.snapshot_root)
            manifest_hash = sha256_file(final_path / "manifest.json")
            pointer = {"snapshot": final_path.name, "manifest_sha256": manifest_hash}
            pointer_temporary = self.snapshot_root / f".CURRENT.{os.getpid()}.tmp"
            pointer_payload = (canonical_json(pointer) + "\n").encode("utf-8")
            with pointer_temporary.open("wb") as handle:
                handle.write(pointer_payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pointer_temporary, self.current_path)
            _fsync_directory(self.snapshot_root)
            return {
                "snapshot": final_path.name,
                "path": str(final_path),
                "manifest_sha256": manifest_hash,
                "manifest": manifest,
            }
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def cleanup_staging(self, older_than_seconds: float = 0.0) -> list[str]:
        removed: list[str] = []
        for path in sorted(self.snapshot_root.glob(".staging-*")):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed.append(str(path))
        return removed
