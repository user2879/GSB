from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

from .cache import CacheStore, _fsync_directory
from .config import canonical_json, sha256_bytes
from .errors import E_SNAPSHOT_INVALID, StudioError
from .rendering import RenderOutcome


class SnapshotStore:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.root = output_root / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.cache = CacheStore(output_root / "cache")

    def pointer_path(self, sample_id: str) -> Path:
        return self.root / sample_id / "CURRENT"

    def sample_root(self, sample_id: str) -> Path:
        path = self.root / sample_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read_pointer(self, sample_id: str) -> dict[str, Any] | None:
        pointer_path = self.pointer_path(sample_id)
        if not pointer_path.is_file():
            return None
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(pointer, dict):
            return None
        return pointer

    def commit(
        self,
        sample_id: str,
        config_sha256: str,
        completed_command_cursor: int,
        outcome: RenderOutcome,
    ) -> dict[str, Any]:
        with self._lock:
            previous = self.read_pointer(sample_id)
            desired_state = {
                "config_sha256": config_sha256,
                "completed_command_cursor": completed_command_cursor,
                "params_fingerprint": outcome.params_fingerprint,
                "final_png_sha256": outcome.final_png_sha256,
                "stage_versions": {
                    stage: summary.get("cache_key")
                    for stage, summary in outcome.stage_summaries.items()
                },
            }
            if previous is not None:
                try:
                    previous_manifest = self.verify(sample_id, previous)
                except StudioError:
                    previous_manifest = None
                if previous_manifest is not None:
                    previous_state = previous_manifest.get("desired_state")
                    if previous_state == desired_state:
                        return {
                            "status": "unchanged",
                            "global_version": previous_manifest["global_version"],
                            "generation": previous["generation"],
                            "previous_version": previous_manifest["global_version"],
                            "replacement": "detected_no_version_change",
                        }

            previous_version = int(previous.get("global_version", 0)) if previous else 0
            previous_version = max(previous_version, self._highest_generation_version(sample_id))
            global_version = previous_version + 1
            sample_root = self.sample_root(sample_id)
            temporary = Path(tempfile.mkdtemp(prefix=".tmp-snapshot-", dir=sample_root))
            try:
                stage_manifest: dict[str, Any] = {}
                for stage, summary in outcome.stage_summaries.items():
                    cache_key = summary["cache_key"]
                    source_artifact = self.cache.artifact_path(stage, cache_key)
                    destination_artifact = temporary / "stages" / stage
                    shutil.copytree(source_artifact, destination_artifact)
                    files = summary.get("files", {})
                    stage_manifest[stage] = {
                        "cache_key": cache_key,
                        "content_sha256": summary.get("content_sha256"),
                        "files": files,
                    }

                final_path = temporary / "final.png"
                final_path.write_bytes(outcome.final_png_bytes)
                final_file_sha = sha256_bytes(outcome.final_png_bytes)
                with final_path.open("rb") as handle:
                    os.fsync(handle.fileno())

                manifest: dict[str, Any] = {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "global_version": global_version,
                    "complete": True,
                    "config_sha256": config_sha256,
                    "completed_command_cursor": completed_command_cursor,
                    "params": outcome.params.as_dict(),
                    "params_fingerprint": outcome.params_fingerprint,
                    "final_png": "final.png",
                    "final_png_file_sha256": final_file_sha,
                    "final_png_sha256": outcome.final_png_sha256,
                    "stage_artifacts": stage_manifest,
                    "desired_state": desired_state,
                }
                manifest_bytes = canonical_json(manifest)
                manifest_path = temporary / "manifest.json"
                with manifest_path.open("wb") as handle:
                    handle.write(manifest_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                _fsync_directory(temporary)
                _fsync_directory(temporary / "stages")

                generation_name = f"gen-{global_version:08d}"
                generation_path = sample_root / generation_name
                if generation_path.exists():
                    raise StudioError(E_SNAPSHOT_INVALID, f"snapshot generation already exists: {generation_name}")
                os.replace(temporary, generation_path)
                _fsync_directory(sample_root)

                pointer = {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "global_version": global_version,
                    "generation": generation_name,
                    "manifest_sha256": sha256_bytes(manifest_bytes),
                }
                self._write_pointer_atomic(sample_id, pointer)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

            return {
                "status": "committed",
                "global_version": global_version,
                "generation": generation_name,
                "previous_version": previous_version,
                "replacement": "atomic_pointer_switch",
            }

    def _write_pointer_atomic(self, sample_id: str, pointer: Mapping[str, Any]) -> None:
        pointer_path = self.pointer_path(sample_id)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".CURRENT-", dir=pointer_path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_json(dict(pointer)))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, pointer_path)
            _fsync_directory(pointer_path.parent)
        except Exception:
            try:
                os.unlink(name)
            except OSError:
                pass
            raise

    def _highest_generation_version(self, sample_id: str) -> int:
        sample_root = self.sample_root(sample_id)
        highest = 0
        for path in sample_root.glob("gen-*"):
            if not path.is_dir():
                continue
            try:
                highest = max(highest, int(path.name.removeprefix("gen-")))
            except ValueError:
                continue
        return highest

    def verify(self, sample_id: str, pointer: Mapping[str, Any] | None = None) -> dict[str, Any]:
        pointer = dict(pointer) if pointer is not None else self.read_pointer(sample_id)
        if pointer is None:
            raise StudioError(E_SNAPSHOT_INVALID, f"no CURRENT snapshot pointer for {sample_id}")
        generation_name = pointer.get("generation")
        if not isinstance(generation_name, str) or Path(generation_name).name != generation_name:
            raise StudioError(E_SNAPSHOT_INVALID, "snapshot pointer generation is invalid")
        generation_path = self.sample_root(sample_id) / generation_name
        manifest_path = generation_path / "manifest.json"
        if not manifest_path.is_file():
            raise StudioError(E_SNAPSHOT_INVALID, f"snapshot manifest missing for {sample_id}")
        manifest_bytes = manifest_path.read_bytes()
        if sha256_bytes(manifest_bytes) != pointer.get("manifest_sha256"):
            raise StudioError(E_SNAPSHOT_INVALID, f"snapshot manifest hash mismatch for {sample_id}")
        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StudioError(E_SNAPSHOT_INVALID, f"snapshot manifest unreadable for {sample_id}") from exc
        if manifest.get("complete") is not True:
            raise StudioError(E_SNAPSHOT_INVALID, f"snapshot is incomplete for {sample_id}")
        if manifest.get("global_version") != pointer.get("global_version"):
            raise StudioError(E_SNAPSHOT_INVALID, f"snapshot version mismatch for {sample_id}")

        final_relative = manifest.get("final_png")
        if not isinstance(final_relative, str) or Path(final_relative).name != final_relative:
            raise StudioError(E_SNAPSHOT_INVALID, "snapshot final PNG path is invalid")
        final_path = generation_path / final_relative
        if not final_path.is_file():
            raise StudioError(E_SNAPSHOT_INVALID, "snapshot final PNG is missing")
        if sha256_bytes(final_path.read_bytes()) != manifest.get("final_png_file_sha256"):
            raise StudioError(E_SNAPSHOT_INVALID, "snapshot final PNG hash mismatch")

        stage_artifacts = manifest.get("stage_artifacts")
        if not isinstance(stage_artifacts, dict):
            raise StudioError(E_SNAPSHOT_INVALID, "snapshot stage artifact manifest is invalid")
        for stage, artifact in sorted(stage_artifacts.items()):
            if not isinstance(artifact, dict):
                raise StudioError(E_SNAPSHOT_INVALID, f"snapshot stage metadata invalid: {stage}")
            artifact_root = generation_path / "stages" / stage
            files = artifact.get("files")
            if not isinstance(files, dict):
                raise StudioError(E_SNAPSHOT_INVALID, f"snapshot stage files invalid: {stage}")
            for filename, expected_hash in sorted(files.items()):
                if Path(filename).name != filename:
                    raise StudioError(E_SNAPSHOT_INVALID, f"snapshot stage filename invalid: {stage}")
                file_path = artifact_root / filename
                if not file_path.is_file():
                    raise StudioError(E_SNAPSHOT_INVALID, f"snapshot stage file missing: {stage}/{filename}")
                if sha256_bytes(file_path.read_bytes()) != expected_hash:
                    raise StudioError(E_SNAPSHOT_INVALID, f"snapshot stage file hash mismatch: {stage}/{filename}")
        return manifest

    def recover(self, sample_id: str, restore_directory: Path | None = None) -> dict[str, Any]:
        pointer = self.read_pointer(sample_id)
        if pointer is None:
            return {"sample_id": sample_id, "status": "no_valid_snapshot"}
        try:
            manifest = self.verify(sample_id, pointer)
        except StudioError as error:
            return {
                "sample_id": sample_id,
                "status": "no_valid_snapshot",
                "error": error.as_failure(),
            }
        generation_path = self.sample_root(sample_id) / pointer["generation"]
        result = {
            "sample_id": sample_id,
            "status": "recovered",
            "recovered_version": manifest["global_version"],
            "generation": pointer["generation"],
            "completed_command_cursor": manifest["completed_command_cursor"],
            "config_sha256": manifest["config_sha256"],
            "final_png_sha256": manifest["final_png_sha256"],
        }
        if restore_directory is not None:
            restore_directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(generation_path / manifest["final_png"], restore_directory / "final.png")
            (restore_directory / "recovery.json").write_bytes(canonical_json(result))
            result["restored_to"] = str(restore_directory)
        return result

    def cleanup_temporary_directories(self) -> list[str]:
        removed: list[str] = []
        if not self.root.is_dir():
            return removed
        for path in sorted(self.root.rglob(".tmp-snapshot-*")):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed.append(str(path))
        return removed
