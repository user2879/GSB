"""Generation-based, atomically committed snapshots."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .canonical import (
    atomic_write_json,
    atomic_write_text,
    b64_decode,
    canonical_json,
    sha256_bytes,
)
from .errors import E_SNAPSHOT_CORRUPT, E_SNAPSHOT_UNCOMMITTED, E_SNAPSHOT_VERSION, fail

SNAPSHOT_SCHEMA = "mmfm-snapshot-v1"


@dataclass
class CommittedSnapshot:
    root: Path
    generation: str
    manifest: dict[str, Any]
    state: dict[str, Any]
    manifest_path: Path


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pointer_path = self.root / "COMMITTED"
        self.lock_path = self.root / ".lock"
        self.commits_this_process = 0
        self.fault_after = int(os.environ.get("MMFM_FAULT_AFTER_COMMIT", "0") or "0")

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_pointer(self) -> dict[str, Any] | None:
        if not self.pointer_path.exists():
            return None
        try:
            pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(E_SNAPSHOT_CORRUPT, "COMMITTED snapshot pointer is unreadable", error=str(exc))
        if not isinstance(pointer, dict) or pointer.get("schema") != SNAPSHOT_SCHEMA:
            fail(E_SNAPSHOT_CORRUPT, "COMMITTED snapshot pointer has an unsupported schema")
        return pointer

    def load_committed(self) -> CommittedSnapshot | None:
        pointer = self._read_pointer()
        if pointer is None:
            return None
        generation = pointer.get("generation")
        if not isinstance(generation, str) or "/" in generation or generation.startswith("."):
            fail(E_SNAPSHOT_CORRUPT, "snapshot generation name is invalid", generation=generation)
        generation_path = self.root / generation
        manifest_path = generation_path / "manifest.json"
        state_path = generation_path / "state.json"
        if not manifest_path.exists() or not state_path.exists():
            fail(
                E_SNAPSHOT_CORRUPT,
                "committed generation is missing manifest or state",
                generation=generation,
            )
        manifest_bytes = manifest_path.read_bytes()
        if pointer.get("manifest_sha256") != sha256_bytes(manifest_bytes):
            fail(E_SNAPSHOT_CORRUPT, "committed manifest hash check failed", generation=generation)
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            state_bytes = state_path.read_bytes()
            state = json.loads(state_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            fail(E_SNAPSHOT_CORRUPT, "committed snapshot JSON is invalid", generation=generation, error=str(exc))
        if manifest.get("state_sha256") != sha256_bytes(state_bytes):
            fail(E_SNAPSHOT_CORRUPT, "committed state hash check failed", generation=generation)
        if manifest.get("generation") != generation or pointer.get("generation") != generation:
            fail(E_SNAPSHOT_VERSION, "snapshot generation identifiers disagree", generation=generation)
        if manifest.get("global_version") != pointer.get("global_version"):
            fail(E_SNAPSHOT_VERSION, "snapshot global version identifiers disagree", generation=generation)
        workspace = state.get("workspace", {})
        if workspace:
            from .workspace import Workspace

            try:
                parsed_workspace = Workspace.from_serialised(workspace)
            except Exception as exc:
                if hasattr(exc, "code"):
                    raise
                fail(E_SNAPSHOT_CORRUPT, "snapshot workspace cannot be reconstructed", error=str(exc))
            if parsed_workspace.config_hash != manifest.get("config_hash"):
                fail(E_SNAPSHOT_VERSION, "snapshot config hash differs from its workspace")
            if parsed_workspace.input_hash != manifest.get("input_hash"):
                fail(E_SNAPSHOT_VERSION, "snapshot input hash differs from its workspace")
        return CommittedSnapshot(
            root=self.root,
            generation=generation,
            manifest=manifest,
            state=state,
            manifest_path=manifest_path,
        )

    def _next_version(self) -> int:
        highest = 0
        pointer = self._read_pointer()
        if pointer:
            highest = int(pointer.get("global_version", 0))
        for path in self.root.iterdir():
            if not path.is_dir() or not path.name.startswith("gen-"):
                continue
            manifest_path = path / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                highest = max(highest, int(manifest.get("global_version", 0)))
            except Exception:
                continue
        return highest + 1

    def _write_outputs(self, generation_path: Path, state: dict[str, Any]) -> None:
        outputs = state.get("outputs", {})
        if not isinstance(outputs, dict):
            return
        for frame_key, frame_output in sorted(outputs.items(), key=lambda item: str(item[0])):
            try:
                frame_number = int(frame_key)
            except (TypeError, ValueError):
                continue
            frame_dir = generation_path / "outputs" / f"frame_{frame_number:04d}"
            if "png_b64" in frame_output:
                (frame_dir).mkdir(parents=True, exist_ok=True)
                atomic_write_bytes_local(frame_dir / "frame.png", b64_decode(frame_output["png_b64"]))
            for chunk in frame_output.get("chunks", []):
                chunk_dir = frame_dir
                chunk_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes_local(
                    chunk_dir / f"chunk_{int(chunk['index']):02d}.png",
                    b64_decode(chunk["png_b64"]),
                )

    def commit(
        self,
        state: dict[str, Any],
        *,
        config_hash: str,
        input_hash: str,
        run_id: str,
    ) -> dict[str, Any]:
        with self._lock():
            previous_pointer = self._read_pointer()
            version = self._next_version()
            generation = f"gen-{version:06d}"
            temporary = self.root / f".tmp-{generation}-{uuid.uuid4().hex}"
            temporary.mkdir(parents=True, exist_ok=False)
            try:
                state_to_write = json.loads(json.dumps(state))
                state_to_write["global_version"] = version
                state_to_write["generation"] = generation
                state_bytes = (canonical_json(state_to_write) + "\n").encode("utf-8")
                atomic_write_bytes_local(temporary / "state.json", state_bytes)
                self._write_outputs(temporary, state_to_write)
                manifest = {
                    "schema": SNAPSHOT_SCHEMA,
                    "generation": generation,
                    "global_version": version,
                    "config_hash": config_hash,
                    "input_hash": input_hash,
                    "run_id": run_id,
                    "recipe_version": state_to_write.get("recipe_version", "unknown"),
                    "state_sha256": sha256_bytes(state_bytes),
                    "config_version_changed": bool(
                        previous_pointer and previous_pointer.get("config_hash") != config_hash
                    ),
                    "parent_generation": previous_pointer.get("generation") if previous_pointer else None,
                }
                manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
                atomic_write_bytes_local(temporary / "manifest.json", manifest_bytes)
                directory_fd = os.open(temporary, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                os.rename(temporary, self.root / generation)
                root_fd = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(root_fd)
                finally:
                    os.close(root_fd)
                pointer = {
                    "schema": SNAPSHOT_SCHEMA,
                    "generation": generation,
                    "global_version": version,
                    "config_hash": config_hash,
                    "input_hash": input_hash,
                    "manifest_sha256": sha256_bytes(manifest_bytes),
                }
                atomic_write_json(self.pointer_path, pointer)
            except Exception:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)
                raise
            self.commits_this_process += 1
            if self.fault_after and self.commits_this_process >= self.fault_after:
                os._exit(137)
            return manifest

    def uncommitted_report(self) -> dict[str, Any]:
        pointer = self._read_pointer()
        committed_generation = pointer.get("generation") if pointer else None
        committed_version = int(pointer.get("global_version", 0)) if pointer else 0
        uncommitted: list[dict[str, Any]] = []
        for path in sorted(self.root.iterdir()):
            if path.name.startswith(".tmp-"):
                uncommitted.append(
                    {
                        "name": path.name,
                        "code": E_SNAPSHOT_UNCOMMITTED,
                        "reason": "temporary generation directory was never committed",
                    }
                )
            elif path.is_dir() and path.name.startswith("gen-") and path.name != committed_generation:
                manifest_path = path / "manifest.json"
                if not manifest_path.exists():
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                # Older generations are historical commits.  Only a newer
                # generation without a matching pointer is an orphan.
                if int(manifest.get("global_version", 0)) > committed_version:
                    uncommitted.append(
                        {
                            "name": path.name,
                            "code": E_SNAPSHOT_UNCOMMITTED,
                            "reason": "generation directory exists but COMMITTED did not advance",
                        }
                    )
        return {
            "committed_generation": committed_generation,
            "committed_global_version": committed_version,
            "uncommitted": uncommitted,
            "ignored_for_recovery": True,
        }


def atomic_write_bytes_local(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
