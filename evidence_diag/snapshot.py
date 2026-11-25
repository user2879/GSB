from __future__ import annotations

import fcntl
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .engine import DiagnosticEngine
from .util import (
    atomic_write_text,
    canonical_json,
    load_json,
    sha256_bytes,
    sha256_json,
    write_json,
)


class SnapshotError(RuntimeError):
    pass


class VersionConflict(SnapshotError):
    pass


@dataclass
class SnapshotBundle:
    state: dict[str, Any]
    result: dict[str, Any]
    manifest: dict[str, Any]
    directory: Path

    def engine(self) -> DiagnosticEngine:
        return DiagnosticEngine.from_state(self.state)


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.versions_dir = self.root / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / ".snapshot.lock"
        self.current_path = self.root / "CURRENT"

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_pointer(self) -> dict[str, Any] | None:
        if not self.current_path.exists():
            return None
        try:
            pointer = load_json(self.current_path)
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        if not isinstance(pointer, dict) or "version" not in pointer:
            return None
        return pointer

    def current_version(self) -> int:
        pointer = self._read_pointer()
        return int(pointer["version"]) if pointer else 0

    def _version_dir(self, version: int) -> Path:
        return self.versions_dir / f"v{version:08d}"

    def _crash_hook(self, stage: str, command_cursor: int) -> None:
        target = os.environ.get("EVIDENCE_DIAG_CRASH_AT", "")
        if not target:
            return
        parts = target.split(":")
        if parts[0] != stage:
            return
        if len(parts) > 1 and parts[1].isdigit():
            if int(parts[1]) != command_cursor:
                return
        os._exit(87)

    def commit(
        self,
        *,
        state: Mapping[str, Any],
        result: Mapping[str, Any],
        execution: Mapping[str, Any],
        expected_parent_version: int,
        base_config_hash: str,
        base_evidence_fingerprint: str,
        command_prefix_hash: str,
    ) -> dict[str, Any]:
        with self._lock():
            pointer = self._read_pointer()
            current = int(pointer["version"]) if pointer else 0
            if current != expected_parent_version:
                raise VersionConflict(
                    f"expected parent snapshot version {expected_parent_version}, found {current}"
                )
            version = current + 1
            final_dir = self._version_dir(version)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            temp_dir = self.root / f".tmp-{version:08d}-{os.getpid()}-{uuid.uuid4().hex}"
            temp_dir.mkdir(parents=True, exist_ok=False)
            try:
                state_without_caches = {
                    key: value
                    for key, value in state.items()
                    if key not in ("caches", "last_result")
                }
                write_json(temp_dir / "state.json", state_without_caches)
                write_json(temp_dir / "result.json", result)
                write_json(temp_dir / "execution.json", execution)

                stage_dir = temp_dir / "stages"
                cache_dir = temp_dir / "caches"
                stage_dir.mkdir()
                cache_dir.mkdir()
                stage_payloads: dict[str, Any] = {
                    "recall": {
                        turn["turn_id"]: {
                            "cache_key": turn["recall_key"],
                            "payload": turn["recall"],
                        }
                        for turn in result["turns"]
                    },
                    "dedupe": {
                        turn["turn_id"]: {
                            "cache_key": turn["dedupe_key"],
                            "payload": turn["dedupe"],
                        }
                        for turn in result["turns"]
                    },
                    "topic": {
                        "topics": result["topics"],
                        "evidence_attribution": result[
                            "evidence_attribution"
                        ],
                        "boundary_keys": result["boundary_keys"],
                    },
                    "confidence": {
                        turn["turn_id"]: {
                            "cache_key": turn["confidence_key"],
                            "payload": turn["confidence"],
                            "status": turn["status"],
                        }
                        for turn in result["turns"]
                    },
                }
                for stage, payload in stage_payloads.items():
                    write_json(stage_dir / f"{stage}.json", payload)
                for stage, payload in state.get("caches", {}).items():
                    write_json(cache_dir / f"{stage}.json", payload)

                self._crash_hook("after-stage-write", int(result["command_cursor"]))

                files: dict[str, str] = {}
                for path in sorted(
                    item for item in temp_dir.rglob("*") if item.is_file()
                ):
                    relative = str(path.relative_to(temp_dir))
                    files[relative] = sha256_bytes(path.read_bytes())
                manifest = {
                    "schema_version": 1,
                    "global_version": version,
                    "engine_version": result["version"],
                    "command_cursor": result["command_cursor"],
                    "config_hash": result["config_hash"],
                    "evidence_library_fingerprint": result[
                        "evidence_library_fingerprint"
                    ],
                    "base_config_hash": base_config_hash,
                    "base_evidence_fingerprint": base_evidence_fingerprint,
                    "command_prefix_hash": command_prefix_hash,
                    "stage_versions": result["stage_versions"],
                    "fixed_point": result["fixed_point"],
                    "files": files,
                    "manifest_hash": "",
                }
                unsigned = dict(manifest)
                unsigned.pop("manifest_hash")
                manifest["manifest_hash"] = sha256_json(unsigned)
                write_json(temp_dir / "manifest.json", manifest)
                os.replace(temp_dir, final_dir)
                self._crash_hook("after-directory-publish", int(result["command_cursor"]))
                pointer_payload = {
                    "version": version,
                    "directory": str(final_dir.relative_to(self.root)),
                    "manifest_hash": manifest["manifest_hash"],
                }
                atomic_write_text(
                    self.current_path,
                    canonical_json(pointer_payload) + "\n",
                )
                return manifest
            except BaseException:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                raise

    def _load_manifest_dir(
        self, directory: Path, *, verify: bool = True
    ) -> SnapshotBundle:
        try:
            manifest = load_json(directory / "manifest.json")
            state_raw = load_json(directory / "state.json")
            result = load_json(directory / "result.json")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"cannot load snapshot {directory}: {exc}") from exc
        if verify:
            unsigned = dict(manifest)
            expected_manifest_hash = unsigned.pop("manifest_hash", "")
            actual_manifest_hash = sha256_json(unsigned)
            if expected_manifest_hash != actual_manifest_hash:
                raise SnapshotError(
                    f"manifest hash mismatch in {directory}"
                )
            for relative, expected in manifest.get("files", {}).items():
                path = directory / relative
                if not path.exists():
                    raise SnapshotError(f"snapshot file missing: {relative}")
                if sha256_bytes(path.read_bytes()) != expected:
                    raise SnapshotError(
                        f"snapshot file hash mismatch: {relative}"
                    )
        caches: dict[str, Any] = {}
        cache_dir = directory / "caches"
        if cache_dir.exists():
            for path in sorted(cache_dir.glob("*.json")):
                caches[path.stem] = load_json(path)
        state = dict(state_raw)
        state["caches"] = caches
        state["last_result"] = result
        return SnapshotBundle(
            state=state,
            result=result,
            manifest=manifest,
            directory=directory,
        )

    def load_current(self, *, verify: bool = True) -> SnapshotBundle | None:
        pointer = self._read_pointer()
        if pointer is None:
            return None
        directory = self.root / str(pointer["directory"])
        return self._load_manifest_dir(directory, verify=verify)

    def load_latest_coherent(self, *, verify: bool = True) -> SnapshotBundle | None:
        try:
            return self.load_current(verify=verify)
        except SnapshotError:
            # A published directory is not a commit until CURRENT atomically
            # points at it. Corrupt or missing CURRENT therefore forces a full
            # recomputation rather than guessing an orphaned version.
            return None

    def recover(
        self,
        *,
        base_config_hash: str,
        base_evidence_fingerprint: str,
        command_prefix_hash: str,
        expected_cursor: int | None = None,
    ) -> dict[str, Any]:
        bundle = self.load_latest_coherent()
        if bundle is None:
            return {
                "decision": "full_recompute",
                "reason": "no_coherent_snapshot",
                "bundle": None,
            }
        manifest = bundle.manifest
        if manifest.get("base_config_hash") != base_config_hash:
            reason = "base_config_hash_mismatch"
        elif manifest.get("base_evidence_fingerprint") != base_evidence_fingerprint:
            reason = "base_evidence_fingerprint_mismatch"
        elif manifest.get("command_prefix_hash") != command_prefix_hash:
            reason = "command_prefix_mismatch"
        elif expected_cursor is not None and manifest.get("command_cursor") != expected_cursor:
            reason = "command_cursor_mismatch"
        else:
            return {
                "decision": "resume",
                "reason": "version_config_evidence_command_match",
                "bundle": bundle,
            }
        return {"decision": "full_recompute", "reason": reason, "bundle": bundle}

    def cleanup_temp(self) -> int:
        removed = 0
        for path in self.root.glob(".tmp-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed

    def reset(self) -> None:
        with self._lock():
            if self.versions_dir.exists():
                shutil.rmtree(self.versions_dir)
            self.versions_dir.mkdir(parents=True, exist_ok=True)
            if self.current_path.exists():
                self.current_path.unlink()
            self.cleanup_temp()
