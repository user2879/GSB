"""Atomic snapshot, operation-log, and patch-cache persistence."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import (
    SCHEMA_VERSION,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    content_hash,
    read_json_bytes,
    sha256_bytes,
    verify_content_hash,
    with_content_hash,
)
from .errors import DomainError, ErrorCode
from .geometry import PatchGeometry
from .history import EditOperation
from .state import GeometryState


MANIFEST_SCHEMA = "cad-surface.manifest.v1"
COMMIT_SCHEMA = "cad-surface.commit.v1"
STATE_FILE_SCHEMA = "cad-surface.state-file.v1"
CACHE_SCHEMA = "cad-surface.patch-cache.v1"
OPERATION_SCHEMA = "cad-surface.operation.v1"
RECOVERY_SCHEMA = "cad-surface.recovery.v1"
COMMIT_DIR_PATTERN = re.compile(r"^v(\d{6,})$")


@dataclass(frozen=True)
class PersistenceReport:
    status: str
    recovered_version: int
    recovered_state_hash: str
    orphan_paths: tuple[str, ...]
    manifest_was_closed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "recovered_version": self.recovered_version,
            "recovered_state_hash": self.recovered_state_hash,
            "orphan_paths": list(self.orphan_paths),
            "manifest_was_closed": self.manifest_was_closed,
        }


@dataclass(frozen=True)
class LoadedCommit:
    state: GeometryState
    operation: EditOperation | None
    metadata: dict[str, Any]


class PersistentStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.commits_dir = self.root / "commits"
        self.staging_dir = self.root / "staging"
        self.manifest_path = self.root / "MANIFEST.json"
        self.recovery_path = self.root / "RECOVERY.json"
        self.session_report_path = self.root / "SESSION_REPORT.json"
        self.revision = 0
        self._manifest_hash: str | None = None

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def initialize(self, state: GeometryState) -> None:
        if self.exists():
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "persistent workspace already has a manifest",
                root=str(self.root),
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.commits_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        commit_meta = self._write_commit_bundle(state, None)
        self._finalize_staging(commit_meta)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "model_schema_version": SCHEMA_VERSION,
            "revision": 1,
            "current_version": state.version,
            "current_state_hash": state.state_hash,
            "undo_stack": [],
            "redo_stack": [],
            "closed": False,
        }
        atomic_write_json(self.manifest_path, with_content_hash(manifest))
        self.revision = 1
        self._manifest_hash = self._read_manifest()["content_hash"]

    def open(self) -> PersistenceReport:
        if not self.exists():
            has_artifacts = any(self.commits_dir.glob("v*")) if self.commits_dir.exists() else False
            has_artifacts = has_artifacts or (self.staging_dir.exists() and any(self.staging_dir.iterdir()))
            if has_artifacts:
                raise DomainError(
                    ErrorCode.SNAPSHOT_STALE,
                    "manifest is missing after an interrupted persistence operation",
                    root=str(self.root),
                )
            raise FileNotFoundError(self.manifest_path)
        manifest = self._read_manifest()
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise DomainError(
                ErrorCode.SCHEMA_VERSION,
                "unsupported manifest schema version",
                expected=MANIFEST_SCHEMA,
                actual=manifest.get("schema_version"),
            )
        if manifest.get("model_schema_version") != SCHEMA_VERSION:
            raise DomainError(
                ErrorCode.SCHEMA_VERSION,
                "unsupported model schema version in manifest",
                expected=SCHEMA_VERSION,
                actual=manifest.get("model_schema_version"),
            )
        self.revision = int(manifest["revision"])
        self._manifest_hash = str(manifest["content_hash"])
        closed = bool(manifest.get("closed", False))
        orphans: list[str] = []
        for entry in sorted(self.root.glob(".*.tmp-*")):
            orphans.append(str(entry))
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        for entry in sorted(self.staging_dir.glob("*")) if self.staging_dir.exists() else []:
            orphans.append(str(entry))
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        known_max = max(
            [int(manifest["current_version"])]
            + [int(item) for item in manifest.get("undo_stack", [])]
            + [int(item) for item in manifest.get("redo_stack", [])]
        )
        for entry in sorted(self.commits_dir.glob("v*")) if self.commits_dir.exists() else []:
            match = COMMIT_DIR_PATTERN.match(entry.name)
            if match and int(match.group(1)) > known_max:
                orphans.append(str(entry))
                shutil.rmtree(entry)
        report = PersistenceReport(
            status="CONTROLLED_EXIT" if closed else "FORCED_TERMINATION",
            recovered_version=int(manifest["current_version"]),
            recovered_state_hash=str(manifest["current_state_hash"]),
            orphan_paths=tuple(orphans),
            manifest_was_closed=closed,
        )
        loaded = self.load_state(report.recovered_version, expected_state_hash=report.recovered_state_hash)
        if loaded.state.state_hash != report.recovered_state_hash:
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "recovered state hash differs from manifest",
                expected=report.recovered_state_hash,
                actual=loaded.state.state_hash,
            )
        atomic_write_json(
            self.recovery_path,
            with_content_hash(
                {
                    "schema_version": RECOVERY_SCHEMA,
                    "status": report.status,
                    "recovered_version": report.recovered_version,
                    "recovered_state_hash": report.recovered_state_hash,
                    "orphan_paths": list(report.orphan_paths),
                    "manifest_was_closed": report.manifest_was_closed,
                    "exit_was_synthesized": False,
                }
            ),
        )
        return report

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise DomainError(ErrorCode.SNAPSHOT_STALE, "manifest is missing", path=str(self.manifest_path))
        manifest, _ = read_json_bytes(self.manifest_path)
        verify_content_hash(manifest)
        return manifest

    def manifest(self) -> dict[str, Any]:
        return dict(self._read_manifest())

    def load_state(self, version: int, *, expected_state_hash: str | None = None) -> LoadedCommit:
        version = int(version)
        commit_dir = self._commit_dir(version)
        commit_path = commit_dir / "commit.json"
        if not commit_path.exists():
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "committed snapshot is missing",
                version=version,
                path=str(commit_path),
            )
        commit, _ = read_json_bytes(commit_path)
        verify_content_hash(commit)
        if commit.get("schema_version") != COMMIT_SCHEMA:
            raise DomainError(
                ErrorCode.SCHEMA_VERSION,
                "unsupported commit schema version",
                expected=COMMIT_SCHEMA,
                actual=commit.get("schema_version"),
            )
        if int(commit.get("version", -1)) != version:
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "commit directory version and payload disagree",
                directory_version=version,
                payload_version=commit.get("version"),
            )
        state_path = commit_dir / str(commit["state_file"])
        state_bytes = state_path.read_bytes()
        actual_state_file_hash = sha256_bytes(state_bytes)
        if actual_state_file_hash != commit.get("state_file_hash"):
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "state snapshot file hash mismatch",
                version=version,
                expected=commit.get("state_file_hash"),
                actual=actual_state_file_hash,
            )
        state_envelope, _ = read_json_bytes(state_path)
        verify_content_hash(state_envelope)
        if state_envelope.get("schema_version") != STATE_FILE_SCHEMA:
            raise DomainError(
                ErrorCode.SCHEMA_VERSION,
                "unsupported state file schema version",
                expected=STATE_FILE_SCHEMA,
                actual=state_envelope.get("schema_version"),
            )
        state_payload = state_envelope.get("payload", {})
        cache_entries = commit.get("cache_index", {})
        geometries: dict[str, PatchGeometry] = {}
        for patch_key, entry in sorted(cache_entries.items()):
            cache_path = commit_dir / str(entry["file"])
            try:
                cache_bytes = cache_path.read_bytes()
            except OSError as exc:
                raise DomainError(
                    ErrorCode.CACHE_HASH,
                    "patch cache file is missing",
                    patch_key=patch_key,
                    version=version,
                    path=str(cache_path),
                ) from exc
            actual_cache_file_hash = sha256_bytes(cache_bytes)
            if actual_cache_file_hash != entry.get("file_hash"):
                raise DomainError(
                    ErrorCode.CACHE_HASH,
                    "patch cache file hash mismatch",
                    patch_key=patch_key,
                    version=version,
                    expected=entry.get("file_hash"),
                    actual=actual_cache_file_hash,
                )
            cache_envelope, _ = read_json_bytes(cache_path)
            try:
                verify_content_hash(cache_envelope)
            except DomainError as exc:
                raise DomainError(
                    ErrorCode.CACHE_HASH,
                    "patch cache content hash mismatch",
                    patch_key=patch_key,
                    version=version,
                    details=exc.details,
                ) from exc
            if cache_envelope.get("content_hash") != entry.get("cache_hash"):
                raise DomainError(
                    ErrorCode.CACHE_HASH,
                    "patch cache envelope hash differs from the commit index",
                    patch_key=patch_key,
                    version=version,
                    expected=entry.get("cache_hash"),
                    actual=cache_envelope.get("content_hash"),
                )
            if cache_envelope.get("schema_version") != CACHE_SCHEMA:
                raise DomainError(
                    ErrorCode.SCHEMA_VERSION,
                    "unsupported patch cache schema version",
                    patch_key=patch_key,
                )
            if int(cache_envelope.get("version", -1)) != version or str(cache_envelope.get("patch_key")) != str(patch_key):
                raise DomainError(
                    ErrorCode.CACHE_HASH,
                    "patch cache index does not match its envelope",
                    patch_key=patch_key,
                    version=version,
                )
            geometry = PatchGeometry.from_cache_payload(cache_envelope["payload"])
            if geometry.geometry_hash != entry.get("geometry_hash"):
                raise DomainError(
                    ErrorCode.CACHE_HASH,
                    "patch cache geometry hash mismatch",
                    patch_key=patch_key,
                    expected=entry.get("geometry_hash"),
                    actual=geometry.geometry_hash,
                )
            geometries[patch_key] = geometry
        state = GeometryState.from_payload(state_payload, geometries)
        if state.state_hash != commit.get("state_hash"):
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "loaded state hash differs from commit record",
                expected=commit.get("state_hash"),
                actual=state.state_hash,
            )
        if expected_state_hash is not None and state.state_hash != expected_state_hash:
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "requested snapshot is stale or has been replaced",
                version=version,
                expected=expected_state_hash,
                actual=state.state_hash,
            )
        operation = self._load_operation(commit_dir, commit)
        return LoadedCommit(state=state, operation=operation, metadata=commit)

    def _load_operation(self, commit_dir: Path, commit: Mapping[str, Any]) -> EditOperation | None:
        operation_entry = commit.get("operation")
        if not operation_entry:
            return None
        operation_path = commit_dir / str(operation_entry["file"])
        operation_bytes = operation_path.read_bytes()
        if sha256_bytes(operation_bytes) != operation_entry.get("file_hash"):
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "operation log file hash mismatch",
                version=commit.get("version"),
            )
        envelope, _ = read_json_bytes(operation_path)
        verify_content_hash(envelope)
        if envelope.get("schema_version") != OPERATION_SCHEMA:
            raise DomainError(
                ErrorCode.SCHEMA_VERSION,
                "unsupported operation schema version",
                actual=envelope.get("schema_version"),
            )
        operation = EditOperation.from_dict(envelope["payload"])
        if operation.op_version != int(commit["version"]) or operation.parent_version != int(commit["parent_version"]):
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "operation version links disagree with commit",
                version=commit.get("version"),
            )
        return operation

    def _commit_dir(self, version: int) -> Path:
        return self.commits_dir / f"v{int(version):06d}"

    def _write_commit_bundle(self, state: GeometryState, operation: EditOperation | None) -> dict[str, Any]:
        staging_path = Path(tempfile.mkdtemp(prefix=f".staging-v{state.version:06d}-", dir=self.staging_dir))
        patches_dir = staging_path / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)
        state_envelope = with_content_hash(
            {
                "schema_version": STATE_FILE_SCHEMA,
                "version": state.version,
                "payload": state.snapshot_payload(),
            }
        )
        state_bytes = canonical_json_bytes(state_envelope)
        state_path = staging_path / "state.json"
        atomic_write_bytes(state_path, state_bytes)
        cache_index: dict[str, Any] = {}
        for patch_key, geometry in sorted(state.geometries.items()):
            cache_envelope = with_content_hash(
                {
                    "schema_version": CACHE_SCHEMA,
                    "version": state.version,
                    "patch_key": patch_key,
                    "payload": geometry.cache_payload(),
                }
            )
            cache_bytes = canonical_json_bytes(cache_envelope)
            relative_path = Path("patches") / f"{patch_key}.json"
            cache_path = staging_path / relative_path
            atomic_write_bytes(cache_path, cache_bytes)
            cache_index[patch_key] = {
                "file": relative_path.as_posix(),
                "file_hash": sha256_bytes(cache_bytes),
                "cache_hash": cache_envelope["content_hash"],
                "geometry_hash": geometry.geometry_hash,
            }
        operation_meta: dict[str, Any] | None = None
        if operation is not None:
            operation_envelope = with_content_hash(
                {
                    "schema_version": OPERATION_SCHEMA,
                    "payload": operation.to_dict(),
                }
            )
            operation_bytes = canonical_json_bytes(operation_envelope)
            operation_path = staging_path / "operation.json"
            atomic_write_bytes(operation_path, operation_bytes)
            operation_meta = {
                "file": "operation.json",
                "file_hash": sha256_bytes(operation_bytes),
                "operation_hash": operation.content_hash(),
            }
        commit_body = {
            "schema_version": COMMIT_SCHEMA,
            "version": state.version,
            "parent_version": state.parent_version,
            "state_file": "state.json",
            "state_file_hash": sha256_bytes(state_bytes),
            "state_hash": state.state_hash,
            "geometry_hash": state.geometry_hash,
            "normals_hash": state.normals_hash,
            "display_hashes": state.display_hashes,
            "cache_index": cache_index,
            "operation": operation_meta,
        }
        commit = with_content_hash(commit_body)
        atomic_write_json(staging_path / "commit.json", commit)
        return {"staging_path": str(staging_path), "version": state.version, "commit": commit}

    def _finalize_staging(self, bundle: Mapping[str, Any]) -> Path:
        staging_path = Path(bundle["staging_path"])
        final_path = self._commit_dir(int(bundle["version"]))
        if final_path.exists():
            shutil.rmtree(staging_path)
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "target commit directory already exists",
                version=bundle["version"],
                path=str(final_path),
            )
        os.replace(staging_path, final_path)
        return final_path

    def persist_commit(
        self,
        state: GeometryState,
        operation: EditOperation,
        *,
        undo_stack: Sequence[int],
        redo_stack: Sequence[int],
    ) -> None:
        self._assert_manifest_current()
        bundle = self._write_commit_bundle(state, operation)
        self._finalize_staging(bundle)
        self._write_manifest(
            {
                "schema_version": MANIFEST_SCHEMA,
                "model_schema_version": SCHEMA_VERSION,
                "revision": self.revision + 1,
                "current_version": state.version,
                "current_state_hash": state.state_hash,
                "undo_stack": [int(item) for item in undo_stack],
                "redo_stack": [int(item) for item in redo_stack],
                "closed": False,
            }
        )

    def save_session(
        self,
        *,
        current_version: int,
        current_state_hash: str,
        undo_stack: Sequence[int],
        redo_stack: Sequence[int],
        closed: bool,
    ) -> None:
        self._assert_manifest_current()
        self._write_manifest(
            {
                "schema_version": MANIFEST_SCHEMA,
                "model_schema_version": SCHEMA_VERSION,
                "revision": self.revision + 1,
                "current_version": int(current_version),
                "current_state_hash": str(current_state_hash),
                "undo_stack": [int(item) for item in undo_stack],
                "redo_stack": [int(item) for item in redo_stack],
                "closed": bool(closed),
            }
        )

    def mark_active(self) -> None:
        manifest = self._assert_manifest_current()
        self._write_manifest(
            {
                "schema_version": MANIFEST_SCHEMA,
                "model_schema_version": SCHEMA_VERSION,
                "revision": self.revision + 1,
                "current_version": int(manifest["current_version"]),
                "current_state_hash": str(manifest["current_state_hash"]),
                "undo_stack": [int(item) for item in manifest.get("undo_stack", [])],
                "redo_stack": [int(item) for item in manifest.get("redo_stack", [])],
                "closed": False,
            }
        )

    def close(self, current: GeometryState, *, undo_stack: Sequence[int], redo_stack: Sequence[int]) -> None:
        self.save_session(
            current_version=current.version,
            current_state_hash=current.state_hash,
            undo_stack=undo_stack,
            redo_stack=redo_stack,
            closed=True,
        )
        atomic_write_json(
            self.session_report_path,
            with_content_hash(
                {
                    "schema_version": "cad-surface.session-report.v1",
                    "exit_kind": "controlled",
                    "final_version": current.version,
                    "final_state_hash": current.state_hash,
                    "geometry_hash": current.geometry_hash,
                    "normals_hash": current.normals_hash,
                    "display_hashes": current.display_hashes,
                    "undo_depth": len(undo_stack),
                    "redo_depth": len(redo_stack),
                    "closed_flag_written": True,
                }
            ),
        )

    def _assert_manifest_current(self) -> dict[str, Any]:
        manifest = self._read_manifest()
        if int(manifest.get("revision", -1)) != self.revision:
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "manifest revision advanced; refusing to overwrite a newer snapshot",
                expected_revision=self.revision,
                actual_revision=manifest.get("revision"),
            )
        if self._manifest_hash is not None and manifest.get("content_hash") != self._manifest_hash:
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "manifest content hash changed since it was loaded",
                expected=self._manifest_hash,
                actual=manifest.get("content_hash"),
            )
        return manifest

    def _write_manifest(self, body: dict[str, Any]) -> None:
        if not self.manifest_path.exists() and self.revision != 0:
            raise DomainError(ErrorCode.SNAPSHOT_STALE, "manifest disappeared during persistence")
        manifest = with_content_hash(body)
        expected_revision = int(body["revision"]) - 1
        if expected_revision != self.revision:
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "manifest revision is not contiguous",
                expected=self.revision,
                actual=expected_revision,
            )
        atomic_write_json(self.manifest_path, manifest)
        self.revision = int(manifest["revision"])
        self._manifest_hash = str(manifest["content_hash"])

    def commit_versions(self) -> tuple[int, ...]:
        if not self.commits_dir.exists():
            return ()
        versions: list[int] = []
        for entry in self.commits_dir.iterdir():
            match = COMMIT_DIR_PATTERN.match(entry.name)
            if match and entry.is_dir() and (entry / "commit.json").exists():
                versions.append(int(match.group(1)))
        return tuple(sorted(versions))

    def load_all_operations(self) -> dict[int, EditOperation]:
        operations: dict[int, EditOperation] = {}
        for version in self.commit_versions():
            operation = self.load_operation(version)
            if operation is not None:
                operations[operation.op_version] = operation
        return operations

    def load_operation(self, version: int) -> EditOperation | None:
        version = int(version)
        commit_dir = self._commit_dir(version)
        commit_path = commit_dir / "commit.json"
        if not commit_path.exists():
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "committed snapshot is missing",
                version=version,
                path=str(commit_path),
            )
        commit, _ = read_json_bytes(commit_path)
        verify_content_hash(commit)
        if int(commit.get("version", -1)) != version:
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "commit directory version and payload disagree",
                directory_version=version,
                payload_version=commit.get("version"),
            )
        return self._load_operation(commit_dir, commit)

    def stage_uncommitted_probe(self, state: GeometryState, operation: EditOperation) -> str:
        """Create a complete but unreferenced staging commit, as if killed before commit."""

        self._assert_manifest_current()
        bundle = self._write_commit_bundle(state, operation)
        return str(bundle["staging_path"])

    def next_version(self) -> int:
        manifest = self._read_manifest()
        versions = list(self.commit_versions())
        versions.extend(int(item) for item in manifest.get("undo_stack", []))
        versions.extend(int(item) for item in manifest.get("redo_stack", []))
        versions.append(int(manifest["current_version"]))
        return max(versions) + 1

    def expected_revision(self) -> int:
        return self.revision

    def read_manifest_for_diagnostics(self) -> dict[str, Any]:
        return self._read_manifest()
