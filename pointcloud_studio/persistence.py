"""Crash-safe committed-state storage and orphan recovery."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .engine import StudioEngine
from .errors import ErrorCode, StudioError
from .formats import encode_normals, encode_points, load_normals, load_points
from .ledger import verify_ledger, verify_ledger_digest
from .mesh import (
    CombinedMesh,
    PatchMesh,
    ReconstructionParams,
    combine_mesh,
    decode_patch_cache,
    patch_cache_name,
)
from .normal_stats_format import encode_neighbor_stats, load_neighbor_stats
from .util import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)

STORE_SCHEMA = "pointcloud-studio.persistence/1"
STATE_DIR_RE = re.compile(r"^state-(\d{6})$")


@dataclass(frozen=True)
class OpenReport:
    clean_previous_exit: bool | None
    current_version: int | None
    recovered_orphans: list[dict[str, Any]]
    recovery_report_path: str | None
    had_stale_closed_marker: bool


@dataclass(frozen=True)
class LoadedState:
    engine: StudioEngine
    version: int
    state_dir: Path
    commit: dict[str, Any]
    points: np.ndarray
    normals: np.ndarray
    confidence: np.ndarray
    ledger: dict[str, Any]


def _normal_stats_payload(engine: StudioEngine) -> bytes:
    return encode_neighbor_stats(
        engine.cloud.neighbor_counts,
        engine.cloud.residuals,
        engine.cloud.error_bounds,
    )


def _source_ids_payload(engine: StudioEngine) -> bytes:
    return np.ascontiguousarray(engine.cloud.source_ids, dtype="<i8").tobytes(order="C")


class AtomicStateStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.states_root = self.root / "states"
        self.staging_root = self.root / "staging"
        self.orphans_root = self.root / "orphans"
        self.recovery_root = self.root / "recovery"
        self.current_path = self.root / "CURRENT"
        self.closed_path = self.root / "closed.json"
        self.open_report: OpenReport | None = None
        self._layout_ready = False

    def _ensure_layout(self) -> None:
        for directory in (
            self.root,
            self.states_root,
            self.staging_root,
            self.orphans_root,
            self.recovery_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._layout_ready = True

    def _read_current(self) -> dict[str, Any] | None:
        if not self.current_path.exists():
            return None
        try:
            value = read_json(self.current_path)
        except Exception as exc:
            raise StudioError(
                ErrorCode.E_CACHE_HASH,
                "CURRENT pointer is unreadable",
                context={"path": str(self.current_path), "reason": str(exc)},
            ) from exc
        if not isinstance(value, dict) or "state" not in value or "version" not in value:
            raise StudioError(
                ErrorCode.E_CACHE_HASH,
                "CURRENT pointer is malformed",
                context={"path": str(self.current_path), "value": value},
            )
        return value

    def open(self) -> OpenReport:
        self._ensure_layout()
        current = self._read_current()
        current_version = int(current["version"]) if current else None
        clean_previous: bool | None = None
        stale_closed = False
        if self.closed_path.exists():
            try:
                closed = read_json(self.closed_path)
            except Exception:
                closed = None
            if isinstance(closed, dict) and closed.get("clean_shutdown") is True:
                clean_previous = (
                    current_version is not None
                    and int(closed.get("last_committed_version", -1)) == current_version
                )
            else:
                clean_previous = False
            stale_closed = True
        elif current_version is not None:
            clean_previous = False

        orphans: list[dict[str, Any]] = []
        recovery_id = self._next_recovery_id()
        quarantine_root = self.orphans_root / f"recovery-{recovery_id:06d}"

        for staging in sorted(self.staging_root.iterdir(), key=lambda path: path.name):
            if not staging.is_dir():
                continue
            destination = quarantine_root / "staging" / staging.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
            orphans.append(
                {
                    "kind": "uncommitted_staging",
                    "original": str(staging),
                    "quarantine": str(destination),
                    "reason": "staging directory was not promoted to a committed state",
                }
            )

        valid_states: list[tuple[int, Path, dict[str, Any]]] = []
        invalid_states: list[tuple[Path, str]] = []
        for directory in sorted(self.states_root.iterdir(), key=lambda path: path.name):
            if not directory.is_dir():
                continue
            match = STATE_DIR_RE.match(directory.name)
            if not match:
                invalid_states.append((directory, "unrecognized state directory name"))
                continue
            commit_path = directory / "COMMIT.json"
            if not commit_path.exists():
                invalid_states.append((directory, "COMMIT.json is missing"))
                continue
            try:
                commit = read_json(commit_path)
            except Exception as exc:
                invalid_states.append((directory, f"COMMIT.json is unreadable: {exc}"))
                continue
            if int(commit.get("version", -1)) != int(match.group(1)):
                invalid_states.append((directory, "COMMIT version does not match directory name"))
                continue
            valid_states.append((int(match.group(1)), directory, commit))

        for directory, reason in invalid_states:
            destination = quarantine_root / "states" / directory.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(directory, destination)
            orphans.append(
                {
                    "kind": "invalid_state",
                    "original": str(directory),
                    "quarantine": str(destination),
                    "reason": reason,
                }
            )

        if current_version is not None:
            for version, directory, commit in valid_states:
                if version > current_version:
                    destination = quarantine_root / "states" / directory.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(directory, destination)
                    orphans.append(
                        {
                            "kind": "unreferenced_newer_state",
                            "original": str(directory),
                            "quarantine": str(destination),
                            "reason": "state was renamed before CURRENT was atomically updated",
                            "version": version,
                        }
                    )
        else:
            for version, directory, commit in valid_states:
                destination = quarantine_root / "states" / directory.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(directory, destination)
                orphans.append(
                    {
                        "kind": "state_without_current_pointer",
                        "original": str(directory),
                        "quarantine": str(destination),
                        "reason": "no atomically committed CURRENT pointer exists",
                        "version": version,
                    }
                )

        recovery_path: Path | None = None
        previous_session_manifest = self.root / "session_manifest.json"
        if previous_session_manifest.exists():
            archived_session = self.recovery_root / f"session-{recovery_id:06d}.json"
            os.replace(previous_session_manifest, archived_session)
        if current_version is not None or stale_closed or orphans:
            recovery_path = self.recovery_root / f"open-{recovery_id:06d}.json"
            report = {
                "schema": "pointcloud-studio.recovery-report/1",
                "recovery_id": recovery_id,
                "observed_at_unix": round(time.time(), 6),
                "clean_previous_exit": clean_previous,
                "closed_marker_present": stale_closed,
                "current_version": current_version,
                "orphans": orphans,
                "recovered_from": "CURRENT committed state" if current_version is not None else "no committed state",
            }
            atomic_write_json(recovery_path, report)
            if self.closed_path.exists():
                archived = self.recovery_root / f"closed-{recovery_id:06d}.json"
                os.replace(self.closed_path, archived)
        self.open_report = OpenReport(
            clean_previous_exit=clean_previous,
            current_version=current_version,
            recovered_orphans=orphans,
            recovery_report_path=str(recovery_path) if recovery_path else None,
            had_stale_closed_marker=stale_closed,
        )
        return self.open_report

    def _next_recovery_id(self) -> int:
        maximum = 0
        if self.recovery_root.exists():
            for path in self.recovery_root.glob("open-*.json"):
                try:
                    maximum = max(maximum, int(path.stem.split("-")[1]))
                except (ValueError, IndexError):
                    continue
        return maximum + 1

    def _state_dir(self, version: int) -> Path:
        return self.states_root / f"state-{version:06d}"

    def _build_file_index(self, state_dir: Path) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for path in sorted(state_dir.rglob("*"), key=lambda item: str(item)):
            if path.is_file() and path.name != "COMMIT.json":
                relative = path.relative_to(state_dir).as_posix()
                result[relative] = {"sha256": sha256_file(path), "size": path.stat().st_size}
        return result

    def commit(self, engine: StudioEngine, *, operation_seq: int | None = None) -> dict[str, Any]:
        self._ensure_layout()
        version = int(engine.version)
        staging = self.staging_root / f"state-{version:06d}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        staging.mkdir(parents=True, exist_ok=False)
        patches_dir = staging / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)
        points_payload = encode_points(engine.cloud.positions)
        normals_payload = encode_normals(engine.cloud.normals, engine.cloud.confidence)
        stats_payload = _normal_stats_payload(engine)
        source_ids_payload = _source_ids_payload(engine)
        atomic_write_bytes(staging / "points.bin", points_payload)
        atomic_write_bytes(staging / "normals.bin", normals_payload)
        atomic_write_bytes(staging / "normal_stats.bin", stats_payload)
        atomic_write_bytes(staging / "source_ids.bin", source_ids_payload)
        for key, patch in sorted(engine.patches.items(), key=lambda item: item[0].encode("ascii")):
            atomic_write_bytes(patches_dir / patch_cache_name(key), patch.cache_payload())

        ledger = dict(engine.ledger)
        state_manifest = {
            "schema": "pointcloud-studio.committed-state/1",
            "version": version,
            "point_count": int(engine.cloud.positions.shape[0]),
            "operation_count": len(engine.history),
            "normal_radius": float(engine.cloud.normal_radius),
            "reconstruction_parameters": engine.cloud.params.to_dict(),
            "mesh_content_sha256": engine.mesh.content_sha256,
            "ledger_sha256": ledger["ledger_sha256"],
            "normal_stats": "normal_stats.bin",
            "source_ids": "source_ids.bin",
        }
        atomic_write_json(staging / "state.json", state_manifest)
        atomic_write_json(staging / "ledger.json", ledger)
        atomic_write_json(staging / "history.json", engine.history)

        file_index = self._build_file_index(staging)
        commit_body = {
            "schema": "pointcloud-studio.commit/1",
            "version": version,
            "files": file_index,
            "point_count": int(engine.cloud.positions.shape[0]),
            "mesh_content_sha256": engine.mesh.content_sha256,
            "ledger_sha256": ledger["ledger_sha256"],
            "committed_operation_seq": operation_seq,
        }
        commit_body["commit_sha256"] = sha256_bytes(canonical_json(commit_body).encode("utf-8"))
        atomic_write_json(staging / "COMMIT.json", commit_body)

        destination = self._state_dir(version)
        if destination.exists():
            raise StudioError(
                ErrorCode.E_CACHE_HASH,
                "refusing to overwrite an existing committed state",
                context={"version": version, "state_dir": str(destination)},
            )
        os.replace(staging, destination)
        current = {
            "schema": "pointcloud-studio.current/1",
            "version": version,
            "state": destination.name,
            "commit_sha256": commit_body["commit_sha256"],
            "mesh_content_sha256": engine.mesh.content_sha256,
            "ledger_sha256": ledger["ledger_sha256"],
        }
        atomic_write_json(self.current_path, current)
        return current

    def close(self, engine: StudioEngine) -> dict[str, Any]:
        current = self._read_current()
        committed_states: list[dict[str, Any]] = []
        for directory in sorted(self.states_root.iterdir(), key=lambda path: path.name):
            commit_path = directory / "COMMIT.json"
            if not directory.is_dir() or not commit_path.exists():
                continue
            try:
                commit = read_json(commit_path)
            except Exception:
                continue
            committed_states.append(
                {
                    "version": int(commit.get("version", -1)),
                    "directory": directory.name,
                    "commit_sha256": commit.get("commit_sha256"),
                    "mesh_content_sha256": commit.get("mesh_content_sha256"),
                    "ledger_sha256": commit.get("ledger_sha256"),
                }
            )
        session_manifest = {
            "schema": "pointcloud-studio.session-manifest/1",
            "clean_shutdown_intent": True,
            "current": current,
            "last_committed_version": int(engine.version),
            "committed_state_count": len(committed_states),
            "committed_states": committed_states,
            "operation_count": len(engine.history),
            "last_cloud_sha256": engine.cloud.content_sha256(),
            "last_mesh_content_sha256": engine.mesh.content_sha256,
            "last_ledger_sha256": engine.ledger["ledger_sha256"],
        }
        atomic_write_json(self.root / "session_manifest.json", session_manifest)
        marker = {
            "schema": "pointcloud-studio.closed/1",
            "clean_shutdown": True,
            "last_committed_version": int(engine.version),
            "last_cloud_sha256": engine.cloud.content_sha256(),
            "last_mesh_content_sha256": engine.mesh.content_sha256,
            "last_ledger_sha256": engine.ledger["ledger_sha256"],
            "history_count": len(engine.history),
        }
        atomic_write_json(self.closed_path, marker)
        return marker

    def _verify_commit_index(self, state_dir: Path, commit: Mapping[str, Any]) -> None:
        files = commit.get("files")
        if not isinstance(files, dict):
            raise StudioError(
                ErrorCode.E_CACHE_HASH,
                "COMMIT file index is missing",
                context={"state_dir": str(state_dir)},
            )
        for relative, expected in files.items():
            path = state_dir / relative
            if not path.exists():
                raise StudioError(
                    ErrorCode.E_CACHE_HASH,
                    "committed state file is missing",
                    context={"state_dir": str(state_dir), "file": relative},
                )
            actual = sha256_file(path)
            if actual != expected.get("sha256"):
                raise StudioError(
                    ErrorCode.E_CACHE_HASH,
                    "committed state file hash mismatch",
                    context={
                        "state_dir": str(state_dir),
                        "file": relative,
                        "expected": expected.get("sha256"),
                        "actual": actual,
                    },
                )

    def load_current(self, *, independent_ledger_check: bool = False) -> LoadedState:
        self._ensure_layout()
        current = self._read_current()
        if current is None:
            raise StudioError(
                ErrorCode.E_CACHE_HASH,
                "no committed state is available",
                context={"root": str(self.root)},
            )
        version = int(current["version"])
        state_dir = self.states_root / str(current["state"])
        commit_path = state_dir / "COMMIT.json"
        if not commit_path.exists():
            raise StudioError(
                ErrorCode.E_CACHE_HASH,
                "CURRENT references a missing state",
                context={"current": current, "state_dir": str(state_dir)},
            )
        commit = read_json(commit_path)
        if commit.get("commit_sha256") != current.get("commit_sha256"):
            raise StudioError(
                ErrorCode.E_CACHE_HASH,
                "CURRENT and COMMIT hashes differ",
                context={
                    "current": current.get("commit_sha256"),
                    "commit": commit.get("commit_sha256"),
                },
            )
        self._verify_commit_index(state_dir, commit)
        state = read_json(state_dir / "state.json")
        params = ReconstructionParams.from_dict(state["reconstruction_parameters"])
        points = load_points(state_dir / "points.bin")
        normals, confidence = load_normals(state_dir / "normals.bin")
        counts, residuals, bounds = load_neighbor_stats(state_dir / "normal_stats.bin")
        source_raw = (state_dir / "source_ids.bin").read_bytes()
        expected_source_size = points.shape[0] * 8
        if len(source_raw) != expected_source_size:
            raise StudioError(
                ErrorCode.E_POINT_CLOUD_FORMAT,
                "source-id sidecar size mismatch",
                context={"path": str(state_dir / "source_ids.bin"), "size": len(source_raw), "expected": expected_source_size},
            )
        source_ids = np.frombuffer(source_raw, dtype="<i8", count=points.shape[0]).astype(np.int64, copy=True)
        for name, array in (
            ("neighbor_counts", counts),
            ("residuals", residuals),
            ("error_bounds", bounds),
        ):
            if array.shape != (points.shape[0],):
                raise StudioError(
                    ErrorCode.E_POINT_CLOUD_FORMAT,
                    "committed normal-stat shape mismatch",
                    context={"array": name, "shape": list(array.shape), "point_count": int(points.shape[0])},
                )
        ledger = read_json(state_dir / "ledger.json")
        verify_ledger_digest(ledger)
        entries = ledger.get("entries", {})
        patches: dict[str, PatchMesh] = {}
        for key, entry in entries.items():
            relative = entry.get("cache_file")
            path = state_dir / str(relative)
            payload = path.read_bytes()
            patch = decode_patch_cache(
                payload,
                expected_content_hash=entry.get("content_sha256"),
                path=str(path),
            )
            if patch.key != key:
                raise StudioError(
                    ErrorCode.E_CACHE_HASH,
                    "patch cache key differs from ledger key",
                    context={"ledger_key": key, "cache_key": patch.key, "path": str(path)},
                )
            patches[key] = patch
        mesh = combine_mesh(patches)
        if mesh.content_sha256 != ledger["mesh"]["content_sha256"]:
            raise StudioError(
                ErrorCode.E_CACHE_HASH,
                "restored mesh hash differs from ledger",
                context={
                    "restored": mesh.content_sha256,
                    "ledger": ledger["mesh"]["content_sha256"],
                },
            )
        history = read_json(state_dir / "history.json")
        engine = StudioEngine.restore(
            points,
            normals,
            confidence,
            counts,
            residuals,
            bounds,
            source_ids,
            normal_radius=float(state["normal_radius"]),
            params=params,
            patches=patches,
            mesh=mesh,
            ledger=ledger,
            version=version,
            history=history,
        )
        if independent_ledger_check:
            verify_ledger(points, normals, confidence, ledger)
        return LoadedState(
            engine=engine,
            version=version,
            state_dir=state_dir,
            commit=commit,
            points=points,
            normals=normals,
            confidence=confidence,
            ledger=ledger,
        )

    def independent_verify_current(self) -> dict[str, Any]:
        loaded = self.load_current(independent_ledger_check=False)
        return verify_ledger(
            loaded.points,
            loaded.normals,
            loaded.confidence,
            loaded.ledger,
        )
