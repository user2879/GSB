"""Atomic content-addressed dependency ledger and restart recovery."""

from __future__ import annotations

import json
import os
import shutil
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import (
    ALGORITHM_VERSION,
    APP_VERSION,
    SCHEMA_CONSISTENCY,
    SCHEMA_LEDGER,
    SCHEMA_REPAIR,
)
from .detection import detection_payload, thresholds_from_mapping
from .errors import BusinessError, ErrorCode
from .evaluation import evaluation_payload
from .io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from .model import DefectEntry, Motion, Snapshot
from .pipeline import build_with_evidence
from .repair import repair_payload


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class CommitInfo:
    commit_id: str
    content_hash: str
    version_label: str
    ledger_sha256: str
    manifest_path: Path
    ledger: dict[str, Any]


@dataclass(frozen=True)
class RecoveryInfo:
    previous_exit_clean: bool
    previous_session_present: bool
    restored_commit_id: str | None
    restored_content_hash: str | None
    orphan_staging_files_removed: int
    orphan_blobs_removed: int
    orphan_bytes_removed: int
    recovery_log_path: str
    current_pointer_present: bool


@dataclass(frozen=True)
class RestoreOutcome:
    snapshot: Snapshot | None
    stale_reason: dict[str, Any] | None


class StateStore:
    """Workspace-relative transaction store.

    A commit becomes visible only when ``current.json`` is atomically replaced.
    Blobs written before that replacement are intentionally treated as orphans
    if the process dies.
    """

    def __init__(self, workspace: Path, namespace: str = "default") -> None:
        self.workspace = Path(workspace).resolve()
        self.namespace = namespace
        self.state_root = self.workspace / "runtime" / "state" / namespace
        self.blob_dir = self.state_root / "blobs"
        self.staging_dir = self.state_root / "staging"
        self.current_path = self.state_root / "current.json"
        self.session_path = self.state_root / "session.json"
        self.closed_path = self.state_root / "closed.json"
        self.logs_dir = self.workspace / "runtime" / "logs"
        self._session_id: str | None = None
        self._recovery: RecoveryInfo | None = None
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.workspace))

    def _log_event(self, payload: dict[str, Any], prefix: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.logs_dir / f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            path,
            {
                "schema": "mocap-runtime-log/v1",
                "app_version": APP_VERSION,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "timestamp_utc": utc_now(),
                **payload,
            },
        )
        return path

    def record_event(self, event: str, payload: dict[str, Any], prefix: str = "recovery") -> str:
        path = self._log_event({"event": event, **payload}, prefix)
        return self._relative(path)

    def read_current_manifest(self) -> dict[str, Any] | None:
        if not self.current_path.exists():
            return None
        try:
            manifest = read_json(self.current_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_STALE,
                "current commit pointer is unreadable",
                {"path": self._relative(self.current_path)},
            ) from exc
        required = {"commit_id", "ledger_blob", "blobs", "version"}
        if not required.issubset(manifest):
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_STALE,
                "current commit pointer is incomplete",
                {"path": self._relative(self.current_path), "missing": sorted(required - set(manifest))},
            )
        return manifest

    def _verify_blob(self, blob_hash: str) -> Path:
        path = self.blob_dir / f"{blob_hash}.json"
        if not path.exists():
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_STALE,
                "committed ledger references a missing blob",
                {"blob": blob_hash, "path": self._relative(path)},
            )
        actual = sha256_file(path)
        if actual != blob_hash:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_STALE,
                "committed blob hash mismatch",
                {"blob": blob_hash, "actual": actual, "path": self._relative(path)},
            )
        return path

    def load_current(self) -> CommitInfo | None:
        manifest = self.read_current_manifest()
        if manifest is None:
            return None
        ledger_hash = str(manifest["ledger_blob"])
        ledger_path = self._verify_blob(ledger_hash)
        for kind, blob_hash in manifest.get("blobs", {}).items():
            self._verify_blob(str(blob_hash))
        ledger = read_json(ledger_path)
        if ledger.get("schema") != SCHEMA_LEDGER:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_STALE,
                "committed ledger has an unexpected schema",
                {"blob": ledger_hash},
            )
        version = ledger.get("version", {})
        return CommitInfo(
            commit_id=str(manifest["commit_id"]),
            content_hash=str(version.get("content_hash", manifest.get("content_hash", ""))),
            version_label=str(version.get("label", manifest.get("version", {}).get("label", ""))),
            ledger_sha256=ledger_hash,
            manifest_path=self.current_path,
            ledger=ledger,
        )

    def _referenced_blobs(self, manifest: dict[str, Any] | None) -> set[str]:
        if manifest is None:
            return set()
        referenced = {str(manifest.get("ledger_blob", ""))}
        referenced.update(str(value) for value in manifest.get("blobs", {}).values())
        referenced.discard("")
        return referenced

    def recover_orphans(self) -> dict[str, Any]:
        manifest = self.read_current_manifest()
        referenced = self._referenced_blobs(manifest)
        staging_removed = 0
        staging_bytes = 0
        for path in self.staging_dir.iterdir():
            if path.is_file():
                staging_bytes += path.stat().st_size
                path.unlink(missing_ok=True)
                staging_removed += 1
        blobs_removed = 0
        bytes_removed = 0
        for path in self.blob_dir.iterdir():
            if not path.is_file() or path.name in {f"{item}.json" for item in referenced}:
                continue
            bytes_removed += path.stat().st_size
            path.unlink(missing_ok=True)
            blobs_removed += 1
        result = {
            "referenced_blob_count": len(referenced),
            "orphan_staging_files_removed": staging_removed,
            "orphan_blobs_removed": blobs_removed,
            "staging_bytes_removed": staging_bytes,
            "orphan_bytes_removed": bytes_removed,
        }
        return result

    def begin_session(self) -> RecoveryInfo:
        manifest = self.read_current_manifest()
        current_commit = str(manifest["commit_id"]) if manifest else None
        current_content = None
        if manifest:
            current_content = str(manifest.get("version", {}).get("content_hash", manifest.get("content_hash", "")))
        closed: dict[str, Any] | None = None
        if self.closed_path.exists():
            try:
                closed = read_json(self.closed_path)
            except (OSError, json.JSONDecodeError):
                closed = None
        previous_session_present = self.session_path.exists()
        previous_clean = bool(
            closed
            and closed.get("clean_exit") is True
            and closed.get("commit_id") == current_commit
        )
        if self.closed_path.exists():
            os.replace(self.closed_path, self.state_root / "closed.previous.json")
        orphan_info = self.recover_orphans()
        self._session_id = uuid.uuid4().hex
        atomic_write_json(
            self.session_path,
            {
                "schema": "mocap-session/v1",
                "session_id": self._session_id,
                "pid": os.getpid(),
                "started_utc": utc_now(),
                "base_commit_id": current_commit,
                "base_content_hash": current_content,
            },
        )
        log_path = self._log_event(
            {
                "event": "startup_recovery",
                "previous_session_present": previous_session_present,
                "previous_exit_clean": previous_clean,
                "previous_was_forced_termination": bool(previous_session_present and not previous_clean),
                "restored_commit_id": current_commit,
                "restored_content_hash": current_content,
                **orphan_info,
            },
            "recovery",
        )
        self._recovery = RecoveryInfo(
            previous_exit_clean=previous_clean,
            previous_session_present=previous_session_present,
            restored_commit_id=current_commit,
            restored_content_hash=current_content,
            orphan_staging_files_removed=int(orphan_info["orphan_staging_files_removed"]),
            orphan_blobs_removed=int(orphan_info["orphan_blobs_removed"]),
            orphan_bytes_removed=int(orphan_info["orphan_bytes_removed"]),
            recovery_log_path=self._relative(log_path),
            current_pointer_present=manifest is not None,
        )
        return self._recovery

    @property
    def recovery(self) -> RecoveryInfo | None:
        return self._recovery

    def _put_blob(self, data: bytes) -> str:
        digest = sha256_bytes(data)
        path = self.blob_dir / f"{digest}.json"
        if path.exists():
            if sha256_file(path) != digest:
                raise BusinessError(
                    ErrorCode.E_SNAPSHOT_STALE,
                    "existing content-addressed blob is corrupt",
                    {"blob": digest, "path": self._relative(path)},
                )
            return digest
        atomic_write_bytes(path, data)
        return digest

    def commit(self, snapshot: Snapshot, truth_entries: tuple[DefectEntry, ...]) -> CommitInfo:
        if self._session_id is None:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_STALE,
                "commit attempted before a session was started",
                {"path": self._relative(self.state_root)},
            )
        detection_data = canonical_json_bytes(
            detection_payload(snapshot.detection, snapshot.evidence, snapshot.record)
        )
        meta = snapshot.to_meta_dict()
        repair_data = canonical_json_bytes(
            repair_payload(meta, snapshot.record, snapshot.repair)
        )
        evaluation_data = canonical_json_bytes(evaluation_payload(snapshot.evaluation))
        detection_hash = self._put_blob(detection_data)
        repair_blob_hash = self._put_blob(repair_data)
        evaluation_blob_hash = self._put_blob(evaluation_data)

        ledger: dict[str, Any] = {
            "schema": SCHEMA_LEDGER,
            "algorithm_version": ALGORITHM_VERSION,
            "version": snapshot.version.to_dict(),
            "record_sha256": snapshot.record.content_sha256,
            "truth_sha256": snapshot.truth.content_sha256,
            "thresholds": snapshot.thresholds.to_dict(),
            "keyframes": sorted(snapshot.keyframes),
            "suppressed_auto": sorted(snapshot.suppressed_auto),
            "detection": {
                "detection_sha256": snapshot.detection.detection_hash,
                "evidence_sha256": snapshot.evidence.evidence_hash,
                "entries": snapshot.detection.entries_as_dicts(),
                "comparison": snapshot.detection.comparison,
            },
            "repair": {
                "repair_sha256": snapshot.repair.repair_hash,
                "frame_hashes": list(snapshot.repair.frame_hashes),
                "changed_joint_frames": [
                    {"frame": frame, "joint": snapshot.record.joint_names[joint]}
                    for frame, joint in snapshot.repair.changed_joint_frames
                ],
                "changed_root_frames": list(snapshot.repair.changed_root_frames),
                "policy": snapshot.repair.policy,
            },
            "evaluation": {
                "evaluation_sha256": snapshot.evaluation.evaluation_hash,
                "summary": snapshot.evaluation.summary,
                "intervals": snapshot.evaluation.interval_metrics,
                "details": [detail.to_dict() for detail in snapshot.evaluation.details],
            },
            "closure": snapshot.closure.to_dict() if snapshot.closure else None,
            "consistency": snapshot.consistency.to_dict() if snapshot.consistency else None,
            "truth_manifest_count": len(truth_entries),
            "blobs": {
                "detection": detection_hash,
                "repair": repair_blob_hash,
                "evaluation": evaluation_blob_hash,
            },
        }
        ledger_data = canonical_json_bytes(ledger)
        ledger_hash = self._put_blob(ledger_data)
        commit_identity = sha256_json(
            {
                "ledger_sha256": ledger_hash,
                "content_hash": snapshot.content_hash,
                "session_id": self._session_id,
            }
        )
        manifest = {
            "schema": "mocap-current-commit/v1",
            "commit_id": commit_identity,
            "created_utc": utc_now(),
            "session_id": self._session_id,
            "version": snapshot.version.to_dict(),
            "content_hash": snapshot.content_hash,
            "ledger_blob": ledger_hash,
            "blobs": ledger["blobs"],
        }
        # The pointer write is the sole commit point.
        atomic_write_json(self.current_path, manifest)
        self._log_event(
            {
                "event": "commit",
                "commit_id": commit_identity,
                "version": snapshot.version.to_dict(),
                "content_hash": snapshot.content_hash,
                "ledger_sha256": ledger_hash,
                "repair_sha256": snapshot.repair.repair_hash,
            },
            "commit",
        )
        return CommitInfo(
            commit_id=commit_identity,
            content_hash=snapshot.content_hash,
            version_label=snapshot.version.label,
            ledger_sha256=ledger_hash,
            manifest_path=self.current_path,
            ledger=ledger,
        )

    def close_cleanly(self) -> Path:
        manifest = self.read_current_manifest()
        commit_id = str(manifest["commit_id"]) if manifest else None
        content_hash = (
            str(manifest.get("version", {}).get("content_hash", manifest.get("content_hash", "")))
            if manifest
            else None
        )
        marker = {
            "schema": "mocap-clean-exit/v1",
            "clean_exit": True,
            "session_id": self._session_id,
            "closed_utc": utc_now(),
            "commit_id": commit_id,
            "content_hash": content_hash,
            "pid": os.getpid(),
        }
        atomic_write_json(self.closed_path, marker)
        self.session_path.unlink(missing_ok=True)
        return self.closed_path

    def restore_committed_snapshot(
        self,
        record: Motion,
        truth: Motion,
        truth_entries: tuple[DefectEntry, ...],
    ) -> Snapshot | None:
        commit = self.load_current()
        if commit is None:
            return None
        ledger = commit.ledger
        if ledger.get("algorithm_version") != ALGORITHM_VERSION:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_STALE,
                "committed ledger was produced by an incompatible algorithm version",
                {
                    "committed": ledger.get("algorithm_version"),
                    "current": ALGORITHM_VERSION,
                    "commit_id": commit.commit_id,
                },
            )
        if ledger.get("record_sha256") != record.content_sha256:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_STALE,
                "committed state belongs to a different record",
                {
                    "committed_record": ledger.get("record_sha256"),
                    "current_record": record.content_sha256,
                },
            )
        thresholds = thresholds_from_mapping(ledger.get("thresholds", {}))
        keyframes = frozenset(int(frame) for frame in ledger.get("keyframes", []))
        version = ledger.get("version", {})
        snapshot = build_with_evidence(
            record,
            truth,
            truth_entries,
            compute_evidence_for_restore(record),
            thresholds,
            keyframes,
            int(version.get("sequence", 0)),
            committed=True,
            suppressed_auto=frozenset(int(frame) for frame in ledger.get("suppressed_auto", [])),
        )
        if snapshot.content_hash != version.get("content_hash"):
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_STALE,
                "recomputed state does not match the committed content hash",
                {
                    "committed": version.get("content_hash"),
                    "recomputed": snapshot.content_hash,
                    "commit_id": commit.commit_id,
                },
            )
        if snapshot.repair.repair_hash != ledger.get("repair", {}).get("repair_sha256"):
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_STALE,
                "restored repair hash differs from the committed ledger",
                {"commit_id": commit.commit_id},
            )
        return snapshot


def compute_evidence_for_restore(record: Motion):
    from .detection import compute_evidence

    return compute_evidence(record)


def simulate_incomplete_write(store: StateStore, name: str = "uncommitted.json") -> Path:
    """Test helper: create blobs/staging data that must never become visible."""

    staging_path = store.staging_dir / name
    atomic_write_bytes(staging_path, b'{"uncommitted":true}\n')
    orphan_blob = store.blob_dir / (sha256_bytes(b"orphan-blob") + ".json")
    atomic_write_bytes(orphan_blob, b"orphan-blob")
    return staging_path
