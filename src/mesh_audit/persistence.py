from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes, digest_json
from .errors import BusinessError, ErrorCode


SNAPSHOT_FORMAT = "mesh-audit-snapshot-v1"
MANIFEST_FORMAT = "mesh-audit-manifest-v1"
STEP_PATTERN = re.compile(r"^step-(\d{4})\.json$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_json_strict(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is not allowed: {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        raise BusinessError(
            ErrorCode.E_SNAPSHOT_INVALID,
            "snapshot JSON cannot be read",
            {"path": str(path), "reason": str(exc)},
        ) from exc


@dataclass(slots=True)
class SnapshotRecord:
    step: int
    path: str
    valid: bool
    reason: str | None
    version_id: str | None
    payload: dict[str, Any] | None

    def to_summary(self) -> dict[str, Any]:
        operator = None
        if self.payload is not None:
            operator = self.payload.get("operator_type")
        return {
            "step": self.step,
            "path": self.path,
            "valid": self.valid,
            "reason": self.reason,
            "operator_type": operator,
        }


@dataclass(slots=True)
class StateInspection:
    state_dir: str
    current_dir: str
    manifest: dict[str, Any] | None
    snapshots: list[SnapshotRecord]
    latest_contiguous_valid_step: int | None
    invalid_snapshot_count: int

    @property
    def all_valid(self) -> bool:
        return self.invalid_snapshot_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_dir": self.state_dir,
            "current_dir": self.current_dir,
            "manifest": self.manifest,
            "latest_contiguous_valid_step": self.latest_contiguous_valid_step,
            "invalid_snapshot_count": self.invalid_snapshot_count,
            "snapshots": [record.to_summary() for record in self.snapshots],
        }


class StateStore:
    def __init__(self, state_dir: str | Path) -> None:
        self.root = Path(state_dir)
        self.current = self.root / "current"
        self.archive = self.root / "archive"

    def _manifest_path(self) -> Path:
        return self.current / "manifest.json"

    def _snapshot_path(self, step: int) -> Path:
        return self.current / f"step-{step:04d}.json"

    def _archive_destination(self, label: str) -> Path:
        self.archive.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)[:24] or "unknown"
        candidate = self.archive / f"{stamp}-{safe_label}"
        counter = 1
        while candidate.exists():
            candidate = self.archive / f"{stamp}-{safe_label}-{counter}"
            counter += 1
        return candidate

    def archive_current(self, reason: str) -> str | None:
        if not self.current.exists():
            return None
        old_version = "unknown"
        manifest_path = self._manifest_path()
        if manifest_path.exists():
            try:
                manifest = read_json_strict(manifest_path)
                old_version = str(manifest.get("version_id", "unknown"))
            except BusinessError:
                old_version = "invalid-manifest"
        destination = self._archive_destination(f"{old_version}-{reason}")
        os.rename(self.current, destination)
        return str(destination)

    def initialize(
        self,
        version_id: str,
        mesh_digest: str,
        script_digest: str,
        total_operators: int,
        restart: bool = False,
    ) -> dict[str, Any]:
        archived_to: str | None = None
        if self.current.exists():
            existing_version: str | None = None
            if self._manifest_path().exists():
                try:
                    existing_manifest = read_json_strict(self._manifest_path())
                    existing_version = str(existing_manifest.get("version_id"))
                except BusinessError:
                    existing_version = None
            if restart or existing_version != version_id:
                archived_to = self.archive_current("version-change")
        self.current.mkdir(parents=True, exist_ok=True)
        manifest_path = self._manifest_path()
        if manifest_path.exists():
            manifest = read_json_strict(manifest_path)
            if manifest.get("version_id") != version_id:
                raise BusinessError(
                    ErrorCode.E_STATE,
                    "manifest version changed unexpectedly",
                    {"expected": version_id, "actual": manifest.get("version_id")},
                )
            return manifest
        manifest = {
            "format": MANIFEST_FORMAT,
            "version_id": version_id,
            "mesh_digest": mesh_digest,
            "script_digest": script_digest,
            "total_operators": total_operators,
            "created_utc": utc_now(),
            "completed_step": -1,
            "archived_previous_to": archived_to,
        }
        atomic_write_json(manifest_path, manifest)
        return manifest

    def write_snapshot(self, step: int, payload: dict[str, Any]) -> Path:
        path = self._snapshot_path(step)
        if path.exists():
            existing = self._read_snapshot_file(path)
            if existing.version_id != str(payload.get("version_id")):
                destination = self._archive_destination(
                    f"snapshot-version-change-step-{step:04d}"
                )
                destination.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination / path.name)
        digest = digest_json(payload)
        wrapper = {
            "format": SNAPSHOT_FORMAT,
            "integrity": {"algorithm": "sha256", "digest": digest},
            "payload": payload,
        }
        atomic_write_json(path, wrapper)
        manifest_path = self._manifest_path()
        manifest = read_json_strict(manifest_path)
        manifest["completed_step"] = max(int(manifest.get("completed_step", -1)), step)
        manifest["updated_utc"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        return path

    def _read_snapshot_file(self, path: Path) -> SnapshotRecord:
        match = STEP_PATTERN.match(path.name)
        if match is None:
            return SnapshotRecord(0, str(path), False, "unexpected snapshot filename", None, None)
        step = int(match.group(1))
        try:
            wrapper = read_json_strict(path)
            if not isinstance(wrapper, dict) or wrapper.get("format") != SNAPSHOT_FORMAT:
                raise BusinessError(
                    ErrorCode.E_SNAPSHOT_INVALID,
                    "snapshot format marker is invalid",
                    {"path": str(path)},
                )
            payload = wrapper.get("payload")
            integrity = wrapper.get("integrity", {})
            if not isinstance(payload, dict):
                raise BusinessError(
                    ErrorCode.E_SNAPSHOT_INVALID,
                    "snapshot payload is missing",
                    {"path": str(path)},
                )
            actual_digest = digest_json(payload)
            expected_digest = str(integrity.get("digest", ""))
            if integrity.get("algorithm") != "sha256" or actual_digest != expected_digest:
                raise BusinessError(
                    ErrorCode.E_SNAPSHOT_INVALID,
                    "snapshot integrity digest does not match",
                    {
                        "path": str(path),
                        "expected": expected_digest,
                        "actual": actual_digest,
                    },
                )
            if int(payload.get("step", -1)) != step:
                raise BusinessError(
                    ErrorCode.E_SNAPSHOT_INVALID,
                    "snapshot filename and payload step disagree",
                    {"path": str(path), "filename_step": step, "payload_step": payload.get("step")},
                )
            return SnapshotRecord(
                step=step,
                path=str(path),
                valid=True,
                reason=None,
                version_id=str(payload.get("version_id")),
                payload=payload,
            )
        except BusinessError as exc:
            return SnapshotRecord(
                step=step,
                path=str(path),
                valid=False,
                reason=str(exc),
                version_id=None,
                payload=None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return SnapshotRecord(
                step=step,
                path=str(path),
                valid=False,
                reason=f"malformed snapshot: {exc}",
                version_id=None,
                payload=None,
            )

    def inspect(self) -> StateInspection:
        manifest: dict[str, Any] | None = None
        if self._manifest_path().exists():
            try:
                raw_manifest = read_json_strict(self._manifest_path())
                if isinstance(raw_manifest, dict):
                    manifest = raw_manifest
            except BusinessError:
                manifest = None
        records: list[SnapshotRecord] = []
        if self.current.exists():
            for path in sorted(self.current.iterdir()):
                if path.is_file() and STEP_PATTERN.match(path.name):
                    record = self._read_snapshot_file(path)
                    if (
                        record.valid
                        and manifest is not None
                        and record.version_id != manifest.get("version_id")
                    ):
                        record = SnapshotRecord(
                            step=record.step,
                            path=record.path,
                            valid=False,
                            reason="snapshot version_id differs from manifest",
                            version_id=record.version_id,
                            payload=None,
                        )
                    records.append(record)
        records.sort(key=lambda record: record.step)
        by_step = {record.step: record for record in records}
        latest: int | None = None
        if 0 in by_step and by_step[0].valid:
            latest = 0
            while True:
                next_step = latest + 1
                if next_step not in by_step or not by_step[next_step].valid:
                    break
                latest = next_step
        invalid_count = sum(1 for record in records if not record.valid)
        return StateInspection(
            state_dir=str(self.root),
            current_dir=str(self.current),
            manifest=manifest,
            snapshots=records,
            latest_contiguous_valid_step=latest,
            invalid_snapshot_count=invalid_count,
        )

    def valid_snapshots_for_resume(self, version_id: str) -> list[SnapshotRecord]:
        inspection = self.inspect()
        if inspection.invalid_snapshot_count > 0:
            invalid = [record.to_summary() for record in inspection.snapshots if not record.valid]
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_INVALID,
                "state contains invalid or tampered snapshots; automatic resume is refused",
                {"invalid_snapshots": invalid},
            )
        if inspection.manifest is None:
            raise BusinessError(
                ErrorCode.E_STATE, "state manifest is missing or unreadable", {"state": str(self.root)}
            )
        if inspection.manifest.get("version_id") != version_id:
            raise BusinessError(
                ErrorCode.E_STATE,
                "state version_id does not match the current mesh and script",
                {
                    "expected": version_id,
                    "actual": inspection.manifest.get("version_id"),
                },
            )
        if inspection.latest_contiguous_valid_step is None:
            raise BusinessError(
                ErrorCode.E_STATE, "no valid initial snapshot is available", {"state": str(self.root)}
            )
        usable: list[SnapshotRecord] = []
        for record in inspection.snapshots:
            if (
                record.valid
                and record.step <= inspection.latest_contiguous_valid_step
                and record.version_id == version_id
            ):
                usable.append(record)
            else:
                break
        return usable

    def load_manifest(self) -> dict[str, Any]:
        return read_json_strict(self._manifest_path())

    def archive_entries(self) -> list[str]:
        if not self.archive.exists():
            return []
        return [str(path) for path in sorted(self.archive.iterdir())]
