"""Versioned, tamper-evident snapshot persistence and recovery."""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canon import (
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    load_json_file,
    load_plain_json_file,
    sha256_canonical,
)
from .errors import BusinessError


SNAPSHOT_FORMAT_VERSION = 1


@dataclass
class RecoveryInfo:
    mode: str
    start_event_seq: int = 0
    snapshot_seq: int | None = None
    snapshot_path: str | None = None
    invalid_snapshots: list[dict] = field(default_factory=list)
    archived_paths: list[str] = field(default_factory=list)
    had_interruption_marker: bool = False
    interruption_marker: dict | None = None
    detail: str | None = None

    def as_dict(self) -> dict:
        result = {
            "mode": self.mode,
            "start_event_seq": self.start_event_seq,
            "snapshot_seq": self.snapshot_seq,
            "snapshot_path": self.snapshot_path,
            "invalid_snapshots": self.invalid_snapshots,
            "archived_paths": self.archived_paths,
            "had_interruption_marker": self.had_interruption_marker,
        }
        if self.interruption_marker is not None:
            result["interruption_marker"] = self.interruption_marker
        if self.detail is not None:
            result["detail"] = self.detail
        return result


class SnapshotManager:
    def __init__(
        self,
        state_dir: Path,
        snapshot_dir: Path,
        version_id: str,
        event_script_digest: str,
    ) -> None:
        self.state_dir = state_dir
        self.snapshot_dir = snapshot_dir
        self.version_id = version_id
        self.event_script_digest = event_script_digest
        self.current_path = state_dir / "snapshot.json"
        self.meta_path = state_dir / "run_meta.json"
        self.log_path = state_dir / "simulation.log.jsonl"
        self.report_path = state_dir / "report.json"
        self.interruption_path = state_dir / "interrupted.json"
        self.version_archive_dir = snapshot_dir / version_id
        self.quarantine_dir = snapshot_dir / "quarantine"
        self.stale_dir = snapshot_dir / "stale"
        self.recovery = RecoveryInfo(mode="fresh")
        self.last_snapshot: dict | None = None

    def prepare(self) -> tuple[dict | None, RecoveryInfo]:
        """Inspect persisted state, returning the newest valid snapshot state."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.version_archive_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.stale_dir.mkdir(parents=True, exist_ok=True)

        old_meta = self._read_json_optional(self.meta_path)
        current_data, current_error = self._read_snapshot(self.current_path)
        marker = self._read_json_optional(self.interruption_path)

        meta_version = old_meta.get("version_id") if isinstance(old_meta, dict) else None
        current_version = current_data.get("version_id") if isinstance(current_data, dict) else None
        reset_reason: str | None = None
        if meta_version is not None and meta_version != self.version_id:
            reset_reason = "version_changed"
        elif meta_version is None and current_version is not None and current_version != self.version_id:
            reset_reason = "version_changed"

        if reset_reason == "version_changed":
            if self.current_path.exists():
                archived = self._archive_stale(self.current_path, "version_changed")
                self.recovery.archived_paths.append(archived)
            if self.meta_path.exists():
                self._archive_stale(self.meta_path, "version_changed_meta")
            if self.interruption_path.exists():
                self._archive_stale(self.interruption_path, "version_changed_marker")
            self.recovery = RecoveryInfo(
                mode="version_reset",
                archived_paths=self.recovery.archived_paths,
                detail="version identifier changed; old state was archived and invalidated",
            )
            self._write_meta()
            return None, self.recovery

        if current_error is not None and current_data is not None:
            self._quarantine(self.current_path, "invalid_current_snapshot", current_error)
        if current_data is not None:
            valid, error = self._validate_snapshot(current_data, self.current_path)
            if valid and current_data["state"].get("event_script_digest") != self.event_script_digest:
                archived = self._archive_invalidated(
                    self.current_path, "event_script_changed", "snapshot event script digest changed"
                )
                self.recovery = RecoveryInfo(
                    mode="event_script_reset",
                    archived_paths=[archived],
                    detail="event script changed while version remained stable; persisted state was invalidated",
                )
                self._write_meta()
                return None, self.recovery

        candidates: list[tuple[dict, Path]] = []
        invalid: list[dict] = []
        if current_data is not None:
            valid, error = self._validate_snapshot(current_data, self.current_path)
            if valid:
                candidates.append((current_data, self.current_path))
            else:
                self._quarantine(
                    self.current_path,
                    "invalid_current_snapshot",
                    error or "invalid snapshot",
                )
                invalid.append(
                    {
                        "path": str(self.current_path),
                        "code": "E_DIGEST",
                        "message": error or "snapshot is invalid",
                    }
                )
        elif current_error is not None:
            invalid.append(
                {
                    "path": str(self.current_path),
                    "code": "E_DIGEST",
                    "message": current_error,
                }
            )

        for path in self._archive_snapshot_paths():
            data, read_error = self._read_snapshot(path)
            if data is None:
                invalid.append(
                    {"path": str(path), "code": "E_DIGEST", "message": read_error or "unreadable snapshot"}
                )
                continue
            valid, error = self._validate_snapshot(data, path)
            if valid:
                candidates.append((data, path))
            else:
                invalid.append(
                    {"path": str(path), "code": "E_DIGEST", "message": error or "snapshot is invalid"}
                )
        self.recovery.invalid_snapshots = invalid

        if candidates:
            chosen_data, chosen_path = max(
                candidates,
                key=lambda item: (
                    int(item[0]["state"].get("event_seq", 0)),
                    int(item[0].get("snapshot_seq", 0)),
                ),
            )
            was_current = chosen_path == self.current_path
            if not was_current and not self.current_path.exists():
                mode = "archive_fallback"
            elif not was_current:
                mode = "archive_fallback"
            else:
                mode = "resume"
            self.recovery.mode = mode
            self.recovery.start_event_seq = int(chosen_data["state"].get("event_seq", 0))
            self.recovery.snapshot_seq = int(chosen_data.get("snapshot_seq", 0))
            self.recovery.snapshot_path = str(chosen_path)
            self.last_snapshot = chosen_data
            self._consume_interruption_marker(marker)
            self._write_meta()
            return copy.deepcopy(chosen_data["state"]), self.recovery

        if current_error is not None or current_data is not None or invalid:
            mode = "digest_restart"
            detail = "no valid snapshot could be recovered; simulation restarts from the beginning"
        else:
            mode = "fresh"
            detail = "no persisted state found"
        self.recovery = RecoveryInfo(
            mode=mode,
            invalid_snapshots=invalid,
            had_interruption_marker=bool(marker),
            interruption_marker=marker if isinstance(marker, dict) else None,
            detail=detail,
        )
        self._consume_interruption_marker(marker)
        self._write_meta()
        return None, self.recovery

    def inspect_integrity(self) -> dict:
        """Validate current and archived snapshots without mutating state."""
        invalid: list[dict] = []
        valid: list[dict] = []
        current_data, current_error = self._read_snapshot(self.current_path)
        if self.current_path.exists():
            if current_data is None:
                invalid.append(
                    {
                        "path": str(self.current_path),
                        "code": "E_DIGEST",
                        "message": current_error or "unreadable current snapshot",
                    }
                )
            else:
                ok, error = self._validate_snapshot(current_data, self.current_path)
                if ok:
                    valid.append(
                        {
                            "path": str(self.current_path),
                            "snapshot_seq": current_data.get("snapshot_seq"),
                            "event_seq": current_data.get("declared_event_seq"),
                        }
                    )
                else:
                    invalid.append(
                        {
                            "path": str(self.current_path),
                            "code": "E_DIGEST",
                            "message": error or "invalid current snapshot",
                        }
                    )
        for path in self._archive_snapshot_paths():
            data, read_error = self._read_snapshot(path)
            if data is None:
                invalid.append(
                    {"path": str(path), "code": "E_DIGEST", "message": read_error or "unreadable snapshot"}
                )
                continue
            ok, error = self._validate_snapshot(data, path)
            if ok:
                valid.append(
                    {
                        "path": str(path),
                        "snapshot_seq": data.get("snapshot_seq"),
                        "event_seq": data.get("declared_event_seq"),
                    }
                )
            else:
                invalid.append(
                    {"path": str(path), "code": "E_DIGEST", "message": error or "invalid snapshot"}
                )
        return {
            "ok": not invalid,
            "current_exists": self.current_path.exists(),
            "interruption_marker_exists": self.interruption_path.exists(),
            "valid_snapshots": valid,
            "invalid_snapshots": invalid,
        }

    def save_snapshot(self, state: dict, reason: str, snapshot_seq: int) -> dict:
        state_copy = copy.deepcopy(state)
        state_copy["snapshot_seq"] = snapshot_seq
        top = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "version_id": self.version_id,
            "snapshot_seq": snapshot_seq,
            "declared_event_seq": state_copy["event_seq"],
            "declared_journal_length": len(state_copy["journal"]),
            "reason": reason,
            "created_at_sim_ms": state_copy["now_ms"],
            "state": state_copy,
        }
        top["digest"] = sha256_canonical(top)
        archive_path = (
            self.version_archive_dir
            / f"snap-{snapshot_seq:08d}-{_safe_name(reason)}-{time.time_ns()}-{os.getpid()}.json"
        )
        atomic_write_json(archive_path, top)
        atomic_write_json(self.current_path, top)
        self.last_snapshot = top
        return top

    def write_log(self, journal: list[dict]) -> None:
        lines = [canonical_json(event) for event in journal]
        text = "\n".join(lines)
        if text:
            text += "\n"
        atomic_write_text(self.log_path, text)

    def write_report(self, report: dict, path: Path | None = None) -> Path:
        target = path or self.report_path
        atomic_write_json(target, report)
        return target

    def write_interruption_marker(self, marker: dict) -> Path:
        atomic_write_json(self.interruption_path, marker)
        return self.interruption_path

    def _read_snapshot(self, path: Path) -> tuple[dict | None, str | None]:
        if not path.exists():
            return None, None
        try:
            data = load_plain_json_file(path, code="E_DIGEST")
        except BusinessError as exc:
            return None, exc.message
        if not isinstance(data, dict):
            return None, "snapshot root is not an object"
        return data, None

    def _read_json_optional(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        try:
            return load_plain_json_file(path)
        except BusinessError:
            return None

    def _validate_snapshot(self, data: dict, path: Path) -> tuple[bool, str | None]:
        digest = data.get("digest")
        if not isinstance(digest, str) or not digest:
            return False, "snapshot digest is missing"
        body = copy.deepcopy(data)
        body.pop("digest", None)
        actual_digest = sha256_canonical(body)
        if digest != actual_digest:
            return False, "snapshot content digest mismatch"
        if data.get("format_version") != SNAPSHOT_FORMAT_VERSION:
            return False, "unsupported snapshot format"
        if data.get("version_id") != self.version_id:
            return False, "snapshot version identifier mismatch"
        state = data.get("state")
        if not isinstance(state, dict):
            return False, "snapshot state is missing"
        declared_event_seq = data.get("declared_event_seq")
        actual_event_seq = state.get("event_seq")
        if (
            not isinstance(declared_event_seq, int)
            or not isinstance(actual_event_seq, int)
            or declared_event_seq != actual_event_seq
        ):
            return False, "declared event sequence does not match actual state"
        journal = state.get("journal")
        if not isinstance(journal, list):
            return False, "snapshot journal is missing"
        if data.get("declared_journal_length") != len(journal):
            return False, "declared journal length does not match actual journal"
        if len(journal) != actual_event_seq:
            return False, "event sequence does not match journal length"
        for expected, event in enumerate(journal, start=1):
            if not isinstance(event, dict) or event.get("seq") != expected:
                return False, "journal sequence is not contiguous"
        snapshot_seq = data.get("snapshot_seq")
        if not isinstance(snapshot_seq, int) or snapshot_seq < 0:
            return False, "snapshot sequence is invalid"
        if state.get("snapshot_seq") != snapshot_seq:
            return False, "snapshot sequence does not match embedded state"
        if state.get("event_script_digest") != self.event_script_digest:
            return False, "snapshot event script digest mismatch"
        return True, None

    def _archive_snapshot_paths(self) -> list[Path]:
        paths = list(self.version_archive_dir.glob("snap-*.json"))
        paths.sort(key=lambda path: path.name)
        return paths

    def _archive_stale(self, source: Path, reason: str) -> str:
        target_dir = self.stale_dir / reason
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{source.name}-{time.time_ns()}-{os.getpid()}"
        shutil.copy2(source, target)
        return str(target)

    def _archive_invalidated(self, source: Path, reason: str, message: str) -> str:
        target_dir = self.quarantine_dir / reason
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{source.name}-{time.time_ns()}-{os.getpid()}"
        shutil.copy2(source, target)
        sidecar = target.with_suffix(target.suffix + ".invalid.json")
        atomic_write_json(
            sidecar,
            {"reason": reason, "message": message, "source": str(source)},
        )
        return str(target)

    def _quarantine(self, source: Path, reason: str, message: str) -> None:
        self._archive_invalidated(source, reason, message)

    def _write_meta(self) -> None:
        existing = self._read_json_optional(self.meta_path)
        run_count = 1
        created_ns = time.time_ns()
        if isinstance(existing, dict) and existing.get("version_id") == self.version_id:
            run_count = int(existing.get("run_count", 0)) + 1
            created_ns = int(existing.get("created_ns", created_ns))
        atomic_write_json(
            self.meta_path,
            {
                "format_version": SNAPSHOT_FORMAT_VERSION,
                "version_id": self.version_id,
                "event_script_digest": self.event_script_digest,
                "run_count": run_count,
                "created_ns": created_ns,
                "updated_ns": time.time_ns(),
            },
        )

    def _consume_interruption_marker(self, marker: Any) -> None:
        if not isinstance(marker, dict):
            return
        self.recovery.had_interruption_marker = True
        self.recovery.interruption_marker = marker
        target_dir = self.version_archive_dir / "interruptions"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"interrupted-{time.time_ns()}-{os.getpid()}.json"
        atomic_write_json(
            target,
            {
                "consumed": True,
                "marker": marker,
                "consumed_at_ns": time.time_ns(),
            },
        )
        try:
            self.interruption_path.unlink()
        except FileNotFoundError:
            pass


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)[:48]
