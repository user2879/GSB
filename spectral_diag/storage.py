"""Atomic snapshot persistence, commit markers, and crash recovery."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, sha256_bytes
from .errors import DiagnosticError, ErrorCode

STORAGE_SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Windows cannot fsync directory handles.  The file itself was fsynced
        # and the pointer replacement still provides the commit boundary.
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_bytes(value))


def read_json(path: Path) -> Any:
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


class Workspace:
    """A local, filesystem-only snapshot store with an explicit commit point."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.snapshots = self.root / "snapshots"
        self.orphans = self.root / "orphans"
        self.runtime = self.root / "runtime"
        self.logs = self.root / "logs"
        self.active_path = self.runtime / "active.json"
        self.current_path = self.root / "CURRENT.json"
        self.committed_path = self.root / "committed-snapshots.json"
        self.recovery_path = self.logs / "recovery.jsonl"

    def ensure(self) -> Workspace:
        for directory in (self.root, self.snapshots, self.orphans, self.runtime, self.logs):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def _validate_run_id(self, run_id: str) -> str:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise DiagnosticError(
                code=ErrorCode.INPUT_INVALID,
                message="run_id contains unsupported characters",
                details={"run_id": run_id, "pattern": RUN_ID_PATTERN.pattern},
            )
        return run_id

    def begin(self, run_id: str) -> dict[str, Any]:
        self.ensure()
        self._validate_run_id(run_id)
        if self.active_path.exists():
            active = read_json(self.active_path)
            raise DiagnosticError(
                code=ErrorCode.INPUT_INVALID,
                message="an unfinished run is active; recover the workspace first",
                details={"active": active},
            )
        active = {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "run_id": run_id,
            "pid": os.getpid(),
            "started_at": utc_now(),
            "state": "running",
            "exit_report": None,
        }
        atomic_write_json(self.active_path, active)
        self._log({"event": "run_started", "run_id": run_id, "pid": os.getpid(), "at": utc_now()})
        return active

    def _snapshot_payload_files(
        self,
        run_id: str,
        pipeline_result: dict[str, Any],
        closed: bool,
    ) -> dict[str, Any]:
        status = str(pipeline_result.get("status"))
        injection = {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "run_id": run_id,
            "injections": pipeline_result.get("injections", []),
            "injection_sha256": pipeline_result.get("injection_sha256"),
        }
        samples = {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "run_id": run_id,
            "samples": pipeline_result.get("samples"),
            "reference_truth": pipeline_result.get("reference_truth"),
            "integration": pipeline_result.get("integration"),
        }
        version = {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "run_id": run_id,
            "version_stamp": pipeline_result.get("version_stamp"),
            "storage_schema_version": STORAGE_SCHEMA_VERSION,
        }
        files: dict[str, Any] = {
            "config.json": pipeline_result.get("config"),
            "scene.json": pipeline_result.get("scene"),
            "samples.json": samples,
            "matrix.json": {
                "schema_version": STORAGE_SCHEMA_VERSION,
                "run_id": run_id,
                "status": status,
                "matrix": pipeline_result.get("matrix"),
            },
            "attribution.json": {
                "schema_version": STORAGE_SCHEMA_VERSION,
                "run_id": run_id,
                "attribution": pipeline_result.get("attribution"),
            },
            "injection.json": injection,
            "version.json": version,
            "report.json": pipeline_result,
        }
        if closed:
            files["CLOSED.json"] = {
                "schema_version": STORAGE_SCHEMA_VERSION,
                "run_id": run_id,
                "closed": True,
                "status": status,
                "closed_at": utc_now(),
                "exit_status": "normal",
            }
        return files

    def stage_and_commit(
        self,
        run_id: str,
        pipeline_result: dict[str, Any],
        *,
        crash_after_stage: bool = False,
        exit_fn: Callable[[int], None] = os._exit,
    ) -> dict[str, Any]:
        """Stage a snapshot; publish CURRENT only for a normal completed run."""

        self.ensure()
        self._validate_run_id(run_id)
        snapshot_id = f"snap-{run_id}"
        final_dir = self.snapshots / snapshot_id
        staging_dir = self.snapshots / f".staging-{run_id}"
        if final_dir.exists() or staging_dir.exists():
            raise DiagnosticError(
                code=ErrorCode.INPUT_INVALID,
                message="snapshot id already exists",
                details={"snapshot_id": snapshot_id},
            )
        staging_dir.mkdir(parents=True)
        files = self._snapshot_payload_files(run_id, pipeline_result, closed=not crash_after_stage)
        file_hashes: dict[str, str] = {}
        for name in sorted(files):
            data = canonical_bytes(files[name])
            atomic_write_bytes(staging_dir / name, data)
            file_hashes[name] = sha256_bytes(data)
        manifest = {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "run_id": run_id,
            "created_at": utc_now(),
            "status": pipeline_result.get("status"),
            "closed": not crash_after_stage,
            "exit_status": "forced_termination" if crash_after_stage else "normal",
            "version_stamp": pipeline_result.get("version_stamp"),
            "files": file_hashes,
            "result_sha256": pipeline_result.get("result_sha256"),
        }
        manifest_data = canonical_bytes(manifest)
        atomic_write_bytes(staging_dir / "manifest.json", manifest_data)
        manifest_hash = sha256_bytes(manifest_data)
        _fsync_directory(staging_dir)
        os.rename(staging_dir, final_dir)
        _fsync_directory(self.snapshots)

        if crash_after_stage:
            self._log(
                {
                    "event": "snapshot_staged_uncommitted",
                    "run_id": run_id,
                    "snapshot_id": snapshot_id,
                    "manifest_sha256": manifest_hash,
                    "at": utc_now(),
                }
            )
            # This models a process terminated after durable writes but before
            # the CURRENT pointer commit.  No close marker or exit report exists.
            exit_fn(86)
            raise RuntimeError("forced-termination callback returned before process exit")

        history: list[str] = []
        if self.current_path.exists():
            try:
                previous_pointer = read_json(self.current_path)
                history = [
                    str(value)
                    for value in previous_pointer.get("history", [])
                    if isinstance(value, str)
                ]
                previous_id = previous_pointer.get("snapshot_id")
                if isinstance(previous_id, str) and previous_id not in history:
                    history.append(previous_id)
            except (OSError, json.JSONDecodeError):
                history = []
        pointer = {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "manifest_sha256": manifest_hash,
            "committed_at": utc_now(),
            "history": history,
        }
        atomic_write_json(self.current_path, pointer)
        committed = []
        if self.committed_path.exists():
            committed = read_json(self.committed_path)
        committed.append(
            {
                "snapshot_id": snapshot_id,
                "manifest_sha256": manifest_hash,
                "run_id": run_id,
                "committed_at": pointer["committed_at"],
            }
        )
        atomic_write_json(self.committed_path, committed)
        self._log(
            {
                "event": "snapshot_committed",
                "run_id": run_id,
                "snapshot_id": snapshot_id,
                "manifest_sha256": manifest_hash,
                "at": utc_now(),
            }
        )
        if self.active_path.exists():
            self.active_path.unlink()
            _fsync_directory(self.runtime)
        self._log({"event": "run_closed", "run_id": run_id, "at": utc_now()})
        return {
            "snapshot_id": snapshot_id,
            "manifest_sha256": manifest_hash,
            "committed": True,
            "closed": True,
            "files": sorted(files) + ["manifest.json"],
        }

    def _load_pointer(self) -> dict[str, Any] | None:
        if not self.current_path.exists():
            return None
        pointer = read_json(self.current_path)
        if int(pointer.get("schema_version", -1)) != STORAGE_SCHEMA_VERSION:
            raise DiagnosticError(
                code=ErrorCode.SNAPSHOT_VERSION,
                message="CURRENT pointer schema version is unsupported",
                details={
                    "expected": STORAGE_SCHEMA_VERSION,
                    "actual": pointer.get("schema_version"),
                },
            )
        return pointer

    def _validate_snapshot(self, snapshot_id: str, expected_manifest_hash: str | None = None) -> dict[str, Any]:
        snapshot_dir = self.snapshots / snapshot_id
        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.exists():
            raise DiagnosticError(
                code=ErrorCode.INPUT_INVALID,
                message="committed snapshot manifest is missing",
                details={"snapshot_id": snapshot_id},
            )
        manifest_bytes = manifest_path.read_bytes()
        actual_manifest_hash = sha256_bytes(manifest_bytes)
        if expected_manifest_hash is not None and actual_manifest_hash != expected_manifest_hash:
            raise DiagnosticError(
                code=ErrorCode.INPUT_INVALID,
                message="snapshot manifest hash does not match CURRENT pointer",
                details={
                    "snapshot_id": snapshot_id,
                    "expected": expected_manifest_hash,
                    "actual": actual_manifest_hash,
                },
            )
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if int(manifest.get("schema_version", -1)) != STORAGE_SCHEMA_VERSION:
            raise DiagnosticError(
                code=ErrorCode.SNAPSHOT_VERSION,
                message="snapshot manifest schema version is unsupported",
                details={
                    "expected": STORAGE_SCHEMA_VERSION,
                    "actual": manifest.get("schema_version"),
                    "snapshot_id": snapshot_id,
                },
            )
        for name, digest in manifest.get("files", {}).items():
            path = snapshot_dir / name
            if not path.exists() or sha256_bytes(path.read_bytes()) != digest:
                raise DiagnosticError(
                    code=ErrorCode.INPUT_INVALID,
                    message="snapshot file failed integrity validation",
                    details={"snapshot_id": snapshot_id, "file": name, "expected": digest},
                )
        return manifest

    def load_current(self) -> dict[str, Any] | None:
        self.ensure()
        pointer = self._load_pointer()
        if pointer is None:
            return None
        snapshot_id = str(pointer["snapshot_id"])
        manifest = self._validate_snapshot(snapshot_id, str(pointer.get("manifest_sha256")))
        if not manifest.get("closed"):
            raise DiagnosticError(
                code=ErrorCode.INPUT_INVALID,
                message="CURRENT points at a snapshot without a normal close marker",
                details={"snapshot_id": snapshot_id},
            )
        snapshot_dir = self.snapshots / snapshot_id
        report = read_json(snapshot_dir / "report.json")
        closed = read_json(snapshot_dir / "CLOSED.json")
        return {
            "pointer": pointer,
            "manifest": manifest,
            "report": report,
            "closed_marker": closed,
            "snapshot_dir": str(snapshot_dir),
        }

    def recover(self) -> dict[str, Any]:
        self.ensure()
        active = read_json(self.active_path) if self.active_path.exists() else None
        current: dict[str, Any] | None = None
        integrity_error: dict[str, Any] | None = None
        try:
            current = self.load_current()
        except DiagnosticError as error:
            integrity_error = error.to_dict()

        referenced = None if current is None else current["manifest"]["snapshot_id"]
        historical: set[str] = set()
        if current is not None:
            historical.update(str(value) for value in current["pointer"].get("history", []))
        if self.committed_path.exists():
            try:
                committed_records = read_json(self.committed_path)
                historical.update(str(item["snapshot_id"]) for item in committed_records)
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                historical = set()
        if referenced is not None:
            historical.add(str(referenced))
        orphan_records: list[dict[str, Any]] = []
        for entry in sorted(self.snapshots.iterdir()):
            if not entry.is_dir() or entry.name in historical:
                continue
            destination = self.orphans / f"{entry.name}-{time.time_ns()}"
            record: dict[str, Any] = {
                "event": "orphan_recovered",
                "name": entry.name,
                "at": utc_now(),
                "destination": str(destination),
            }
            manifest_path = entry / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = read_json(manifest_path)
                    record.update(
                        {
                            "run_id": manifest.get("run_id"),
                            "closed": manifest.get("closed"),
                            "exit_status": manifest.get("exit_status"),
                            "schema_version": manifest.get("schema_version"),
                        }
                    )
                except (OSError, json.JSONDecodeError) as error:
                    record["manifest_error"] = str(error)
            shutil.move(str(entry), str(destination))
            orphan_records.append(record)
            self._log(record)

        interrupted = active is not None and not (
            current is not None and current["manifest"].get("run_id") == active.get("run_id")
        )
        if active is not None:
            self._log(
                {
                    "event": "unfinished_run_detected",
                    "run_id": active.get("run_id"),
                    "pid": active.get("pid"),
                    "started_at": active.get("started_at"),
                    "at": utc_now(),
                    "forced_exit_report": None,
                    "note": "no normal exit report exists; this is not fabricated",
                }
            )
            self.active_path.unlink()
            _fsync_directory(self.runtime)

        report = {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "recovered_at": utc_now(),
            "state": (
                "committed"
                if current is not None and integrity_error is None
                else "no_committed_state"
                if current is None
                else "commit_integrity_error"
            ),
            "interrupted_run": active if interrupted else None,
            "normal_exit_report_fabricated": False,
            "current": None
            if current is None
            else {
                "snapshot_id": current["manifest"]["snapshot_id"],
                "manifest": current["manifest"],
                "closed_marker": current["closed_marker"],
                "result_sha256": current["report"].get("result_sha256"),
                "status": current["report"].get("status"),
            },
            "orphans": orphan_records,
            "orphan_count": len(orphan_records),
            "integrity_error": integrity_error,
        }
        atomic_write_json(self.logs / "last_recovery.json", report)
        self._log({"event": "recovery_completed", "state": report["state"], "at": utc_now()})
        if integrity_error is not None:
            raise DiagnosticError(
                code=ErrorCode(integrity_error["code"]),
                message=integrity_error["message"],
                details={"recovery": report},
            )
        return report

    def load_snapshot_file(self, snapshot_id: str, name: str) -> Any:
        manifest = self._validate_snapshot(snapshot_id)
        if name not in manifest.get("files", {}):
            raise DiagnosticError(
                code=ErrorCode.INPUT_INVALID,
                message="snapshot does not contain the requested file",
                details={"snapshot_id": snapshot_id, "file": name},
            )
        return read_json(self.snapshots / snapshot_id / name)

    def _log(self, record: dict[str, Any]) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        with open(self.recovery_path, "ab") as handle:
            handle.write(canonical_bytes(record))
            handle.flush()
            os.fsync(handle.fileno())
