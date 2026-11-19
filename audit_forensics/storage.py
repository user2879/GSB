"""Atomic file replacement, checksummed envelopes, and workspace layout."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_text
from .errors import (
    E_CHECKPOINT_CORRUPT,
    E_CHECKPOINT_INVALID,
    E_CHECKPOINT_MISSING,
    E_SNAPSHOT_INVALID,
    E_SNAPSHOT_STALE,
    E_STATE_SCHEMA_MISMATCH,
    business_error,
)


CHECKPOINT_SCHEMA_VERSION = "audit-checkpoint.v1"
CASE_SNAPSHOT_SCHEMA_VERSION = "audit-case-snapshot.v1"


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path

    @property
    def state_dir(self) -> Path:
        return self.root / ".audit"

    @property
    def checkpoint(self) -> Path:
        return self.state_dir / "checkpoint.json"

    @property
    def previous_checkpoint(self) -> Path:
        return self.state_dir / "checkpoint.prev.json"

    @property
    def case_snapshot(self) -> Path:
        return self.state_dir / "case_snapshot.json"

    @property
    def lock_file(self) -> Path:
        return self.state_dir / "workspace.lock"

    @property
    def report(self) -> Path:
        return self.root / "report.json"

    @property
    def graph(self) -> Path:
        return self.root / "case_graph.json"

    def ensure(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(
    path_like: str | Path,
    content: str,
    *,
    pause_key: str | None = None,
) -> None:
    """Write and fsync a temporary sibling, then replace the target atomically."""

    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        pause_at = os.environ.get("AUDIT_FORENSICS_TEST_PAUSE_AT")
        if pause_key and pause_at == pause_key:
            ready = os.environ.get("AUDIT_FORENSICS_TEST_READY_FILE")
            if ready:
                Path(ready).write_text(
                    json.dumps(
                        {
                            "path": str(path),
                            "temporary": temporary,
                            "pause_key": pause_key,
                        },
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
            time.sleep(
                max(
                    0.0,
                    float(os.environ.get("AUDIT_FORENSICS_TEST_PAUSE_SECONDS", "5")),
                )
            )

        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_json(
    path_like: str | Path,
    value: Any,
    *,
    pause_key: str | None = None,
) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path_like, content, pause_key=pause_key)


def cleanup_temporary_files(state_dir: Path) -> list[str]:
    removed: list[str] = []
    if not state_dir.exists():
        return removed
    for candidate in state_dir.glob(".*.tmp"):
        try:
            candidate.unlink()
            removed.append(str(candidate))
        except OSError:
            pass
    return removed


def make_envelope(schema_version: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "payload": payload,
        "checksum": sha256_text(canonical_json(payload)),
    }


@dataclass
class Envelope:
    schema_version: str
    payload: dict[str, Any]
    checksum: str
    schema_matches: bool
    path: Path


class EnvelopeError(ValueError):
    def __init__(self, message: str, *, code: str = E_CHECKPOINT_INVALID) -> None:
        super().__init__(message)
        self.code = code


def read_envelope(path_like: str | Path, expected_schema: str) -> Envelope:
    path = Path(path_like)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EnvelopeError(
            f"state file does not exist: {path}", code=E_CHECKPOINT_MISSING
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvelopeError(
            f"cannot parse state file {path}: {exc}",
            code=E_CHECKPOINT_CORRUPT,
        ) from exc
    if not isinstance(raw, dict):
        raise EnvelopeError(
            f"state file root is not an object: {path}",
            code=E_CHECKPOINT_CORRUPT,
        )
    schema_version = raw.get("schema_version")
    payload = raw.get("payload")
    checksum = raw.get("checksum")
    if not isinstance(schema_version, str) or not isinstance(payload, dict):
        raise EnvelopeError(
            f"state file is missing schema_version/payload: {path}",
            code=E_CHECKPOINT_CORRUPT,
        )
    actual_checksum = sha256_text(canonical_json(payload))
    if not isinstance(checksum, str) or checksum != actual_checksum:
        raise EnvelopeError(
            f"state checksum mismatch: {path}", code=E_CHECKPOINT_CORRUPT
        )
    return Envelope(
        schema_version=schema_version,
        payload=payload,
        checksum=checksum,
        schema_matches=(schema_version == expected_schema),
        path=path,
    )


@dataclass
class CheckpointLoad:
    has_state: bool
    payload: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    path: str = ""
    schema_version: str = ""
    recovered_from_previous: bool = False
    force_rebuild: bool = False
    errors: list[dict[str, Any]] = field(default_factory=list)
    notices: list[dict[str, Any]] = field(default_factory=list)


def load_checkpoint(paths: WorkspacePaths) -> CheckpointLoad:
    candidates = [paths.checkpoint, paths.previous_checkpoint]
    first_error: dict[str, Any] | None = None
    for index, candidate in enumerate(candidates):
        if not candidate.exists():
            continue
        try:
            envelope = read_envelope(candidate, CHECKPOINT_SCHEMA_VERSION)
        except EnvelopeError as exc:
            if first_error is None:
                first_error = business_error(
                    exc.code,
                    str(exc),
                    source=str(candidate),
                )
            continue

        notices: list[dict[str, Any]] = []
        force_rebuild = False
        if not envelope.schema_matches:
            force_rebuild = True
            notices.append(
                business_error(
                    E_STATE_SCHEMA_MISMATCH,
                    "checkpoint structure version differs; rebuilding derived state",
                    source=str(candidate),
                    details={
                        "found": envelope.schema_version,
                        "expected": CHECKPOINT_SCHEMA_VERSION,
                    },
                )
            )
        if index > 0:
            notices.append(
                business_error(
                    E_CHECKPOINT_CORRUPT,
                    "current checkpoint was unavailable; recovered previous complete checkpoint",
                    source=str(candidate),
                )
            )
        return CheckpointLoad(
            has_state=True,
            payload=envelope.payload,
            checksum=envelope.checksum,
            path=str(candidate),
            schema_version=envelope.schema_version,
            recovered_from_previous=index > 0,
            force_rebuild=force_rebuild,
            errors=(
                []
                if index > 0
                else ([] if first_error is None else [first_error])
            ),
            notices=notices,
        )

    return CheckpointLoad(
        has_state=False,
        errors=[] if first_error is None else [first_error],
    )


def write_checkpoint(paths: WorkspacePaths, payload: dict[str, Any]) -> dict[str, Any]:
    paths.ensure()
    if paths.checkpoint.exists():
        try:
            current = read_envelope(paths.checkpoint, CHECKPOINT_SCHEMA_VERSION)
        except EnvelopeError:
            pass
        else:
            atomic_write_json(
                paths.previous_checkpoint,
                make_envelope(current.schema_version, current.payload),
            )
    envelope = make_envelope(CHECKPOINT_SCHEMA_VERSION, payload)
    atomic_write_json(paths.checkpoint, envelope, pause_key="checkpoint")
    return envelope


def write_case_snapshot(paths: WorkspacePaths, payload: dict[str, Any]) -> dict[str, Any]:
    paths.ensure()
    envelope = make_envelope(CASE_SNAPSHOT_SCHEMA_VERSION, payload)
    atomic_write_json(paths.case_snapshot, envelope, pause_key="case_snapshot")
    return envelope


@dataclass
class SnapshotLoad:
    valid: bool
    payload: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    errors: list[dict[str, Any]] = field(default_factory=list)
    notices: list[dict[str, Any]] = field(default_factory=list)


def load_case_snapshot(
    paths: WorkspacePaths,
    *,
    checkpoint_checksum: str,
    checkpoint_generation: int,
) -> SnapshotLoad:
    if not paths.case_snapshot.exists():
        return SnapshotLoad(
            valid=False,
            errors=[
                business_error(
                    E_SNAPSHOT_INVALID,
                    "case snapshot is missing",
                    source=str(paths.case_snapshot),
                )
            ],
        )
    try:
        envelope = read_envelope(
            paths.case_snapshot, CASE_SNAPSHOT_SCHEMA_VERSION
        )
    except EnvelopeError as exc:
        return SnapshotLoad(
            valid=False,
            errors=[
                business_error(exc.code, str(exc), source=str(paths.case_snapshot))
            ],
        )
    payload = envelope.payload
    errors: list[dict[str, Any]] = []
    notices: list[dict[str, Any]] = []
    if not envelope.schema_matches:
        errors.append(
            business_error(
                E_SNAPSHOT_INVALID,
                "case snapshot structure version differs",
                source=str(paths.case_snapshot),
                details={
                    "found": envelope.schema_version,
                    "expected": CASE_SNAPSHOT_SCHEMA_VERSION,
                },
            )
        )
    if payload.get("checkpoint_checksum") != checkpoint_checksum:
        errors.append(
            business_error(
                E_SNAPSHOT_STALE,
                "case snapshot refers to a different checkpoint",
                source=str(paths.case_snapshot),
            )
        )
    if payload.get("generation") != checkpoint_generation:
        errors.append(
            business_error(
                E_SNAPSHOT_STALE,
                "case snapshot generation differs from checkpoint generation",
                source=str(paths.case_snapshot),
                details={
                    "snapshot_generation": payload.get("generation"),
                    "checkpoint_generation": checkpoint_generation,
                },
            )
        )
    if errors:
        notices.append(
            business_error(
                E_SNAPSHOT_STALE,
                "derived case snapshot must be rebuilt from checkpoint",
                source=str(paths.case_snapshot),
            )
        )
    return SnapshotLoad(
        valid=not errors,
        payload=payload,
        checksum=envelope.checksum,
        errors=errors,
        notices=notices,
    )
