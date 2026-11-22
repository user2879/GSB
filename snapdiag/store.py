"""SQLite-backed session, copy-on-write snapshot and rollback engine."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .bitmap import (
    bitmap_checksum,
    bitmap_length,
    empty_bitmap,
    popcount,
    set_bit,
    set_bits,
    validate_bitmap,
)
from .errors import (
    DiagnosticError,
    E_CHAIN,
    E_EXISTS,
    E_FORMAT,
    E_GEOM,
    E_INTEGRITY,
    E_STATE,
    E_VERSION,
)
from .geometry import (
    SUPPORTED_BLOCK_SIZES,
    VolumeGeometry,
    generate_volume,
    validate_geometry,
)
from .manifest import (
    FORMAT_VERSIONS,
    ManifestResolver,
    NormalizedManifest,
    SnapshotManifest,
    build_manifest_model,
    decode_manifest,
    encode_manifest,
    encode_manifest_text,
    load_manifest_file,
    make_snapshot_manifest,
    peek_format_version,
)
from .util import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_text,
    copy_file_atomic,
    display_digest,
    ensure_readonly,
    ensure_writable,
    fsync_directory,
    make_tree_writable,
    sha256_bytes,
    sha256_file,
    unique_suffix,
    utc_now,
)


SCHEMA_VERSION = 1
DEFAULT_THRESHOLD = 0.25
SESSION_FILE = "session.json"
MANIFEST_FILE = "manifest.json"
DEFAULT_MANIFEST_FORMAT = "m3"


def _bool_to_float(value: Any) -> bool:
    return isinstance(value, bool)


def _extract_geometry(spec: Mapping[str, Any]) -> tuple[str, int, int]:
    volume = spec.get("volume")
    if not isinstance(volume, Mapping):
        raise DiagnosticError(
            E_GEOM,
            "session volume declaration is missing",
            field="volume",
        )
    image = volume.get("image")
    if not isinstance(image, str) or not image:
        raise DiagnosticError(
            E_GEOM,
            "volume.image must be a relative file name",
            field="volume.image",
        )
    return image, volume.get("block_size"), volume.get("block_count")  # type: ignore[return-value]


@dataclass(frozen=True)
class SessionSpec:
    session_dir: Path
    image_name: str
    block_size: int
    block_count: int
    manifest_format: str
    threshold: Fraction
    rollback_target: str | None
    snapshot_plan: tuple[str, ...]
    writes: tuple[Mapping[str, Any], ...]
    steps: tuple[Mapping[str, Any], ...]
    raw: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, session_dir: Path, value: Mapping[str, Any]) -> "SessionSpec":
        if not isinstance(value, Mapping):
            raise DiagnosticError(E_FORMAT, "session manifest must be an object", field="session")
        schema = value.get("schema_version", 1)
        if schema != 1:
            raise DiagnosticError(
                E_VERSION,
                "unsupported session schema_version",
                field="schema_version",
                details={"expected": 1, "actual": schema},
            )
        image_name, raw_size, raw_count = _extract_geometry(value)
        # Geometry validation is completed after the image has been generated or copied.
        if isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise DiagnosticError(E_GEOM, "volume.block_size must be an integer", field="block_size")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise DiagnosticError(E_GEOM, "volume.block_count must be an integer", field="block_count")
        if raw_size not in SUPPORTED_BLOCK_SIZES:
            raise DiagnosticError(
                E_GEOM,
                "block_size is not in the supported set",
                field="block_size",
                details={"actual": raw_size, "supported": sorted(SUPPORTED_BLOCK_SIZES)},
            )
        if raw_count <= 0:
            raise DiagnosticError(
                E_GEOM,
                "block_count must be greater than zero",
                field="block_count",
                details={"actual": raw_count},
            )
        manifest_format = value.get("manifest_format", DEFAULT_MANIFEST_FORMAT)
        if manifest_format not in FORMAT_VERSIONS:
            raise DiagnosticError(
                E_FORMAT,
                "manifest_format is unknown",
                field="manifest_format",
                details={"actual": manifest_format, "supported": list(FORMAT_VERSIONS)},
            )
        threshold_raw = value.get("rollback_threshold", DEFAULT_THRESHOLD)
        if _bool_to_float(threshold_raw) or not isinstance(threshold_raw, (int, float, str)):
            raise DiagnosticError(
                E_FORMAT,
                "rollback_threshold must be numeric",
                field="rollback_threshold",
            )
        try:
            threshold = Fraction(str(threshold_raw))
        except (ValueError, ZeroDivisionError):
            raise DiagnosticError(
                E_FORMAT,
                "rollback_threshold must be a finite ratio",
                field="rollback_threshold",
                details={"actual": threshold_raw},
            )
        if threshold < 0 or threshold > 1:
            raise DiagnosticError(
                E_FORMAT,
                "rollback_threshold must be between 0 and 1",
                field="rollback_threshold",
                details={"actual": threshold_raw},
            )
        target = value.get("rollback_target")
        if target is not None and (not isinstance(target, str) or not target):
            raise DiagnosticError(
                E_FORMAT,
                "rollback_target must be a non-empty string or null",
                field="rollback_target",
            )
        plan_value = value.get("snapshot_plan", [])
        if not isinstance(plan_value, list) or any(not isinstance(item, str) or not item for item in plan_value):
            raise DiagnosticError(
                E_FORMAT,
                "snapshot_plan must be a list of non-empty names",
                field="snapshot_plan",
            )
        writes_value = value.get("writes", [])
        if not isinstance(writes_value, list) or any(not isinstance(item, Mapping) for item in writes_value):
            raise DiagnosticError(
                E_FORMAT,
                "writes must be a list of objects",
                field="writes",
            )
        steps_value = value.get("steps", [])
        if not isinstance(steps_value, list) or any(not isinstance(item, Mapping) for item in steps_value):
            raise DiagnosticError(
                E_FORMAT,
                "steps must be a list of objects",
                field="steps",
            )
        return cls(
            session_dir=session_dir,
            image_name=image_name,
            block_size=raw_size,
            block_count=raw_count,
            manifest_format=manifest_format,
            threshold=threshold,
            rollback_target=target,
            snapshot_plan=tuple(plan_value),
            writes=tuple(writes_value),
            steps=tuple(steps_value),
            raw=dict(value),
        )


class JsonEventLogger:
    """Append-only JSON-lines diagnostics log with a stable envelope."""

    def __init__(self, session_dir: Path):
        self.path = session_dir / "logs" / "diagnostics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        event: str,
        *,
        code: str | None = None,
        operation: str | None = None,
        snapshot_id: str | None = None,
        field: str | None = None,
        block_index: int | None = None,
        expected: Any = None,
        actual: Any = None,
        **details: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": utc_now(),
            "event": event,
            "code": code,
            "operation": operation,
            "snapshot_id": snapshot_id,
            "field": field,
            "block_index": block_index,
            "expected": expected,
            "actual": actual,
            "details": details,
        }
        line = canonical_json_text(record).encode("utf-8") + b"\n"
        with open(self.path, "ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def emit_error(self, error: DiagnosticError, *, operation: str) -> dict[str, Any]:
        details = dict(error.details)
        block_index = details.pop("block_index", None)
        expected = details.pop("expected_digest", details.pop("expected", None))
        actual = details.pop("actual_digest", details.pop("actual", None))
        return self.emit(
            "error",
            code=error.code,
            operation=operation,
            snapshot_id=error.snapshot_id,
            field=error.field,
            block_index=block_index if isinstance(block_index, int) else None,
            expected=expected,
            actual=actual,
            message=error.message,
            **details,
        )

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except FileNotFoundError:
            return []
        records: list[dict[str, Any]] = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"event": "corrupt_log_line", "raw": line.rstrip("\n")})
        return records[-limit:] if limit is not None else records


class SimulatedInterruption(RuntimeError):
    """Raised by an explicit fault point to model a forcibly killed process."""


def _schema_sql() -> str:
    return """
    CREATE TABLE meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE snapshots (
        snapshot_id TEXT PRIMARY KEY,
        parent_snapshot_id TEXT,
        created_seq INTEGER NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        block_count INTEGER NOT NULL,
        block_size INTEGER NOT NULL,
        bitmap BLOB NOT NULL,
        bitmap_length INTEGER NOT NULL,
        bitmap_checksum TEXT NOT NULL,
        snapshot_version INTEGER NOT NULL,
        checkpoint_digest TEXT NOT NULL,
        state TEXT NOT NULL,
        FOREIGN KEY (parent_snapshot_id) REFERENCES snapshots(snapshot_id)
    );
    CREATE TABLE snapshot_blocks (
        snapshot_id TEXT NOT NULL,
        block_index INTEGER NOT NULL,
        ref TEXT NOT NULL,
        digest TEXT NOT NULL,
        PRIMARY KEY (snapshot_id, block_index),
        FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE
    );
    CREATE TABLE snapshot_views (
        snapshot_id TEXT NOT NULL,
        block_index INTEGER NOT NULL,
        ref TEXT NOT NULL,
        digest TEXT NOT NULL,
        PRIMARY KEY (snapshot_id, block_index),
        FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE
    );
    CREATE TABLE operations (
        operation_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        target_snapshot TEXT,
        phase TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        temp_path TEXT,
        prior_sha256 TEXT,
        expected_sha256 TEXT,
        requires_rerun INTEGER NOT NULL DEFAULT 0,
        plan_json TEXT NOT NULL,
        details_json TEXT NOT NULL
    );
    CREATE TABLE audits (
        audit_version INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        state_version INTEGER NOT NULL,
        snapshot_versions_json TEXT NOT NULL,
        manifest_digest TEXT NOT NULL,
        status TEXT NOT NULL,
        conclusion_json TEXT NOT NULL
    );
    CREATE INDEX idx_snapshot_blocks_ref ON snapshot_blocks(ref);
    CREATE INDEX idx_snapshot_views_ref ON snapshot_views(ref);
    CREATE INDEX idx_operations_phase ON operations(phase, status);
    CREATE INDEX idx_audits_created ON audits(created_at, audit_version);
    """


@dataclass(frozen=True)
class RollbackPlan:
    selected_path: str
    threshold: str
    ratio: str
    ratio_float: float
    dirty_blocks: int
    total_blocks: int
    snapshot_count: int
    chain_depth: int
    boundary_hit: bool
    tie_rule: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_path": self.selected_path,
            "threshold": self.threshold,
            "ratio": self.ratio,
            "ratio_float": self.ratio_float,
            "dirty_blocks": self.dirty_blocks,
            "total_blocks": self.total_blocks,
            "snapshot_count": self.snapshot_count,
            "chain_depth": self.chain_depth,
            "boundary_hit": self.boundary_hit,
            "tie_rule": self.tie_rule,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RollbackResult:
    operation_id: str
    target_snapshot: str
    plan: RollbackPlan
    prior_sha256: str
    result_sha256: str
    checkpoint_sha256: str
    restored_blocks: tuple[int, ...]
    verification: Mapping[str, Any]
    recovered: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "target_snapshot": self.target_snapshot,
            "path": self.plan.as_dict(),
            "prior_sha256": self.prior_sha256,
            "result_sha256": self.result_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "restored_blocks": list(self.restored_blocks),
            "verification": dict(self.verification),
            "recovered": self.recovered,
        }


class Session:
    """A local, deterministic snapshot diagnostic session."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.db_path = self.root / "session.sqlite3"
        self.manifest_path = self.root / MANIFEST_FILE
        self.objects_dir = self.root / "objects"
        self.backup_objects_dir = self.root / "backups" / "objects"
        self.backup_manifests_dir = self.root / "backups" / "manifests"
        self.tmp_dir = self.root / "tmp"
        self.logger = JsonEventLogger(self.root)
        self._meta_cache: dict[str, str] | None = None

    # ------------------------------------------------------------------ setup
    @classmethod
    def initialize(
        cls,
        root: str | os.PathLike[str],
        *,
        block_size: int,
        block_count: int,
        image_name: str = "volume.bin",
        source_image: str | os.PathLike[str] | None = None,
        seed: int = 20261123,
        manifest_format: str = DEFAULT_MANIFEST_FORMAT,
        threshold: float | str = DEFAULT_THRESHOLD,
        snapshot_plan: Sequence[str] = (),
        rollback_target: str | None = None,
        writes: Sequence[Mapping[str, Any]] = (),
        steps: Sequence[Mapping[str, Any]] = (),
        reset: bool = False,
    ) -> "Session":
        session_root = Path(root).resolve()
        if session_root.exists() and any(session_root.iterdir()):
            if not reset:
                raise DiagnosticError(
                    E_STATE,
                    "session directory is not empty; use reset explicitly",
                    field="session_dir",
                    details={"path": str(session_root)},
                )
            cls._reset_directory(session_root)
        session_root.mkdir(parents=True, exist_ok=True)
        image_path = session_root / image_name
        if image_path.parent.resolve() != session_root:
            raise DiagnosticError(
                E_GEOM,
                "volume image must reside directly in the session directory",
                field="volume.image",
            )
        if source_image is not None:
            source = Path(source_image)
            expected = block_size * block_count
            if not source.exists() or source.stat().st_size != expected:
                raise DiagnosticError(
                    E_GEOM,
                    "source image does not match the declared volume geometry",
                    field="image_length",
                    details={"path": str(source), "expected": expected},
                )
            shutil.copyfile(source, image_path)
        else:
            generate_volume(image_path, block_size, block_count, seed=seed)
        geometry = validate_geometry(image_path, block_size, block_count)
        if manifest_format not in FORMAT_VERSIONS:
            raise DiagnosticError(
                E_FORMAT,
                "manifest_format is unknown",
                field="manifest_format",
                details={"supported": list(FORMAT_VERSIONS)},
            )
        try:
            threshold_fraction = Fraction(str(threshold))
        except (ValueError, ZeroDivisionError):
            raise DiagnosticError(E_FORMAT, "rollback threshold is invalid", field="rollback_threshold")
        if threshold_fraction < 0 or threshold_fraction > 1:
            raise DiagnosticError(
                E_FORMAT,
                "rollback threshold must be between 0 and 1",
                field="rollback_threshold",
            )
        for directory in (
            session_root / "objects",
            session_root / "backups" / "objects",
            session_root / "backups" / "manifests",
            session_root / "tmp",
            session_root / "logs",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        spec_document = {
            "schema_version": 1,
            "session_id": f"session-{unique_suffix()}",
            "volume": {
                "image": image_name,
                "block_size": block_size,
                "block_count": block_count,
            },
            "manifest_format": manifest_format,
            "rollback_threshold": float(threshold_fraction),
            "snapshot_plan": list(snapshot_plan),
            "rollback_target": rollback_target,
            "writes": list(writes),
            "steps": list(steps),
        }
        atomic_write_json(session_root / SESSION_FILE, spec_document)
        initial_backup = session_root / "backups" / "initial-volume.bin"
        copy_file_atomic(image_path, initial_backup, readonly=True)
        initial_sha = sha256_file(image_path)
        session = cls(session_root)
        session._create_database(spec_document, geometry, initial_sha)
        session.export_current_manifest()
        session.logger.emit(
            "session_initialized",
            operation="init",
            block_size=block_size,
            block_count=block_count,
            threshold=str(threshold_fraction),
            manifest_format=manifest_format,
        )
        return session

    @staticmethod
    def _reset_directory(path: Path) -> None:
        marker = path / SESSION_FILE
        allowed_partial = {
            "logs",
            "tmp",
            "objects",
            "backups",
            "volume.bin",
            "manifest.json",
            "session.sqlite3",
            "session.sqlite3-wal",
            "session.sqlite3-shm",
        }
        if not marker.exists() and any(path.iterdir()):
            unknown = sorted(child.name for child in path.iterdir() if child.name not in allowed_partial)
            if unknown:
                raise DiagnosticError(
                    E_STATE,
                    "refusing to reset a directory that is not a snapdiag session",
                    field="session_dir",
                    details={"path": str(path), "unknown_entries": unknown},
                )
        make_tree_writable(path)
        for child in list(path.iterdir()):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                ensure_writable(child)
                child.unlink()

    def _create_database(
        self,
        spec: Mapping[str, Any],
        geometry: VolumeGeometry,
        initial_sha: str,
    ) -> None:
        with self._connect() as connection:
            connection.executescript(_schema_sql())
            now = utc_now()
            meta = {
                "schema_version": str(SCHEMA_VERSION),
                "session_id": str(spec["session_id"]),
                "created_at": now,
                "image_name": geometry.image_path.name,
                "block_size": str(geometry.block_size),
                "block_count": str(geometry.block_count),
                "initial_sha256": initial_sha,
                "manifest_format": str(spec.get("manifest_format", DEFAULT_MANIFEST_FORMAT)),
                "manifest_file": MANIFEST_FILE,
                "manifest_generation": "0",
                "state_version": "0",
                "threshold": str(spec.get("rollback_threshold", DEFAULT_THRESHOLD)),
            }
            connection.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", sorted(meta.items()))

    @classmethod
    def open(cls, root: str | os.PathLike[str], *, validate_volume: bool = True) -> "Session":
        session = cls(Path(root))
        if not session.db_path.exists():
            raise DiagnosticError(
                E_STATE,
                "session database does not exist",
                field="session",
                details={"path": str(session.db_path)},
            )
        meta = session.meta()
        expected_schema = int(meta.get("schema_version", "0"))
        if expected_schema != SCHEMA_VERSION:
            raise DiagnosticError(
                E_VERSION,
                "session database schema version is unsupported",
                field="schema_version",
                details={"expected": SCHEMA_VERSION, "actual": expected_schema},
            )
        if validate_volume:
            session.geometry()
        return session

    def meta(self) -> dict[str, str]:
        if self._meta_cache is not None:
            return dict(self._meta_cache)
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        self._meta_cache = {str(row["key"]): str(row["value"]) for row in rows}
        return dict(self._meta_cache)

    @property
    def block_size(self) -> int:
        return int(self.meta()["block_size"])

    @property
    def block_count(self) -> int:
        return int(self.meta()["block_count"])

    @property
    def threshold(self) -> Fraction:
        return Fraction(self.meta().get("threshold", str(DEFAULT_THRESHOLD)))

    @property
    def state_version(self) -> int:
        return int(self.meta().get("state_version", "0"))

    def geometry(self) -> VolumeGeometry:
        image_name = self.meta().get("image_name", "volume.bin")
        image_path = self.root / image_name
        return validate_geometry(image_path, self.block_size, self.block_count)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def close(self) -> None:
        self._meta_cache = None

    # ------------------------------------------------------------------ objects
    def _object_path(self, digest: str) -> Path:
        return self.objects_dir / f"{digest}.blk"

    def _backup_object_path(self, digest: str) -> Path:
        return self.backup_objects_dir / f"{digest}.blk"

    def _store_object(self, data: bytes, digest: str) -> None:
        if sha256_bytes(data) != digest:
            raise DiagnosticError(
                E_INTEGRITY,
                "internal object digest mismatch",
                field="ref",
                details={"expected": digest, "actual": sha256_bytes(data)},
            )
        object_path = self._object_path(digest)
        if not object_path.exists():
            atomic_write_bytes(
                object_path,
                data,
                readonly=True,
                sync=False,
                sync_parent=False,
            )
        elif sha256_file(object_path) != digest:
            raise DiagnosticError(
                E_INTEGRITY,
                "content-addressed object store is corrupted",
                field="ref",
                details={"expected": digest, "actual": sha256_file(object_path)},
            )
        backup = self._backup_object_path(digest)
        if not backup.exists():
            copy_file_atomic(
                object_path,
                backup,
                readonly=True,
                sync=False,
                sync_parent=False,
            )

    def restore_object_from_backup(self, digest: str) -> bool:
        backup = self._backup_object_path(digest)
        if not backup.exists():
            return False
        if sha256_file(backup) != digest:
            raise DiagnosticError(
                E_INTEGRITY,
                "read-only object backup is corrupted",
                field="ref",
                details={"expected": digest, "actual": sha256_file(backup)},
            )
        copy_file_atomic(backup, self._object_path(digest))
        return True

    def read_ref_object(self, digest: str, *, from_backup: bool = False) -> bytes:
        path = self._backup_object_path(digest) if from_backup else self._object_path(digest)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise DiagnosticError(
                E_INTEGRITY,
                "snapshot data object is missing",
                field="ref",
                details={"expected_digest": digest, "actual_digest": None, "path": str(path)},
            )

    def verify_ref_object(self, digest: str, *, from_backup: bool = False) -> tuple[bool, str | None]:
        try:
            data = self.read_ref_object(digest, from_backup=from_backup)
        except DiagnosticError:
            return False, None
        actual = sha256_bytes(data)
        return actual == digest, actual

    # ---------------------------------------------------------------- snapshots
    def snapshot_rows(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM snapshots ORDER BY created_seq").fetchall()
        return [dict(row) for row in rows]

    def snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
        if row is None:
            raise DiagnosticError(
                E_CHAIN,
                "snapshot does not exist",
                snapshot_id=snapshot_id,
                field="snapshot_id",
            )
        return dict(row)

    def snapshot_blocks(self, snapshot_id: str) -> dict[int, dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT block_index, ref, digest FROM snapshot_blocks WHERE snapshot_id = ? ORDER BY block_index",
                (snapshot_id,),
            ).fetchall()
        return {
            int(row["block_index"]): {"ref": str(row["ref"]), "digest": str(row["digest"])}
            for row in rows
        }

    def snapshot_views(self, snapshot_id: str) -> dict[int, dict[str, str]]:
        """Return the immutable full checkpoint reference view."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT block_index, ref, digest FROM snapshot_views WHERE snapshot_id = ? ORDER BY block_index",
                (snapshot_id,),
            ).fetchall()
        return {
            int(row["block_index"]): {"ref": str(row["ref"]), "digest": str(row["digest"])}
            for row in rows
        }

    def resolve_snapshot(self, snapshot_id: str) -> Mapping[str, Any] | None:
        """Resolver interface consumed by historical manifest normalization."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
            if row is None:
                return None
            block_rows = connection.execute(
                "SELECT block_index, ref, digest FROM snapshot_blocks WHERE snapshot_id = ? ORDER BY block_index",
                (snapshot_id,),
            ).fetchall()
        return {
            "parent_snapshot_id": row["parent_snapshot_id"],
            "snapshot_version": int(row["snapshot_version"]),
            "bitmap": bytes(row["bitmap"]),
            "digest_table": {int(item["block_index"]): str(item["digest"]) for item in block_rows},
            "block_refs": {int(item["block_index"]): str(item["ref"]) for item in block_rows},
            "checkpoint_digest": str(row["checkpoint_digest"]),
            "created_seq": int(row["created_seq"]),
        }

    def _next_created_seq(self, connection: sqlite3.Connection) -> int:
        row = connection.execute("SELECT COALESCE(MAX(created_seq), 0) + 1 AS next FROM snapshots").fetchone()
        return int(row["next"])

    def _increment_state_version(self, connection: sqlite3.Connection) -> int:
        row = connection.execute("SELECT value FROM meta WHERE key = 'state_version'").fetchone()
        current = int(row["value"]) if row is not None else 0
        next_version = current + 1
        connection.execute(
            "INSERT INTO meta(key, value) VALUES ('state_version', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(next_version),),
        )
        return next_version

    def _validate_existing_chain(self, *, error_code: str = E_CHAIN) -> None:
        rows = self.snapshot_rows()
        by_id = {row["snapshot_id"]: row for row in rows}
        for row in rows:
            parent = row["parent_snapshot_id"]
            if parent is not None and parent not in by_id:
                raise DiagnosticError(
                    error_code,
                    "snapshot chain is broken: parent reference is missing",
                    snapshot_id=str(row["snapshot_id"]),
                    field="parent_snapshot_id",
                    details={
                        "snapshot_id": str(row["snapshot_id"]),
                        "missing_parent": str(parent),
                    },
                )
        state: dict[str, int] = {}

        def visit(snapshot_id: str, path: tuple[str, ...]) -> None:
            marker = state.get(snapshot_id, 0)
            if marker == 2:
                return
            if marker == 1:
                cycle = list(path[path.index(snapshot_id) :]) + [snapshot_id] if snapshot_id in path else [snapshot_id]
                raise DiagnosticError(
                    error_code,
                    "snapshot chain contains a cycle",
                    snapshot_id=snapshot_id,
                    field="parent_snapshot_id",
                    details={"cycle": cycle},
                )
            state[snapshot_id] = 1
            parent = by_id[snapshot_id]["parent_snapshot_id"]
            if parent is not None:
                visit(str(parent), path + (snapshot_id,))
            state[snapshot_id] = 2

        for snapshot_id in sorted(by_id):
            visit(snapshot_id, ())

    def create_snapshot(
        self,
        snapshot_id: str,
        *,
        parent_snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise DiagnosticError(
                E_FORMAT,
                "snapshot_id must be a non-empty string",
                snapshot_id=None,
                field="snapshot_id",
            )
        snapshot_id = snapshot_id.strip()
        geometry = self.geometry()
        volume_data = geometry.read_all()
        full_view: list[tuple[int, str, str]] = []
        for index in range(geometry.block_count):
            start = index * geometry.block_size
            data = volume_data[start : start + geometry.block_size]
            digest = sha256_bytes(data)
            self._store_object(data, digest)
            full_view.append((index, digest, digest))
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
            if existing is not None:
                raise DiagnosticError(
                    E_EXISTS,
                    "snapshot name already exists",
                    snapshot_id=snapshot_id,
                    field="snapshot_id",
                )
            if parent_snapshot_id is not None:
                parent = connection.execute(
                    "SELECT 1 FROM snapshots WHERE snapshot_id = ?", (parent_snapshot_id,)
                ).fetchone()
                if parent is None:
                    raise DiagnosticError(
                        E_CHAIN,
                        "snapshot chain is broken: parent reference is missing",
                        snapshot_id=snapshot_id,
                        field="parent_snapshot_id",
                        details={
                            "snapshot_id": snapshot_id,
                            "missing_parent": parent_snapshot_id,
                        },
                    )
            checkpoint = sha256_file(geometry.image_path)
            bitmap = bytes(empty_bitmap(geometry.block_count))
            sequence = self._next_created_seq(connection)
            now = utc_now()
            connection.execute(
                """
                INSERT INTO snapshots(
                    snapshot_id, parent_snapshot_id, created_seq, created_at,
                    block_count, block_size, bitmap, bitmap_length, bitmap_checksum,
                    snapshot_version, checkpoint_digest, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 'active')
                """,
                (
                    snapshot_id,
                    parent_snapshot_id,
                    sequence,
                    now,
                    geometry.block_count,
                    geometry.block_size,
                    bitmap,
                    bitmap_length(geometry.block_count),
                    bitmap_checksum(bitmap),
                    checkpoint,
                ),
            )
            connection.executemany(
                "INSERT INTO snapshot_views(snapshot_id, block_index, ref, digest) VALUES (?, ?, ?, ?)",
                [
                    (snapshot_id, index, ref, digest)
                    for index, ref, digest in full_view
                ],
            )
            self._increment_state_version(connection)
        self._validate_existing_chain()
        self.export_current_manifest()
        self.logger.emit(
            "snapshot_created",
            operation="create-snapshot",
            snapshot_id=snapshot_id,
            parent_snapshot_id=parent_snapshot_id,
            checkpoint_sha256=checkpoint,
        )
        return self.snapshot(snapshot_id)

    # ------------------------------------------------------------------ writes
    def write_block_hex(
        self,
        block_index: int,
        hex_value: str,
        *,
        failpoint: str | None = None,
    ) -> dict[str, Any]:
        from .geometry import decode_hex_block

        geometry = self.geometry()
        geometry.validate_block_index(block_index)
        data = decode_hex_block(hex_value, geometry.block_size)
        return self.write_block(block_index, data, failpoint=failpoint)

    def write_block(self, block_index: int, data: bytes, *, failpoint: str | None = None) -> dict[str, Any]:
        geometry = self.geometry()
        geometry.validate_block_index(block_index)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise DiagnosticError(
                E_GEOM,
                "block payload must be bytes",
                field="block_data",
                details={"block_index": block_index},
            )
        payload = bytes(data)
        if len(payload) != geometry.block_size:
            raise DiagnosticError(
                E_GEOM,
                "block payload length must equal block_size",
                field="block_data",
                details={"expected": geometry.block_size, "actual": len(payload)},
            )
        old_data = geometry.read_block(block_index)
        old_digest = sha256_bytes(old_data)
        # COW undo metadata commits first.  If the process dies before the
        # in-place write, the undo record still points at the current bytes and
        # a later rollback is harmless.  This avoids copying a multi-GiB image
        # for every 4 KiB block write.
        captured: list[str] = []
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM snapshots ORDER BY created_seq").fetchall()
            for row in rows:
                snapshot_id = str(row["snapshot_id"])
                present = connection.execute(
                    "SELECT 1 FROM snapshot_blocks WHERE snapshot_id = ? AND block_index = ?",
                    (snapshot_id, block_index),
                ).fetchone()
                if present is not None:
                    continue
                self._store_object(old_data, old_digest)
                connection.execute(
                    "INSERT INTO snapshot_blocks(snapshot_id, block_index, ref, digest) VALUES (?, ?, ?, ?)",
                    (snapshot_id, block_index, old_digest, old_digest),
                )
                bitmap = bytearray(row["bitmap"])
                set_bit(bitmap, block_index)
                version = int(row["snapshot_version"]) + 1
                connection.execute(
                    """
                    UPDATE snapshots SET bitmap = ?, bitmap_checksum = ?, snapshot_version = ?
                    WHERE snapshot_id = ?
                    """,
                    (bytes(bitmap), bitmap_checksum(bitmap), version, snapshot_id),
                )
                captured.append(snapshot_id)
            state_version = self._increment_state_version(connection)
        with open(geometry.image_path, "r+b") as handle:
            handle.seek(block_index * geometry.block_size)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.export_current_manifest()
        self.logger.emit(
            "block_written",
            operation="write-block",
            block_index=block_index,
            old_digest=f"sha256:{old_digest}",
            new_digest=f"sha256:{sha256_bytes(payload)}",
            captured_snapshots=sorted(captured),
            state_version=state_version,
        )
        if failpoint == "after_cow_commit":
            raise RuntimeError("simulated write interruption after metadata commit")
        return {
            "block_index": block_index,
            "old_digest": display_digest(old_digest),
            "new_digest": display_digest(sha256_bytes(payload)),
            "captured_snapshots": sorted(captured),
            "state_version": state_version,
        }

    # --------------------------------------------------------------- manifests
    def build_manifest_model(self, *, resolver: ManifestResolver | None = None) -> NormalizedManifest:
        rows = self.snapshot_rows()
        snapshots: list[SnapshotManifest] = []
        for row in rows:
            blocks = self.snapshot_blocks(str(row["snapshot_id"]))
            refs = {index: item["ref"] for index, item in blocks.items()}
            digests = {index: item["digest"] for index, item in blocks.items()}
            snapshots.append(
                make_snapshot_manifest(
                    snapshot_id=str(row["snapshot_id"]),
                    block_count=int(row["block_count"]),
                    bitmap=bytes(row["bitmap"]),
                    digest_table=digests,
                    block_refs=refs,
                    parent_snapshot_id=row["parent_snapshot_id"],
                    snapshot_version=int(row["snapshot_version"]),
                    source_format="m3",
                )
            )
        return build_manifest_model(
            block_size=self.block_size,
            block_count=self.block_count,
            snapshots=snapshots,
            source_format="m3",
        )

    def manifest_text(self, version: str = "m3") -> str:
        return encode_manifest_text(self.build_manifest_model(), version)

    def export_current_manifest(self, *, version: str = "m3") -> dict[str, Any]:
        model = self.build_manifest_model()
        document = encode_manifest(model, version)
        atomic_write_json(self.manifest_path, document)
        ensure_writable(self.manifest_path)
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM meta WHERE key='manifest_generation'").fetchone()
            generation = int(row["value"]) + 1 if row is not None else 1
            connection.execute(
                "INSERT INTO meta(key,value) VALUES ('manifest_generation', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(generation),),
            )
        self._meta_cache = None
        return document

    def read_manifest(self, *, strict_version: bool = True) -> NormalizedManifest:
        try:
            data = self.manifest_path.read_bytes()
        except FileNotFoundError:
            raise DiagnosticError(
                E_FORMAT,
                "snapshot manifest file is missing",
                snapshot_id="<manifest>",
                field="file",
                details={"path": str(self.manifest_path)},
            )
        raw_version = peek_format_version(data)
        expected_version = self.meta().get("manifest_format", DEFAULT_MANIFEST_FORMAT)
        if strict_version and raw_version is not None and raw_version != expected_version:
            match = re.search(r'"snapshot_id"\s*:\s*"([^"]+)"', data.decode("utf-8", errors="replace"))
            raise DiagnosticError(
                E_VERSION,
                "snapshot manifest version conflicts with the session version",
                snapshot_id=match.group(1) if match else "<manifest>",
                field="format_version",
                details={
                    "expected": expected_version,
                    "actual": raw_version,
                    "manifest_path": str(self.manifest_path),
                },
            )
        model = decode_manifest(data, resolver=self)
        if strict_version and model.source_format != expected_version:
            raise DiagnosticError(
                E_VERSION,
                "snapshot manifest version conflicts with the session version",
                snapshot_id=model.snapshots[0].snapshot_id if model.snapshots else "<manifest>",
                field="format_version",
                details={"expected": expected_version, "actual": model.source_format},
            )
        return model

    def manifest_consistency(self) -> dict[str, Any]:
        database_model = self.build_manifest_model()
        try:
            file_model = self.read_manifest(strict_version=True)
        except DiagnosticError as error:
            return {
                "consistent": False,
                "error": error.to_dict(),
                "database_manifest_digest": f"sha256:{database_model.manifest_digest}",
            }
        return {
            "consistent": file_model.fingerprint() == database_model.fingerprint(),
            "file_source_format": file_model.source_format,
            "file_manifest_digest": f"sha256:{file_model.manifest_digest}",
            "database_manifest_digest": f"sha256:{database_model.manifest_digest}",
            "filled_fields": list(file_model.filled_fields),
        }

    def list_manifest_backups(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if not self.backup_manifests_dir.exists():
            return result
        for path in sorted(self.backup_manifests_dir.glob("manifest-*.json"), key=lambda item: item.name):
            stat = path.stat()
            result.append(
                {
                    "name": path.name,
                    "path": str(path.relative_to(self.root)),
                    "size": stat.st_size,
                    "sha256": sha256_file(path),
                    "readonly": not bool(stat.st_mode & 0o222),
                }
            )
        return result

    def _backup_manifest(self, reason: str) -> str | None:
        if not self.manifest_path.exists():
            return None
        generation = self.meta().get("manifest_generation", "0")
        timestamp = utc_now().replace(":", "").replace("-", "").replace(".", "")
        safe_reason = re.sub(r"[^a-zA-Z0-9_-]+", "-", reason)[:40]
        destination = self.backup_manifests_dir / (
            f"manifest-{int(generation):06d}-{timestamp}-{safe_reason}-{unique_suffix()}.json"
        )
        copy_file_atomic(self.manifest_path, destination, readonly=True)
        ensure_readonly(destination)
        return str(destination.relative_to(self.root))

    # ----------------------------------------------------- manifest replacement
    @staticmethod
    def _snapshot_signature(snapshot: SnapshotManifest) -> tuple[Any, ...]:
        return (
            bytes(snapshot.bitmap).hex(),
            tuple(sorted(snapshot.digest_table.items())),
            tuple(sorted(snapshot.block_refs.items())),
            snapshot.parent_snapshot_id,
            snapshot.snapshot_version,
        )

    def _current_signatures(self) -> dict[str, tuple[Any, ...]]:
        return {
            snapshot.snapshot_id: self._snapshot_signature(snapshot)
            for snapshot in self.build_manifest_model().snapshots
        }

    def manifest_diff(
        self,
        incoming: NormalizedManifest | None = None,
    ) -> dict[str, list[str] | dict[str, list[str]]]:
        current = self._current_signatures()
        if incoming is None:
            incoming_signatures = current
        else:
            incoming_signatures = {
                snapshot.snapshot_id: self._snapshot_signature(snapshot)
                for snapshot in incoming.snapshots
            }
        added = sorted(set(incoming_signatures) - set(current))
        removed = sorted(set(current) - set(incoming_signatures))
        replaced = sorted(
            snapshot_id
            for snapshot_id in set(current) & set(incoming_signatures)
            if current[snapshot_id] != incoming_signatures[snapshot_id]
        )
        return {"added": added, "removed": removed, "replaced": replaced}

    def import_manifest(self, source: str | os.PathLike[str], *, reason: str = "import") -> dict[str, Any]:
        source_path = Path(source)
        incoming = load_manifest_file(source_path, resolver=self)
        diff = self.manifest_diff(incoming)
        incoming_by_id = incoming.by_id()
        existing_rows = {str(row["snapshot_id"]): row for row in self.snapshot_rows()}
        missing_new = sorted(set(incoming_by_id) - set(existing_rows))
        for snapshot_id in missing_new:
            snapshot = incoming_by_id[snapshot_id]
            if incoming.source_format != "m3":
                raise DiagnosticError(
                    E_FORMAT,
                    "historical manifest cannot introduce an unknown snapshot without disk references",
                    snapshot_id=snapshot_id,
                    field="snapshot_id",
                )
            for index, ref in snapshot.block_refs.items():
                if not self._object_path(ref).exists():
                    raise DiagnosticError(
                        E_INTEGRITY,
                        "manifest references a missing data block object",
                        snapshot_id=snapshot_id,
                        field="block_refs",
                        details={"block_index": index, "expected_digest": ref},
                    )
        backup = self._backup_manifest(reason)
        current_volume_sha = sha256_file(self.geometry().image_path)
        existing_views = {
            snapshot_id: self.snapshot_views(snapshot_id) for snapshot_id in existing_rows
        }
        # Parents must be inserted before children because the self-reference
        # uses immediate foreign-key checking. Remaining ties are stable by id.
        incoming_parents = {
            snapshot.snapshot_id: snapshot.parent_snapshot_id for snapshot in incoming.snapshots
        }
        insert_order: list[SnapshotManifest] = []
        remaining = {snapshot.snapshot_id: snapshot for snapshot in incoming.snapshots}
        while remaining:
            ready = sorted(
                snapshot_id
                for snapshot_id in remaining
                if incoming_parents[snapshot_id] is None or incoming_parents[snapshot_id] not in remaining
            )
            if not ready:
                raise DiagnosticError(
                    E_CHAIN,
                    "manifest snapshot chain cannot be inserted due to a cycle",
                    field="parent_snapshot_id",
                )
            for snapshot_id in ready:
                insert_order.append(remaining.pop(snapshot_id))
        with self._connect() as connection:
            connection.execute("PRAGMA defer_foreign_keys = ON")
            connection.execute("DELETE FROM snapshot_blocks")
            connection.execute("DELETE FROM snapshots")
            used_sequences = {
                int(row["created_seq"]) for row in existing_rows.values() if row is not None
            }
            next_seq = max(used_sequences, default=0) + 1
            for snapshot in insert_order:
                existing = existing_rows.get(snapshot.snapshot_id)
                if existing is not None:
                    checkpoint = str(existing["checkpoint_digest"])
                    created_at = str(existing["created_at"])
                    sequence = int(existing["created_seq"])
                    full_view = [
                        (index, item["ref"], item["digest"])
                        for index, item in sorted(existing_views[snapshot.snapshot_id].items())
                    ]
                    if len(full_view) != self.block_count:
                        raise DiagnosticError(
                            E_STATE,
                            "existing snapshot is missing its immutable full view",
                            snapshot_id=snapshot.snapshot_id,
                            field="snapshot_views",
                        )
                else:
                    # Reconstruct the checkpoint view by applying this entry's
                    # undo references to the current volume.
                    view = bytearray(self.geometry().read_all())
                    for index, ref in sorted(snapshot.block_refs.items()):
                        data = self.read_ref_object(ref)
                        if sha256_bytes(data) != ref:
                            raise DiagnosticError(
                                E_INTEGRITY,
                                "manifest data reference failed digest verification",
                                snapshot_id=snapshot.snapshot_id,
                                field="block_refs",
                                details={
                                    "block_index": index,
                                    "expected_digest": ref,
                                    "actual_digest": sha256_bytes(data),
                                },
                            )
                        start = index * self.block_size
                        view[start : start + self.block_size] = data
                    checkpoint = sha256_bytes(bytes(view))
                    created_at = utc_now()
                    sequence = next_seq
                    next_seq += 1
                    full_view = []
                    for index in range(self.block_count):
                        start = index * self.block_size
                        data = bytes(view[start : start + self.block_size])
                        digest = sha256_bytes(data)
                        self._store_object(data, digest)
                        full_view.append((index, digest, digest))
                connection.execute(
                    """
                    INSERT INTO snapshots(
                        snapshot_id, parent_snapshot_id, created_seq, created_at,
                        block_count, block_size, bitmap, bitmap_length, bitmap_checksum,
                        snapshot_version, checkpoint_digest, state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                    """,
                    (
                        snapshot.snapshot_id,
                        snapshot.parent_snapshot_id,
                        sequence,
                        created_at,
                        snapshot.block_count,
                        self.block_size,
                        bytes(snapshot.bitmap),
                        snapshot.bitmap_length,
                        snapshot.bitmap_checksum,
                        snapshot.snapshot_version,
                        checkpoint,
                    ),
                )
                connection.executemany(
                    "INSERT INTO snapshot_blocks(snapshot_id, block_index, ref, digest) VALUES (?, ?, ?, ?)",
                    [
                        (snapshot.snapshot_id, index, ref, snapshot.digest_table[index])
                        for index, ref in sorted(snapshot.block_refs.items())
                    ],
                )
                connection.executemany(
                    "INSERT INTO snapshot_views(snapshot_id, block_index, ref, digest) VALUES (?, ?, ?, ?)",
                    [
                        (snapshot.snapshot_id, index, ref, digest)
                        for index, ref, digest in full_view
                    ],
                )
            state_version = self._increment_state_version(connection)
        self._validate_existing_chain()
        self.export_current_manifest()
        report = {
            "ok": True,
            "operation": "import-manifest",
            "reason": reason,
            "source_format": incoming.source_format,
            "filled_fields": list(incoming.filled_fields),
            "fill_basis": dict(sorted(incoming.fill_basis.items())),
            "added": diff["added"],
            "removed": diff["removed"],
            "replaced": diff["replaced"],
            "backup": backup,
            "state_version": state_version,
            "source_manifest_digest": f"sha256:{incoming.manifest_digest}",
            "current_volume_sha256": current_volume_sha,
        }
        self.logger.emit(
            "manifest_imported",
            operation="import-manifest",
            source_format=incoming.source_format,
            added=diff["added"],
            removed=diff["removed"],
            replaced=diff["replaced"],
            backup=backup,
        )
        return report

    def reconcile_manifest(self, *, reason: str = "format-or-digest-mismatch") -> dict[str, Any]:
        old_model: NormalizedManifest | None = None
        parse_error: dict[str, Any] | None = None
        try:
            old_model = self.read_manifest(strict_version=False)
        except DiagnosticError as error:
            parse_error = error.to_dict()
        database_model = self.build_manifest_model()
        diff = self.manifest_diff(old_model) if old_model is not None else {
            "added": [],
            "removed": [],
            "replaced": [],
        }
        backup = self._backup_manifest(reason)
        document = self.export_current_manifest()
        report = {
            "ok": True,
            "operation": "reconcile-manifest",
            "reason": reason,
            "parse_error": parse_error,
            "old_manifest_digest": (
                f"sha256:{old_model.manifest_digest}" if old_model is not None else None
            ),
            "new_manifest_digest": f"sha256:{database_model.manifest_digest}",
            "format_version": document["format_version"],
            "added": diff["added"],
            "removed": diff["removed"],
            "replaced": diff["replaced"],
            "backup": backup,
        }
        self.logger.emit(
            "manifest_reconciled",
            operation="reconcile-manifest",
            reason=reason,
            backup=backup,
            added=diff["added"],
            removed=diff["removed"],
            replaced=diff["replaced"],
        )
        return report

    # --------------------------------------------------------------- rollback
    def _chain_rows(self) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
        rows = self.snapshot_rows()
        by_id = {str(row["snapshot_id"]): row for row in rows}
        children: dict[str, list[str]] = {snapshot_id: [] for snapshot_id in by_id}
        for snapshot_id, row in by_id.items():
            parent = row["parent_snapshot_id"]
            if parent is not None and parent in children:
                children[str(parent)].append(snapshot_id)
        for values in children.values():
            values.sort(key=lambda item: int(by_id[item]["created_seq"]))
        return by_id, children

    def _ancestor_chain(self, snapshot_id: str) -> list[str]:
        by_id, _ = self._chain_rows()
        if snapshot_id not in by_id:
            raise DiagnosticError(
                E_CHAIN,
                "rollback target snapshot does not exist",
                snapshot_id=snapshot_id,
                field="rollback_target",
            )
        result: list[str] = []
        current: str | None = snapshot_id
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise DiagnosticError(
                    E_CHAIN,
                    "snapshot chain contains a cycle",
                    snapshot_id=current,
                    field="parent_snapshot_id",
                    details={"cycle_member": current},
                )
            seen.add(current)
            row = by_id.get(current)
            if row is None:
                raise DiagnosticError(
                    E_CHAIN,
                    "snapshot chain is broken: parent reference is missing",
                    snapshot_id=result[-1] if result else snapshot_id,
                    field="parent_snapshot_id",
                    details={"missing_parent": current},
                )
            result.append(current)
            current = row["parent_snapshot_id"]
        return result

    def _descendants(self, snapshot_id: str) -> list[str]:
        by_id, children = self._chain_rows()
        result: list[str] = []
        stack = list(children.get(snapshot_id, []))
        while stack:
            current = stack.pop()
            result.append(current)
            stack.extend(children.get(current, []))
        # Newest first so target undo records are applied last.
        result.sort(key=lambda item: int(by_id[item]["created_seq"]), reverse=True)
        return result

    def plan_rollback(
        self,
        target_snapshot: str,
        *,
        threshold: float | str | None = None,
        path: str = "auto",
    ) -> RollbackPlan:
        if path not in ("auto", "bitmap", "full"):
            raise DiagnosticError(
                E_FORMAT,
                "rollback path must be auto, bitmap or full",
                snapshot_id=target_snapshot,
                field="path",
                details={"actual": path},
            )
        target = self.snapshot(target_snapshot)
        bitmap = bytes(target["bitmap"])
        validate_bitmap(
            bitmap,
            int(target["block_count"]),
            snapshot_id=target_snapshot,
            bitmap_length_value=int(target["bitmap_length"]),
        )
        dirty = popcount(bitmap)
        total = int(target["block_count"])
        threshold_fraction = Fraction(str(threshold)) if threshold is not None else self.threshold
        if threshold_fraction < 0 or threshold_fraction > 1:
            raise DiagnosticError(
                E_FORMAT,
                "rollback threshold must be between 0 and 1",
                field="rollback_threshold",
                details={"actual": str(threshold_fraction)},
            )
        ratio = Fraction(dirty, total) if total else Fraction(0, 1)
        boundary = ratio == threshold_fraction
        if path == "auto":
            # Explicit stable tie rule: an exact boundary selects the full-scan
            # path because the contract says "below" the threshold is bitmap.
            selected = "bitmap" if ratio < threshold_fraction else "full"
            tie_rule = "ratio < threshold -> bitmap; ratio == threshold -> full"
            if boundary:
                reason = (
                    f"boundary ratio {ratio} equals threshold {threshold_fraction}; "
                    "stable tie-break selects full-scan"
                )
            elif selected == "bitmap":
                reason = f"ratio {ratio} is below threshold {threshold_fraction}; bitmap scan is cheaper"
            else:
                reason = f"ratio {ratio} is at or above threshold {threshold_fraction}; full comparison avoids sparse bitmap overhead"
        else:
            selected = path
            tie_rule = "explicit path override"
            reason = f"explicit path override selected {selected}"
        chain = self._ancestor_chain(target_snapshot)
        return RollbackPlan(
            selected_path=selected,
            threshold=str(threshold_fraction),
            ratio=str(ratio),
            ratio_float=float(ratio),
            dirty_blocks=dirty,
            total_blocks=total,
            snapshot_count=len(self.snapshot_rows()),
            chain_depth=len(chain) - 1,
            boundary_hit=boundary,
            tie_rule=tie_rule,
            reason=reason,
        )

    def _integrity_error(
        self,
        *,
        snapshot_id: str,
        block_index: int,
        expected: str,
        actual: str | None,
        ref: str,
    ) -> DiagnosticError:
        return DiagnosticError(
            E_INTEGRITY,
            "snapshot block data digest verification failed",
            snapshot_id=snapshot_id,
            field="block_refs",
            details={
                "block_index": block_index,
                "expected_digest": f"sha256:{expected}",
                "actual_digest": f"sha256:{actual}" if actual is not None else None,
                "ref": f"sha256:{ref}",
            },
        )

    def _read_verified_ref(
        self,
        *,
        snapshot_id: str,
        block_index: int,
        expected_digest: str,
        ref: str,
    ) -> bytes:
        try:
            data = self.read_ref_object(ref)
        except DiagnosticError:
            raise self._integrity_error(
                snapshot_id=snapshot_id,
                block_index=block_index,
                expected=expected_digest,
                actual=None,
                ref=ref,
            )
        actual = sha256_bytes(data)
        if actual != expected_digest or actual != ref:
            raise self._integrity_error(
                snapshot_id=snapshot_id,
                block_index=block_index,
                expected=expected_digest,
                actual=actual,
                ref=ref,
            )
        return data

    def _target_refs(self, snapshot_id: str) -> dict[int, dict[str, str]]:
        target = self.snapshot(snapshot_id)
        refs = self.snapshot_blocks(snapshot_id)
        bits = set(set_bits(bytes(target["bitmap"]), int(target["block_count"])))
        if set(refs) != bits:
            raise DiagnosticError(
                E_CHAIN,
                "snapshot bitmap and data block reference set disagree",
                snapshot_id=snapshot_id,
                field="block_refs",
                details={
                    "missing_refs": sorted(bits - set(refs)),
                    "extra_refs": sorted(set(refs) - bits),
                },
            )
        return refs

    def _apply_refs_to_image(
        self,
        image: bytearray,
        *,
        snapshot_id: str,
        refs: Mapping[int, Mapping[str, str]],
    ) -> None:
        for block_index, item in sorted(refs.items()):
            data = self._read_verified_ref(
                snapshot_id=snapshot_id,
                block_index=block_index,
                expected_digest=item["digest"],
                ref=item["ref"],
            )
            start = block_index * self.block_size
            image[start : start + self.block_size] = data

    def rollback(
        self,
        target_snapshot: str,
        *,
        threshold: float | str | None = None,
        path: str = "auto",
        retry: bool = False,
        failpoint: str | None = None,
    ) -> RollbackResult:
        recovery = self.recover_session()
        blocked = [item for item in recovery if item.get("requires_rerun")]
        if blocked and not retry:
            raise DiagnosticError(
                E_STATE,
                "an interrupted rollback requires an explicit retry",
                field="operation",
                details={"operations": blocked, "retry_required": True},
            )
        if blocked and retry:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE operations SET status='superseded', requires_rerun=0, updated_at=? WHERE status='incomplete_unknown'",
                    (utc_now(),),
                )
        geometry = self.geometry()
        plan = self.plan_rollback(target_snapshot, threshold=threshold, path=path)
        target = self.snapshot(target_snapshot)
        target_refs = self._target_refs(target_snapshot)
        # Verify the target before doing any temp-image work so corruption is
        # reported deterministically by the lowest affected block number.
        for block_index, item in sorted(target_refs.items()):
            self._read_verified_ref(
                snapshot_id=target_snapshot,
                block_index=block_index,
                expected_digest=item["digest"],
                ref=item["ref"],
            )

        before_bytes = geometry.read_all()
        prior_sha = sha256_bytes(before_bytes)
        result_image = bytearray(before_bytes)
        chain_applied: list[dict[str, Any]] = []
        if plan.selected_path == "full":
            for descendant in self._descendants(target_snapshot):
                descendant_refs = self.snapshot_blocks(descendant)
                self._apply_refs_to_image(
                    result_image,
                    snapshot_id=descendant,
                    refs=descendant_refs,
                )
                chain_applied.append(
                    {
                        "snapshot_id": descendant,
                        "blocks": sorted(descendant_refs),
                    }
                )
            self._apply_refs_to_image(
                result_image,
                snapshot_id=target_snapshot,
                refs=target_refs,
            )
            # Full-scan comparison is the cost model of this path; the result
            # still reports the stable target bitmap as the restored set.
            compared = len(result_image) // geometry.block_size
            differing = sum(
                1
                for index in range(compared)
                if result_image[index * geometry.block_size : (index + 1) * geometry.block_size]
                != before_bytes[index * geometry.block_size : (index + 1) * geometry.block_size]
            )
        else:
            self._apply_refs_to_image(
                result_image,
                snapshot_id=target_snapshot,
                refs=target_refs,
            )
            compared = len(target_refs)
            differing = sum(
                1
                for index in target_refs
                if result_image[index * geometry.block_size : (index + 1) * geometry.block_size]
                != before_bytes[index * geometry.block_size : (index + 1) * geometry.block_size]
            )

        result_sha = sha256_bytes(bytes(result_image))
        checkpoint_sha = str(target["checkpoint_digest"])
        if result_sha != checkpoint_sha:
            raise DiagnosticError(
                E_INTEGRITY,
                "rollback result does not match the snapshot checkpoint digest",
                snapshot_id=target_snapshot,
                field="checkpoint_digest",
                details={
                    "expected_digest": f"sha256:{checkpoint_sha}",
                    "actual_digest": f"sha256:{result_sha}",
                },
            )
        expected_block_digests = [
            sha256_bytes(
                bytes(
                    result_image[
                        index * geometry.block_size : (index + 1) * geometry.block_size
                    ]
                )
            )
            for index in range(geometry.block_count)
        ]
        operation_id = f"rollback-{utc_now().replace(':', '').replace('-', '')}-{unique_suffix()}"
        temp_path = self.tmp_dir / f"{operation_id}.image"
        atomic_write_bytes(temp_path, bytes(result_image))
        marker_path = self.tmp_dir / f"{operation_id}.json"
        details = {
            "target_snapshot": target_snapshot,
            "expected_block_digests": expected_block_digests,
            "restored_blocks": sorted(target_refs),
            "chain_applied": chain_applied,
            "compared_blocks": compared,
            "differing_blocks": differing,
        }
        marker = {
            "operation_id": operation_id,
            "kind": "rollback",
            "phase": "prepared",
            "status": "incomplete",
            "temp_path": str(temp_path.relative_to(self.root)),
            "prior_sha256": prior_sha,
            "expected_sha256": result_sha,
            "plan": plan.as_dict(),
            "details": details,
            "updated_at": utc_now(),
        }
        with self._connect() as connection:
            now = utc_now()
            connection.execute(
                """
                INSERT INTO operations(
                    operation_id, kind, target_snapshot, phase, status, created_at,
                    updated_at, temp_path, prior_sha256, expected_sha256,
                    requires_rerun, plan_json, details_json
                ) VALUES (?, 'rollback', ?, 'prepared', 'incomplete', ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    operation_id,
                    target_snapshot,
                    now,
                    now,
                    str(temp_path.relative_to(self.root)),
                    prior_sha,
                    result_sha,
                    canonical_json_text(plan.as_dict()),
                    canonical_json_text(details),
                ),
            )
        atomic_write_json(marker_path, marker)
        self.logger.emit(
            "rollback_prepared",
            operation="rollback",
            snapshot_id=target_snapshot,
            operation_id=operation_id,
            path=plan.selected_path,
            threshold=plan.threshold,
            ratio=plan.ratio,
            prior_sha256=prior_sha,
            expected_sha256=result_sha,
        )
        if failpoint == "after_prepare":
            raise SimulatedInterruption("simulated termination after rollback prepare")
        os.replace(temp_path, geometry.image_path)
        fsync_directory(geometry.image_path.parent)
        if failpoint == "after_replace":
            raise SimulatedInterruption("simulated termination after volume replacement")
        verification = self.verify_volume_blocks(expected_block_digests)
        if not verification["ok"]:
            first = verification["issues"][0]
            raise DiagnosticError(
                E_INTEGRITY,
                "post-rollback block verification failed",
                snapshot_id=target_snapshot,
                field="block_digest",
                details=first,
            )
        with self._connect() as connection:
            now = utc_now()
            connection.execute(
                "UPDATE operations SET phase='completed', status='completed', requires_rerun=0, updated_at=? WHERE operation_id=?",
                (now, operation_id),
            )
        marker.update({"phase": "completed", "status": "completed", "updated_at": utc_now()})
        atomic_write_json(marker_path, marker)
        self.logger.emit(
            "rollback_completed",
            operation="rollback",
            snapshot_id=target_snapshot,
            operation_id=operation_id,
            path=plan.selected_path,
            result_sha256=result_sha,
            restored_blocks=len(target_refs),
            verified_blocks=verification["checked_blocks"],
        )
        return RollbackResult(
            operation_id=operation_id,
            target_snapshot=target_snapshot,
            plan=plan,
            prior_sha256=prior_sha,
            result_sha256=result_sha,
            checkpoint_sha256=checkpoint_sha,
            restored_blocks=tuple(sorted(target_refs)),
            verification=verification,
        )

    def verify_volume_blocks(self, expected_digests: Sequence[str]) -> dict[str, Any]:
        geometry = self.geometry()
        if len(expected_digests) != geometry.block_count:
            raise DiagnosticError(
                E_STATE,
                "expected digest table length does not match the volume",
                field="expected_block_digests",
                details={"expected": geometry.block_count, "actual": len(expected_digests)},
            )
        issues: list[dict[str, Any]] = []
        data = geometry.read_all()
        for index, expected in enumerate(expected_digests):
            start = index * geometry.block_size
            actual = sha256_bytes(data[start : start + geometry.block_size])
            if actual != expected:
                issues.append(
                    {
                        "block_index": index,
                        "expected_digest": f"sha256:{expected}",
                        "actual_digest": f"sha256:{actual}",
                    }
                )
        return {
            "ok": not issues,
            "checked_blocks": geometry.block_count,
            "issue_count": len(issues),
            "issues": issues,
            "volume_sha256": sha256_bytes(data),
        }

    def operation_rows(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM operations ORDER BY created_at, operation_id"
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["plan"] = json.loads(item.pop("plan_json"))
            item["details"] = json.loads(item.pop("details_json"))
            item["requires_rerun"] = bool(item["requires_rerun"])
            result.append(item)
        return result

    def recover_session(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM operations WHERE status IN ('incomplete', 'unknown', 'incomplete_unknown') ORDER BY created_at"
            ).fetchall()
        recovered: list[dict[str, Any]] = []
        geometry = self.geometry()
        current_sha = sha256_file(geometry.image_path)
        for row in rows:
            operation_id = str(row["operation_id"])
            expected_sha = row["expected_sha256"]
            prior_sha = row["prior_sha256"]
            temp_relative = row["temp_path"]
            temp_path = self.root / temp_relative if temp_relative else None
            details = json.loads(row["details_json"])
            marker_path = self.tmp_dir / f"{operation_id}.json"
            marker: dict[str, Any] = {}
            marker_error: str | None = None
            if marker_path.exists():
                try:
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    marker_error = str(exc)
            marker_matches = (
                marker_path.exists()
                and marker_error is None
                and marker.get("prior_sha256") in (None, prior_sha)
                and marker.get("expected_sha256") in (None, expected_sha)
            )
            if not marker_matches:
                phase = "unknown"
                status = "unknown"
                requires_rerun = 1
                outcome = "temporary_marker_missing_or_inconsistent"
                verification = {
                    "ok": False,
                    "checked_blocks": 0,
                    "issue_count": 1,
                    "issues": [
                        {
                            "field": "tmp_marker",
                            "message": marker_error or "temporary operation marker is missing or inconsistent",
                        }
                    ],
                }
            elif current_sha == expected_sha:
                digest_table = details.get("expected_block_digests", [])
                verification = self.verify_volume_blocks(digest_table) if digest_table else {
                    "ok": True,
                    "checked_blocks": 0,
                    "issue_count": 0,
                    "issues": [],
                }
                phase = "recovered_completed"
                status = "completed" if verification["ok"] else "unknown"
                requires_rerun = 0 if verification["ok"] else 1
                outcome = "recovered_committed_volume" if verification["ok"] else "unknown"
            elif current_sha == prior_sha:
                if temp_path is not None and temp_path.exists():
                    try:
                        ensure_writable(temp_path)
                        temp_path.unlink()
                    except OSError:
                        pass
                phase = "rolled_back"
                status = "incomplete_unknown"
                requires_rerun = 1
                outcome = "volume_unchanged_requires_rerun"
                verification = {
                    "ok": False,
                    "checked_blocks": 0,
                    "issue_count": 1,
                    "issues": [{"field": "rollback", "message": "interrupted before atomic replacement"}],
                }
            else:
                phase = "unknown"
                status = "unknown"
                requires_rerun = 1
                outcome = "volume_digest_matches_neither_prior_nor_expected"
                verification = {
                    "ok": False,
                    "checked_blocks": 0,
                    "issue_count": 1,
                    "issues": [{"field": "volume", "message": outcome}],
                }
            with self._connect() as connection:
                connection.execute(
                    "UPDATE operations SET phase=?, status=?, requires_rerun=?, updated_at=? WHERE operation_id=?",
                    (phase, status, requires_rerun, utc_now(), operation_id),
                )
            marker.update(
                {
                    "operation_id": operation_id,
                    "phase": phase,
                    "status": status,
                    "requires_rerun": bool(requires_rerun),
                    "recovery_outcome": outcome,
                    "recovered_at": utc_now(),
                    "verification": verification,
                    "marker_present": marker_path.exists(),
                }
            )
            atomic_write_json(marker_path, marker)
            item = {
                "operation_id": operation_id,
                "target_snapshot": row["target_snapshot"],
                "phase": phase,
                "status": status,
                "outcome": outcome,
                "requires_rerun": bool(requires_rerun),
                "current_sha256": current_sha,
                "prior_sha256": prior_sha,
                "expected_sha256": expected_sha,
                "verification": verification,
                "marker_present": marker_path.exists(),
            }
            recovered.append(item)
            code = E_STATE if requires_rerun else None
            self.logger.emit(
                "rollback_recovered",
                code=code,
                operation="recover",
                snapshot_id=row["target_snapshot"],
                operation_id=operation_id,
                phase=phase,
                status=status,
                requires_rerun=bool(requires_rerun),
                outcome=outcome,
            )
        return recovered

    def repair_objects(self) -> dict[str, Any]:
        refs = sorted(
            {
                str(row["ref"])
                for row in self._all_block_rows()
            }
        )
        repaired: list[str] = []
        missing_backup: list[str] = []
        for ref in refs:
            ok, _ = self.verify_ref_object(ref)
            if ok:
                continue
            if self.restore_object_from_backup(ref):
                repaired.append(ref)
            else:
                missing_backup.append(ref)
        report = {
            "ok": not missing_backup,
            "checked_objects": len(refs),
            "repaired_objects": repaired,
            "missing_backups": missing_backup,
        }
        self.logger.emit("objects_repaired", operation="repair", **report)
        return report

    def _all_block_rows(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT ref FROM (
                    SELECT ref FROM snapshot_blocks
                    UNION
                    SELECT ref FROM snapshot_views
                ) ORDER BY ref
                """
            ).fetchall()

    # ------------------------------------------------------------------ audits
    def snapshot_versions(self) -> dict[str, int]:
        return {
            str(row["snapshot_id"]): int(row["snapshot_version"])
            for row in self.snapshot_rows()
        }

    def audit(self, *, strict_manifest: bool = False) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        snapshot_results: list[dict[str, Any]] = []
        state_version = self.state_version
        try:
            self._validate_existing_chain(error_code=E_CHAIN)
        except DiagnosticError as error:
            issues.append(error.to_dict())
        database_model = self.build_manifest_model()
        file_model: NormalizedManifest | None = None
        try:
            file_model = self.read_manifest(strict_version=True)
            if file_model.fingerprint() != database_model.fingerprint():
                issues.append(
                    {
                        "ok": False,
                        "code": E_VERSION,
                        "snapshot_id": "<manifest>",
                        "field": "manifest_digest",
                        "message": "manifest content digest differs from disk state",
                        "expected": f"sha256:{database_model.manifest_digest}",
                        "actual": f"sha256:{file_model.manifest_digest}",
                    }
                )
        except DiagnosticError as error:
            issues.append(error.to_dict())
        rows = self.snapshot_rows()
        initial_sha = self.meta().get("initial_sha256")
        volume_data = self.geometry().read_all()
        chain_replay = bytearray(volume_data)
        for replay_row in reversed(rows):
            replay_snapshot_id = str(replay_row["snapshot_id"])
            for block_index, item in sorted(self.snapshot_blocks(replay_snapshot_id).items()):
                try:
                    data = self.read_ref_object(item["ref"])
                except DiagnosticError:
                    continue
                if sha256_bytes(data) != item["digest"]:
                    continue
                start = block_index * int(replay_row["block_size"])
                chain_replay[start : start + int(replay_row["block_size"])] = data
        chain_replay_sha = sha256_bytes(bytes(chain_replay))
        chain_replay_ok = bool(initial_sha) and chain_replay_sha == initial_sha
        if not chain_replay_ok:
            issues.append(
                {
                    "ok": False,
                    "code": E_CHAIN,
                    "snapshot_id": str(rows[0]["snapshot_id"]) if rows else "<manifest>",
                    "field": "chain_replay_sha256",
                    "message": "snapshot undo chain cannot replay back to the initial volume",
                    "expected": f"sha256:{initial_sha}" if initial_sha else None,
                    "actual": f"sha256:{chain_replay_sha}",
                }
            )
        for row in rows:
            snapshot_id = str(row["snapshot_id"])
            bitmap = bytes(row["bitmap"])
            snapshot_issues: list[dict[str, Any]] = []
            try:
                validate_bitmap(
                    bitmap,
                    int(row["block_count"]),
                    snapshot_id=snapshot_id,
                    bitmap_length_value=int(row["bitmap_length"]),
                )
            except DiagnosticError as error:
                snapshot_issues.append(error.to_dict())
            if bitmap_checksum(bitmap) != str(row["bitmap_checksum"]):
                snapshot_issues.append(
                    {
                        "ok": False,
                        "code": E_FORMAT,
                        "snapshot_id": snapshot_id,
                        "field": "bitmap_checksum",
                        "message": "stored bitmap checksum does not match bitmap",
                        "expected": f"sha256:{bitmap_checksum(bitmap)}",
                        "actual": f"sha256:{row['bitmap_checksum']}",
                    }
                )
            bits = set(set_bits(bitmap, int(row["block_count"])))
            refs = self.snapshot_blocks(snapshot_id)
            if set(refs) != bits:
                snapshot_issues.append(
                    {
                        "ok": False,
                        "code": E_CHAIN,
                        "snapshot_id": snapshot_id,
                        "field": "block_refs",
                        "message": "bitmap and data reference sets disagree",
                        "missing_refs": sorted(bits - set(refs)),
                        "extra_refs": sorted(set(refs) - bits),
                    }
                )
            replay = bytearray(volume_data)
            for block_index, item in sorted(refs.items()):
                ok, actual = self.verify_ref_object(item["ref"])
                if not ok or actual != item["digest"]:
                    snapshot_issues.append(
                        {
                            "ok": False,
                            "code": E_INTEGRITY,
                            "snapshot_id": snapshot_id,
                            "field": "block_refs",
                            "block_index": block_index,
                            "message": "data reference digest verification failed",
                            "expected": f"sha256:{item['digest']}",
                            "actual": f"sha256:{actual}" if actual else None,
                        }
                    )
                    continue
                data = self.read_ref_object(item["ref"])
                start = block_index * int(row["block_size"])
                replay[start : start + int(row["block_size"])] = data
            view_refs = self.snapshot_views(snapshot_id)
            if set(view_refs) != set(range(int(row["block_count"]))):
                snapshot_issues.append(
                    {
                        "ok": False,
                        "code": E_CHAIN,
                        "snapshot_id": snapshot_id,
                        "field": "snapshot_views",
                        "message": "snapshot does not have a complete immutable checkpoint view",
                        "expected": int(row["block_count"]),
                        "actual": len(view_refs),
                    }
                )
            replay = bytearray(int(row["block_size"]) * int(row["block_count"]))
            for block_index, item in sorted(view_refs.items()):
                ok, actual = self.verify_ref_object(item["ref"])
                if not ok or actual != item["digest"]:
                    snapshot_issues.append(
                        {
                            "ok": False,
                            "code": E_INTEGRITY,
                            "snapshot_id": snapshot_id,
                            "field": "snapshot_views",
                            "block_index": block_index,
                            "message": "checkpoint view data reference failed digest verification",
                            "expected": f"sha256:{item['digest']}",
                            "actual": f"sha256:{actual}" if actual else None,
                        }
                    )
                    continue
                data = self.read_ref_object(item["ref"])
                start = block_index * int(row["block_size"])
                replay[start : start + int(row["block_size"])] = data
            replay_sha = sha256_bytes(bytes(replay))
            checkpoint_ok = replay_sha == str(row["checkpoint_digest"])
            if not checkpoint_ok:
                snapshot_issues.append(
                    {
                        "ok": False,
                        "code": E_CHAIN,
                        "snapshot_id": snapshot_id,
                        "field": "checkpoint_digest",
                        "message": "snapshot chain cannot replay to the recorded checkpoint",
                        "expected": f"sha256:{row['checkpoint_digest']}",
                        "actual": f"sha256:{replay_sha}",
                    }
                )
            if row["parent_snapshot_id"] is None and initial_sha and str(row["checkpoint_digest"]) != initial_sha:
                snapshot_issues.append(
                    {
                        "ok": False,
                        "code": E_CHAIN,
                        "snapshot_id": snapshot_id,
                        "field": "checkpoint_digest",
                        "message": "root snapshot does not replay to the initial volume",
                        "expected": f"sha256:{initial_sha}",
                        "actual": f"sha256:{row['checkpoint_digest']}",
                    }
                )
            if file_model is not None:
                manifest_entry = file_model.by_id().get(snapshot_id)
                if manifest_entry is None:
                    snapshot_issues.append(
                        {
                            "ok": False,
                            "code": E_VERSION,
                            "snapshot_id": snapshot_id,
                            "field": "snapshot_id",
                            "message": "snapshot is missing from the manifest",
                        }
                    )
                elif int(manifest_entry.snapshot_version) != int(row["snapshot_version"]):
                    snapshot_issues.append(
                        {
                            "ok": False,
                            "code": E_VERSION,
                            "snapshot_id": snapshot_id,
                            "field": "snapshot_version",
                            "message": "manifest snapshot version does not match SQLite",
                            "expected": int(row["snapshot_version"]),
                            "actual": int(manifest_entry.snapshot_version),
                        }
                    )
            issues.extend(snapshot_issues)
            snapshot_results.append(
                {
                    "snapshot_id": snapshot_id,
                    "snapshot_version": int(row["snapshot_version"]),
                    "parent_snapshot_id": row["parent_snapshot_id"],
                    "bitmap_bits": len(bits),
                    "reference_count": len(refs),
                    "full_view_count": len(view_refs),
                    "checkpoint_sha256": f"sha256:{row['checkpoint_digest']}",
                    "replay_sha256": f"sha256:{replay_sha}",
                    "ok": not snapshot_issues,
                    "issues": snapshot_issues,
                }
            )
        status = "pass" if not issues else "fail"
        conclusion = {
            "ok": not issues,
            "status": status,
            "state_version": state_version,
            "snapshot_versions": self.snapshot_versions(),
            "manifest_digest": f"sha256:{database_model.manifest_digest}",
            "manifest_format": file_model.source_format if file_model is not None else None,
            "chain_replay_sha256": f"sha256:{chain_replay_sha}",
            "initial_volume_sha256": f"sha256:{initial_sha}" if initial_sha else None,
            "chain_replay_ok": chain_replay_ok,
            "snapshots": snapshot_results,
            "issues": issues,
            "issue_count": len(issues),
        }
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO audits(
                    created_at, state_version, snapshot_versions_json,
                    manifest_digest, status, conclusion_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    state_version,
                    canonical_json_text(conclusion["snapshot_versions"]),
                    database_model.manifest_digest,
                    status,
                    canonical_json_text(conclusion),
                ),
            )
            audit_version = int(cursor.lastrowid)
        conclusion["audit_version"] = audit_version
        self.logger.emit(
            "audit_completed" if status == "pass" else "audit_failed",
            code=None if status == "pass" else (issues[0].get("code") if issues else E_STATE),
            operation="audit",
            audit_version=audit_version,
            status=status,
            issue_count=len(issues),
        )
        if strict_manifest and issues:
            first = issues[0]
            raise DiagnosticError(
                first.get("code", E_STATE),
                str(first.get("message", "audit failed")),
                snapshot_id=first.get("snapshot_id"),
                field=first.get("field"),
                details={"audit_version": audit_version, "issues": issues},
            )
        return conclusion

    def latest_audit(self, *, strict: bool = True) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM audits ORDER BY audit_version DESC LIMIT 1"
            ).fetchone()
        if row is None:
            raise DiagnosticError(
                E_STATE,
                "no audit conclusion has been persisted",
                field="audit",
            )
        conclusion = json.loads(row["conclusion_json"])
        conclusion["audit_version"] = int(row["audit_version"])
        conclusion["created_at"] = row["created_at"]
        stored_versions = json.loads(row["snapshot_versions_json"])
        current_versions = self.snapshot_versions()
        current_manifest_digest = self.build_manifest_model().manifest_digest
        manifest_matches = str(row["manifest_digest"]) == current_manifest_digest
        differences = {
            "state_version": {
                "stored": int(row["state_version"]),
                "current": self.state_version,
            },
            "snapshot_versions": {
                "stored": stored_versions,
                "current": current_versions,
            },
            "manifest_digest": {
                "stored": f"sha256:{row['manifest_digest']}",
                "current": f"sha256:{current_manifest_digest}",
            },
        }
        versions_match = (
            int(row["state_version"]) == self.state_version
            and stored_versions == current_versions
            and manifest_matches
        )
        conclusion["current_state_version"] = self.state_version
        conclusion["version_current"] = versions_match
        if strict and not versions_match:
            raise DiagnosticError(
                E_VERSION,
                "persisted audit conclusion is stale for the current snapshot versions",
                field="audit_version",
                details={
                    "audit_version": int(row["audit_version"]),
                    "differences": differences,
                    "message": "run a new audit; old conclusions are never overwritten",
                },
            )
        return conclusion

    def audit_history(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT audit_version, created_at, state_version, status, manifest_digest FROM audits ORDER BY audit_version DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
