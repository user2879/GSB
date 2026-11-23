from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .content import LEAF_SIZE
from .digest import MerkleBuilder, leaf_digest, merkle_root
from .events import monotonic_ms
from .models import ObjectSpec, SessionSpec


FORMAT_VERSION = 1
TAG_LEDGER = b"OSSDIAG-LEDGER-v1\x00"
TAG_PARTIAL = b"OSSDIAG-PARTIAL-v1\x00"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_digest(session: SessionSpec) -> str:
    return hashlib.sha256(canonical_json(session.canonical_dict()).encode("utf-8")).hexdigest()


def object_spec_digest(spec: ObjectSpec) -> str:
    return hashlib.sha256(canonical_json(spec.canonical_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReconcileResult:
    replaced: bool
    code: str | None
    reason: str | None
    issues: tuple[dict[str, Any], ...]
    replaced_shards: tuple[tuple[str, int], ...]
    snapshot_id: int | None
    backup_path: str | None
    recovered_unknown: tuple[dict[str, Any], ...] = ()
    reclaimed_leases: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "replaced": self.replaced,
            "code": self.code,
            "reason": self.reason,
            "issues": list(self.issues),
            "replaced_shards": [{"object_key": key, "shard_index": index} for key, index in self.replaced_shards],
            "snapshot_id": self.snapshot_id,
            "backup_path": self.backup_path,
            "recovered_unknown": list(self.recovered_unknown),
            "reclaimed_leases": list(self.reclaimed_leases),
        }


class Ledger:
    """SQLite ledger with full-replacement recovery semantics."""

    def __init__(self, path: str | Path, data_dir: str | Path, backup_dir: str | Path | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = Path(backup_dir) if backup_dir else self.path.parent / "ledger_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = self._connect()
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def _create_schema(self) -> None:
        statements = (
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            """
            CREATE TABLE IF NOT EXISTS objects (
                object_key TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                spec_digest TEXT NOT NULL,
                total_bytes INTEGER NOT NULL,
                shard_size INTEGER NOT NULL,
                batch_size INTEGER NOT NULL,
                selected_path TEXT NOT NULL,
                path_reason TEXT,
                path_scores_json TEXT,
                tie_boundary INTEGER NOT NULL DEFAULT 0,
                scale_exponent INTEGER NOT NULL,
                object_digest TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                updated_mono_ms INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS shards (
                object_key TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                content_digest TEXT,
                partial_digest TEXT,
                lease_version INTEGER NOT NULL DEFAULT 0,
                renew_count INTEGER NOT NULL DEFAULT 0,
                completed_mono_ms INTEGER,
                progress_cursor INTEGER NOT NULL,
                digest_duration_ns INTEGER NOT NULL DEFAULT 0,
                attempt INTEGER NOT NULL DEFAULT 0,
                unknown INTEGER NOT NULL DEFAULT 0,
                updated_mono_ms INTEGER NOT NULL,
                PRIMARY KEY (object_key, shard_index),
                FOREIGN KEY (object_key) REFERENCES objects(object_key) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS batch_commits (
                commit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_key TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                lease_version INTEGER NOT NULL,
                renew_count INTEGER NOT NULL,
                completed_mono_ms INTEGER NOT NULL,
                progress_cursor INTEGER NOT NULL,
                digest_duration_ns INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                UNIQUE (object_key, shard_index, byte_start, byte_end)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS shard_leaves (
                object_key TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                leaf_index INTEGER NOT NULL,
                digest TEXT NOT NULL,
                PRIMARY KEY (object_key, shard_index, leaf_index)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS inflight (
                run_id TEXT PRIMARY KEY,
                object_key TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                lease_version INTEGER NOT NULL,
                renew_count INTEGER NOT NULL,
                started_mono_ms INTEGER NOT NULL,
                phase TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lease_runtime (
                object_key TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                version INTEGER NOT NULL,
                granted_ms INTEGER NOT NULL,
                expires_ms INTEGER NOT NULL,
                renew_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                updated_ms INTEGER NOT NULL,
                PRIMARY KEY (object_key, shard_index, version)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_path TEXT NOT NULL,
                reason TEXT NOT NULL,
                replaced_shards_json TEXT NOT NULL,
                created_mono_ms INTEGER NOT NULL,
                read_only INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS replacement_shards (
                snapshot_id INTEGER NOT NULL,
                object_key TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                PRIMARY KEY (snapshot_id, object_key, shard_index)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_batches_shard ON batch_commits(object_key, shard_index, byte_start)",
            "CREATE INDEX IF NOT EXISTS idx_leases_object ON lease_runtime(object_key, shard_index)",
        )
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in statements:
                    self.conn.execute(statement)
                now = monotonic_ms()
                self.conn.execute(
                    "INSERT OR IGNORE INTO meta(key,value) VALUES('format_version',?)",
                    (str(FORMAT_VERSION),),
                )
                self.conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('spec_digest','')")
                self.conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('ledger_digest','')")
                self.conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('created_mono_ms',?)", (str(now),))
                self.conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('updated_mono_ms',?)", (str(now),))
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")
            object_count = int(self.conn.execute("SELECT COUNT(*) AS n FROM objects").fetchone()["n"])
            snapshot_count = int(self.conn.execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()["n"])
            if self._meta_get_locked("ledger_digest") == "" and object_count == 0 and snapshot_count == 0:
                with self._transaction():
                    self._refresh_digest_locked()

    def _meta_get_locked(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _meta_set_locked(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def meta(self) -> dict[str, str]:
        with self._lock:
            rows = self.conn.execute("SELECT key,value FROM meta ORDER BY key").fetchall()
            return {str(row["key"]): str(row["value"]) for row in rows}

    def _table_rows_locked(self, table: str, order_by: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}").fetchall()
        return [dict(row) for row in rows]

    def _compute_digest_locked(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(TAG_LEDGER)
        hasher.update(self._meta_get_locked("format_version").encode("ascii"))
        hasher.update((self._meta_get_locked("spec_digest") or "").encode("ascii"))
        tables = (
            ("objects", "object_key"),
            ("shards", "object_key, shard_index"),
            ("batch_commits", "commit_id"),
            ("shard_leaves", "object_key, shard_index, leaf_index"),
            ("inflight", "run_id"),
            ("lease_runtime", "object_key, shard_index, version"),
            ("snapshots", "snapshot_id"),
            ("replacement_shards", "snapshot_id, object_key, shard_index"),
        )
        for table, order in tables:
            hasher.update(table.encode("ascii") + b"\x00")
            for row in self._table_rows_locked(table, order):
                hasher.update(canonical_json(row).encode("utf-8") + b"\n")
        return hasher.hexdigest()

    def _refresh_digest_locked(self) -> None:
        digest = self._compute_digest_locked()
        self._meta_set_locked("ledger_digest", digest)
        self._meta_set_locked("updated_mono_ms", str(monotonic_ms()))

    def ledger_digest(self) -> str:
        with self._lock:
            return self._compute_digest_locked()

    def snapshot_meta(self) -> dict[str, Any]:
        with self._lock:
            meta = {key: value for key, value in self.meta().items()}
            meta["recomputed_digest"] = self._compute_digest_locked()
            meta["snapshots"] = self.list_snapshots()
            return meta

    def object_specs(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._table_rows_locked("objects", "object_key")

    def object_row(self, object_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM objects WHERE object_key=?", (object_key,)).fetchone()
            return None if row is None else dict(row)

    def shard_rows(self, object_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM shards WHERE object_key=? ORDER BY shard_index", (object_key,)
            ).fetchall()
            return [dict(row) for row in rows]

    def all_shard_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._table_rows_locked("shards", "object_key, shard_index")

    def batch_rows(self, object_key: str, shard_index: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM batch_commits WHERE object_key=? AND shard_index=? ORDER BY byte_start",
                (object_key, shard_index),
            ).fetchall()
            return [dict(row) for row in rows]

    def leaf_hex_values(self, object_key: str, shard_index: int) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT digest FROM shard_leaves WHERE object_key=? AND shard_index=? ORDER BY leaf_index",
                (object_key, shard_index),
            ).fetchall()
            return [str(row["digest"]) for row in rows]

    def inflight_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._table_rows_locked("inflight", "run_id")

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM snapshots ORDER BY snapshot_id").fetchall()
            return [dict(row) for row in rows]

    def replacement_shards(self, snapshot_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT object_key,shard_index FROM replacement_shards WHERE snapshot_id=? ORDER BY object_key,shard_index",
                (snapshot_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def lease_stats(self, object_key: str) -> tuple[int, int]:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) AS grants, COALESCE(SUM(renew_count),0) AS renews FROM lease_runtime WHERE object_key=?",
                (object_key,),
            ).fetchone()
            return int(row["grants"]), int(row["renews"])

    def max_lease_version(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT COALESCE(MAX(version),0) AS version FROM lease_runtime").fetchone()
            return int(row["version"])

    def shard_path(self, object_key: str, shard_index: int) -> Path:
        object_hash = hashlib.sha256(object_key.encode("utf-8")).hexdigest()[:24]
        return self.data_dir / "objects" / object_hash / f"shard_{shard_index:08d}.bin"

    def lease_changed(self, lease: Any, status: str) -> None:
        with self._transaction():
            self.conn.execute(
                """
                INSERT INTO lease_runtime(object_key,shard_index,version,granted_ms,expires_ms,renew_count,status,updated_ms)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(object_key,shard_index,version) DO UPDATE SET
                    expires_ms=excluded.expires_ms,
                    renew_count=excluded.renew_count,
                    status=excluded.status,
                    updated_ms=excluded.updated_ms
                """,
                (
                    lease.object_key,
                    lease.shard_index,
                    lease.version,
                    lease.granted_ms,
                    lease.expires_ms,
                    lease.renew_count,
                    status,
                    monotonic_ms(),
                ),
            )
            self._refresh_digest_locked()

    def _verify_meta_locked(self, expected_spec_digest: str) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        format_value = self._meta_get_locked("format_version")
        if format_value != str(FORMAT_VERSION):
            issues.append(
                {
                    "kind": "format_version",
                    "expected": FORMAT_VERSION,
                    "actual": format_value,
                }
            )
        stored_spec_digest = self._meta_get_locked("spec_digest") or ""
        if stored_spec_digest != expected_spec_digest:
            issues.append(
                {
                    "kind": "spec_digest",
                    "expected": expected_spec_digest,
                    "actual": stored_spec_digest,
                }
            )
        stored_digest = self._meta_get_locked("ledger_digest") or ""
        computed_digest = self._compute_digest_locked()
        if stored_digest != computed_digest:
            issues.append(
                {
                    "kind": "ledger_digest",
                    "expected": stored_digest,
                    "actual": computed_digest,
                }
            )
        return issues

    def _verify_disk_locked(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for shard in self._table_rows_locked("shards", "object_key, shard_index"):
            object_key = str(shard["object_key"])
            shard_index = int(shard["shard_index"])
            start = int(shard["byte_start"])
            end = int(shard["byte_end"])
            cursor = int(shard["progress_cursor"])
            path = self.shard_path(object_key, shard_index)
            expected_size = (end - start) if shard["status"] == "complete" else (cursor - start)
            actual_size = path.stat().st_size if path.exists() else 0
            if actual_size != expected_size:
                issues.append(
                    {
                        "kind": "disk_size",
                        "object_key": object_key,
                        "shard_index": shard_index,
                        "expected": expected_size,
                        "actual": actual_size,
                    }
                )
                continue
            if expected_size == 0:
                continue
            stored_leaves = self.conn.execute(
                "SELECT leaf_index,digest FROM shard_leaves WHERE object_key=? AND shard_index=? ORDER BY leaf_index",
                (object_key, shard_index),
            ).fetchall()
            stored_by_index = {int(row["leaf_index"]): str(row["digest"]) for row in stored_leaves}
            leaf_count = (expected_size + LEAF_SIZE - 1) // LEAF_SIZE
            first_leaf = (start - start) // LEAF_SIZE
            expected_leaf_count = (cursor - start) // LEAF_SIZE
            if shard["status"] != "complete" and len(stored_by_index) != expected_leaf_count:
                issues.append(
                    {
                        "kind": "leaf_metadata_gap",
                        "object_key": object_key,
                        "shard_index": shard_index,
                        "expected": expected_leaf_count,
                        "actual": len(stored_by_index),
                    }
                )
            builder = MerkleBuilder()
            with path.open("rb") as handle:
                absolute = start
                for local_index in range(leaf_count):
                    wanted = min(LEAF_SIZE, expected_size - local_index * LEAF_SIZE)
                    data = handle.read(wanted)
                    if len(data) != wanted:
                        issues.append(
                            {
                                "kind": "disk_short_read",
                                "object_key": object_key,
                                "shard_index": shard_index,
                                "leaf_index": local_index,
                            }
                        )
                        break
                    leaf_no = (absolute - start) // LEAF_SIZE
                    leaf_len = min(LEAF_SIZE, end - absolute)
                    digest_hex = leaf_digest(object_key, shard_index, leaf_no, leaf_len, data).hex()
                    builder.add(bytes.fromhex(digest_hex))
                    stored = stored_by_index.get(leaf_no)
                    if stored is not None and stored != digest_hex:
                        issues.append(
                            {
                                "kind": "disk_leaf_digest",
                                "object_key": object_key,
                                "shard_index": shard_index,
                                "leaf_index": leaf_no,
                            }
                        )
                    absolute += wanted
            if shard["status"] == "complete" and shard["content_digest"]:
                root = builder.root().hex()
                if root != shard["content_digest"]:
                    issues.append(
                        {
                            "kind": "disk_content_digest",
                            "object_key": object_key,
                            "shard_index": shard_index,
                            "expected": shard["content_digest"],
                            "actual": root,
                        }
                    )
        return issues

    def reconcile(self, session: SessionSpec, *, force_snapshot_conflict: bool = False) -> ReconcileResult:
        expected = spec_digest(session)
        with self._lock:
            issues = self._verify_meta_locked(expected)
            stored_spec = self._meta_get_locked("spec_digest") or ""
            object_count = int(self.conn.execute("SELECT COUNT(*) AS n FROM objects").fetchone()["n"])
            if not force_snapshot_conflict and not stored_spec and object_count == 0:
                issues = [issue for issue in issues if issue.get("kind") != "spec_digest"]
            if force_snapshot_conflict:
                issues.insert(
                    0,
                    {
                        "kind": "injected_snapshot_conflict",
                        "detail": "command-line parameter requested a format/version conflict",
                    },
                )
            if issues:
                replaced_shards = tuple(
                    (str(row["object_key"]), int(row["shard_index"]))
                    for row in self.conn.execute(
                        "SELECT object_key,shard_index FROM shards ORDER BY object_key,shard_index"
                    ).fetchall()
                )
                reason = "injected snapshot conflict" if force_snapshot_conflict else issues[0]["kind"]
                snapshot_id, backup_path = self._replace_ledger_locked(reason, replaced_shards)
                self._meta_set_locked("spec_digest", expected)
                self._refresh_digest_locked()
                return ReconcileResult(
                    replaced=True,
                    code="E_SNAPSHOT",
                    reason=reason,
                    issues=tuple(issues),
                    replaced_shards=replaced_shards,
                    snapshot_id=snapshot_id,
                    backup_path=backup_path,
                )

            if stored_spec == "":
                with self._transaction():
                    self._meta_set_locked("spec_digest", expected)
                    self._refresh_digest_locked()
            unknown, reclaimed = self._recover_inflight_locked()
            if unknown or reclaimed:
                with self._transaction():
                    self._refresh_digest_locked()
            disk_issues = self._verify_disk_locked()
            if disk_issues:
                replaced_shards = tuple(
                    (str(row["object_key"]), int(row["shard_index"]))
                    for row in self.conn.execute(
                        "SELECT object_key,shard_index FROM shards ORDER BY object_key,shard_index"
                    ).fetchall()
                )
                snapshot_id, backup_path = self._replace_ledger_locked(disk_issues[0]["kind"], replaced_shards)
                with self._transaction():
                    self._meta_set_locked("spec_digest", expected)
                    self._refresh_digest_locked()
                return ReconcileResult(
                    replaced=True,
                    code="E_SNAPSHOT",
                    reason=disk_issues[0]["kind"],
                    issues=tuple(disk_issues),
                    replaced_shards=replaced_shards,
                    snapshot_id=snapshot_id,
                    backup_path=backup_path,
                    recovered_unknown=tuple(unknown),
                    reclaimed_leases=tuple(reclaimed),
                )
            return ReconcileResult(
                replaced=False,
                code=None,
                reason=None,
                issues=(),
                replaced_shards=(),
                snapshot_id=None,
                backup_path=None,
                recovered_unknown=tuple(unknown),
                reclaimed_leases=tuple(reclaimed),
            )

    def _replace_ledger_locked(self, reason: str, replaced_shards: Sequence[tuple[str, int]]) -> tuple[int, str]:
        stamp = time.time_ns()
        backup_path = self.backup_dir / f"ledger.{stamp}.sqlite"
        destination = sqlite3.connect(backup_path)
        try:
            self.conn.backup(destination)
        finally:
            destination.close()
        try:
            os.chmod(backup_path, 0o444)
        except OSError:
            pass
        self.conn.close()
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        self.conn = self._connect()
        self._create_schema()
        replaced_json = canonical_json(
            [{"object_key": key, "shard_index": index} for key, index in replaced_shards]
        )
        with self._transaction():
            cursor = self.conn.execute(
                """
                INSERT INTO snapshots(backup_path,reason,replaced_shards_json,created_mono_ms,read_only)
                VALUES(?,?,?,?,1)
                """,
                (str(backup_path.resolve()), reason, replaced_json, monotonic_ms()),
            )
            snapshot_id = int(cursor.lastrowid)
            for object_key, shard_index in replaced_shards:
                self.conn.execute(
                    "INSERT INTO replacement_shards(snapshot_id,object_key,shard_index) VALUES(?,?,?)",
                    (snapshot_id, object_key, shard_index),
                )
            self._meta_set_locked("format_version", str(FORMAT_VERSION))
            self._meta_set_locked("spec_digest", "")
            self._refresh_digest_locked()
        return snapshot_id, str(backup_path.resolve())

    def _recover_inflight_locked(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        inflight = self._table_rows_locked("inflight", "run_id")
        unknown: list[dict[str, Any]] = []
        shard_keys: set[tuple[str, int]] = set()
        if inflight:
            with self._transaction():
                for row in inflight:
                    object_key = str(row["object_key"])
                    shard_index = int(row["shard_index"])
                    shard = self.conn.execute(
                        "SELECT progress_cursor,byte_start,byte_end FROM shards WHERE object_key=? AND shard_index=?",
                        (object_key, shard_index),
                    ).fetchone()
                    if shard is not None:
                        cursor = int(shard["progress_cursor"])
                        self.conn.execute(
                            """
                            UPDATE shards SET status='unknown', unknown=1, attempt=attempt+1,
                                updated_mono_ms=? WHERE object_key=? AND shard_index=?
                            """,
                            (monotonic_ms(), object_key, shard_index),
                        )
                        shard_keys.add((object_key, shard_index))
                        unknown.append(
                            {
                                "object_key": object_key,
                                "shard_index": shard_index,
                                "committed_cursor": cursor,
                                "attempted_start": int(row["byte_start"]),
                                "attempted_end": int(row["byte_end"]),
                                "lease_version": int(row["lease_version"]),
                                "run_id": str(row["run_id"]),
                                "state": "unknown_uncommitted_batch",
                                "requires_rerun": True,
                            }
                        )
                self.conn.execute("DELETE FROM inflight")
                self._refresh_digest_locked()
        for object_key, shard_index in sorted(shard_keys):
            self._truncate_shard_to_cursor(object_key, shard_index)

        reclaimed: list[dict[str, Any]] = []
        active = self.conn.execute(
            "SELECT * FROM lease_runtime WHERE status='active' ORDER BY object_key,shard_index,version"
        ).fetchall()
        if active:
            with self._transaction():
                for row in active:
                    reclaimed.append(
                        {
                            "object_key": str(row["object_key"]),
                            "shard_index": int(row["shard_index"]),
                            "version": int(row["version"]),
                            "action": "quota_reclaimed",
                        }
                    )
                self.conn.execute(
                    "UPDATE lease_runtime SET status='reclaimed_after_crash', updated_ms=? WHERE status='active'",
                    (monotonic_ms(),),
                )
                self._refresh_digest_locked()
        return unknown, reclaimed

    def _truncate_shard_to_cursor(self, object_key: str, shard_index: int) -> None:
        with self._lock:
            row = self.conn.execute(
                "SELECT progress_cursor,byte_start FROM shards WHERE object_key=? AND shard_index=?",
                (object_key, shard_index),
            ).fetchone()
            if row is None:
                return
            target = int(row["progress_cursor"]) - int(row["byte_start"])
        path = self.shard_path(object_key, shard_index)
        if path.exists():
            with path.open("r+b") as handle:
                handle.truncate(max(0, target))
                handle.flush()
                os.fsync(handle.fileno())

    def prepare_session(
        self,
        session: SessionSpec,
        selections: dict[str, Any],
    ) -> None:
        expected = spec_digest(session)
        with self._transaction():
            stored = self._meta_get_locked("spec_digest") or ""
            if stored and stored != expected:
                raise RuntimeError("session specification changed without ledger replacement")
            changed = stored != expected
            if changed:
                self._meta_set_locked("spec_digest", expected)
            now = monotonic_ms()
            for spec in session.objects:
                selection = selections[spec.object_key]
                desired_spec_json = canonical_json(spec.canonical_dict())
                desired_spec_digest = object_spec_digest(spec)
                desired_scores = canonical_json([estimate.as_dict() for estimate in selection.estimates])
                desired_boundary = 1 if selection.boundary_case else 0
                row = self.conn.execute(
                    "SELECT * FROM objects WHERE object_key=?", (spec.object_key,)
                ).fetchone()
                if row is None:
                    changed = True
                    self.conn.execute(
                        """
                        INSERT INTO objects(object_key,spec_json,spec_digest,total_bytes,shard_size,batch_size,
                            selected_path,path_reason,path_scores_json,tie_boundary,scale_exponent,status,updated_mono_ms)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            spec.object_key,
                            desired_spec_json,
                            desired_spec_digest,
                            spec.total_bytes,
                            spec.shard_size,
                            spec.batch_size,
                            selection.path,
                            selection.reason,
                            desired_scores,
                            desired_boundary,
                            session.scale_exponent,
                            "pending",
                            now,
                        ),
                    )
                    for shard_index in range(spec.shard_count):
                        start, end = spec.shard_bounds(shard_index)
                        self.conn.execute(
                            """
                            INSERT INTO shards(object_key,shard_index,byte_start,byte_end,status,progress_cursor,updated_mono_ms)
                            VALUES(?,?,?,?,?,?,?)
                            """,
                            (spec.object_key, shard_index, start, end, "pending", start, now),
                        )
                else:
                    object_changed = any(
                        (
                            row["selected_path"] != selection.path,
                            row["path_reason"] != selection.reason,
                            row["path_scores_json"] != desired_scores,
                            int(row["tie_boundary"]) != desired_boundary,
                            int(row["scale_exponent"]) != session.scale_exponent,
                            row["spec_json"] != desired_spec_json,
                            row["spec_digest"] != desired_spec_digest,
                        )
                    )
                    if object_changed:
                        changed = True
                        self.conn.execute(
                            """
                            UPDATE objects SET selected_path=?,path_reason=?,path_scores_json=?,tie_boundary=?,scale_exponent=?,
                                updated_mono_ms=?,spec_json=?,spec_digest=?,total_bytes=?,shard_size=?,batch_size=?
                            WHERE object_key=?
                            """,
                            (
                                selection.path,
                                selection.reason,
                                desired_scores,
                                desired_boundary,
                                session.scale_exponent,
                                now,
                                desired_spec_json,
                                desired_spec_digest,
                                spec.total_bytes,
                                spec.shard_size,
                                spec.batch_size,
                                spec.object_key,
                            ),
                        )
                    for shard_index in range(spec.shard_count):
                        start, end = spec.shard_bounds(shard_index)
                        shard = self.conn.execute(
                            "SELECT byte_start,byte_end FROM shards WHERE object_key=? AND shard_index=?",
                            (spec.object_key, shard_index),
                        ).fetchone()
                        if shard is None:
                            changed = True
                            self.conn.execute(
                                """
                                INSERT INTO shards(object_key,shard_index,byte_start,byte_end,status,progress_cursor,updated_mono_ms)
                                VALUES(?,?,?,?,?,?,?)
                                """,
                                (spec.object_key, shard_index, start, end, "pending", start, now),
                            )
                        elif int(shard["byte_start"]) != start or int(shard["byte_end"]) != end:
                            changed = True
                            self.conn.execute(
                                """
                                UPDATE shards SET byte_start=?,byte_end=?,updated_mono_ms=?
                                WHERE object_key=? AND shard_index=?
                                """,
                                (start, end, now, spec.object_key, shard_index),
                            )
            if changed:
                self._refresh_digest_locked()

    def begin_batch(
        self,
        *,
        run_id: str,
        object_key: str,
        shard_index: int,
        byte_start: int,
        byte_end: int,
        lease_version: int,
        renew_count: int,
        phase: str = "constructing",
    ) -> None:
        with self._transaction():
            self.conn.execute(
                """
                INSERT OR REPLACE INTO inflight(run_id,object_key,shard_index,byte_start,byte_end,
                    lease_version,renew_count,started_mono_ms,phase)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    object_key,
                    shard_index,
                    byte_start,
                    byte_end,
                    lease_version,
                    renew_count,
                    monotonic_ms(),
                    phase,
                ),
            )
            self._refresh_digest_locked()

    def commit_batch(
        self,
        *,
        run_id: str,
        object_key: str,
        shard_index: int,
        byte_start: int,
        byte_end: int,
        content_digest_hex: str,
        lease_version: int,
        renew_count: int,
        progress_cursor: int,
        digest_duration_ns: int,
        leaves: Iterable[tuple[int, str]],
        final_content_digest_hex: str | None,
    ) -> None:
        completed = monotonic_ms()
        with self._transaction():
            inflight = self.conn.execute("SELECT * FROM inflight WHERE run_id=?", (run_id,)).fetchone()
            if inflight is None:
                raise RuntimeError(f"missing inflight intent for {run_id}")
            existing = self.conn.execute(
                """
                SELECT content_digest FROM batch_commits
                WHERE object_key=? AND shard_index=? AND byte_start=? AND byte_end=?
                """,
                (object_key, shard_index, byte_start, byte_end),
            ).fetchone()
            if existing is not None and str(existing["content_digest"]) != content_digest_hex:
                raise ValueError("replayed batch digest mismatch")
            if existing is None:
                self.conn.execute(
                    """
                    INSERT INTO batch_commits(object_key,shard_index,byte_start,byte_end,content_digest,
                        lease_version,renew_count,completed_mono_ms,progress_cursor,digest_duration_ns,run_id)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        object_key,
                        shard_index,
                        byte_start,
                        byte_end,
                        content_digest_hex,
                        lease_version,
                        renew_count,
                        completed,
                        progress_cursor,
                        digest_duration_ns,
                        run_id,
                    ),
                )
            for leaf_index, digest_hex in leaves:
                self.conn.execute(
                    """
                    INSERT INTO shard_leaves(object_key,shard_index,leaf_index,digest) VALUES(?,?,?,?)
                    ON CONFLICT(object_key,shard_index,leaf_index) DO UPDATE SET digest=excluded.digest
                    """,
                    (object_key, shard_index, leaf_index, digest_hex),
                )
            shard = self.conn.execute(
                "SELECT progress_cursor,partial_digest,digest_duration_ns FROM shards WHERE object_key=? AND shard_index=?",
                (object_key, shard_index),
            ).fetchone()
            if shard is None:
                raise RuntimeError("missing shard row")
            previous_partial = str(shard["partial_digest"] or "")
            chain = hashlib.sha256()
            chain.update(TAG_PARTIAL)
            chain.update(bytes.fromhex(previous_partial) if previous_partial else b"\x00" * 32)
            chain.update(bytes.fromhex(content_digest_hex))
            new_partial = chain.hexdigest()
            complete = final_content_digest_hex is not None
            self.conn.execute(
                """
                UPDATE shards SET progress_cursor=?, partial_digest=?, lease_version=?, renew_count=?,
                    completed_mono_ms=?, digest_duration_ns=digest_duration_ns+?, content_digest=?,
                    status=?, unknown=0, updated_mono_ms=?
                WHERE object_key=? AND shard_index=?
                """,
                (
                    progress_cursor,
                    new_partial,
                    lease_version,
                    renew_count,
                    completed if complete else None,
                    digest_duration_ns,
                    final_content_digest_hex,
                    "complete" if complete else "uploading",
                    completed,
                    object_key,
                    shard_index,
                ),
            )
            self.conn.execute("DELETE FROM inflight WHERE run_id=?", (run_id,))
            if complete:
                self.conn.execute(
                    """
                    UPDATE objects SET status=CASE WHEN EXISTS(
                        SELECT 1 FROM shards WHERE object_key=? AND status!='complete'
                    ) THEN 'uploading' ELSE 'complete' END, updated_mono_ms=? WHERE object_key=?
                    """,
                    (object_key, completed, object_key),
                )
            else:
                self.conn.execute(
                    "UPDATE objects SET status='uploading',updated_mono_ms=? WHERE object_key=?",
                    (completed, object_key),
                )
            self._refresh_digest_locked()

    def abort_inflight(
        self,
        run_id: str,
        *,
        reason: str,
        mark_unknown: bool = True,
    ) -> dict[str, Any] | None:
        with self._transaction():
            row = self.conn.execute("SELECT * FROM inflight WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return None
            record = dict(row)
            object_key = str(row["object_key"])
            shard_index = int(row["shard_index"])
            if mark_unknown:
                self.conn.execute(
                    """
                    UPDATE shards SET status='unknown',unknown=1,attempt=attempt+1,updated_mono_ms=?
                    WHERE object_key=? AND shard_index=?
                    """,
                    (monotonic_ms(), object_key, shard_index),
                )
                self.conn.execute(
                    "UPDATE objects SET status='failed',updated_mono_ms=? WHERE object_key=?",
                    (monotonic_ms(), object_key),
                )
            self.conn.execute("DELETE FROM inflight WHERE run_id=?", (run_id,))
            self._refresh_digest_locked()
        record["reason"] = reason
        record["requires_rerun"] = mark_unknown
        self._truncate_shard_to_cursor(object_key, shard_index)
        return record

    def mark_object_digest(self, object_key: str, digest_hex: str) -> None:
        with self._transaction():
            self.conn.execute(
                "UPDATE objects SET object_digest=?,status='complete',updated_mono_ms=? WHERE object_key=?",
                (digest_hex, monotonic_ms(), object_key),
            )
            self._refresh_digest_locked()

    def mark_object_failed(self, object_key: str, reason: str) -> None:
        with self._transaction():
            self.conn.execute(
                "UPDATE objects SET status='failed',updated_mono_ms=? WHERE object_key=?",
                (monotonic_ms(), object_key),
            )
            self._refresh_digest_locked()

    def verify_completed_object(self, object_key: str) -> dict[str, Any]:
        row = self.object_row(object_key)
        if row is None:
            return {"ok": False, "reason": "missing_object"}
        shards = self.shard_rows(object_key)
        if not shards or any(shard["status"] != "complete" for shard in shards):
            return {"ok": False, "reason": "incomplete_shards"}
        descriptors: list[tuple[int, int, int, bytes]] = []
        for shard in shards:
            path = self.shard_path(object_key, int(shard["shard_index"]))
            builder = MerkleBuilder()
            size = int(shard["byte_end"]) - int(shard["byte_start"])
            with path.open("rb") as handle:
                absolute = int(shard["byte_start"])
                remaining = size
                while remaining:
                    wanted = min(LEAF_SIZE, remaining)
                    data = handle.read(wanted)
                    if len(data) != wanted:
                        return {"ok": False, "reason": "short_read", "shard_index": shard["shard_index"]}
                    leaf_no = (absolute - int(shard["byte_start"])) // LEAF_SIZE
                    leaf_len = min(LEAF_SIZE, int(shard["byte_end"]) - absolute)
                    builder.add(leaf_digest(object_key, int(shard["shard_index"]), leaf_no, leaf_len, data))
                    absolute += wanted
                    remaining -= wanted
            root = builder.root()
            if root.hex() != shard["content_digest"]:
                return {"ok": False, "reason": "shard_digest_mismatch", "shard_index": shard["shard_index"]}
            descriptors.append(
                (int(shard["shard_index"]), int(shard["byte_start"]), int(shard["byte_end"]), root)
            )
        from .digest import object_digest

        digest = object_digest(object_key, int(row["total_bytes"]), descriptors).hex()
        ledger_digest_value = row["object_digest"]
        return {
            "ok": ledger_digest_value == digest,
            "reason": None if ledger_digest_value == digest else "object_digest_mismatch",
            "object_digest": digest,
            "ledger_object_digest": ledger_digest_value,
        }

    def replace_for_test_or_operator(self, reason: str, force_snapshot_conflict: bool = False) -> ReconcileResult:
        """Explicit operator-facing replacement; still preserves a read-only backup."""

        with self._lock:
            replaced_shards = tuple(
                (str(row["object_key"]), int(row["shard_index"]))
                for row in self.conn.execute(
                    "SELECT object_key,shard_index FROM shards ORDER BY object_key,shard_index"
                ).fetchall()
            )
            snapshot_id, backup_path = self._replace_ledger_locked(reason, replaced_shards)
            issues = ({"kind": reason},) if reason else ()
            return ReconcileResult(
                replaced=True,
                code="E_SNAPSHOT",
                reason=reason,
                issues=issues,
                replaced_shards=replaced_shards,
                snapshot_id=snapshot_id,
                backup_path=backup_path,
            )
