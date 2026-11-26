"""SQLite atomic outcome ledger, snapshots and crash recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable

from .errors import DivergenceError, GSBError, SnapshotError
from .samples import canonical_json_bytes, sha256_hex

LEDGER_FORMAT_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
    shard_id TEXT PRIMARY KEY,
    session_version INTEGER NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    record_json TEXT NOT NULL,
    aggregate_digest TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    resource_units INTEGER NOT NULL,
    water_json TEXT NOT NULL,
    error_json TEXT NOT NULL,
    progress_cursor INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    units INTEGER NOT NULL,
    status TEXT NOT NULL,
    issued_ns INTEGER NOT NULL,
    deadline_ns INTEGER NOT NULL,
    closed_ns INTEGER,
    close_reason TEXT
);
CREATE TABLE IF NOT EXISTS uncommitted (
    shard_id TEXT PRIMARY KEY,
    lease_id TEXT NOT NULL UNIQUE,
    node_id TEXT NOT NULL,
    units INTEGER NOT NULL,
    started_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS unknown_shards (
    shard_id TEXT PRIMARY KEY,
    lease_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    units INTEGER NOT NULL,
    started_ns INTEGER NOT NULL,
    reason TEXT NOT NULL,
    recovered_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recovered_ns INTEGER NOT NULL,
    unknown_count INTEGER NOT NULL,
    reclaimed_units INTEGER NOT NULL,
    report_json TEXT NOT NULL
);
"""


def _migrate_schema(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(unknown_shards)").fetchall()}
    if "started_ns" not in columns:
        conn.execute(
            "ALTER TABLE unknown_shards ADD COLUMN started_ns INTEGER NOT NULL DEFAULT 0"
        )


def _canonical_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def _state_digest(conn: sqlite3.Connection) -> str:
    session_version = int(_meta_get(conn, "session_version") or 0)
    rows = conn.execute(
        "SELECT shard_id, content_digest, record_json FROM records ORDER BY shard_id"
    ).fetchall()
    payload = {
        "format_version": LEDGER_FORMAT_VERSION,
        "session_version": session_version,
        "records": [
            {
                "shard_id": str(row[0]),
                "content_digest": str(row[1]),
                "record_json": str(row[2]),
            }
            for row in rows
        ],
    }
    return sha256_hex(canonical_json_bytes(payload))


def _verify_record(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise SnapshotError("ledger record must be a JSON object")
    if record.get("format_version") != LEDGER_FORMAT_VERSION:
        raise SnapshotError(
            "record format_version mismatch",
            details={"expected": LEDGER_FORMAT_VERSION, "actual": record.get("format_version")},
        )
    expected = record.get("content_digest")
    if not isinstance(expected, str):
        raise SnapshotError("record is missing content_digest")
    core = dict(record)
    core.pop("content_digest", None)
    actual = sha256_hex(canonical_json_bytes(core))
    if actual != expected:
        raise DivergenceError(
            str(record.get("shard_id", "<unknown>")),
            "content_digest",
            expected,
            actual,
        )


class Ledger:
    """A single-session ledger.  Every state-changing method owns one transaction."""

    def __init__(self, path: str | Path, *, session_version: int = 1) -> None:
        if session_version < 1:
            raise ValueError("session_version must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_version = session_version
        self.conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(_SCHEMA)
        _migrate_schema(self.conn)
        existing_version = _meta_get(self.conn, "session_version")
        if existing_version is None:
            with self._transaction() as conn:
                _meta_set(conn, "format_version", str(LEDGER_FORMAT_VERSION))
                _meta_set(conn, "session_version", str(session_version))
                _meta_set(conn, "created_ns", str(time.time_ns()))
                _meta_set(conn, "state_digest", _state_digest(conn))
        elif int(existing_version) != session_version:
            self.conn.close()
            raise SnapshotError(
                "ledger session version differs; restore a whole snapshot instead of merging",
                details={
                    "existing_session_version": int(existing_version),
                    "requested_session_version": session_version,
                    "path": str(self.path),
                },
            )
        format_version = _meta_get(self.conn, "format_version")
        if format_version != str(LEDGER_FORMAT_VERSION):
            self.conn.close()
            raise SnapshotError(
                "ledger format version mismatch",
                details={"expected": LEDGER_FORMAT_VERSION, "actual": format_version},
            )

    class _Tx:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def __enter__(self) -> sqlite3.Connection:
            self.conn.execute("BEGIN IMMEDIATE")
            return self.conn

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()

    def _transaction(self) -> "Ledger._Tx":
        return Ledger._Tx(self.conn)

    def close(self) -> None:
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def state_digest(self) -> str:
        digest = _meta_get(self.conn, "state_digest")
        if digest is None:
            raise SnapshotError("ledger is missing its state_digest")
        return digest

    def reserve_lease(
        self,
        *,
        lease_id: str,
        shard_id: str,
        node_id: str,
        units: int,
        issued_ns: int,
        deadline_ns: int,
    ) -> None:
        if units <= 0 or deadline_ns <= issued_ns:
            raise ValueError("invalid lease reservation")
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO leases(lease_id,shard_id,node_id,units,status,issued_ns,deadline_ns)"
                " VALUES(?,?,?,?, 'active', ?,?)",
                (lease_id, shard_id, node_id, units, issued_ns, deadline_ns),
            )
            conn.execute(
                "INSERT INTO uncommitted(shard_id,lease_id,node_id,units,started_ns)"
                " VALUES(?,?,?,?,?)",
                (shard_id, lease_id, node_id, units, issued_ns),
            )
            _meta_set(conn, "state_digest", _state_digest(conn))

    def renew_lease(self, lease_id: str, *, deadline_ns: int) -> None:
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE leases SET deadline_ns=? WHERE lease_id=? AND status='active'",
                (deadline_ns, lease_id),
            )
            if cursor.rowcount != 1:
                raise GSBError("lease is not active", details={"lease_id": lease_id})

    def release_lease(self, lease_id: str, *, reason: str) -> bool:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT shard_id,status FROM leases WHERE lease_id=?", (lease_id,)
            ).fetchone()
            if row is None or row[1] in {"released", "cancelled", "abandoned"}:
                return False
            conn.execute(
                "UPDATE leases SET status='released', closed_ns=?, close_reason=?"
                " WHERE lease_id=? AND status IN ('active','expired')",
                (time.time_ns(), reason, lease_id),
            )
            conn.execute("DELETE FROM uncommitted WHERE lease_id=?", (lease_id,))
            _meta_set(conn, "state_digest", _state_digest(conn))
            return True

    def complete_shard(self, record: dict[str, Any], *, lease_id: str) -> None:
        _verify_record(record)
        record_json = _canonical_text(record)
        with self._transaction() as conn:
            lease = conn.execute(
                "SELECT shard_id,status FROM leases WHERE lease_id=?", (lease_id,)
            ).fetchone()
            if lease is None:
                raise GSBError("committed record has no lease", details={"lease_id": lease_id})
            if str(record["shard_id"]) != str(lease[0]):
                raise GSBError(
                    "record and lease shard mismatch",
                    details={"record_shard": record["shard_id"], "lease_shard": lease[0]},
                )
            conn.execute(
                "INSERT INTO records(shard_id,session_version,format_version,content_digest,"
                "record_json,aggregate_digest,lease_id,resource_units,water_json,error_json,"
                "progress_cursor) VALUES(?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(shard_id) DO UPDATE SET "
                "session_version=excluded.session_version,format_version=excluded.format_version,"
                "content_digest=excluded.content_digest,record_json=excluded.record_json,"
                "aggregate_digest=excluded.aggregate_digest,lease_id=excluded.lease_id,"
                "resource_units=excluded.resource_units,water_json=excluded.water_json,"
                "error_json=excluded.error_json,progress_cursor=excluded.progress_cursor",
                (
                    record["shard_id"],
                    self.session_version,
                    LEDGER_FORMAT_VERSION,
                    record["content_digest"],
                    record_json,
                    record["aggregate_digest"],
                    lease_id,
                    int(record["resource_units"]),
                    _canonical_text(record["water"]),
                    _canonical_text(record["error_attribution"]),
                    int(record["progress_cursor"]),
                ),
            )
            conn.execute(
                "UPDATE leases SET status='released', closed_ns=?, close_reason='completed'"
                " WHERE lease_id=?",
                (time.time_ns(), lease_id),
            )
            conn.execute("DELETE FROM uncommitted WHERE lease_id=?", (lease_id,))
            _meta_set(conn, "state_digest", _state_digest(conn))

    def import_records(self, records: Iterable[dict[str, Any]]) -> None:
        """Seed already authoritative records, still as one atomic transaction."""

        canonical_records = list(records)
        for record in canonical_records:
            _verify_record(record)
        with self._transaction() as conn:
            for record in canonical_records:
                conn.execute(
                    "INSERT INTO records(shard_id,session_version,format_version,content_digest,"
                    "record_json,aggregate_digest,lease_id,resource_units,water_json,error_json,"
                    "progress_cursor) VALUES(?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(shard_id) DO UPDATE SET "
                    "session_version=excluded.session_version,format_version=excluded.format_version,"
                    "content_digest=excluded.content_digest,record_json=excluded.record_json,"
                    "aggregate_digest=excluded.aggregate_digest,lease_id=excluded.lease_id,"
                    "resource_units=excluded.resource_units,water_json=excluded.water_json,"
                    "error_json=excluded.error_json,progress_cursor=excluded.progress_cursor",
                    (
                        record["shard_id"],
                        self.session_version,
                        LEDGER_FORMAT_VERSION,
                        record["content_digest"],
                        _canonical_text(record),
                        record["aggregate_digest"],
                        str(record["lease"]["lease_id"]),
                        int(record["resource_units"]),
                        _canonical_text(record["water"]),
                        _canonical_text(record["error_attribution"]),
                        int(record["progress_cursor"]),
                    ),
                )
            _meta_set(conn, "state_digest", _state_digest(conn))

    def records(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        rows = self.conn.execute(
            "SELECT shard_id,record_json,content_digest FROM records ORDER BY shard_id"
        ).fetchall()
        for shard_id, record_json, content_digest in rows:
            record = json.loads(record_json)
            _verify_record(record)
            if record["content_digest"] != content_digest:
                raise DivergenceError(str(shard_id), "content_digest", content_digest, record["content_digest"])
            result[str(shard_id)] = record
        return result

    def record_bytes(self) -> dict[str, bytes]:
        return {shard_id: canonical_json_bytes(record) for shard_id, record in self.records().items()}

    def recover(self, *, session_version: int | None = None) -> dict[str, Any]:
        requested_version = self.session_version if session_version is None else session_version
        if requested_version != self.session_version:
            raise SnapshotError("recovery session version mismatch")
        recovered_ns = time.time_ns()
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT u.shard_id,u.lease_id,u.node_id,u.units,u.started_ns,"
                "r.content_digest FROM uncommitted u LEFT JOIN records r ON r.shard_id=u.shard_id"
                " ORDER BY u.shard_id"
            ).fetchall()
            reclaimed_units = 0
            for shard_id, lease_id, node_id, units, started_ns, committed_digest in rows:
                if committed_digest is not None:
                    # A committed record and its uncommitted marker cannot be
                    # split by the normal transaction.  Treat it as committed.
                    conn.execute("DELETE FROM uncommitted WHERE shard_id=?", (shard_id,))
                    conn.execute(
                        "UPDATE leases SET status='released',closed_ns=?,close_reason='recovered_after_commit'"
                        " WHERE lease_id=?",
                        (recovered_ns, lease_id),
                    )
                    continue
                reason = "uncommitted_after_forced_termination"
                conn.execute(
                    "INSERT INTO unknown_shards(shard_id,lease_id,node_id,units,started_ns,reason,recovered_ns)"
                    " VALUES(?,?,?,?,?,?,?) ON CONFLICT(shard_id) DO UPDATE SET "
                    "lease_id=excluded.lease_id,node_id=excluded.node_id,units=excluded.units,"
                    "started_ns=excluded.started_ns,"
                    "reason=excluded.reason,recovered_ns=excluded.recovered_ns",
                    (shard_id, lease_id, node_id, units, started_ns, reason, recovered_ns),
                )
                conn.execute(
                    "UPDATE leases SET status='abandoned',closed_ns=?,close_reason=?"
                    " WHERE lease_id=? AND status!='released'",
                    (recovered_ns, reason, lease_id),
                )
                conn.execute("DELETE FROM uncommitted WHERE shard_id=?", (shard_id,))
                reclaimed_units += int(units)
            unknown = [
                {
                    "shard_id": str(row[0]),
                    "lease_id": str(row[1]),
                    "node_id": str(row[2]),
                    "units": int(row[3]),
                    "started_ns": int(row[4]),
                    "reason": str(row[5]),
                    "requires_rerun": True,
                }
                for row in conn.execute(
                    "SELECT shard_id,lease_id,node_id,units,started_ns,reason"
                    " FROM unknown_shards ORDER BY shard_id"
                ).fetchall()
            ]
            committed = [
                str(row[0])
                for row in conn.execute("SELECT shard_id FROM records ORDER BY shard_id").fetchall()
            ]
            report = {
                "code": "RECOVERED",
                "session_version": self.session_version,
                "committed_shards": committed,
                "unknown_shards": sorted(unknown, key=lambda item: item["shard_id"]),
                "reclaimed_units": reclaimed_units,
                "exit_report_available": False,
                "exit_report_note": (
                    "No exit report is generated for a forced termination; only complete "
                    "transactions are authoritative."
                ),
            }
            if rows:
                conn.execute(
                    "INSERT INTO recovery_events(recovered_ns,unknown_count,reclaimed_units,report_json)"
                    " VALUES(?,?,?,?)",
                    (recovered_ns, len(unknown), reclaimed_units, _canonical_text(report)),
                )
            _meta_set(conn, "state_digest", _state_digest(conn))
        return report

    def create_snapshot(self, snapshot_path: str | Path) -> dict[str, Any]:
        target = Path(snapshot_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
        if temporary.exists():
            temporary.unlink()
        destination = sqlite3.connect(temporary)
        try:
            self.conn.backup(destination)
            destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            digest = _state_digest(destination)
            with destination:
                _meta_set(destination, "snapshot_digest", digest)
                _meta_set(destination, "snapshot_version", str(self.session_version))
                _meta_set(destination, "snapshot_created_ns", str(time.time_ns()))
            destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            destination.close()
        os.replace(temporary, target)
        return {
            "snapshot_path": str(target),
            "session_version": self.session_version,
            "snapshot_digest": digest,
            "format_version": LEDGER_FORMAT_VERSION,
            "records": len(self.records()),
        }

    def checkpoint(self) -> None:
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def inspect_snapshot(snapshot_path: str | Path) -> dict[str, Any]:
    path = Path(snapshot_path)
    if not path.exists():
        raise SnapshotError("snapshot does not exist", details={"path": str(path)})
    conn = sqlite3.connect(path)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise SnapshotError("snapshot SQLite integrity check failed", details={"result": integrity})
        format_text = _meta_get(conn, "format_version")
        session_text = _meta_get(conn, "session_version")
        stored_digest = _meta_get(conn, "snapshot_digest")
        if format_text != str(LEDGER_FORMAT_VERSION):
            raise SnapshotError(
                "snapshot format version mismatch",
                details={"expected": LEDGER_FORMAT_VERSION, "actual": format_text},
            )
        if session_text is None or stored_digest is None:
            raise SnapshotError("snapshot metadata is incomplete")
        actual_digest = _state_digest(conn)
        if actual_digest != stored_digest:
            raise SnapshotError(
                "snapshot content digest mismatch",
                details={"expected": stored_digest, "actual": actual_digest},
            )
        records = conn.execute("SELECT record_json FROM records ORDER BY shard_id").fetchall()
        for (record_json,) in records:
            _verify_record(json.loads(record_json))
        return {
            "format_version": int(format_text),
            "session_version": int(session_text),
            "snapshot_digest": stored_digest,
            "record_count": len(records),
            "path": str(path),
        }
    except sqlite3.Error as exc:
        raise SnapshotError("cannot inspect snapshot", details={"error": str(exc)}) from exc
    finally:
        conn.close()


def restore_snapshot(
    snapshot_path: str | Path,
    target_path: str | Path,
    *,
    expected_version: int | None = None,
) -> dict[str, Any]:
    """Wholly replace ``target_path`` from a validated snapshot.

    No record is merged incrementally.  A version or digest difference in the
    live target is reported and the target is replaced by the snapshot.
    """

    info = inspect_snapshot(snapshot_path)
    if expected_version is not None and info["session_version"] != expected_version:
        raise SnapshotError(
            "snapshot version does not match the expected version",
            details={"expected": expected_version, "actual": info["session_version"]},
        )
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    prior_version: int | None = None
    prior_digest: str | None = None
    if target.exists():
        try:
            conn = sqlite3.connect(target)
            version_text = _meta_get(conn, "session_version")
            prior_version = int(version_text) if version_text is not None else None
            prior_digest = _state_digest(conn)
            conn.close()
        except (sqlite3.Error, ValueError):
            prior_version = None
            prior_digest = None
    temporary = target.with_name(target.name + f".restore-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    source = sqlite3.connect(snapshot_path)
    destination = sqlite3.connect(temporary)
    try:
        source.backup(destination)
        destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        source.close()
        destination.close()
    os.replace(temporary, target)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(target) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    version_mismatch = prior_version != info["session_version"]
    digest_mismatch = prior_digest != info["snapshot_digest"]
    return {
        "restored": True,
        "whole_replacement": True,
        "merge_used": False,
        "version_mismatch": version_mismatch,
        "digest_mismatch": digest_mismatch,
        "prior_session_version": prior_version,
        "prior_state_digest": prior_digest,
        **info,
    }


def compare_record_sets(
    actual: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
    *,
    fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Field-by-field consistency check used after incremental reruns."""

    compare_fields = fields or (
        "aggregate_digest",
        "sample_digest",
        "lease",
        "resource_units",
        "water",
        "error_attribution",
        "participating_rows",
        "progress_cursor",
    )
    all_shards = sorted(set(actual) | set(expected))
    for shard_id in all_shards:
        if shard_id not in actual:
            raise DivergenceError(shard_id, "record", "present", "missing")
        if shard_id not in expected:
            raise DivergenceError(shard_id, "record", "missing", "present")
        for field in compare_fields:
            left = actual[shard_id].get(field)
            right = expected[shard_id].get(field)
            if canonical_json_bytes(left) != canonical_json_bytes(right):
                raise DivergenceError(shard_id, field, right, left)
    return {
        "consistent": True,
        "shards_compared": len(all_shards),
        "fields_compared": list(compare_fields),
        "first_mismatch": None,
    }


def verify_preserved_bytes(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    shard_ids: Iterable[str],
) -> dict[str, Any]:
    checked = sorted(set(shard_ids))
    for shard_id in checked:
        if canonical_json_bytes(before[shard_id]) != canonical_json_bytes(after[shard_id]):
            raise DivergenceError(
                shard_id,
                "preserved_record_bytes",
                before[shard_id],
                after[shard_id],
            )
    return {"preserved": True, "shards_checked": checked}
