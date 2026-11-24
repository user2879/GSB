from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import LEDGER_FORMAT_VERSION
from .errors import LeaseError
from .spec import PipelineSpec
from .util import canonical_json_bytes, canonical_sha256, is_int, mono_ms, wall_ms


SCHEMA_VERSION = 1
TERMINAL_LEASE_STATUSES = (
    "EXPIRED",
    "RELEASED",
    "INVALIDATED",
    "UNKNOWN",
    "CANCELLED",
)


class Ledger:
    """SQLite event ledger and transactional projection.

    One connection is intentionally serialized by a re-entrant lock.  Every
    state mutation starts ``BEGIN IMMEDIATE``, inserts an audit event, updates
    nodes/leases/claims/sources/progress, and finally seals the ledger content
    digest in the same transaction.  Readers never observe a partial state.
    """

    def __init__(self, path: str | Path, *, create: bool = True) -> None:
        self.path = Path(path).resolve()
        if not create and not self.path.exists():
            raise LeaseError(
                "E_LEDGER",
                "ledger does not exist",
                field="ledger.path",
                expected="existing SQLite ledger",
                actual=str(self.path),
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path),
            timeout=15,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=OFF")
        self._conn.execute("PRAGMA busy_timeout=15000")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_uid TEXT UNIQUE,
                    kind TEXT NOT NULL,
                    mono_ms INTEGER NOT NULL,
                    wall_ms INTEGER NOT NULL,
                    node_id TEXT,
                    artifact_key TEXT,
                    lease_id TEXT,
                    request_seq INTEGER,
                    rationale TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    capacity_slots INTEGER NOT NULL,
                    lease_limit INTEGER NOT NULL,
                    used_slots INTEGER NOT NULL,
                    active_leases INTEGER NOT NULL,
                    updated_seq INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leases (
                    lease_id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    request_seq INTEGER NOT NULL,
                    artifact_keys_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    granted_mono_ms INTEGER,
                    expires_mono_ms INTEGER,
                    closed_mono_ms INTEGER,
                    terminal_reason TEXT,
                    updated_seq INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS claims (
                    artifact_key TEXT PRIMARY KEY,
                    lease_id TEXT NOT NULL,
                    node_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    node_id TEXT NOT NULL,
                    artifact_key TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    refs_json TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    updated_seq INTEGER NOT NULL,
                    PRIMARY KEY (node_id, artifact_key)
                );
                CREATE TABLE IF NOT EXISTS sources (
                    node_id TEXT NOT NULL,
                    artifact_key TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    refs_json TEXT NOT NULL,
                    valid INTEGER NOT NULL,
                    generation INTEGER NOT NULL,
                    updated_seq INTEGER NOT NULL,
                    PRIMARY KEY (node_id, artifact_key)
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    artifact_key TEXT PRIMARY KEY,
                    canonical_digest TEXT NOT NULL,
                    refs_json TEXT NOT NULL,
                    source_nodes_json TEXT NOT NULL,
                    reusable INTEGER NOT NULL,
                    selected_node TEXT,
                    verdicts_json TEXT NOT NULL,
                    updated_seq INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS progress (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    edit_hash TEXT NOT NULL,
                    total_events INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    completed_keys_json TEXT NOT NULL,
                    updated_seq INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, seq);
                CREATE INDEX IF NOT EXISTS idx_leases_status ON leases(status, node_id);
                """
            )
            schema_value = self._meta_get("schema_version")
            if schema_value is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES('clean_shutdown', '1')"
                )
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES('session_number', '0')"
                )
                self._conn.execute(
                    "INSERT INTO progress(id, edit_hash, total_events, phase, completed_keys_json, updated_seq) "
                    "VALUES(1, '', 0, 'new', '[]', 0)"
                )
                digest = self._content_digest(self._conn)
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('content_digest', ?)",
                    (digest,),
                )
            else:
                if int(schema_value) != SCHEMA_VERSION:
                    raise LeaseError(
                        "E_LEDGER",
                        "ledger schema version is not supported",
                        field="meta.schema_version",
                        expected=SCHEMA_VERSION,
                        actual=schema_value,
                    )
                format_value = self._meta_get("format_version")
                if format_value is None or int(format_value) != LEDGER_FORMAT_VERSION:
                    raise LeaseError(
                        "E_LEDGER",
                        "ledger format version is not supported",
                        field="meta.format_version",
                        expected=LEDGER_FORMAT_VERSION,
                        actual=format_value,
                    )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise

    def _meta_get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _meta_set(self, connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        kind: str,
        *,
        node_id: str | None = None,
        artifact_key: str | None = None,
        lease_id: str | None = None,
        request_seq: int | None = None,
        rationale: str = "",
        payload: dict[str, Any] | None = None,
        now: int | None = None,
    ) -> int:
        timestamp = mono_ms() if now is None else now
        cursor = connection.execute(
            "INSERT INTO events(kind, mono_ms, wall_ms, node_id, artifact_key, lease_id, "
            "request_seq, rationale, payload_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                kind,
                timestamp,
                wall_ms(),
                node_id,
                artifact_key,
                lease_id,
                request_seq,
                rationale,
                canonical_json_bytes(payload or {}).decode("utf-8"),
            ),
        )
        seq = int(cursor.lastrowid)
        connection.execute(
            "UPDATE events SET event_uid = ? WHERE seq = ?", (f"evt-{seq}", seq)
        )
        return seq

    def _progress_state(self, connection: sqlite3.Connection) -> dict[str, Any]:
        row = connection.execute("SELECT * FROM progress WHERE id = 1").fetchone()
        if row is None:
            return {
                "edit_hash": "",
                "total_events": 0,
                "phase": "new",
                "completed_keys": [],
            }
        return {
            "edit_hash": row["edit_hash"],
            "total_events": int(row["total_events"]),
            "phase": row["phase"],
            "completed_keys": json.loads(row["completed_keys_json"]),
        }

    def _set_progress(
        self,
        connection: sqlite3.Connection,
        seq: int,
        *,
        edit_hash: str | None = None,
        phase: str | None = None,
        completed_keys: set[str] | list[str] | None = None,
    ) -> None:
        current = self._progress_state(connection)
        keys = (
            sorted(set(completed_keys))
            if completed_keys is not None
            else sorted(current["completed_keys"])
        )
        connection.execute(
            "INSERT INTO progress(id, edit_hash, total_events, phase, completed_keys_json, updated_seq) "
            "VALUES(1, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "edit_hash=excluded.edit_hash, total_events=excluded.total_events, "
            "phase=excluded.phase, completed_keys_json=excluded.completed_keys_json, "
            "updated_seq=excluded.updated_seq",
            (
                edit_hash if edit_hash is not None else current["edit_hash"],
                seq,
                phase if phase is not None else current["phase"],
                json.dumps(keys, ensure_ascii=False, separators=(",", ":")),
                seq,
            ),
        )

    def _content_digest(self, connection: sqlite3.Connection) -> str:
        def rows(sql: str, columns: tuple[str, ...]) -> list[dict[str, Any]]:
            result = []
            for row in connection.execute(sql).fetchall():
                result.append({column: row[column] for column in columns})
            return result

        content = {
            "format_version": LEDGER_FORMAT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "events": rows(
                "SELECT seq,event_uid,kind,mono_ms,node_id,artifact_key,lease_id,"
                "request_seq,rationale,payload_json FROM events ORDER BY seq",
                (
                    "seq",
                    "event_uid",
                    "kind",
                    "mono_ms",
                    "node_id",
                    "artifact_key",
                    "lease_id",
                    "request_seq",
                    "rationale",
                    "payload_json",
                ),
            ),
            "nodes": rows(
                "SELECT node_id,host,port,capacity_slots,lease_limit,used_slots,"
                "active_leases,updated_seq FROM nodes ORDER BY node_id",
                (
                    "node_id",
                    "host",
                    "port",
                    "capacity_slots",
                    "lease_limit",
                    "used_slots",
                    "active_leases",
                    "updated_seq",
                ),
            ),
            "leases": rows(
                "SELECT lease_id,node_id,request_seq,artifact_keys_json,status,"
                "granted_mono_ms,expires_mono_ms,closed_mono_ms,terminal_reason,updated_seq "
                "FROM leases ORDER BY lease_id",
                (
                    "lease_id",
                    "node_id",
                    "request_seq",
                    "artifact_keys_json",
                    "status",
                    "granted_mono_ms",
                    "expires_mono_ms",
                    "closed_mono_ms",
                    "terminal_reason",
                    "updated_seq",
                ),
            ),
            "claims": rows(
                "SELECT artifact_key,lease_id,node_id FROM claims ORDER BY artifact_key",
                ("artifact_key", "lease_id", "node_id"),
            ),
            "summaries": rows(
                "SELECT node_id,artifact_key,digest,refs_json,generation,updated_seq "
                "FROM summaries ORDER BY artifact_key,node_id",
                (
                    "node_id",
                    "artifact_key",
                    "digest",
                    "refs_json",
                    "generation",
                    "updated_seq",
                ),
            ),
            "sources": rows(
                "SELECT node_id,artifact_key,digest,refs_json,valid,generation,updated_seq "
                "FROM sources ORDER BY artifact_key,node_id",
                (
                    "node_id",
                    "artifact_key",
                    "digest",
                    "refs_json",
                    "valid",
                    "generation",
                    "updated_seq",
                ),
            ),
            "decisions": rows(
                "SELECT artifact_key,canonical_digest,refs_json,source_nodes_json,reusable,"
                "selected_node,verdicts_json,updated_seq FROM decisions ORDER BY artifact_key",
                (
                    "artifact_key",
                    "canonical_digest",
                    "refs_json",
                    "source_nodes_json",
                    "reusable",
                    "selected_node",
                    "verdicts_json",
                    "updated_seq",
                ),
            ),
            "progress": rows(
                "SELECT edit_hash,total_events,phase,completed_keys_json,updated_seq "
                "FROM progress WHERE id = 1",
                (
                    "edit_hash",
                    "total_events",
                    "phase",
                    "completed_keys_json",
                    "updated_seq",
                ),
            ),
        }
        return canonical_sha256(content)

    def _seal(
        self,
        connection: sqlite3.Connection,
        seq: int,
        *,
        edit_hash: str | None = None,
        phase: str | None = None,
        completed_keys: set[str] | list[str] | None = None,
    ) -> str:
        self._set_progress(
            connection,
            seq,
            edit_hash=edit_hash,
            phase=phase,
            completed_keys=completed_keys,
        )
        digest = self._content_digest(connection)
        self._meta_set(connection, "content_digest", digest)
        return digest

    def start_session(self, edit_hash: str) -> dict[str, Any]:
        with self._tx() as connection:
            prior_clean = self._meta_get("clean_shutdown") != "0"
            session_number = int(self._meta_get("session_number") or "0") + 1
            self._meta_set(connection, "session_number", str(session_number))
            self._meta_set(connection, "clean_shutdown", "0")
            seq = self._insert_event(
                connection,
                "SESSION_STARTED",
                rationale="process session opened before runtime reconciliation",
                payload={
                    "session_number": session_number,
                    "prior_clean_shutdown": prior_clean,
                    "edit_hash": edit_hash,
                },
            )
            self._seal(
                connection,
                seq,
                edit_hash=edit_hash,
                phase="session_started",
            )
            return {
                "session_number": session_number,
                "prior_clean_shutdown": prior_clean,
                "requires_recovery": not prior_clean,
                "seq": seq,
            }

    def sync_node_specs(self, spec: PipelineSpec) -> list[str]:
        changed: list[str] = []
        with self._tx() as connection:
            for node in sorted(spec.nodes, key=lambda item: item.node_id):
                row = connection.execute(
                    "SELECT * FROM nodes WHERE node_id = ?", (node.node_id,)
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO nodes(node_id,host,port,capacity_slots,lease_limit,"
                        "used_slots,active_leases,updated_seq) VALUES(?,?,?,?,?,0,0,0)",
                        (
                            node.node_id,
                            node.host,
                            node.port,
                            node.capacity_slots,
                            node.lease_limit,
                        ),
                    )
                    changed.append(node.node_id)
                else:
                    values = (
                        node.host,
                        node.port,
                        node.capacity_slots,
                        node.lease_limit,
                    )
                    old = (
                        row["host"],
                        int(row["port"]),
                        int(row["capacity_slots"]),
                        int(row["lease_limit"]),
                    )
                    if values != old:
                        connection.execute(
                            "UPDATE nodes SET host=?,port=?,capacity_slots=?,lease_limit=? "
                            "WHERE node_id=?",
                            (*values, node.node_id),
                        )
                        changed.append(node.node_id)
            if changed:
                seq = self._insert_event(
                    connection,
                    "NODE_SPEC_SYNC",
                    rationale="node declarations reconciled with the ledger",
                    payload={"changed_nodes": sorted(changed)},
                )
                self._seal(connection, seq)
            return sorted(changed)

    def reconcile_spec(self, spec: PipelineSpec) -> dict[str, Any]:
        """Prune current-state projections for nodes/artifacts removed by an edit.

        Historical lease rows and events are retained.  A grant event carries the
        node capacity and lease limit that applied at grant time, so the verifier
        can replay leases belonging to a since-removed node without the current
        specification declaring that node.
        """

        current_nodes = set(spec.node_by_id)
        current_keys = set(spec.artifact_by_key)
        with self._tx() as connection:
            open_removed = connection.execute(
                "SELECT COUNT(*) AS count FROM leases WHERE status IN ('PENDING','ACTIVE') "
                f"AND node_id NOT IN ({','.join('?' for _ in current_nodes)})",
                tuple(sorted(current_nodes)),
            ).fetchone()
            if open_removed is not None and int(open_removed["count"]) > 0:
                raise LeaseError(
                    "E_LEDGER",
                    "cannot retire a node with open leases",
                    field="nodes.node_id",
                    expected="no PENDING/ACTIVE leases for removed nodes",
                    actual=int(open_removed["count"]),
                )
            old_nodes = {
                row["node_id"]
                for row in connection.execute("SELECT node_id FROM nodes").fetchall()
            }
            old_keys = {
                row["artifact_key"]
                for row in connection.execute(
                    "SELECT DISTINCT artifact_key FROM summaries"
                ).fetchall()
            }
            removed_nodes = sorted(old_nodes - current_nodes)
            removed_keys = sorted(old_keys - current_keys)
            if not removed_nodes and not removed_keys:
                return {
                    "removed_nodes": [],
                    "removed_artifacts": [],
                    "event_seq": None,
                }
            for node_id in removed_nodes:
                connection.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))
            for key in removed_keys:
                connection.execute("DELETE FROM decisions WHERE artifact_key=?", (key,))
            connection.execute(
                "DELETE FROM summaries WHERE node_id NOT IN ("
                + ",".join("?" for _ in current_nodes)
                + ") OR artifact_key NOT IN ("
                + ",".join("?" for _ in current_keys)
                + ")",
                (*sorted(current_nodes), *sorted(current_keys)),
            )
            connection.execute(
                "DELETE FROM sources WHERE node_id NOT IN ("
                + ",".join("?" for _ in current_nodes)
                + ") OR artifact_key NOT IN ("
                + ",".join("?" for _ in current_keys)
                + ")",
                (*sorted(current_nodes), *sorted(current_keys)),
            )
            connection.execute(
                "DELETE FROM claims WHERE node_id NOT IN ("
                + ",".join("?" for _ in current_nodes)
                + ") OR artifact_key NOT IN ("
                + ",".join("?" for _ in current_keys)
                + ")",
                (*sorted(current_nodes), *sorted(current_keys)),
            )
            seq = self._insert_event(
                connection,
                "SPEC_RECONCILED",
                rationale="incremental edit removed no-longer-declared runtime projections",
                payload={
                    "removed_nodes": removed_nodes,
                    "removed_artifacts": removed_keys,
                },
            )
            completed = set(self._progress_state(connection)["completed_keys"]) & current_keys
            self._seal(connection, seq, completed_keys=completed)
            return {
                "removed_nodes": removed_nodes,
                "removed_artifacts": removed_keys,
                "event_seq": seq,
            }

    def recover_unknown_leases(self, reason: str) -> dict[str, Any]:
        recovered: list[dict[str, Any]] = []
        affected: set[str] = set()
        with self._tx() as connection:
            rows = connection.execute(
                "SELECT * FROM leases WHERE status IN ('PENDING','ACTIVE') "
                "ORDER BY lease_id"
            ).fetchall()
            last_seq = 0
            for row in rows:
                lease_id = row["lease_id"]
                node_id = row["node_id"]
                keys = json.loads(row["artifact_keys_json"])
                was_active = row["status"] == "ACTIVE"
                closed_at = mono_ms()
                if was_active:
                    connection.execute(
                        "UPDATE nodes SET used_slots=MAX(0, used_slots-?), "
                        "active_leases=MAX(0, active_leases-?), updated_seq=? "
                        "WHERE node_id=?",
                        (len(keys), 1, last_seq, node_id),
                    )
                    connection.execute("DELETE FROM claims WHERE lease_id=?", (lease_id,))
                connection.execute(
                    "UPDATE leases SET status='UNKNOWN', closed_mono_ms=?, "
                    "terminal_reason=?, updated_seq=? WHERE lease_id=?",
                    (closed_at, reason, last_seq, lease_id),
                )
                for key in keys:
                    connection.execute(
                        "UPDATE sources SET valid=0, updated_seq=? WHERE artifact_key=?",
                        (last_seq, key),
                    )
                    affected.add(key)
                last_seq = self._insert_event(
                    connection,
                    "LEASE_UNKNOWN",
                    node_id=node_id,
                    artifact_key=keys[0] if keys else None,
                    lease_id=lease_id,
                    request_seq=int(row["request_seq"]),
                    rationale=reason,
                    payload={
                        "prior_status": row["status"],
                        "artifact_keys": keys,
                        "capacity_owner_released": was_active,
                        "requires_reapply": True,
                        "invalidated_keys": keys,
                    },
                    now=closed_at,
                )
                recovered.append(
                    {
                        "lease_id": lease_id,
                        "node_id": node_id,
                        "artifact_keys": keys,
                        "prior_status": row["status"],
                        "event_seq": last_seq,
                    }
                )
            if recovered:
                current = self._progress_state(connection)["completed_keys"]
                connection.execute(
                    "UPDATE nodes SET updated_seq=? WHERE updated_seq=0", (last_seq,)
                )
                connection.execute(
                    "UPDATE leases SET updated_seq=? WHERE updated_seq=0", (last_seq,)
                )
                connection.execute(
                    "UPDATE sources SET updated_seq=? WHERE updated_seq=0", (last_seq,)
                )
                self._seal(
                    connection,
                    last_seq,
                    phase="recovered_unknown",
                    completed_keys=set(current) - affected,
                )
            return {
                "reason": reason,
                "recovered": recovered,
                "affected_keys": sorted(affected),
            }

    def orphan_node_leases(
        self, node_id: str, reason: str, *, invalidate_keys: set[str] | None = None
    ) -> list[dict[str, Any]]:
        invalidate = set(invalidate_keys or ())
        result: list[dict[str, Any]] = []
        with self._tx() as connection:
            rows = connection.execute(
                "SELECT * FROM leases WHERE node_id=? AND status='ACTIVE' "
                "ORDER BY lease_id",
                (node_id,),
            ).fetchall()
            last_seq = 0
            for row in rows:
                keys = json.loads(row["artifact_keys_json"])
                closed_at = mono_ms()
                connection.execute(
                    "UPDATE nodes SET used_slots=MAX(0, used_slots-?), "
                    "active_leases=MAX(0, active_leases-?), updated_seq=? WHERE node_id=?",
                    (len(keys), 1, last_seq, node_id),
                )
                connection.execute("DELETE FROM claims WHERE lease_id=?", (row["lease_id"],))
                connection.execute(
                    "UPDATE leases SET status='UNKNOWN', closed_mono_ms=?, "
                    "terminal_reason=?, updated_seq=? WHERE lease_id=?",
                    (closed_at, reason, last_seq, row["lease_id"]),
                )
                truly_invalidated = sorted(set(keys) & invalidate)
                for key in truly_invalidated:
                    connection.execute(
                        "UPDATE sources SET valid=0, updated_seq=? WHERE artifact_key=?",
                        (last_seq, key),
                    )
                last_seq = self._insert_event(
                    connection,
                    "LEASE_UNKNOWN",
                    node_id=node_id,
                    artifact_key=keys[0] if keys else None,
                    lease_id=row["lease_id"],
                    request_seq=int(row["request_seq"]),
                    rationale=reason,
                    payload={
                        "artifact_keys": keys,
                        "capacity_owner_released": True,
                        "requires_reapply": True,
                        "invalidated_keys": truly_invalidated,
                    },
                    now=closed_at,
                )
                result.append(
                    {
                        "lease_id": row["lease_id"],
                        "artifact_keys": keys,
                        "invalidated_keys": truly_invalidated,
                        "event_seq": last_seq,
                    }
                )
            if result:
                current = set(self._progress_state(connection)["completed_keys"])
                connection.execute(
                    "UPDATE nodes SET updated_seq=? WHERE updated_seq=0", (last_seq,)
                )
                connection.execute(
                    "UPDATE leases SET updated_seq=? WHERE updated_seq=0", (last_seq,)
                )
                connection.execute(
                    "UPDATE sources SET updated_seq=? WHERE updated_seq=0", (last_seq,)
                )
                self._seal(
                    connection,
                    last_seq,
                    edit_hash=self._progress_state(connection)["edit_hash"],
                    completed_keys=current - invalidate,
                )
            return result

    def sync_digests(
        self, snapshot: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        affected: set[str] = set()
        canonical_changed: set[str] = set()
        changed_nodes: dict[str, list[str]] = {}
        summary_changes: list[dict[str, Any]] = []
        with self._tx() as connection:
            all_keys = sorted(
                {
                    key
                    for node_data in snapshot.values()
                    for key in node_data["digests"]
                }
            )
            canonical_before: dict[str, tuple[str, str] | None] = {}
            for key in all_keys:
                old = connection.execute(
                    "SELECT digest,refs_json FROM summaries WHERE artifact_key=? "
                    "ORDER BY node_id LIMIT 1",
                    (key,),
                ).fetchone()
                canonical_before[key] = (
                    None if old is None else (old["digest"], old["refs_json"])
                )
            for node_id in sorted(snapshot):
                node_data = snapshot[node_id]
                digests = node_data["digests"]
                refs = node_data["refs"]
                for key in sorted(digests):
                    digest = str(digests[key])
                    refs_list = sorted(refs.get(key, []))
                    refs_json = json.dumps(
                        refs_list, ensure_ascii=False, separators=(",", ":")
                    )
                    row = connection.execute(
                        "SELECT * FROM summaries WHERE node_id=? AND artifact_key=?",
                        (node_id, key),
                    ).fetchone()
                    if row is None:
                        generation = 1
                        connection.execute(
                            "INSERT INTO summaries(node_id,artifact_key,digest,refs_json,"
                            "generation,updated_seq) VALUES(?,?,?,?,?,0)",
                            (node_id, key, digest, refs_json, generation),
                        )
                        affected.add(key)
                        changed_nodes.setdefault(node_id, []).append(key)
                        summary_changes.append(
                            {
                                "node_id": node_id,
                                "artifact_key": key,
                                "old_digest": None,
                                "new_digest": digest,
                            }
                        )
                    elif row["digest"] != digest or row["refs_json"] != refs_json:
                        generation = int(row["generation"]) + 1
                        connection.execute(
                            "UPDATE summaries SET digest=?,refs_json=?,generation=?,"
                            "updated_seq=0 WHERE node_id=? AND artifact_key=?",
                            (digest, refs_json, generation, node_id, key),
                        )
                        connection.execute(
                            "UPDATE sources SET valid=0, updated_seq=0 "
                            "WHERE node_id=? AND artifact_key=?",
                            (node_id, key),
                        )
                        affected.add(key)
                        changed_nodes.setdefault(node_id, []).append(key)
                        summary_changes.append(
                            {
                                "node_id": node_id,
                                "artifact_key": key,
                                "old_digest": row["digest"],
                                "new_digest": digest,
                            }
                        )

            # Canonical digest is the copy generated by the lexicographically
            # first node.  This makes the reusable source set independent of
            # dictionary/hash traversal order.
            for key in all_keys:
                canonical_row = connection.execute(
                    "SELECT * FROM summaries WHERE artifact_key=? ORDER BY node_id LIMIT 1",
                    (key,),
                ).fetchone()
                if canonical_row is None:
                    continue
                canonical_digest = canonical_row["digest"]
                canonical_refs = canonical_row["refs_json"]
                previous = canonical_before.get(key)
                if previous is not None and previous != (
                    canonical_digest,
                    canonical_refs,
                ):
                    canonical_changed.add(key)
                valid_sources = connection.execute(
                    "SELECT * FROM sources WHERE artifact_key=? AND valid=1", (key,)
                ).fetchall()
                if not valid_sources:
                    continue
                if any(
                    source["digest"] != canonical_digest
                    or source["refs_json"] != canonical_refs
                    for source in valid_sources
                ):
                    connection.execute(
                        "UPDATE sources SET valid=0, updated_seq=0 WHERE artifact_key=?",
                        (key,),
                    )
                    affected.add(key)
                    canonical_changed.add(key)
                    continue
                matching_nodes = [
                    row["node_id"]
                    for row in connection.execute(
                        "SELECT node_id FROM summaries WHERE artifact_key=? AND "
                        "digest=? AND refs_json=? ORDER BY node_id",
                        (key, canonical_digest, canonical_refs),
                    ).fetchall()
                ]
                placeholders = ",".join("?" for _ in matching_nodes)
                connection.execute(
                    f"UPDATE sources SET valid=0, updated_seq=0 WHERE artifact_key=? "
                    f"AND node_id NOT IN ({placeholders})",
                    (key, *matching_nodes),
                )
                for node_id in matching_nodes:
                    connection.execute(
                        "UPDATE sources SET valid=1, updated_seq=0 "
                        "WHERE node_id=? AND artifact_key=?",
                        (node_id, key),
                    )

            invalidated_leases: list[dict[str, Any]] = []
            if affected:
                active_rows = connection.execute(
                    "SELECT * FROM leases WHERE status='ACTIVE' ORDER BY lease_id"
                ).fetchall()
                for row in active_rows:
                    keys = json.loads(row["artifact_keys_json"])
                    lease_affected = set(keys) & affected
                    node_changed = row["node_id"] in changed_nodes and bool(
                        set(changed_nodes[row["node_id"]]) & set(keys)
                    )
                    if not lease_affected or not (node_changed or canonical_changed.intersection(keys)):
                        continue
                    closed_at = mono_ms()
                    connection.execute(
                        "UPDATE nodes SET used_slots=MAX(0, used_slots-?), "
                        "active_leases=MAX(0, active_leases-?), updated_seq=0 WHERE node_id=?",
                        (len(keys), 1, row["node_id"]),
                    )
                    connection.execute("DELETE FROM claims WHERE lease_id=?", (row["lease_id"],))
                    connection.execute(
                        "UPDATE leases SET status='INVALIDATED', closed_mono_ms=?, "
                        "terminal_reason=?, updated_seq=0 WHERE lease_id=?",
                        (
                            closed_at,
                            "digest or reference convergence invalidated lease",
                            row["lease_id"],
                        ),
                    )
                    invalidated_leases.append(
                        {"lease_id": row["lease_id"], "artifact_keys": keys}
                    )
                    self._insert_event(
                        connection,
                        "LEASE_INVALIDATED",
                        node_id=row["node_id"],
                        artifact_key=keys[0] if keys else None,
                        lease_id=row["lease_id"],
                        request_seq=int(row["request_seq"]),
                        rationale="digest/reference change invalidated cache reuse",
                        payload={
                            "artifact_keys": keys,
                            "affected_keys": sorted(lease_affected),
                            "capacity_owner_released": True,
                        },
                        now=closed_at,
                    )

            valid_keys = {
                row["artifact_key"]
                for row in connection.execute(
                    "SELECT DISTINCT artifact_key FROM sources WHERE valid=1"
                ).fetchall()
            }
            reuse_invalidated = sorted(set(affected) - valid_keys)
            seq = self._insert_event(
                connection,
                "SYNC_DIGESTS",
                rationale="node copies regenerated and compared with the ledger",
                payload={
                    "affected_keys": sorted(affected),
                    "canonical_changed_keys": sorted(canonical_changed),
                    "reuse_invalidated_keys": reuse_invalidated,
                    "summary_changes": summary_changes,
                    "invalidated_leases": invalidated_leases,
                },
            )
            for node_id, keys in changed_nodes.items():
                connection.execute(
                    "UPDATE nodes SET updated_seq=? WHERE node_id=?", (seq, node_id)
                )
            connection.execute(
                "UPDATE summaries SET updated_seq=?", (seq,)
            )
            connection.execute("UPDATE sources SET updated_seq=? WHERE updated_seq=0", (seq,))
            connection.execute("UPDATE leases SET updated_seq=? WHERE updated_seq=0", (seq,))
            current_completed = set(self._progress_state(connection)["completed_keys"])
            # Recompute completion from valid sources: an artifact is completed
            # only when at least one canonical copy is valid.
            self._seal(
                connection,
                seq,
                completed_keys=current_completed & valid_keys,
            )
            return {
                "affected_keys": sorted(affected),
                "canonical_changed_keys": sorted(canonical_changed),
                "summary_changes": summary_changes,
                "invalidated_leases": invalidated_leases,
                "event_seq": seq,
            }

    def begin_pending(
        self,
        lease_id: str,
        node_id: str,
        artifact_key: str,
        request_seq: int,
        expected_digest: str,
        refs: list[str],
    ) -> int:
        with self._tx() as connection:
            now = mono_ms()
            connection.execute(
                "INSERT INTO leases(lease_id,node_id,request_seq,artifact_keys_json,status,"
                "granted_mono_ms,expires_mono_ms,closed_mono_ms,terminal_reason,updated_seq) "
                "VALUES(?,?,?,?,'PENDING',NULL,NULL,NULL,NULL,0)",
                (
                    lease_id,
                    node_id,
                    request_seq,
                    json.dumps([artifact_key], ensure_ascii=False, separators=(",", ":")),
                ),
            )
            seq = self._insert_event(
                connection,
                "LEASE_REQUESTED",
                node_id=node_id,
                artifact_key=artifact_key,
                lease_id=lease_id,
                request_seq=request_seq,
                rationale="real TCP lease request prepared before socket call",
                payload={"expected_digest": expected_digest, "references": refs},
                now=now,
            )
            self._seal(connection, seq, phase="leasing")
            return seq

    def cancel_pending(self, lease_id: str, reason: str) -> bool:
        with self._tx() as connection:
            row = connection.execute(
                "SELECT * FROM leases WHERE lease_id=? AND status='PENDING'", (lease_id,)
            ).fetchone()
            if row is None:
                return False
            closed_at = mono_ms()
            connection.execute(
                "UPDATE leases SET status='CANCELLED',closed_mono_ms=?,terminal_reason=?,"
                "updated_seq=0 WHERE lease_id=?",
                (closed_at, reason, lease_id),
            )
            seq = self._insert_event(
                connection,
                "LEASE_CANCELLED",
                node_id=row["node_id"],
                artifact_key=json.loads(row["artifact_keys_json"])[0],
                lease_id=lease_id,
                request_seq=int(row["request_seq"]),
                rationale=reason,
                payload={"capacity_owner_released": False},
                now=closed_at,
            )
            connection.execute("UPDATE leases SET updated_seq=? WHERE lease_id=?", (seq, lease_id))
            self._seal(connection, seq)
            return True

    def commit_grant(
        self,
        lease_id: str,
        node_id: str,
        artifact_keys: list[str],
        digests: dict[str, str],
        refs: dict[str, list[str]],
        expires_mono_ms: int,
        *,
        source_nodes: list[str],
        node_capacity_ack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        keys = sorted(set(artifact_keys))
        if len(keys) != len(artifact_keys):
            raise LeaseError(
                "E_LEDGER",
                "grant contains duplicate artifact keys",
                field="artifact_keys",
                expected="unique keys",
            )
        with self._tx() as connection:
            lease = connection.execute(
                "SELECT * FROM leases WHERE lease_id=? AND status='PENDING'", (lease_id,)
            ).fetchone()
            if lease is None:
                raise LeaseError(
                    "E_LEDGER",
                    "pending lease is not available for grant",
                    field="lease_id",
                    expected="committed PENDING lease",
                    actual=lease_id,
                    node_id=node_id,
                )
            node = connection.execute(
                "SELECT * FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if node is None:
                raise LeaseError(
                    "E_LEDGER",
                    "node is absent from ledger",
                    field="node_id",
                    expected="declared node",
                    actual=node_id,
                    node_id=node_id,
                )
            used_before = int(node["used_slots"])
            active_before = int(node["active_leases"])
            if active_before + 1 > int(node["lease_limit"]):
                raise LeaseError(
                    "E_LEASE_LIMIT",
                    "ledger node lease limit would be exceeded",
                    field="lease_limit",
                    expected=f"active leases < {node['lease_limit']}",
                    actual=active_before,
                    node_id=node_id,
                )
            if used_before + len(keys) > int(node["capacity_slots"]):
                raise LeaseError(
                    "E_CAPACITY",
                    "ledger node capacity would be exceeded",
                    field="capacity_slots",
                    expected=f"free slots >= {len(keys)}",
                    actual=int(node["capacity_slots"]) - used_before,
                    node_id=node_id,
                )
            for key in keys:
                claim = connection.execute(
                    "SELECT * FROM claims WHERE artifact_key=?", (key,)
                ).fetchone()
                if claim is not None:
                    raise LeaseError(
                        "E_LEASE_CLAIMED",
                        "artifact key already has a unique active capacity owner",
                        field="artifact_key",
                        expected="unclaimed artifact key",
                        actual={"lease_id": claim["lease_id"], "node_id": claim["node_id"]},
                        artifact_key=key,
                        node_id=node_id,
                    )
            now = mono_ms()
            connection.execute(
                "UPDATE nodes SET used_slots=used_slots+?,active_leases=active_leases+?,"
                "updated_seq=0 WHERE node_id=?",
                (len(keys), 1, node_id),
            )
            connection.execute(
                "UPDATE leases SET status='ACTIVE',granted_mono_ms=?,expires_mono_ms=?,"
                "updated_seq=0 WHERE lease_id=?",
                (now, expires_mono_ms, lease_id),
            )
            for key in keys:
                connection.execute(
                    "INSERT INTO claims(artifact_key,lease_id,node_id) VALUES(?,?,?)",
                    (key, lease_id, node_id),
                )
            for node_source in sorted(set(source_nodes)):
                for key in keys:
                    summary = connection.execute(
                        "SELECT * FROM summaries WHERE node_id=? AND artifact_key=?",
                        (node_source, key),
                    ).fetchone()
                    if summary is None:
                        continue
                    connection.execute(
                        "INSERT INTO sources(node_id,artifact_key,digest,refs_json,valid,"
                        "generation,updated_seq) VALUES(?,?,?,?,1,?,0) "
                        "ON CONFLICT(node_id,artifact_key) DO UPDATE SET "
                        "digest=excluded.digest,refs_json=excluded.refs_json,valid=1,"
                        "generation=excluded.generation,updated_seq=0",
                        (
                            node_source,
                            key,
                            summary["digest"],
                            summary["refs_json"],
                            int(summary["generation"]),
                        ),
                    )
            current_completed = set(self._progress_state(connection)["completed_keys"])
            current_completed.update(keys)
            seq = self._insert_event(
                connection,
                "LEASE_GRANTED",
                node_id=node_id,
                artifact_key=keys[0],
                lease_id=lease_id,
                request_seq=int(lease["request_seq"]),
                rationale="unique service node arbitration granted the sole claim",
                payload={
                    "artifact_keys": keys,
                    "digests": {key: digests[key] for key in keys},
                    "references": {key: sorted(refs[key]) for key in keys},
                    "source_nodes": sorted(set(source_nodes)),
                    "expires_mono_ms": expires_mono_ms,
                    "capacity_before": {
                        "used_slots": used_before,
                        "active_leases": active_before,
                    },
                    "capacity_after": {
                        "used_slots": used_before + len(keys),
                        "active_leases": active_before + 1,
                    },
                    "node_capacity_ack": node_capacity_ack or {},
                    "node_capacity_slots": int(node["capacity_slots"]),
                    "node_lease_limit": int(node["lease_limit"]),
                },
                now=now,
            )
            connection.execute(
                "UPDATE nodes SET updated_seq=? WHERE node_id=?", (seq, node_id)
            )
            connection.execute(
                "UPDATE leases SET updated_seq=? WHERE lease_id=?", (seq, lease_id)
            )
            connection.execute(
                "UPDATE sources SET updated_seq=? WHERE updated_seq=0", (seq,)
            )
            digest = self._seal(
                connection, seq, phase="lease_active", completed_keys=current_completed
            )
            return {
                "lease_id": lease_id,
                "node_id": node_id,
                "event_seq": seq,
                "ledger_digest": digest,
                "capacity": {
                    "used_slots": used_before + len(keys),
                    "active_leases": active_before + 1,
                    "free_slots": int(node["capacity_slots"]) - used_before - len(keys),
                },
            }

    def renew_lease(self, lease_id: str, expires_mono_ms: int) -> dict[str, Any]:
        with self._tx() as connection:
            row = connection.execute(
                "SELECT * FROM leases WHERE lease_id=? AND status='ACTIVE'", (lease_id,)
            ).fetchone()
            if row is None:
                raise LeaseError(
                    "E_LEASE_UNKNOWN",
                    "active lease cannot be renewed",
                    field="lease_id",
                    expected="ACTIVE lease",
                    actual=lease_id,
                )
            old_expiry = int(row["expires_mono_ms"])
            if expires_mono_ms <= mono_ms():
                raise LeaseError(
                    "E_LEASE_RENEW",
                    "renewed expiry is not in the local monotonic future",
                    field="expires_mono_ms",
                    expected="future monotonic millisecond integer",
                    actual=expires_mono_ms,
                )
            connection.execute(
                "UPDATE leases SET expires_mono_ms=?,updated_seq=0 WHERE lease_id=?",
                (expires_mono_ms, lease_id),
            )
            seq = self._insert_event(
                connection,
                "LEASE_RENEWED",
                node_id=row["node_id"],
                artifact_key=json.loads(row["artifact_keys_json"])[0],
                lease_id=lease_id,
                request_seq=int(row["request_seq"]),
                rationale="lease lifetime extended without changing capacity ownership",
                payload={
                    "previous_expires_mono_ms": old_expiry,
                    "expires_mono_ms": expires_mono_ms,
                    "capacity_owner_unchanged": True,
                },
            )
            connection.execute("UPDATE leases SET updated_seq=? WHERE lease_id=?", (seq, lease_id))
            self._seal(connection, seq)
            return {"lease_id": lease_id, "expires_mono_ms": expires_mono_ms, "event_seq": seq}

    def _close_active_lease(
        self,
        connection: sqlite3.Connection,
        lease_id: str,
        kind: str,
        reason: str,
        *,
        now: int,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT * FROM leases WHERE lease_id=? AND status='ACTIVE'", (lease_id,)
        ).fetchone()
        if row is None:
            return None
        keys = json.loads(row["artifact_keys_json"])
        node_id = row["node_id"]
        node = connection.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
        used_before = int(node["used_slots"])
        active_before = int(node["active_leases"])
        status = {
            "LEASE_EXPIRED": "EXPIRED",
            "QUOTA_RELEASED": "RELEASED",
        }.get(kind, "INVALIDATED")
        connection.execute(
            "UPDATE nodes SET used_slots=MAX(0,used_slots-?),"
            "active_leases=MAX(0,active_leases-?),updated_seq=0 WHERE node_id=?",
            (len(keys), 1, node_id),
        )
        connection.execute("DELETE FROM claims WHERE lease_id=?", (lease_id,))
        connection.execute(
            "UPDATE leases SET status=?,closed_mono_ms=?,terminal_reason=?,updated_seq=0 "
            "WHERE lease_id=?",
            (status, now, reason, lease_id),
        )
        seq = self._insert_event(
            connection,
            kind,
            node_id=node_id,
            artifact_key=keys[0] if keys else None,
            lease_id=lease_id,
            request_seq=int(row["request_seq"]),
            rationale=reason,
            payload={
                "artifact_keys": keys,
                "capacity_before": {
                    "used_slots": used_before,
                    "active_leases": active_before,
                },
                "capacity_after": {
                    "used_slots": max(0, used_before - len(keys)),
                    "active_leases": max(0, active_before - 1),
                },
                "capacity_owner_released": True,
                "exclusive_owner": True,
            },
            now=now,
        )
        connection.execute("UPDATE nodes SET updated_seq=? WHERE node_id=?", (seq, node_id))
        connection.execute("UPDATE leases SET updated_seq=? WHERE lease_id=?", (seq, lease_id))
        self._seal(connection, seq)
        return {
            "lease_id": lease_id,
            "node_id": node_id,
            "event_seq": seq,
            "artifact_keys": keys,
            "capacity_after": {
                "used_slots": max(0, used_before - len(keys)),
                "active_leases": max(0, active_before - 1),
            },
        }

    def expire_lease(self, lease_id: str, *, now: int | None = None) -> dict[str, Any] | None:
        current = mono_ms() if now is None else now
        with self._tx() as connection:
            row = connection.execute(
                "SELECT * FROM leases WHERE lease_id=? AND status='ACTIVE'", (lease_id,)
            ).fetchone()
            if row is None:
                return None
            if current < int(row["expires_mono_ms"]):
                return None
            return self._close_active_lease(
                connection,
                lease_id,
                "LEASE_EXPIRED",
                "local monotonic deadline reached; capacity and lease count released together",
                now=current,
            )

    def release_lease(self, lease_id: str, reason: str = "quota reclaim requested") -> dict[str, Any] | None:
        with self._tx() as connection:
            return self._close_active_lease(
                connection,
                lease_id,
                "QUOTA_RELEASED",
                reason,
                now=mono_ms(),
            )

    def record_decisions(
        self,
        spec: PipelineSpec,
        decisions: list[dict[str, Any]],
        *,
        completed_keys: set[str],
        phase: str,
    ) -> dict[str, Any]:
        with self._tx() as connection:
            for decision in decisions:
                connection.execute(
                    "INSERT INTO decisions(artifact_key,canonical_digest,refs_json,"
                    "source_nodes_json,reusable,selected_node,verdicts_json,updated_seq) "
                    "VALUES(?,?,?,?,?,?,?,0) ON CONFLICT(artifact_key) DO UPDATE SET "
                    "canonical_digest=excluded.canonical_digest,refs_json=excluded.refs_json,"
                    "source_nodes_json=excluded.source_nodes_json,reusable=excluded.reusable,"
                    "selected_node=excluded.selected_node,verdicts_json=excluded.verdicts_json,"
                    "updated_seq=0",
                    (
                        decision["artifact_key"],
                        decision["canonical_digest"],
                        json.dumps(
                            sorted(decision["references"]),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        json.dumps(
                            sorted(decision["source_nodes"]),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        1 if decision["reusable"] else 0,
                        decision["selected_node"],
                        json.dumps(
                            decision["verdicts"],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )
            seq = self._insert_event(
                connection,
                "DECISIONS_RECORDED",
                rationale="reuse, arbitration and ownership conclusions committed together",
                payload={
                    "artifact_count": len(decisions),
                    "completed_keys": sorted(completed_keys),
                    "phase": phase,
                },
            )
            connection.execute("UPDATE decisions SET updated_seq=?", (seq,))
            digest = self._seal(
                connection,
                seq,
                edit_hash=spec.content_sha256,
                phase=phase,
                completed_keys=completed_keys,
            )
            return {"event_seq": seq, "ledger_digest": digest}

    def close_session(self, *, status: str = "complete", report_path: str | None = None) -> dict[str, Any]:
        with self._tx() as connection:
            seq = self._insert_event(
                connection,
                "SESSION_CLOSED",
                rationale="clean shutdown marker; any prior uncommitted runtime state is excluded",
                payload={"status": status, "report_path": report_path},
            )
            self._meta_set(connection, "clean_shutdown", "1")
            digest = self._seal(connection, seq, phase="closed")
            return {"event_seq": seq, "ledger_digest": digest}

    def next_expiry(self) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(expires_mono_ms) AS expiry FROM leases WHERE status='ACTIVE'"
            ).fetchone()
            return None if row is None or row["expiry"] is None else int(row["expiry"])

    def active_leases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM leases WHERE status='ACTIVE' ORDER BY lease_id"
            ).fetchall()
            return [self._lease_row(row) for row in rows]

    def has_open_leases(self) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 AS present FROM leases WHERE status IN ('PENDING','ACTIVE') LIMIT 1"
            ).fetchone()
            return row is not None

    def lease(self, lease_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM leases WHERE lease_id=?", (lease_id,)
            ).fetchone()
            return None if row is None else self._lease_row(row)

    def list_leases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM leases ORDER BY lease_id").fetchall()
            return [self._lease_row(row) for row in rows]

    def _lease_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "lease_id": row["lease_id"],
            "node_id": row["node_id"],
            "request_seq": int(row["request_seq"]),
            "artifact_keys": json.loads(row["artifact_keys_json"]),
            "status": row["status"],
            "granted_mono_ms": row["granted_mono_ms"],
            "expires_mono_ms": row["expires_mono_ms"],
            "closed_mono_ms": row["closed_mono_ms"],
            "terminal_reason": row["terminal_reason"],
            "updated_seq": int(row["updated_seq"]),
        }

    def nodes(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "node_id": row["node_id"],
                    "host": row["host"],
                    "port": int(row["port"]),
                    "capacity_slots": int(row["capacity_slots"]),
                    "lease_limit": int(row["lease_limit"]),
                    "used_slots": int(row["used_slots"]),
                    "active_leases": int(row["active_leases"]),
                    "free_slots": int(row["capacity_slots"]) - int(row["used_slots"]),
                }
                for row in self._conn.execute(
                    "SELECT * FROM nodes ORDER BY node_id"
                ).fetchall()
            ]

    def summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "node_id": row["node_id"],
                    "artifact_key": row["artifact_key"],
                    "digest": row["digest"],
                    "references": json.loads(row["refs_json"]),
                    "generation": int(row["generation"]),
                }
                for row in self._conn.execute(
                    "SELECT * FROM summaries ORDER BY artifact_key,node_id"
                ).fetchall()
            ]

    def sources(self, *, valid_only: bool = False) -> list[dict[str, Any]]:
        clause = " WHERE valid=1" if valid_only else ""
        with self._lock:
            return [
                {
                    "node_id": row["node_id"],
                    "artifact_key": row["artifact_key"],
                    "digest": row["digest"],
                    "references": json.loads(row["refs_json"]),
                    "valid": bool(row["valid"]),
                    "generation": int(row["generation"]),
                }
                for row in self._conn.execute(
                    f"SELECT * FROM sources{clause} ORDER BY artifact_key,node_id"
                ).fetchall()
            ]

    def decisions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "artifact_key": row["artifact_key"],
                    "canonical_digest": row["canonical_digest"],
                    "references": json.loads(row["refs_json"]),
                    "source_nodes": json.loads(row["source_nodes_json"]),
                    "reusable": bool(row["reusable"]),
                    "selected_node": row["selected_node"],
                    "verdicts": json.loads(row["verdicts_json"]),
                }
                for row in self._conn.execute(
                    "SELECT * FROM decisions ORDER BY artifact_key"
                ).fetchall()
            ]

    def progress(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM progress WHERE id=1").fetchone()
            return {
                "edit_hash": row["edit_hash"],
                "total_events": int(row["total_events"]),
                "phase": row["phase"],
                "completed_keys": json.loads(row["completed_keys_json"]),
                "updated_seq": int(row["updated_seq"]),
            }

    def events(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM events ORDER BY seq"
            if limit is not None:
                sql = (
                    "SELECT * FROM (SELECT * FROM events ORDER BY seq DESC LIMIT ?) "
                    "ORDER BY seq"
                )
                rows = self._conn.execute(sql, (limit,)).fetchall()
            else:
                rows = self._conn.execute(sql).fetchall()
            return [
                {
                    "seq": int(row["seq"]),
                    "event_uid": row["event_uid"],
                    "kind": row["kind"],
                    "mono_ms": int(row["mono_ms"]),
                    "wall_ms": int(row["wall_ms"]),
                    "node_id": row["node_id"],
                    "artifact_key": row["artifact_key"],
                    "lease_id": row["lease_id"],
                    "request_seq": row["request_seq"],
                    "rationale": row["rationale"],
                    "payload": json.loads(row["payload_json"]),
                }
                for row in rows
            ]

    def meta(self) -> dict[str, str]:
        with self._lock:
            return {
                row["key"]: row["value"]
                for row in self._conn.execute("SELECT * FROM meta ORDER BY key").fetchall()
            }

    def verify_content_digest(self) -> str:
        with self._lock:
            stored = self._meta_get("content_digest")
            actual = self._content_digest(self._conn)
            if stored != actual:
                raise LeaseError(
                    "E_VERIFY",
                    "ledger content digest does not match its committed records",
                    field="meta.content_digest",
                    expected=stored,
                    actual=actual,
                )
            return actual

    def checkpoint(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
