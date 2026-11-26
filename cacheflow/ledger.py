from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from . import FORMAT_VERSION
from .canonical import canonical_bytes, sha256_hex


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY,
  event_type TEXT NOT NULL,
  lease_id TEXT,
  node_id TEXT,
  artifact_key TEXT,
  quota_version INTEGER,
  event_json TEXT NOT NULL,
  result_json TEXT,
  status TEXT NOT NULL,
  error_code TEXT
);

CREATE TABLE IF NOT EXISTS lease_current (
  lease_id TEXT PRIMARY KEY,
  node_id TEXT NOT NULL,
  artifact_key TEXT NOT NULL,
  quota_version INTEGER NOT NULL,
  state TEXT NOT NULL,
  generation INTEGER NOT NULL,
  last_event_seq INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS transitions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_seq INTEGER NOT NULL,
  ordinal INTEGER NOT NULL,
  lease_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  trigger TEXT NOT NULL,
  reason TEXT NOT NULL,
  state_before TEXT NOT NULL,
  state_after TEXT NOT NULL,
  generation_before INTEGER NOT NULL,
  generation_after INTEGER NOT NULL,
  carried_label INTEGER,
  quota_version INTEGER NOT NULL,
  digest TEXT NOT NULL,
  UNIQUE(event_seq, ordinal, lease_id)
);

CREATE TABLE IF NOT EXISTS holds (
  hold_id TEXT PRIMARY KEY,
  node_id TEXT NOT NULL,
  slot_id INTEGER NOT NULL,
  lease_id TEXT NOT NULL,
  artifact_key TEXT NOT NULL,
  shard_index INTEGER NOT NULL,
  status TEXT NOT NULL,
  opened_seq INTEGER NOT NULL,
  closed_seq INTEGER,
  reclaim_reason TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_active_slot
ON holds(node_id, slot_id)
WHERE status IN ('provisional', 'active');

CREATE UNIQUE INDEX IF NOT EXISTS idx_active_artifact_shard
ON holds(artifact_key, shard_index)
WHERE status IN ('provisional', 'active');

CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY,
  event_seq INTEGER NOT NULL,
  ordinal INTEGER NOT NULL,
  artifact_key TEXT NOT NULL,
  owner_node_id TEXT,
  basis TEXT NOT NULL,
  generation INTEGER,
  quota_version INTEGER NOT NULL,
  decision_json TEXT NOT NULL,
  digest TEXT NOT NULL,
  UNIQUE(event_seq, ordinal, artifact_key)
);

CREATE TABLE IF NOT EXISTS reclaims (
  reclaim_id TEXT PRIMARY KEY,
  event_seq INTEGER NOT NULL,
  ordinal INTEGER NOT NULL,
  lease_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  reason_kind TEXT NOT NULL,
  generation_before INTEGER NOT NULL,
  generation_after INTEGER NOT NULL,
  hold_ids_json TEXT NOT NULL,
  digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inflight_requests (
  request_id TEXT PRIMARY KEY,
  event_seq INTEGER,
  node_id TEXT,
  lease_id TEXT,
  op TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  wait_started_ms INTEGER NOT NULL,
  wait_ended_ms INTEGER NOT NULL,
  status TEXT NOT NULL,
  result_json TEXT
);

CREATE TABLE IF NOT EXISTS wait_records (
  seq INTEGER PRIMARY KEY,
  node_id TEXT,
  group_name TEXT,
  wait_started_ms INTEGER NOT NULL,
  wait_ended_ms INTEGER NOT NULL,
  wait_ms INTEGER NOT NULL,
  exclusive INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS commits (
  commit_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_seq INTEGER,
  kind TEXT NOT NULL,
  format_version INTEGER NOT NULL,
  prev_digest TEXT NOT NULL,
  content_digest TEXT NOT NULL,
  canonical_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
  snapshot_key TEXT PRIMARY KEY,
  cutoff_seq INTEGER NOT NULL,
  format_version INTEGER NOT NULL,
  digest TEXT NOT NULL,
  canonical_json TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout=30000")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        self.initialize_schema()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def initialize_schema(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)
            self.conn.execute(f"PRAGMA user_version={FORMAT_VERSION}")
            row = self.conn.execute("SELECT value FROM schema_meta WHERE key='format_version'").fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO schema_meta(key,value) VALUES('format_version',?)",
                    (str(FORMAT_VERSION),),
                )

    def initialize_spec(self, spec_digest: str, spec_bundle: dict[str, Any]) -> None:
        canonical = canonical_bytes(spec_bundle).decode("utf-8")
        digest = sha256_hex(canonical_bytes(spec_bundle))
        with self.conn:
            for key, value in (
                ("spec_digest", spec_digest),
                ("spec_canonical_digest", digest),
                ("format_version", str(FORMAT_VERSION)),
            ):
                self.conn.execute(
                    "INSERT INTO schema_meta(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, value),
                )
            self._insert_commit(
                self.conn,
                event_seq=0,
                kind="spec_initialized",
                payload={"spec_digest": spec_digest, "spec": spec_bundle},
            )

    @staticmethod
    def read_only_connection(path: str | Path) -> sqlite3.Connection:
        ledger_path = Path(path).resolve()
        conn = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _last_digest(self, conn: sqlite3.Connection) -> str:
        row = conn.execute("SELECT content_digest FROM commits ORDER BY commit_id DESC LIMIT 1").fetchone()
        return "0" * 64 if row is None else str(row["content_digest"])

    def _insert_commit(
        self,
        conn: sqlite3.Connection,
        *,
        event_seq: int | None,
        kind: str,
        payload: dict[str, Any],
    ) -> str:
        prev_digest = self._last_digest(conn)
        body = {
            "format_version": FORMAT_VERSION,
            "prev_digest": prev_digest,
            "event_seq": event_seq,
            "kind": kind,
            "payload": payload,
        }
        canonical = canonical_bytes(body).decode("utf-8")
        digest = sha256_hex(canonical.encode("utf-8"))
        conn.execute(
            "INSERT INTO commits(event_seq,kind,format_version,prev_digest,content_digest,canonical_json) "
            "VALUES(?,?,?,?,?,?)",
            (event_seq, kind, FORMAT_VERSION, prev_digest, digest, canonical),
        )
        return digest

    def mark_wait_records(self, records: Iterable[dict[str, Any]]) -> None:
        rows = list(records)
        if not rows:
            return
        with self.conn:
            for record in rows:
                self.conn.execute(
                    "INSERT OR REPLACE INTO wait_records"
                    "(seq,node_id,group_name,wait_started_ms,wait_ended_ms,wait_ms,exclusive) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (
                        int(record["seq"]),
                        record.get("node_id"),
                        record.get("group"),
                        int(record["wait_started_ms"]),
                        int(record["wait_ended_ms"]),
                        int(record["wait_ms"]),
                        1 if record.get("exclusive") else 0,
                    ),
                )

    def begin_inflight(
        self,
        *,
        request_id: str,
        event_seq: int | None,
        node_id: str | None,
        lease_id: str | None,
        op: str,
        payload: dict[str, Any],
        wait_started_ms: int,
        wait_ended_ms: int,
        provisional_holds: list[dict[str, Any]] | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO inflight_requests"
                "(request_id,event_seq,node_id,lease_id,op,payload_json,wait_started_ms,wait_ended_ms,status,result_json) "
                "VALUES(?,?,?,?,?,?,?,?, 'inflight', NULL)",
                (
                    request_id,
                    event_seq,
                    node_id,
                    lease_id,
                    op,
                    json.dumps(payload, sort_keys=True),
                    wait_started_ms,
                    wait_ended_ms,
                ),
            )
            for hold in provisional_holds or []:
                self._upsert_hold(conn=self.conn, hold=hold, default_status="provisional")
            self._insert_commit(
                self.conn,
                event_seq=event_seq,
                kind="request_inflight",
                payload={
                    "request_id": request_id,
                    "op": op,
                    "provisional_holds": provisional_holds or [],
                },
            )

    def _upsert_hold(self, *, conn: sqlite3.Connection, hold: dict[str, Any], default_status: str) -> None:
        conn.execute(
            "INSERT INTO holds(hold_id,node_id,slot_id,lease_id,artifact_key,shard_index,status,opened_seq,closed_seq,reclaim_reason) "
            "VALUES(:hold_id,:node_id,:slot_id,:lease_id,:artifact_key,:shard_index,:status,:opened_seq,:closed_seq,:reclaim_reason) "
            "ON CONFLICT(hold_id) DO UPDATE SET "
            "status=excluded.status, closed_seq=excluded.closed_seq, reclaim_reason=excluded.reclaim_reason",
            {
                "hold_id": hold["hold_id"],
                "node_id": hold["node_id"],
                "slot_id": int(hold["slot_id"]),
                "lease_id": hold["lease_id"],
                "artifact_key": hold["artifact_key"],
                "shard_index": int(hold["shard_index"]),
                "status": hold.get("status", default_status),
                "opened_seq": int(hold["opened_seq"]),
                "closed_seq": hold.get("closed_seq"),
                "reclaim_reason": hold.get("reclaim_reason"),
            },
        )

    def commit_bundle(
        self,
        *,
        event: dict[str, Any],
        status: str,
        result: dict[str, Any] | None = None,
        error_code: str | None = None,
        transitions: list[dict[str, Any]] | None = None,
        leases: list[dict[str, Any]] | None = None,
        holds: list[dict[str, Any]] | None = None,
        decisions: list[dict[str, Any]] | None = None,
        reclaims: list[dict[str, Any]] | None = None,
        inflight_id: str | None = None,
    ) -> str:
        event_seq = int(event["seq"])
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO events"
                "(seq,event_type,lease_id,node_id,artifact_key,quota_version,event_json,result_json,status,error_code) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    event_seq,
                    event.get("type"),
                    event.get("lease_id"),
                    event.get("node_id"),
                    event.get("artifact_key"),
                    event.get("quota_version"),
                    canonical_bytes(event).decode("utf-8"),
                    None if result is None else canonical_bytes(result).decode("utf-8"),
                    status,
                    error_code,
                ),
            )

            for lease in leases or []:
                self.conn.execute(
                    "INSERT INTO lease_current(lease_id,node_id,artifact_key,quota_version,state,generation,last_event_seq) "
                    "VALUES(:lease_id,:node_id,:artifact_key,:quota_version,:state,:generation,:last_event_seq) "
                    "ON CONFLICT(lease_id) DO UPDATE SET node_id=excluded.node_id,"
                    "artifact_key=excluded.artifact_key,quota_version=excluded.quota_version,"
                    "state=excluded.state,generation=excluded.generation,last_event_seq=excluded.last_event_seq",
                    lease,
                )

            for transition in transitions or []:
                digest = sha256_hex(canonical_bytes(transition))
                row = dict(transition)
                row["digest"] = digest
                self.conn.execute(
                    "INSERT OR REPLACE INTO transitions(event_seq,ordinal,lease_id,node_id,trigger,reason,"
                    "state_before,state_after,generation_before,generation_after,carried_label,quota_version,digest) "
                    "VALUES(:event_seq,:ordinal,:lease_id,:node_id,:trigger,:reason,:state_before,:state_after,"
                    ":generation_before,:generation_after,:carried_label,:quota_version,:digest)",
                    row,
                )

            for hold in holds or []:
                self._upsert_hold(conn=self.conn, hold=hold, default_status="active")

            for decision in decisions or []:
                decision_row = dict(decision)
                body = decision_row.pop("decision")
                digest = sha256_hex(canonical_bytes(body))
                self.conn.execute(
                    "INSERT OR REPLACE INTO decisions(decision_id,event_seq,ordinal,artifact_key,owner_node_id,"
                    "basis,generation,quota_version,decision_json,digest) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        decision_row["decision_id"],
                        decision_row["event_seq"],
                        decision_row["ordinal"],
                        decision_row["artifact_key"],
                        decision_row.get("owner_node_id"),
                        decision_row["basis"],
                        decision_row.get("generation"),
                        int(decision_row["quota_version"]),
                        canonical_bytes(body).decode("utf-8"),
                        digest,
                    ),
                )

            for reclaim in reclaims or []:
                digest = sha256_hex(canonical_bytes({k: v for k, v in reclaim.items() if k != "digest"}))
                self.conn.execute(
                    "INSERT OR REPLACE INTO reclaims(reclaim_id,event_seq,ordinal,lease_id,node_id,reason_kind,"
                    "generation_before,generation_after,hold_ids_json,digest) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        reclaim["reclaim_id"],
                        int(reclaim["event_seq"]),
                        int(reclaim["ordinal"]),
                        reclaim["lease_id"],
                        reclaim["node_id"],
                        reclaim["reason_kind"],
                        int(reclaim["generation_before"]),
                        int(reclaim["generation_after"]),
                        canonical_bytes(reclaim.get("hold_ids", [])).decode("utf-8"),
                        digest,
                    ),
                )

            if inflight_id is not None:
                self.conn.execute(
                    "UPDATE inflight_requests SET status='completed', result_json=? WHERE request_id=?",
                    (None if result is None else canonical_bytes(result).decode("utf-8"), inflight_id),
                )

            return self._insert_commit(
                self.conn,
                event_seq=event_seq,
                kind=f"event_{event.get('type')}_{status}",
                payload={
                    "event": event,
                    "status": status,
                    "result": result,
                    "error_code": error_code,
                    "transitions": transitions or [],
                    "leases": leases or [],
                    "holds": holds or [],
                    "decisions": decisions or [],
                    "reclaims": reclaims or [],
                },
            )

    def recover_unknown_inflight(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM inflight_requests WHERE status='inflight' ORDER BY request_id"
        ).fetchall()
        recovered: list[dict[str, Any]] = []
        for row in rows:
            request_id = row["request_id"]
            hold_rows = self.conn.execute(
                "SELECT * FROM holds WHERE lease_id=? AND status='provisional' ORDER BY shard_index",
                (row["lease_id"],),
            ).fetchall()
            hold_ids = [item["hold_id"] for item in hold_rows]
            with self.conn:
                self.conn.execute(
                    "UPDATE inflight_requests SET status='unknown_rerun_required' WHERE request_id=?",
                    (request_id,),
                )
                for hold_row in hold_rows:
                    self.conn.execute(
                        "UPDATE holds SET status='reclaimed_unknown', reclaim_reason='unknown_request_rerun_first' "
                        "WHERE hold_id=?",
                        (hold_row["hold_id"],),
                    )
                reclaim = {
                    "reclaim_id": f"unknown:{request_id}",
                    "event_seq": row["event_seq"] if row["event_seq"] is not None else -1,
                    "ordinal": 0,
                    "lease_id": row["lease_id"] or request_id,
                    "node_id": row["node_id"] or "unknown",
                    "reason_kind": "unknown_request",
                    "generation_before": 0,
                    "generation_after": 0,
                    "hold_ids": hold_ids,
                }
                self.conn.execute(
                    "INSERT OR REPLACE INTO reclaims(reclaim_id,event_seq,ordinal,lease_id,node_id,reason_kind,"
                    "generation_before,generation_after,hold_ids_json,digest) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        reclaim["reclaim_id"],
                        reclaim["event_seq"],
                        0,
                        reclaim["lease_id"],
                        reclaim["node_id"],
                        reclaim["reason_kind"],
                        0,
                        0,
                        canonical_bytes(hold_ids).decode(),
                        sha256_hex(canonical_bytes(reclaim)),
                    ),
                )
                self._insert_commit(
                    self.conn,
                    event_seq=row["event_seq"],
                    kind="unknown_inflight_reclaimed",
                    payload=reclaim,
                )
            recovered.append(reclaim)
        return recovered

    def save_snapshot(self, *, cutoff_seq: int, report: dict[str, Any]) -> tuple[str, bytes]:
        raw = canonical_bytes(report)
        digest = sha256_hex(raw)
        with self.conn:
            for key in ("latest", f"cutoff:{cutoff_seq}"):
                self.conn.execute(
                    "INSERT INTO snapshots(snapshot_key,cutoff_seq,format_version,digest,canonical_json) "
                    "VALUES(?,?,?,?,?) ON CONFLICT(snapshot_key) DO UPDATE SET cutoff_seq=excluded.cutoff_seq,"
                    "digest=excluded.digest,canonical_json=excluded.canonical_json",
                    (key, cutoff_seq, FORMAT_VERSION, digest, raw.decode("utf-8")),
                )
            self._insert_commit(
                self.conn,
                event_seq=cutoff_seq,
                kind="snapshot_saved",
                payload={"cutoff_seq": cutoff_seq, "digest": digest},
            )
        return digest, raw

    @staticmethod
    def read_node_recovery(path: str | Path, node_id: str) -> dict[str, dict[str, Any]]:
        if not Path(path).exists():
            return {}
        conn = Ledger.read_only_connection(path)
        try:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            if "lease_current" not in tables:
                return {}
            rows = conn.execute(
                "SELECT lease_id,state,generation,quota_version,artifact_key,last_event_seq "
                "FROM lease_current WHERE node_id=? ORDER BY lease_id",
                (node_id,),
            ).fetchall()
            return {
                row["lease_id"]: {
                    "state": row["state"],
                    "generation": row["generation"],
                    "quota_version": row["quota_version"],
                    "artifact_key": row["artifact_key"],
                    "last_event_seq": row["last_event_seq"],
                }
                for row in rows
            }
        finally:
            conn.close()
