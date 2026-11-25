from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from . import LEDGER_FORMAT_VERSION
from .canonical import canonical_text, sha256_hex
from .errors import E_LIMIT, MeshError
from .ownership import OwnershipAdjudicator
from .state_machine import ACTIVE_STATES


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_format (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  format_version INTEGER NOT NULL,
  declaration_digest TEXT NOT NULL,
  content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
  request_id TEXT PRIMARY KEY,
  op TEXT NOT NULL,
  node_id TEXT NOT NULL,
  lease_id TEXT,
  slot_id TEXT,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  response_json TEXT,
  error_json TEXT,
  wait_started_ms REAL,
  wait_ended_ms REAL,
  wait_ms REAL,
  recovery_state TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS reservations (
  reservation_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  slot_id TEXT,
  node_id TEXT,
  quota_version INTEGER,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS commits (
  seq INTEGER PRIMARY KEY,
  digest TEXT NOT NULL UNIQUE,
  prev_digest TEXT,
  kind TEXT NOT NULL,
  request_id TEXT,
  event_seq INTEGER,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS leases (
  lease_id TEXT PRIMARY KEY,
  node_id TEXT NOT NULL,
  slot_id TEXT NOT NULL,
  state TEXT NOT NULL,
  generation INTEGER NOT NULL,
  quota_version INTEGER,
  last_event_seq INTEGER,
  history_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transitions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lease_id TEXT NOT NULL,
  event_seq INTEGER,
  request_id TEXT,
  before_state TEXT,
  after_state TEXT NOT NULL,
  reason TEXT NOT NULL,
  generation_before INTEGER,
  generation_after INTEGER NOT NULL,
  record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ownership (
  slot_id TEXT PRIMARY KEY,
  lease_id TEXT NOT NULL UNIQUE,
  node_id TEXT NOT NULL,
  generation INTEGER NOT NULL,
  event_seq INTEGER NOT NULL,
  basis_generation INTEGER
);
CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY,
  event_seq INTEGER NOT NULL,
  slot_id TEXT NOT NULL,
  lease_id TEXT NOT NULL,
  generation INTEGER NOT NULL,
  reason_category TEXT NOT NULL,
  winning_claim_id TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_records (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT NOT NULL,
  action TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class Ledger:
    def __init__(
        self,
        path: str | Path,
        spec: Optional[dict[str, Any]] = None,
        *,
        readonly: bool = False,
    ) -> None:
        self.path = Path(path)
        self.spec = spec or {}
        self.readonly = readonly
        self.lock = threading.RLock()
        if readonly:
            uri = f"file:{self.path.resolve()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.row_factory = sqlite3.Row
        if not readonly:
            self._initialize()
        self.adjudicator = self._load_adjudicator()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def _initialize(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.execute(f"PRAGMA user_version={int(LEDGER_FORMAT_VERSION)}")
        declaration_digest = self.spec.get("declaration_digest")
        if not declaration_digest and self.spec:
            from .spec import digest_declaration

            declaration_digest = digest_declaration(self.spec)
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
            if declaration_digest:
                conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES('declaration_digest', ?)",
                    (declaration_digest,),
                )
            row = conn.execute("SELECT value FROM meta WHERE key='head_digest'").fetchone()
            if row is None:
                genesis = sha256_hex(
                    {
                        "genesis": "leasemesh-ledger",
                        "format_version": LEDGER_FORMAT_VERSION,
                        "declaration_digest": declaration_digest,
                    }
                )
                conn.execute("INSERT INTO meta(key, value) VALUES('head_digest', ?)", (genesis,))
                conn.execute("INSERT INTO meta(key, value) VALUES('content_digest', ?)", (genesis,))
                conn.execute(
                    "INSERT INTO ledger_format(id, format_version, declaration_digest, content_digest) "
                    "VALUES (1, ?, ?, ?)",
                    (int(LEDGER_FORMAT_VERSION), declaration_digest or "", genesis),
                )
            head_row = conn.execute("SELECT value FROM meta WHERE key='head_digest'").fetchone()
            conn.execute(
                "INSERT OR IGNORE INTO ledger_format(id, format_version, declaration_digest, content_digest) "
                "VALUES (1, ?, ?, ?)",
                (
                    int(LEDGER_FORMAT_VERSION),
                    declaration_digest or "",
                    str(head_row["value"]) if head_row else "",
                ),
            )
            row = conn.execute("SELECT value FROM meta WHERE key='projection_json'").fetchone()
            if row is None:
                result = OwnershipAdjudicator(self.spec).result()
                text = canonical_text(result)
                conn.execute("INSERT INTO meta(key, value) VALUES('projection_json', ?)", (text,))
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES('projection_digest', ?)",
                    (sha256_hex(text.encode("utf-8")),),
                )

    def meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _append_commit(
        self,
        conn: sqlite3.Connection,
        *,
        kind: str,
        payload: dict[str, Any],
        request_id: Optional[str] = None,
        event_seq: Optional[int] = None,
    ) -> int:
        existing = conn.execute("SELECT seq FROM commits ORDER BY seq LIMIT 1").fetchone()
        if existing is None:
            prev_digest = None
        else:
            row = conn.execute("SELECT value FROM meta WHERE key='head_digest'").fetchone()
            prev_digest = str(row["value"]) if row else None
        seq_row = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM commits").fetchone()
        seq = int(seq_row["next_seq"])
        envelope = {
            "seq": seq,
            "kind": kind,
            "request_id": request_id,
            "event_seq": event_seq,
            "payload": payload,
            "prev_digest": prev_digest,
        }
        digest = sha256_hex(envelope)
        conn.execute(
            """
            INSERT INTO commits(seq, digest, prev_digest, kind, request_id, event_seq, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (seq, digest, prev_digest, kind, request_id, event_seq, canonical_text(payload)),
        )
        conn.execute("UPDATE meta SET value=? WHERE key='head_digest'", (digest,))
        conn.execute("UPDATE meta SET value=? WHERE key='content_digest'", (digest,))
        conn.execute("UPDATE ledger_format SET content_digest=? WHERE id=1", (digest,))
        return seq

    def _load_adjudicator(self) -> OwnershipAdjudicator:
        adjudicator = OwnershipAdjudicator(self.spec)
        if not self.path.exists():
            return adjudicator
        rows = self.conn.execute(
            "SELECT payload_json, kind FROM commits WHERE kind IN ('lease_result','domain_event') ORDER BY seq"
        ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            event = payload.get("event")
            if event is not None:
                events.append(event)
        adjudicator.apply_all(events)
        return adjudicator

    def plan_request(self, request: dict[str, Any]) -> None:
        request_id = request["request_id"]
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT status FROM requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if existing is not None:
                return

            if request.get("op") == "grant":
                slot_id = request.get("slot_id")
                owner = conn.execute("SELECT lease_id FROM ownership WHERE slot_id=?", (slot_id,)).fetchone()
                if owner is not None:
                    raise MeshError(
                        f"slot {slot_id} is already owned by {owner['lease_id']}",
                        code=E_LIMIT,
                        field="slot_id",
                    )
                held = conn.execute(
                    "SELECT reservation_id FROM reservations WHERE slot_id=? AND status='held'",
                    (slot_id,),
                ).fetchone()
                if held is not None:
                    raise MeshError(
                        f"slot {slot_id} has an in-doubt held reservation",
                        code=E_LIMIT,
                        field="slot_id",
                    )
                limit = int(self.spec.get("per_node_lease_limit", 10**9))
                active = conn.execute(
                    "SELECT COUNT(*) AS count FROM leases WHERE node_id=? AND state IN ('granted','renewed')",
                    (request["node_id"],),
                ).fetchone()["count"]
                held_count = conn.execute(
                    "SELECT COUNT(*) AS count FROM reservations WHERE node_id=? AND status='held' AND kind='capacity'",
                    (request["node_id"],),
                ).fetchone()["count"]
                if int(active) + int(held_count) >= limit:
                    raise MeshError(
                        f"node {request['node_id']} lease limit {limit} exceeded",
                        code=E_LIMIT,
                        field="per_node_lease_limit",
                    )
                conn.execute(
                    """
                    INSERT INTO reservations(reservation_id, request_id, kind, slot_id, node_id,
                                             quota_version, status, payload_json)
                    VALUES (?, ?, 'capacity', ?, ?, ?, 'held', ?)
                    """,
                    (
                        f"res-{request_id}",
                        request_id,
                        slot_id,
                        request["node_id"],
                        request.get("quota_version", self.spec.get("quota_version")),
                        canonical_text(request),
                    ),
                )

            conn.execute(
                """
                INSERT INTO requests(request_id, op, node_id, lease_id, slot_id, status,
                                     payload_json, wait_started_ms)
                VALUES (?, ?, ?, ?, ?, 'planned', ?, ?)
                """,
                (
                    request_id,
                    request["op"],
                    request["node_id"],
                    request.get("lease_id"),
                    request.get("slot_id"),
                    canonical_text(request),
                    request.get("wait_started_ms"),
                ),
            )
            self._append_commit(
                conn,
                kind="request_planned",
                request_id=request_id,
                payload={"request": request},
            )

    def commit_live_result(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        wait: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        request_id = request["request_id"]
        op = request["op"]
        kind = {"grant": "lease_granted", "renew": "lease_renewed", "release": "lease_released"}[op]
        generation_after = int(response["generation"])
        event: dict[str, Any] = {
            "seq": None,
            "kind": kind,
            "lease_id": request["lease_id"],
            "node_id": request["node_id"],
            "slot_id": request.get("slot_id"),
            "generation": generation_after,
            "quota_version": response.get("quota_version", self.spec.get("quota_version")),
            "request_id": request_id,
        }
        if op != "grant":
            event["basis_generation"] = response.get("generation_before", generation_after - 1)
        if op == "release" and not event.get("slot_id"):
            lease_row = self.conn.execute(
                "SELECT slot_id FROM leases WHERE lease_id=?", (request["lease_id"],)
            ).fetchone()
            if lease_row is not None:
                event["slot_id"] = lease_row["slot_id"]

        with self.lock:
            try:
                wait = wait or {}
                with self.transaction() as conn:
                    row = conn.execute(
                        "SELECT status FROM requests WHERE request_id=?", (request_id,)
                    ).fetchone()
                    if row is None:
                        raise MeshError(f"request {request_id} was not planned", code="E_VERIFY")
                    if row["status"] not in {"planned", "unknown"}:
                        return {"committed": False, "status": row["status"], "event": event}

                    next_seq_row = conn.execute(
                        "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM commits"
                    ).fetchone()
                    event_seq = int(next_seq_row["next_seq"])
                    event["seq"] = event_seq
                    self.adjudicator.apply(event)
                    result = self.adjudicator.result()
                    seq = self._append_commit(
                        conn,
                        kind="lease_result",
                        request_id=request_id,
                        event_seq=event_seq,
                        payload={"event": event, "request": request, "response": response, "wait": wait},
                    )
                    conn.execute(
                        """
                        UPDATE requests SET status='committed', response_json=?, wait_started_ms=?,
                          wait_ended_ms=?, wait_ms=?, error_json=NULL WHERE request_id=?
                        """,
                        (
                            canonical_text(response),
                            wait.get("wait_started_ms"),
                            wait.get("wait_ended_ms"),
                            wait.get("wait_ms"),
                            request_id,
                        ),
                    )
                    conn.execute(
                        "UPDATE reservations SET status='converted', updated_at=datetime('now') "
                        "WHERE request_id=? AND status='held'",
                        (request_id,),
                    )
                    self._replace_projection(conn, result)
                    conn.execute(
                        "INSERT INTO meta(key,value) VALUES('last_commit_seq', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (str(seq),),
                    )
            except Exception:
                self.adjudicator = self._load_adjudicator()
                raise
        return {"committed": True, "event": event, "projection": result}

    def record_rejection(
        self,
        request: dict[str, Any],
        error: dict[str, Any] | MeshError,
        wait: Optional[dict[str, Any]] = None,
    ) -> None:
        error_dict = error.to_dict() if isinstance(error, MeshError) else dict(error)
        with self.transaction() as conn:
            conn.execute(
                "UPDATE requests SET status='rejected', error_json=?, wait_ms=? WHERE request_id=?",
                (canonical_text(error_dict), (wait or {}).get("wait_ms"), request["request_id"]),
            )
            conn.execute(
                "UPDATE reservations SET status='released_rejected' WHERE request_id=? AND status='held'",
                (request["request_id"],),
            )
            self._append_commit(
                conn,
                kind="request_rejected",
                request_id=request["request_id"],
                payload={"request": request, "error": error_dict, "wait": wait or {}},
            )

    def recover_pending(self) -> dict[str, Any]:
        unknown: list[dict[str, Any]] = []
        rerun_requests: list[dict[str, Any]] = []
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT request_id, payload_json FROM requests WHERE status IN ('planned','unknown') ORDER BY request_id"
            ).fetchall()
            for row in rows:
                request = json.loads(row["payload_json"])
                request_id = row["request_id"]
                reclaimed = conn.execute(
                    "UPDATE reservations SET status='reclaimed', updated_at=datetime('now') "
                    "WHERE request_id=? AND status='held'",
                    (request_id,),
                )
                conn.execute(
                    "UPDATE requests SET status='unknown', recovery_state='rerun_required' WHERE request_id=?",
                    (request_id,),
                )
                payload = {
                    "request_id": request_id,
                    "action": "mark_unknown_and_reclaim_quota",
                    "reservations_reclaimed": int(reclaimed.rowcount or 0),
                    "request": request,
                }
                conn.execute(
                    "INSERT INTO recovery_records(request_id, action, payload_json) VALUES(?, 'unknown_rerun', ?)",
                    (request_id, canonical_text(payload)),
                )
                self._append_commit(
                    conn,
                    kind="recovery",
                    request_id=request_id,
                    payload=payload,
                )
                unknown.append(payload)
                rerun = dict(request)
                rerun["rerun_required_for"] = request_id
                rerun["request_id"] = f"{request_id}--rerun"
                rerun_requests.append(rerun)
        return {
            "unknown_requests": unknown,
            "rerun_requests": rerun_requests,
            "count": len(unknown),
            "note": "SIGKILL produces no exit report; this report is created only by explicit restart recovery.",
        }

    def import_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        imported = 0
        for event in events:
            with self.lock:
                self.adjudicator.apply(event)
                result = self.adjudicator.result()
                with self.transaction() as conn:
                    self._append_commit(
                        conn,
                        kind="domain_event",
                        event_seq=int(event["seq"]),
                        payload={"event": event},
                    )
                    self._replace_projection(conn, result)
                imported += 1
        return {"imported": imported, "projection": self.adjudicator.result()}

    def _replace_projection(self, conn: sqlite3.Connection, result: dict[str, Any]) -> None:
        text = canonical_text(result)
        conn.execute(
            "INSERT INTO meta(key,value) VALUES('projection_json', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (text,),
        )
        digest = sha256_hex(text.encode("utf-8"))
        conn.execute(
            "INSERT INTO meta(key,value) VALUES('projection_digest', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (digest,),
        )

        conn.execute("DELETE FROM leases")
        conn.execute("DELETE FROM ownership")
        conn.execute("DELETE FROM decisions")
        conn.execute("DELETE FROM transitions")

        for lease in result["leases"]:
            conn.execute(
                """
                INSERT INTO leases(lease_id,node_id,slot_id,state,generation,quota_version,
                                   last_event_seq,history_json)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    lease["lease_id"],
                    lease["node_id"],
                    lease["slot_id"],
                    lease["state"],
                    int(lease["generation"]),
                    lease.get("quota_version"),
                    lease.get("last_event_seq"),
                    canonical_text(lease.get("history", [])),
                ),
            )
            for record in lease.get("history", []):
                conn.execute(
                    """
                    INSERT INTO transitions(lease_id,event_seq,request_id,before_state,after_state,
                                            reason,generation_before,generation_after,record_json)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        record["lease_id"],
                        record.get("event_seq"),
                        record.get("request_id"),
                        record.get("before_state"),
                        record["after_state"],
                        record["reason"],
                        record.get("generation_before"),
                        int(record["generation_after"]),
                        canonical_text(record),
                    ),
                )

        for owner in result["owners"]:
            conn.execute(
                """
                INSERT INTO ownership(slot_id,lease_id,node_id,generation,event_seq,basis_generation)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    owner["slot_id"],
                    owner["lease_id"],
                    owner["node_id"],
                    int(owner["generation"]),
                    int(owner["event_seq"]),
                    owner.get("basis_generation"),
                ),
            )

        for decision in result["decisions"]:
            conn.execute(
                """
                INSERT INTO decisions(decision_id,event_seq,slot_id,lease_id,generation,
                                      reason_category,winning_claim_id,payload_json)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    decision["decision_id"],
                    int(decision["event_seq"]),
                    decision["slot_id"],
                    decision["lease_id"],
                    int(decision["generation"]),
                    decision["reason_category"],
                    decision["winning_claim_id"],
                    canonical_text(decision),
                ),
            )

    def active_capacity_snapshot(self) -> dict[str, Any]:
        return self.adjudicator.result()["capacity"]


def ledger_is_initialized(path: str | Path) -> bool:
    return Path(path).exists() and Path(path).stat().st_size > 0
