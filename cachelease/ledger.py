from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .canonical import ZERO_DIGEST, canonical_json, rolling_digest, sha256_hex
from .errors import ReplayError, SpecError
from .spec import Config, Event


LEDGER_FORMAT_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY,
    first_ordinal INTEGER NOT NULL,
    last_ordinal INTEGER NOT NULL,
    first_seq INTEGER NOT NULL,
    last_seq INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('open','committed','unknown')),
    event_digest TEXT NOT NULL,
    ledger_digest TEXT NOT NULL,
    started_ms INTEGER NOT NULL,
    committed_ms INTEGER
);
CREATE TABLE IF NOT EXISTS progress (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    last_ordinal INTEGER NOT NULL,
    last_seq INTEGER NOT NULL,
    last_batch INTEGER NOT NULL,
    ledger_digest TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    ordinal INTEGER PRIMARY KEY,
    seq INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_digest TEXT NOT NULL,
    event_json TEXT NOT NULL,
    outcome_json TEXT NOT NULL,
    status TEXT NOT NULL,
    error_code TEXT,
    at_ms INTEGER NOT NULL,
    FOREIGN KEY(batch_id) REFERENCES batches(batch_id)
);
CREATE INDEX IF NOT EXISTS events_seq_idx ON events(seq, ordinal);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    artifact_key TEXT NOT NULL,
    apply_seq INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    quota_version INTEGER NOT NULL,
    digest TEXT NOT NULL,
    status TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0 CHECK(stale IN (0,1)),
    disposition TEXT,
    parent_lease_id TEXT,
    opened_ms INTEGER NOT NULL,
    closed_ms INTEGER
);
CREATE INDEX IF NOT EXISTS leases_active_idx ON leases(node_id, artifact_key, status, stale);
CREATE TABLE IF NOT EXISTS reservations (
    reservation_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    artifact_key TEXT NOT NULL,
    apply_seq INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    quota_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0 CHECK(stale IN (0,1)),
    disposition TEXT,
    grace_deadline_ms INTEGER NOT NULL,
    created_ms INTEGER NOT NULL,
    closed_ms INTEGER
);
CREATE INDEX IF NOT EXISTS reservations_active_idx ON reservations(node_id, status, stale);
CREATE TABLE IF NOT EXISTS capacity_changes (
    ordinal INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    old_quota_version INTEGER NOT NULL,
    new_quota_version INTEGER NOT NULL,
    old_capacity_slots INTEGER NOT NULL,
    new_capacity_slots INTEGER NOT NULL,
    old_lease_limit INTEGER NOT NULL,
    new_lease_limit INTEGER NOT NULL,
    policy TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    PRIMARY KEY(ordinal, node_id)
);
CREATE TABLE IF NOT EXISTS watermarks (
    ordinal INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    quota_version INTEGER NOT NULL,
    used_slots INTEGER NOT NULL,
    free_slots INTEGER NOT NULL,
    active_leases INTEGER NOT NULL,
    lease_remaining INTEGER NOT NULL,
    watermark_units INTEGER NOT NULL,
    quantization_error_upper INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0 CHECK(stale IN (0,1)),
    elapsed_ms INTEGER NOT NULL,
    observation TEXT NOT NULL,
    observation_digest TEXT NOT NULL,
    PRIMARY KEY(ordinal, node_id)
);
CREATE INDEX IF NOT EXISTS watermarks_current_idx ON watermarks(node_id, quota_version, stale, ordinal);
CREATE TABLE IF NOT EXISTS arbitrations (
    ordinal INTEGER PRIMARY KEY,
    artifact_key TEXT NOT NULL,
    chosen_node TEXT,
    rule TEXT NOT NULL,
    candidates_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS queue_observations (
    ordinal INTEGER PRIMARY KEY,
    seq INTEGER NOT NULL,
    node_id TEXT,
    artifact_key TEXT NOT NULL,
    stage TEXT NOT NULL,
    wait_start_ms INTEGER NOT NULL,
    wait_end_ms INTEGER NOT NULL,
    wait_ms INTEGER NOT NULL,
    global_in_flight INTEGER NOT NULL,
    node_in_flight INTEGER NOT NULL,
    stage_in_flight INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_actions (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER,
    kind TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS state_snapshots (
    batch_id INTEGER PRIMARY KEY,
    last_ordinal INTEGER NOT NULL,
    last_seq INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    snapshot_digest TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path, config: Config | None = None, *, create: bool = True) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.executescript(SCHEMA)
        if config is not None:
            self.initialize(config)
        elif not create:
            self._require_initialized()

    def initialize(self, config: Config) -> None:
        with self._transaction():
            format_row = self.connection.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
            if format_row is None:
                self.connection.execute("INSERT INTO meta(key, value) VALUES('format_version', ?)", (str(LEDGER_FORMAT_VERSION),))
                self.connection.execute("INSERT INTO meta(key, value) VALUES('config_digest', ?)", (config.config_digest,))
                self.connection.execute("INSERT INTO meta(key, value) VALUES('scale', ?)", (str(config.scale),))
                self.connection.execute("INSERT INTO meta(key, value) VALUES('ledger_digest', ?)", (ZERO_DIGEST,))
                self.connection.execute("INSERT INTO meta(key, value) VALUES('content_digest', ?)", (ZERO_DIGEST,))
                self.connection.execute(
                    "INSERT INTO progress(singleton, last_ordinal, last_seq, last_batch, ledger_digest, updated_ms) VALUES(1,0,0,0,?,?)",
                    (ZERO_DIGEST, _mono_ms()),
                )
            else:
                self._require_compatible(config)
                self.connection.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES('content_digest', ?)", (ZERO_DIGEST,)
                )

    def _require_initialized(self) -> None:
        row = self.connection.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
        if row is None:
            raise SpecError("ledger is not initialized", self.path)

    def _require_compatible(self, config: Config) -> None:
        version = self.meta_int("format_version")
        if version != LEDGER_FORMAT_VERSION:
            raise SpecError(
                f"ledger format version {version} is not supported (expected {LEDGER_FORMAT_VERSION})", self.path
            )
        digest = self.meta("config_digest")
        if digest != config.config_digest:
            raise SpecError("ledger config digest does not match the supplied declaration", self.path)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def meta(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def meta_int(self, key: str) -> int:
        value = self.meta(key)
        if value is None:
            raise SpecError(f"ledger meta key {key!r} is missing", self.path)
        return int(value)

    def progress(self) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM progress WHERE singleton=1").fetchone()
        if row is None:
            raise SpecError("ledger progress is missing", self.path)
        return dict(row)

    def begin_batch(self, batch_id: int, events: Iterable[Event]) -> dict[str, Any]:
        event_list = list(events)
        if not event_list:
            raise ValueError("batch must contain at least one event")
        first_ordinal = event_list[0].ordinal
        last_ordinal = event_list[-1].ordinal
        first_seq = event_list[0].seq
        last_seq = event_list[-1].seq
        event_digest = _event_set_digest(event_list)
        with self._transaction():
            existing = self.connection.execute("SELECT status FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
            if existing is not None and existing["status"] == "committed":
                raise SpecError(f"batch {batch_id} is already committed", self.path)
            self.connection.execute(
                """
                INSERT INTO batches(batch_id, first_ordinal, last_ordinal, first_seq, last_seq, status,
                                    event_digest, ledger_digest, started_ms, committed_ms)
                VALUES(?,?,?,?,?,'open',?,?,?,NULL)
                ON CONFLICT(batch_id) DO UPDATE SET
                    first_ordinal=excluded.first_ordinal,
                    last_ordinal=excluded.last_ordinal,
                    first_seq=excluded.first_seq,
                    last_seq=excluded.last_seq,
                    status='open', event_digest=excluded.event_digest,
                    ledger_digest=excluded.ledger_digest,
                    started_ms=excluded.started_ms,
                    committed_ms=NULL
                """,
                (
                    batch_id,
                    first_ordinal,
                    last_ordinal,
                    first_seq,
                    last_seq,
                    event_digest,
                    ZERO_DIGEST,
                    _mono_ms(),
                ),
            )
        return {
            "batch_id": batch_id,
            "first_ordinal": first_ordinal,
            "last_ordinal": last_ordinal,
            "event_digest": event_digest,
        }

    def commit_batch(
        self,
        batch_id: int,
        *,
        events: list[dict[str, Any]],
        lease_upserts: list[dict[str, Any]] | None = None,
        lease_updates: list[dict[str, Any]] | None = None,
        reservation_upserts: list[dict[str, Any]] | None = None,
        reservation_updates: list[dict[str, Any]] | None = None,
        capacity_changes: list[dict[str, Any]] | None = None,
        watermarks: list[dict[str, Any]] | None = None,
        stale_watermarks: list[dict[str, Any]] | None = None,
        arbitrations: list[dict[str, Any]] | None = None,
        queue_observations: list[dict[str, Any]] | None = None,
        snapshot: dict[str, Any],
    ) -> str:
        if not events:
            raise ValueError("committed batch must contain events")
        progress = self.progress()
        last_ordinal = int(progress["last_ordinal"])
        for event in events:
            if int(event["ordinal"]) <= last_ordinal:
                raise SpecError(
                    f"event ordinal {event['ordinal']} is not after ledger progress {last_ordinal}", self.path
                )
        previous_digest = str(progress["ledger_digest"])
        batch_row = self.connection.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
        if batch_row is None or batch_row["status"] != "open":
            raise SpecError(f"batch {batch_id} is not open", self.path)
        event_payload = canonical_json(
            {
                "batch": batch_id,
                "events": [
                    {"digest": event["event_digest"], "outcome": event["outcome"]}
                    for event in sorted(events, key=lambda item: int(item["ordinal"]))
                ],
            }
        ).encode("utf-8")
        new_digest = rolling_digest(previous_digest, event_payload)
        snapshot_json = canonical_json(snapshot)
        snapshot_digest = sha256_hex(snapshot_json.encode("utf-8"))
        content_entries = sorted(
            (
                {
                    "node_id": node_id,
                    "artifact_key": str(entry["artifact_key"]),
                    "digest": str(entry["digest"]),
                }
                for node_id, node_state in snapshot.get("nodes", {}).items()
                for entry in node_state.get("entries", [])
            ),
            key=lambda item: (item["node_id"], item["artifact_key"]),
        )
        content_digest = sha256_hex(canonical_json(content_entries).encode("utf-8"))
        now = _mono_ms()
        last_event = max(events, key=lambda item: int(item["ordinal"]))

        with self._transaction():
            for row in lease_upserts or []:
                self.connection.execute(
                    """
                    INSERT INTO leases(lease_id, node_id, artifact_key, apply_seq, ordinal, quota_version,
                                       digest, status, stale, disposition, parent_lease_id, opened_ms, closed_ms)
                    VALUES(:lease_id,:node_id,:artifact_key,:apply_seq,:ordinal,:quota_version,
                           :digest,:status,:stale,:disposition,:parent_lease_id,:opened_ms,:closed_ms)
                    ON CONFLICT(lease_id) DO UPDATE SET
                        status=excluded.status, stale=excluded.stale, disposition=excluded.disposition,
                        closed_ms=excluded.closed_ms
                    """,
                    _lease_defaults(row),
                )
            for row in lease_updates or []:
                self.connection.execute(
                    "UPDATE leases SET status=:status, stale=:stale, disposition=:disposition, closed_ms=:closed_ms WHERE lease_id=:lease_id",
                    {
                        "lease_id": row["lease_id"],
                        "status": row["status"],
                        "stale": int(row.get("stale", 0)),
                        "disposition": row.get("disposition"),
                        "closed_ms": row.get("closed_ms", now),
                    },
                )
            for row in reservation_upserts or []:
                self.connection.execute(
                    """
                    INSERT INTO reservations(reservation_id, node_id, artifact_key, apply_seq, ordinal,
                                             quota_version, status, stale, disposition, grace_deadline_ms,
                                             created_ms, closed_ms)
                    VALUES(:reservation_id,:node_id,:artifact_key,:apply_seq,:ordinal,:quota_version,
                           :status,:stale,:disposition,:grace_deadline_ms,:created_ms,:closed_ms)
                    ON CONFLICT(reservation_id) DO UPDATE SET
                        status=excluded.status, stale=excluded.stale, disposition=excluded.disposition,
                        closed_ms=excluded.closed_ms
                    """,
                    _reservation_defaults(row),
                )
            for row in reservation_updates or []:
                self.connection.execute(
                    "UPDATE reservations SET status=:status, stale=:stale, disposition=:disposition, closed_ms=:closed_ms WHERE reservation_id=:reservation_id",
                    {
                        "reservation_id": row["reservation_id"],
                        "status": row["status"],
                        "stale": int(row.get("stale", 0)),
                        "disposition": row.get("disposition"),
                        "closed_ms": row.get("closed_ms", now),
                    },
                )
            for row in capacity_changes or []:
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO capacity_changes(ordinal,node_id,old_quota_version,new_quota_version,
                        old_capacity_slots,new_capacity_slots,old_lease_limit,new_lease_limit,policy,detail_json)
                    VALUES(:ordinal,:node_id,:old_quota_version,:new_quota_version,:old_capacity_slots,
                           :new_capacity_slots,:old_lease_limit,:new_lease_limit,:policy,:detail_json)
                    """,
                    row,
                )
            for row in watermarks or []:
                payload = {
                    "ordinal": int(row["ordinal"]),
                    "seq": int(row["seq"]),
                    "node_id": row["node_id"],
                    "quota_version": int(row["quota_version"]),
                    "used_slots": int(row["used_slots"]),
                    "free_slots": int(row["free_slots"]),
                    "active_leases": int(row["active_leases"]),
                    "lease_remaining": int(row["lease_remaining"]),
                    "watermark_units": int(row["watermark_units"]),
                    "quantization_error_upper": int(row["quantization_error_upper"]),
                    "elapsed_ms": int(row.get("elapsed_ms", 0)),
                }
                stale_flag = int(row.get("stale", 0))
                observation = canonical_json(payload)
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO watermarks(ordinal,seq,node_id,quota_version,used_slots,free_slots,
                        active_leases,lease_remaining,watermark_units,quantization_error_upper,stale,
                        elapsed_ms,observation,observation_digest)
                    VALUES(:ordinal,:seq,:node_id,:quota_version,:used_slots,:free_slots,:active_leases,
                           :lease_remaining,:watermark_units,:quantization_error_upper,:stale,:elapsed_ms,
                           :observation,:observation_digest)
                    """,
                    {
                        **payload,
                        "stale": stale_flag,
                        "observation": observation,
                        "observation_digest": sha256_hex(observation.encode("utf-8")),
                    },
                )
            for row in stale_watermarks or []:
                self.connection.execute(
                    "UPDATE watermarks SET stale=1 WHERE node_id=? AND quota_version<=?",
                    (row["node_id"], int(row["quota_version"])),
                )
            for row in arbitrations or []:
                self.connection.execute(
                    "INSERT OR REPLACE INTO arbitrations(ordinal,artifact_key,chosen_node,rule,candidates_json) VALUES(?,?,?,?,?)",
                    (
                        row["ordinal"],
                        row["artifact_key"],
                        row.get("chosen_node"),
                        row["rule"],
                        canonical_json(row["candidates"]),
                    ),
                )
            for row in queue_observations or []:
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO queue_observations(ordinal,seq,node_id,artifact_key,stage,wait_start_ms,
                        wait_end_ms,wait_ms,global_in_flight,node_in_flight,stage_in_flight)
                    VALUES(:ordinal,:seq,:node_id,:artifact_key,:stage,:wait_start_ms,:wait_end_ms,:wait_ms,
                           :global_in_flight,:node_in_flight,:stage_in_flight)
                    """,
                    row,
                )
            for event in events:
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO events(ordinal,seq,batch_id,event_type,event_digest,event_json,
                        outcome_json,status,error_code,at_ms)
                    VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        int(event["ordinal"]),
                        int(event["seq"]),
                        batch_id,
                        event["event_type"],
                        event["event_digest"],
                        canonical_json(event["event"]),
                        canonical_json(event["outcome"]),
                        event.get("status", "ok"),
                        event.get("error_code"),
                        now,
                    ),
                )
            self.connection.execute(
                "UPDATE batches SET status='committed', ledger_digest=?, committed_ms=? WHERE batch_id=?",
                (new_digest, now, batch_id),
            )
            self.connection.execute(
                "UPDATE progress SET last_ordinal=?, last_seq=?, last_batch=?, ledger_digest=?, updated_ms=? WHERE singleton=1",
                (int(last_event["ordinal"]), int(last_event["seq"]), batch_id, new_digest, now),
            )
            self.connection.execute("UPDATE meta SET value=? WHERE key='ledger_digest'", (new_digest,))
            self.connection.execute("UPDATE meta SET value=? WHERE key='content_digest'", (content_digest,))
            self.connection.execute(
                "INSERT OR REPLACE INTO state_snapshots(batch_id,last_ordinal,last_seq,snapshot_json,snapshot_digest) VALUES(?,?,?,?,?)",
                (batch_id, int(last_event["ordinal"]), int(last_event["seq"]), snapshot_json, snapshot_digest),
            )
        return new_digest

    def mark_open_batches_unknown(self, reason: str) -> list[dict[str, Any]]:
        open_rows = self.connection.execute("SELECT * FROM batches WHERE status='open' ORDER BY batch_id").fetchall()
        if not open_rows:
            return []
        with self._transaction():
            for row in open_rows:
                self.connection.execute("UPDATE batches SET status='unknown' WHERE batch_id=?", (row["batch_id"],))
                self.connection.execute(
                    "INSERT INTO recovery_actions(batch_id,kind,detail_json,created_ms) VALUES(?,?,?,?)",
                    (
                        row["batch_id"],
                        "UNKNOWN_UNCOMMITTED_BATCH",
                        canonical_json(
                            {
                                "batch_id": row["batch_id"],
                                "first_ordinal": row["first_ordinal"],
                                "last_ordinal": row["last_ordinal"],
                                "reason": reason,
                                "action": "rerun_from_committed_progress",
                            }
                        ),
                        _mono_ms(),
                    ),
                )
        return [dict(row) for row in open_rows]

    def recovery_actions(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM recovery_actions ORDER BY action_id").fetchall()
        return [dict(row) for row in rows]

    def record_recovery_action(self, batch_id: int | None, kind: str, detail: dict[str, Any]) -> None:
        with self._transaction():
            self.connection.execute(
                "INSERT INTO recovery_actions(batch_id,kind,detail_json,created_ms) VALUES(?,?,?,?)",
                (batch_id, kind, canonical_json(detail), _mono_ms()),
            )

    def latest_snapshot(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM state_snapshots ORDER BY last_ordinal DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return json.loads(str(row["snapshot_json"]))

    def snapshot_for_batch(self, batch_id: int) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT snapshot_json FROM state_snapshots WHERE batch_id=?", (batch_id,)).fetchone()
        return None if row is None else json.loads(str(row["snapshot_json"]))

    def active_leases(self, artifact: str | None = None, node_id: str | None = None) -> list[dict[str, Any]]:
        clauses = ["status='active'", "stale=0"]
        params: list[Any] = []
        if artifact is not None:
            clauses.append("artifact_key=?")
            params.append(artifact)
        if node_id is not None:
            clauses.append("node_id=?")
            params.append(node_id)
        rows = self.connection.execute(
            f"SELECT * FROM leases WHERE {' AND '.join(clauses)} ORDER BY node_id, artifact_key, lease_id",
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def lease(self, lease_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
        return None if row is None else dict(row)

    def active_reservations(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM reservations WHERE status='reserved' ORDER BY node_id, reservation_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def committed_events(self, up_to_ordinal: int | None = None) -> list[dict[str, Any]]:
        if up_to_ordinal is None:
            rows = self.connection.execute(
                "SELECT e.* FROM events e JOIN batches b ON b.batch_id=e.batch_id "
                "WHERE b.status='committed' ORDER BY e.ordinal"
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT e.* FROM events e JOIN batches b ON b.batch_id=e.batch_id "
                "WHERE b.status='committed' AND e.ordinal<=? ORDER BY e.ordinal",
                (up_to_ordinal,),
            ).fetchall()
        return [dict(row) for row in rows]

    def event_hashes(self) -> dict[int, str]:
        return {
            int(row["ordinal"]): str(row["event_digest"])
            for row in self.connection.execute("SELECT ordinal,event_digest FROM events ORDER BY ordinal")
        }

    def latest_watermarks(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT w.* FROM watermarks w
            JOIN (SELECT node_id, MAX(ordinal) AS ordinal FROM watermarks WHERE stale=0 GROUP BY node_id) latest
              ON latest.node_id=w.node_id AND latest.ordinal=w.ordinal
            ORDER BY w.node_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def watermark_for(self, ordinal: int, node_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM watermarks WHERE ordinal=? AND node_id=?", (ordinal, node_id)
        ).fetchone()
        return None if row is None else dict(row)

    def replay_and_validate(self, expected_nodes: set[str]) -> dict[str, Any]:
        progression = self.progress()
        rows = self.committed_events()
        if not rows:
            if int(progression["last_ordinal"]) != 0:
                raise ReplayError(0, "progress is non-zero but no committed events exist")
            return {"events": 0, "last_seq": 0, "watermarks": 0}
        previous = ZERO_DIGEST
        expected_ordinal = 1
        for row in rows:
            ordinal = int(row["ordinal"])
            seq = int(row["seq"])
            if ordinal != expected_ordinal:
                raise ReplayError(seq, f"event ordinal gap: expected {expected_ordinal}, got {ordinal}")
            expected_ordinal += 1
            event_json = str(row["event_json"])
            try:
                event_payload = json.loads(event_json)
            except json.JSONDecodeError as exc:
                raise ReplayError(seq, f"stored event JSON is invalid: {exc}") from exc
            computed_event_digest = sha256_hex(canonical_json(event_payload).encode("utf-8"))
            if computed_event_digest != row["event_digest"]:
                raise ReplayError(seq, "event digest does not match stored event payload")
            for node_id in sorted(expected_nodes):
                watermark = self.watermark_for(ordinal, node_id)
                if watermark is None:
                    raise ReplayError(seq, f"missing watermark row for node {node_id}")
                observation_payload = {
                    key: watermark[key]
                    for key in (
                        "ordinal", "seq", "node_id", "quota_version", "used_slots", "free_slots",
                        "active_leases", "lease_remaining", "watermark_units",
                        "quantization_error_upper", "elapsed_ms",
                    )
                }
                expected_observation = canonical_json(observation_payload)
                if expected_observation != watermark["observation"]:
                    raise ReplayError(seq, f"watermark observation changed for node {node_id}")
                expected_digest = sha256_hex(expected_observation.encode("utf-8"))
                if expected_digest != watermark["observation_digest"]:
                    raise ReplayError(seq, f"watermark digest mismatch for node {node_id}")

        batch_rows = self.connection.execute(
            "SELECT * FROM batches WHERE status='committed' ORDER BY batch_id"
        ).fetchall()
        for batch in batch_rows:
            payload = canonical_json(
                {
                    "batch": int(batch["batch_id"]),
                    "events": [
                        {"digest": row["event_digest"], "outcome": json.loads(row["outcome_json"])}
                        for row in self.connection.execute(
                            "SELECT event_digest,outcome_json FROM events WHERE batch_id=? ORDER BY ordinal",
                            (batch["batch_id"],),
                        )
                    ],
                }
            ).encode("utf-8")
            previous = rolling_digest(previous, payload)
            if previous != batch["ledger_digest"]:
                raise ReplayError(
                    int(batch["first_seq"]),
                    f"ledger digest chain mismatch at batch {batch['batch_id']}",
                )
        if previous != progression["ledger_digest"]:
            raise ReplayError(int(progression["last_seq"]), "progress ledger digest differs from replay chain")
        last_row = rows[-1]
        if int(progression["last_ordinal"]) != int(last_row["ordinal"]):
            raise ReplayError(int(last_row["seq"]), "progress ordinal is behind the last committed event")
        for watermark in self.latest_watermarks():
            if int(watermark["ordinal"]) > int(progression["last_ordinal"]):
                raise ReplayError(int(watermark["seq"]), "watermark is beyond committed progress")
        snapshot = self.latest_snapshot()
        if snapshot is not None:
            if int(snapshot.get("last_ordinal", -1)) != int(progression["last_ordinal"]):
                raise ReplayError(int(progression["last_seq"]), "state snapshot is not aligned with progress")
            for node_id in sorted(expected_nodes):
                node_state = snapshot.get("nodes", {}).get(node_id)
                if node_state is None:
                    raise ReplayError(int(progression["last_seq"]), f"snapshot has no state for node {node_id}")
                latest_wm = next((item for item in self.latest_watermarks() if item["node_id"] == node_id), None)
                if latest_wm is None:
                    raise ReplayError(int(progression["last_seq"]), f"no current watermark for node {node_id}")
                for field_name in (
                    "used_slots", "free_slots", "active_leases", "lease_remaining",
                    "watermark_units", "quantization_error_upper", "quota_version",
                ):
                    expected_value = (
                        node_state["quota_version"]
                        if field_name == "quota_version"
                        else node_state["watermark"][field_name]
                    )
                    if int(expected_value) != int(latest_wm[field_name]):
                        raise ReplayError(
                            int(progression["last_seq"]),
                            f"snapshot watermark {field_name} diverges for node {node_id}",
                        )
                snapshot_reservations = sorted(
                    str(item["reservation_id"]) for item in node_state.get("reservations", [])
                )
                ledger_reservations = sorted(
                    str(item["reservation_id"])
                    for item in self.active_reservations()
                    if str(item["node_id"]) == node_id
                )
                if snapshot_reservations != ledger_reservations:
                    raise ReplayError(
                        int(progression["last_seq"]),
                        f"snapshot reservation account diverges for node {node_id}",
                    )
            content_entries = sorted(
                (
                    {
                        "node_id": node_id,
                        "artifact_key": str(entry["artifact_key"]),
                        "digest": str(entry["digest"]),
                    }
                    for node_id, node_state in snapshot.get("nodes", {}).items()
                    for entry in node_state.get("entries", [])
                ),
                key=lambda item: (item["node_id"], item["artifact_key"]),
            )
            content_digest = sha256_hex(canonical_json(content_entries).encode("utf-8"))
            if content_digest != self.meta("content_digest"):
                raise ReplayError(int(progression["last_seq"]), "content digest does not match snapshot entries")
        return {
            "events": len(rows),
            "last_seq": int(rows[-1]["seq"]),
            "watermarks": len(rows) * len(expected_nodes),
        }


def _event_set_digest(events: Iterable[Event]) -> str:
    payload = [{"ordinal": event.ordinal, "digest": event.digest} for event in events]
    return sha256_hex(canonical_json(payload).encode("utf-8"))


def _lease_defaults(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.setdefault("status", "active")
    result.setdefault("stale", 0)
    result.setdefault("disposition", None)
    result.setdefault("parent_lease_id", None)
    result.setdefault("opened_ms", _mono_ms())
    result.setdefault("closed_ms", None)
    return result


def _reservation_defaults(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.setdefault("status", "reserved")
    result.setdefault("stale", 0)
    result.setdefault("disposition", None)
    result.setdefault("created_ms", _mono_ms())
    result.setdefault("closed_ms", None)
    return result


def _mono_ms() -> int:
    return time.monotonic_ns() // 1_000_000
