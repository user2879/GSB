"""SQLite ledger persistence.

A request batch is written in a single transaction: all lease events, the
post-batch lease states, per-pool watermarks and the progress cursor commit
atomically.  A batch row is inserted up-front with status ``OPEN`` and flipped
to ``COMMITTED`` in the same transaction, so after a forced kill an ``OPEN``
row unambiguously marks work whose effects never became durable and must be
re-run.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .diagnostics import DEFAULT_SCALE
from .errors import E_SNAPSHOT, LeaseError
from .model import Pool

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    started_ms INTEGER NOT NULL,
    committed_ms INTEGER
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    request_id TEXT,
    key TEXT,
    pool_id TEXT,
    lease_id TEXT,
    at_ms INTEGER NOT NULL,
    revision INTEGER,
    priority INTEGER,
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lease_states (
    lease_id TEXT NOT NULL,
    batch_id INTEGER NOT NULL,
    state TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (lease_id, batch_id)
);
CREATE TABLE IF NOT EXISTS watermarks (
    batch_id INTEGER NOT NULL,
    scope TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (batch_id, scope)
);
CREATE TABLE IF NOT EXISTS progress (
    batch_id INTEGER PRIMARY KEY,
    last_request_id TEXT NOT NULL,
    last_event_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    name TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    digest TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path),
                                    isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------- metadata
    def put_meta(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, sort_keys=True)))
        self.conn.commit()

    def get_meta(self, key: str, default=None):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    # ------------------------------------------------------------- batches
    def begin_batch(self, started_ms: int) -> int:
        cur = self.conn.execute(
            "INSERT INTO batches(status, started_ms) VALUES('OPEN', ?)",
            (started_ms,))
        return int(cur.lastrowid)

    def commit_batch(self, batch_id: int, *, events: list[dict],
                     leases: dict, watermarks: list[dict],
                     last_request_id: str, committed_ms: int,
                     snapshot: dict | None = None) -> None:
        try:
            self.conn.execute("BEGIN")
            self._insert_events(batch_id, events)
            self._insert_leases(batch_id, leases)
            self._insert_watermarks(batch_id, watermarks)
            last_seq = max((event["seq"] for event in events), default=0)
            self.conn.execute(
                "INSERT INTO progress(batch_id, last_request_id, last_event_seq)"
                " VALUES(?, ?, ?)",
                (batch_id, last_request_id, last_seq))
            self.conn.execute(
                "UPDATE batches SET status='COMMITTED', committed_ms=? "
                "WHERE batch_id=?", (committed_ms, batch_id))
            if snapshot is not None:
                self.conn.execute(
                    "INSERT INTO snapshots(name, payload, digest) "
                    "VALUES('session', ?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET payload=excluded.payload, "
                    "digest=excluded.digest",
                    (json.dumps(snapshot["payload"], sort_keys=True),
                     snapshot["digest"]))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _insert_events(self, batch_id: int, events: list[dict]) -> None:
        self.conn.executemany(
            "INSERT INTO events(seq, batch_id, event_type, request_id, key, "
            "pool_id, lease_id, at_ms, revision, priority, detail) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(event["seq"], batch_id, event["event_type"],
              event.get("request_id"), event.get("key"),
              event.get("pool_id"), event.get("lease_id"), event["at_ms"],
              event.get("revision"), event.get("priority"),
              json.dumps(event.get("detail", {}), sort_keys=True))
             for event in events])

    def _insert_leases(self, batch_id: int, leases: dict) -> None:
        self.conn.executemany(
            "INSERT INTO lease_states(lease_id, batch_id, state, payload) "
            "VALUES(?, ?, ?, ?)",
            [(lease_id, batch_id, payload["state"],
              json.dumps(payload, sort_keys=True))
             for lease_id, payload in sorted(leases.items())])

    def _insert_watermarks(self, batch_id: int, watermarks: list[dict]) -> None:
        self.conn.executemany(
            "INSERT INTO watermarks(batch_id, scope, payload) VALUES(?, ?, ?)",
            [(batch_id, mark["scope"], json.dumps(mark, sort_keys=True))
             for mark in watermarks])

    # --------------------------------------------------------------- reads
    def list_batches(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM batches ORDER BY batch_id"))

    def open_batch_ids(self) -> list[int]:
        return [int(row["batch_id"]) for row in self.conn.execute(
            "SELECT batch_id FROM batches WHERE status='OPEN' ORDER BY batch_id")]

    def committed_batches(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM batches WHERE status='COMMITTED' ORDER BY batch_id"))

    def events(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY seq").fetchall()
        return [self._event_row(row) for row in rows]

    def events_for_batch(self, batch_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE batch_id=? ORDER BY seq",
            (batch_id,)).fetchall()
        return [self._event_row(row) for row in rows]

    def latest_leases(self) -> dict:
        rows = self.conn.execute(
            "SELECT ls.payload FROM lease_states ls JOIN ("
            "  SELECT lease_id, MAX(batch_id) AS max_batch FROM lease_states "
            "  GROUP BY lease_id"
            ") latest ON latest.lease_id=ls.lease_id "
            "AND latest.max_batch=ls.batch_id").fetchall()
        return {json.loads(row["payload"])["lease_id"]:
                json.loads(row["payload"]) for row in rows}

    def latest_watermarks(self) -> list[dict]:
        batch_row = self.conn.execute(
            "SELECT MAX(batch_id) AS bid FROM batches "
            "WHERE status='COMMITTED'").fetchone()
        if batch_row["bid"] is None:
            return []
        rows = self.conn.execute(
            "SELECT payload FROM watermarks WHERE batch_id=? ORDER BY scope",
            (batch_row["bid"],)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def watermarks_for_batch(self, batch_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT payload FROM watermarks WHERE batch_id=? ORDER BY scope",
            (batch_id,)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def progress(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM progress ORDER BY batch_id DESC LIMIT 1").fetchone()
        return None if row is None else dict(row)

    def request_sequence(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT request_id FROM events WHERE request_id IS NOT NULL "
            "ORDER BY seq").fetchall()
        seen: list[str] = []
        unique: set[str] = set()
        for row in rows:
            rid = row["request_id"]
            if rid not in unique:
                unique.add(rid)
                seen.append(rid)
        return seen

    def load_snapshot_row(self) -> dict | None:
        row = self.conn.execute(
            "SELECT payload, digest FROM snapshots WHERE name='session'"
        ).fetchone()
        if row is None:
            return None
        return {"payload": json.loads(row["payload"]),
                "digest": row["digest"]}

    def save_snapshot_row(self, snapshot: dict) -> None:
        self.conn.execute(
            "INSERT INTO snapshots(name, payload, digest) VALUES('session', ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET payload=excluded.payload, "
            "digest=excluded.digest",
            (json.dumps(snapshot["payload"], sort_keys=True),
             snapshot["digest"]))
        self.conn.commit()

    def replace_snapshot_payload(self, payload: dict, digest: str) -> None:
        """Test/demo helper used to inject E_SNAPSHOT digest corruption."""
        rowcount = self.conn.execute(
            "UPDATE snapshots SET payload=?, digest=? WHERE name='session'",
            (json.dumps(payload, sort_keys=True), digest)).rowcount
        self.conn.commit()
        if rowcount == 0:
            raise LeaseError(E_SNAPSHOT, "no snapshot row exists to corrupt")


    def reset_to_last_committed(self) -> dict:
        """Discard every OPEN (non-durable) batch and reset derived tables.

        Used when re-running after a forced termination: only fully committed
        records survive, and the unknown OPEN work is removed so the request
        sequence can be adjudicated again from the durable progress cursor.
        """
        open_ids = self.open_batch_ids()
        if not open_ids:
            return {"removed_batches": []}
        placeholders = ",".join("?" for _ in open_ids)
        self.conn.execute("BEGIN")
        for table in ("events", "lease_states", "watermarks", "progress"):
            self.conn.execute(
                f"DELETE FROM {table} WHERE batch_id IN ({placeholders})",
                open_ids)
        self.conn.execute(
            f"DELETE FROM batches WHERE batch_id IN ({placeholders})",
            open_ids)
        self.conn.commit()
        return {"removed_batches": open_ids}

    # ------------------------------------------------------------- recovery
    def recovery_report(self) -> dict:
        open_ids = self.open_batch_ids()
        committed = self.committed_batches()
        return {
            "committed_batches": [int(row["batch_id"]) for row in committed],
            "unknown_open_batches": open_ids,
            "rerun_required": bool(open_ids),
            "note": ("OPEN batches never committed; their grants/rollbacks "
                     "are not durable and must be re-run") if open_ids
                    else "all batches committed",
            "progress": self.progress(),
        }

    @staticmethod
    def _event_row(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["detail"] = json.loads(data["detail"])
        return data


def describe_pools_for_meta(pools: dict[str, Pool]) -> dict:
    return {pid: pool.to_dict() for pid, pool in sorted(pools.items())}
