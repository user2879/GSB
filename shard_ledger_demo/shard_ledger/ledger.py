"""SQLite ledger with format versioning and a chained content digest."""

import hashlib
import json
import os
import sqlite3

from .clock import now_ms, wall_iso
from .errors import SnapshotError

LEDGER_FORMAT_VERSION = 1


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_row(previous: str, kind: str, payload: dict) -> str:
    body = previous + "|" + kind + "|" + canonical(payload)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, path: str):
        self.path = path
        fresh = not os.path.exists(path) or os.path.getsize(path) == 0
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        if fresh:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
            self.conn.execute("INSERT INTO meta(key, value) VALUES('head_digest', ?)", ("GENESIS",))
            self.conn.commit()
        self._verify_format()
        self.verify_digest_chain()

    def _create_tables(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records(
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                digest TEXT NOT NULL,
                wall_ts TEXT NOT NULL,
                mono_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS outcomes(
                shard_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                status TEXT NOT NULL,
                config_version INTEGER NOT NULL,
                summary TEXT,
                units INTEGER,
                event_seq INTEGER,
                mono_ms INTEGER,
                digest TEXT NOT NULL,
                PRIMARY KEY(shard_id, generation)
            );
            CREATE TABLE IF NOT EXISTS unknown_shards(
                shard_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                reason TEXT NOT NULL,
                mono_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots(
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_seq INTEGER NOT NULL,
                head_digest TEXT NOT NULL,
                wall_ts TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def _verify_format(self):
        row = self.conn.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
        if row is None or int(row[0]) != LEDGER_FORMAT_VERSION:
            raise SnapshotError("unknown ledger format version", entity=self.path, field="format_version")

    def _head(self) -> str:
        return self.conn.execute("SELECT value FROM meta WHERE key='head_digest'").fetchone()[0]

    def append_record(self, kind: str, payload: dict):
        digest = digest_row(self._head(), kind, payload)
        self.conn.execute(
            "INSERT INTO records(kind, payload, digest, wall_ts, mono_ms) VALUES(?, ?, ?, ?, ?)",
            (kind, canonical(payload), digest, wall_iso(), now_ms()),
        )
        self.conn.execute("UPDATE meta SET value=? WHERE key='head_digest'", (digest,))
        return digest

    def append_event(self, event: dict):
        with self.conn:
            self.append_record("event", event)

    def append_lease(self, lease: dict, event_seq: int):
        with self.conn:
            payload = {"event_seq": event_seq, **lease}
            self.append_record("lease", payload)

    def append_propagation(self, event_seq: int, revision: int, basis: list):
        with self.conn:
            self.append_record("propagation", {"event_seq": event_seq, "revision": revision, "basis": basis})

    def write_outcome(self, outcome: dict):
        with self.conn:
            digest = self.append_record("outcome", outcome)
            self.conn.execute(
                """INSERT OR REPLACE INTO outcomes
                   (shard_id, generation, status, config_version, summary, units,
                    event_seq, mono_ms, digest)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    outcome["shard_id"],
                    outcome["generation"],
                    outcome["status"],
                    outcome["config_version"],
                    canonical(outcome["summary"]) if outcome.get("summary") is not None else None,
                    outcome.get("units"),
                    outcome["event_seq"],
                    now_ms(),
                    digest,
                ),
            )

    def mark_unknown(self, shard_id: str, generation: int, reason: str):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO unknown_shards VALUES(?, ?, ?, ?)",
                (shard_id, generation, reason, now_ms()),
            )
            self.append_record("unknown", {"shard_id": shard_id, "generation": generation, "reason": reason})

    def write_snapshot(self, event_seq: int):
        with self.conn:
            head = self._head()
            self.conn.execute(
                "INSERT INTO snapshots(event_seq, head_digest, wall_ts) VALUES(?, ?, ?)",
                (event_seq, head, wall_iso()),
            )
            self.append_record("snapshot", {"event_seq": event_seq, "head_digest": head})

    def verify_digest_chain(self):
        previous = "GENESIS"
        for seq, kind, payload, digest in self.conn.execute(
            "SELECT seq, kind, payload, digest FROM records ORDER BY seq"
        ):
            expected = digest_row(previous, kind, json.loads(payload))
            if expected != digest:
                raise SnapshotError(
                    "ledger content digest mismatch",
                    entity=f"record:{seq}",
                    field="digest",
                    expected=expected,
                    actual=digest,
                )
            previous = digest
        head = self._head()
        if head != previous:
            raise SnapshotError("ledger head digest mismatch", entity=self.path, field="head_digest")
        return previous

    def latest_outcomes(self) -> dict:
        rows = self.conn.execute(
            "SELECT shard_id, generation, status, config_version, summary, units, digest FROM outcomes"
        ).fetchall()
        latest = {}
        for shard_id, generation, status, version, summary, units, digest in rows:
            current = latest.get(shard_id)
            if current is None or generation > current[0]:
                latest[shard_id] = (generation, status, version, summary, units, digest)
        return latest

    def all_outcomes(self) -> list:
        return self.conn.execute(
            "SELECT shard_id, generation, status, config_version, summary, units, digest FROM outcomes ORDER BY shard_id, generation"
        ).fetchall()

    def unknown_shards(self) -> list:
        return self.conn.execute(
            "SELECT shard_id, generation, reason FROM unknown_shards ORDER BY shard_id"
        ).fetchall()

    def snapshots(self) -> list:
        return self.conn.execute(
            "SELECT snapshot_id, event_seq, head_digest, wall_ts FROM snapshots ORDER BY snapshot_id"
        ).fetchall()

    def processed_events(self) -> set:
        result = set()
        for (payload,) in self.conn.execute("SELECT payload FROM records WHERE kind='event' ORDER BY seq"):
            result.add(json.loads(payload)["seq"])
        return result

    def close(self):
        self.conn.close()
