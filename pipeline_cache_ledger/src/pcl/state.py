"""SQLite-backed session state: registrations, hits, invalidation, schedule.

The scheduler invokes judgments on worker threads, so the connection allows
cross-thread use and every access is guarded by one reentrant lock.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

STATE_DB_NAME = "session.sqlite3"


class StateStore:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / STATE_DB_NAME
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    @contextmanager
    def _tx(self, commit=False):
        with self._lock:
            yield self.conn
            if commit:
                self.conn.commit()

    def _create_schema(self):
        with self._tx(commit=True) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    key TEXT PRIMARY KEY,
                    stage_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    digest TEXT NOT NULL,
                    refs_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_format INTEGER,
                    backfilled_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS hits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    seq INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS invalidations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    root_key TEXT NOT NULL,
                    invalidated_key TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    trail_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS schedule (
                    key TEXT PRIMARY KEY,
                    stage_id TEXT NOT NULL,
                    queued_tick INTEGER NOT NULL,
                    start_tick INTEGER NOT NULL,
                    end_tick INTEGER NOT NULL,
                    slot INTEGER NOT NULL,
                    wait_ticks INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS meta (
                    name TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    # ------------------------------------------------------------- artifacts
    def upsert_artifact(self, key, stage_id, version, digest, refs, status,
                        source_format=None, backfilled=()):
        with self._tx(commit=True) as conn:
            conn.execute(
                """
                INSERT INTO artifacts(key, stage_id, version, digest, refs_json,
                                      status, source_format, backfilled_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    stage_id=excluded.stage_id, version=excluded.version,
                    digest=excluded.digest, refs_json=excluded.refs_json,
                    status=excluded.status, source_format=excluded.source_format,
                    backfilled_json=excluded.backfilled_json
                """,
                (key, stage_id, version, digest,
                 json.dumps(refs, sort_keys=True), status, source_format,
                 json.dumps(sorted(backfilled))),
            )

    def get_artifact(self, key):
        with self._tx() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE key=?", (key,)
            ).fetchone()
            return dict(row) if row else None

    def all_artifacts(self):
        with self._tx() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM artifacts ORDER BY key"
            ).fetchall()]

    def mark_status(self, key, status):
        with self._tx(commit=True) as conn:
            conn.execute(
                "UPDATE artifacts SET status=? WHERE key=?", (status, key)
            )

    # ------------------------------------------------------------------ hits
    def record_hit(self, key, outcome, reason, digest, seq):
        with self._tx(commit=True) as conn:
            conn.execute(
                "INSERT INTO hits(key, outcome, reason, digest, seq) "
                "VALUES (?,?,?,?,?)",
                (key, outcome, reason, digest, seq),
            )

    def hit_history(self):
        with self._tx() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT key, outcome, reason, digest, seq FROM hits "
                "ORDER BY seq, key"
            ).fetchall()]

    def last_outcomes(self):
        query = (
            "SELECT key, outcome, reason, digest FROM hits h "
            "WHERE seq=(SELECT MAX(seq) FROM hits h2 WHERE h2.key=h.key) "
            "ORDER BY key"
        )
        with self._tx() as conn:
            return {
                r["key"]: dict(r)
                for r in conn.execute(query).fetchall()
            }

    # -------------------------------------------------------- invalidations
    def record_invalidation(self, root_key, order, trail):
        with self._tx(commit=True) as conn:
            conn.execute(
                "DELETE FROM invalidations WHERE root_key=?", (root_key,)
            )
            for position, invalidated_key in enumerate(order):
                conn.execute(
                    "INSERT INTO invalidations(root_key, invalidated_key, "
                    "position, trail_json) VALUES (?, ?, ?, ?)",
                    (root_key, invalidated_key, position,
                     json.dumps(order)),
                )

    def invalidation_trails(self):
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT DISTINCT root_key, trail_json FROM invalidations "
                "ORDER BY root_key"
            ).fetchall()
            return {r["root_key"]: json.loads(r["trail_json"]) for r in rows}

    # -------------------------------------------------------------- schedule
    def save_schedule(self, records):
        with self._tx(commit=True) as conn:
            conn.execute("DELETE FROM schedule")
            for rec in records:
                conn.execute(
                    "INSERT INTO schedule(key, stage_id, queued_tick, "
                    "start_tick, end_tick, slot, wait_ticks, outcome, reason) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (rec.key, rec.stage_id, rec.queued_tick, rec.start_tick,
                     rec.end_tick, rec.slot, rec.wait_ticks,
                     rec.outcome, rec.reason),
                )

    def schedule_rows(self):
        with self._tx() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM schedule ORDER BY queued_tick, key"
            ).fetchall()]

    # ------------------------------------------------------------------ meta
    def set_meta(self, name, value):
        with self._tx(commit=True) as conn:
            conn.execute(
                "INSERT INTO meta(name, value) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                (name, str(value)),
            )

    def get_meta(self, name, default=None):
        with self._tx() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE name=?", (name,)
            ).fetchone()
            return row["value"] if row else default

    def export_state(self):
        with self._tx() as conn:
            meta_rows = conn.execute("SELECT name, value FROM meta").fetchall()
            return {
                "artifacts": self.all_artifacts(),
                "hits": self.hit_history(),
                "invalidations": self.invalidation_trails(),
                "schedule": self.schedule_rows(),
                "meta": {r["name"]: r["value"] for r in meta_rows},
            }

    def replace_with(self, state_doc):
        from dataclasses import dataclass

        @dataclass
        class Rec:
            key: str
            stage_id: str
            queued_tick: int
            start_tick: int
            end_tick: int
            slot: int
            wait_ticks: int
            outcome: str
            reason: str

        with self._tx(commit=True) as conn:
            conn.executescript(
                "DELETE FROM artifacts; DELETE FROM hits; "
                "DELETE FROM invalidations; DELETE FROM schedule; "
                "DELETE FROM meta;"
            )
        for art in state_doc.get("artifacts", []):
            self.upsert_artifact(
                art["key"], art["stage_id"], art["version"], art["digest"],
                json.loads(art["refs_json"]), art["status"],
                art.get("source_format"),
                json.loads(art.get("backfilled_json", "[]")),
            )
        for hit in state_doc.get("hits", []):
            self.record_hit(
                hit["key"], hit["outcome"], hit["reason"],
                hit["digest"], hit["seq"],
            )
        for root_key, trail in state_doc.get("invalidations", {}).items():
            self.record_invalidation(root_key, trail, trail)
        if state_doc.get("schedule"):
            self.save_schedule([Rec(**row) for row in state_doc["schedule"]])
        for name, value in state_doc.get("meta", {}).items():
            self.set_meta(name, value)

    def close(self):
        with self._lock:
            self.conn.close()
