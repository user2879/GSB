"""SQLite ledger: authoritative event store, cache state and query journal.

The ledger is the single source of truth. Ingestion, cache updates and
journal inserts happen inside one ``BEGIN IMMEDIATE`` transaction so that a
forced termination can only expose fully committed records. Rows read but not
yet committed are simply absent after restart and must be reloaded.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from typing import Any

from .config import Event, SessionSpec, canonical_dumps
from .engine import SessionEngine

SCHEMA_VERSION = 1


class Ledger:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self.conn.close()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS param_versions(
                version INTEGER PRIMARY KEY,
                out_of_order_ms INTEGER NOT NULL,
                latency_budget_ms INTEGER NOT NULL,
                scale INTEGER NOT NULL,
                rule_digest TEXT NOT NULL,
                status TEXT NOT NULL,
                created_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS raw_events(
                param_version INTEGER NOT NULL,
                partition TEXT NOT NULL,
                seq INTEGER NOT NULL,
                ts_ms INTEGER NOT NULL,
                value INTEGER NOT NULL,
                source_line INTEGER NOT NULL,
                committed_ms INTEGER NOT NULL,
                PRIMARY KEY(param_version, partition, seq)
            );
            CREATE TABLE IF NOT EXISTS partition_state(
                param_version INTEGER NOT NULL,
                partition TEXT NOT NULL,
                watermark_ms INTEGER NOT NULL,
                max_ts_seen_ms INTEGER NOT NULL,
                finalized_count INTEGER NOT NULL,
                late_count INTEGER NOT NULL,
                raw_sum TEXT NOT NULL,
                sum_fixed INTEGER NOT NULL,
                sum_saturated INTEGER NOT NULL,
                cursor_seq INTEGER NOT NULL,
                PRIMARY KEY(param_version, partition)
            );
            CREATE TABLE IF NOT EXISTS watermark_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                param_version INTEGER NOT NULL,
                partition TEXT NOT NULL,
                old_watermark_ms INTEGER NOT NULL,
                new_watermark_ms INTEGER NOT NULL,
                trigger_seq INTEGER NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS late_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                param_version INTEGER NOT NULL,
                partition TEXT NOT NULL,
                seq INTEGER NOT NULL,
                ts_ms INTEGER NOT NULL,
                watermark_ms INTEGER NOT NULL,
                source_line INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS query_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                param_version INTEGER NOT NULL,
                query_seq INTEGER NOT NULL,
                partition TEXT,
                latency_ms INTEGER NOT NULL,
                budget_ms INTEGER NOT NULL,
                over_budget INTEGER NOT NULL,
                injected_ms INTEGER NOT NULL,
                clock_start_ns INTEGER NOT NULL
            );
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else row["value"]

    def put_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))

    def latest_param_version(self) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(version) AS v FROM param_versions").fetchone()
        return None if row["v"] is None else row["v"]

    def insert_param_version(self, version: int, spec: SessionSpec,
                             rule_digest: str, created_ms: int) -> None:
        self.conn.execute(
            "INSERT INTO param_versions(version,out_of_order_ms,"
            "latency_budget_ms,scale,rule_digest,status,created_ms)"
            " VALUES(?,?,?,?,?, 'active', ?)",
            (version, spec.out_of_order_ms, spec.latency_budget_ms,
             spec.scale, rule_digest, created_ms))

    def mark_versions_stale(self, current: int) -> None:
        self.conn.execute(
            "UPDATE param_versions SET status='stale' WHERE version<>?",
            (current,))

    # ---------------------------------------------------------------- ingest
    def commit_ingest(self, param_version: int, spec: SessionSpec,
                      events: list[Event], engine: SessionEngine,
                      committed_ms: int) -> None:
        """Atomically persist raw events, logs and refreshed cache state."""
        with self.conn:  # BEGIN/COMMIT; rollback on any exception
            for event in events:
                self.conn.execute(
                    "INSERT INTO raw_events(param_version,partition,seq,"
                    "ts_ms,value,source_line,committed_ms) VALUES(?,?,?,?,?,?,?)",
                    (param_version, event.partition, event.seq, event.ts_ms,
                     event.value, event.source_line, committed_ms))
            self._flush_steps(param_version, engine)
            self._flush_late(param_version, engine)
            self._refresh_state(param_version, spec, engine)

    def _flush_steps(self, param_version: int,
                     engine: SessionEngine) -> None:
        for key in sorted(engine.parts):
            for step in engine.parts[key].steps:
                self.conn.execute(
                    "INSERT INTO watermark_log(param_version,partition,"
                    "old_watermark_ms,new_watermark_ms,trigger_seq,reason)"
                    " VALUES(?,?,?,?,?,?)",
                    (param_version, step.partition, step.old_watermark_ms,
                     step.new_watermark_ms, step.trigger_seq, step.reason))
        for key in engine.parts:
            engine.parts[key].steps.clear()

    def _flush_late(self, param_version: int,
                    engine: SessionEngine) -> None:
        for key in sorted(engine.parts):
            for late in engine.parts[key].late_events:
                self.conn.execute(
                    "INSERT INTO late_log(param_version,partition,seq,"
                    "ts_ms,watermark_ms,source_line) VALUES(?,?,?,?,?,?)",
                    (param_version, key, late.seq, late.ts_ms,
                     late.watermark_ms, late.source_line))
            engine.parts[key].late_events.clear()

    def _refresh_state(self, param_version: int, spec: SessionSpec,
                       engine: SessionEngine) -> None:
        for key in sorted(engine.parts):
            part = engine.parts[key]
            fixed = part.fixed_sum()
            self.conn.execute(
                "INSERT INTO partition_state(param_version,partition,"
                "watermark_ms,max_ts_seen_ms,finalized_count,late_count,"
                "raw_sum,sum_fixed,sum_saturated,cursor_seq) "
                "VALUES(?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(param_version,partition) DO UPDATE SET "
                "watermark_ms=excluded.watermark_ms,"
                "max_ts_seen_ms=excluded.max_ts_seen_ms,"
                "finalized_count=excluded.finalized_count,"
                "late_count=excluded.late_count,"
                "raw_sum=excluded.raw_sum,"
                "sum_fixed=excluded.sum_fixed,"
                "sum_saturated=excluded.sum_saturated,"
                "cursor_seq=excluded.cursor_seq",
                (param_version, key, part.watermark_ms, part.max_ts_seen,
                 part.finalized_count, part.late_count, str(part.raw_sum),
                 fixed, int(fixed != part.raw_sum * (10 ** spec.scale)),
                 part.cursor))

    # ------------------------------------------------------------- query log
    def record_query(self, param_version: int, query_seq: int,
                     partition: str | None, latency_ms: int, budget_ms: int,
                     over_budget: bool, injected_ms: int,
                     clock_start_ns: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO query_log(param_version,query_seq,partition,"
                "latency_ms,budget_ms,over_budget,injected_ms,clock_start_ns)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (param_version, query_seq, partition, latency_ms, budget_ms,
                 int(over_budget), injected_ms, clock_start_ns))

    def fetch_queries(self, param_version: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM query_log WHERE param_version=? ORDER BY id",
            (param_version,)).fetchall())

    def next_query_seq(self, param_version: int) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(query_seq),0)+1 AS n FROM query_log "
            "WHERE param_version=?", (param_version,)).fetchone()
        return row["n"]

    # ------------------------------------------------------------- cache IO
    def cache_views(self, param_version: int,
                    spec: SessionSpec) -> list[dict[str, Any]]:
        """Incremental cache path: read the maintained partition_state."""
        rows = self.conn.execute(
            "SELECT * FROM partition_state WHERE param_version=? "
            "ORDER BY partition", (param_version,)).fetchall()
        views: list[dict[str, Any]] = []
        if not rows:
            # Zero-state version: expose every declared partition as empty
            # so cache and replay share an identical partition set.
            return [{
                "partition": rule.key,
                "watermark_ms": 0,
                "max_ts_seen_ms": 0,
                "finalized_count": 0,
                "late_count": 0,
                "pending_count": 0,
                "sum_raw": 0,
                "sum_fixed": 0,
                "sum_saturated": False,
                "scale": spec.scale,
                "cursor_seq": -1,
            } for rule in spec.rules]
        for row in rows:
            views.append({
                "partition": row["partition"],
                "watermark_ms": row["watermark_ms"],
                "max_ts_seen_ms": row["max_ts_seen_ms"],
                "finalized_count": row["finalized_count"],
                "late_count": row["late_count"],
                "pending_count": 0,
                "sum_raw": int(row["raw_sum"]),
                "sum_fixed": row["sum_fixed"],
                "sum_saturated": bool(row["sum_saturated"]),
                "scale": spec.scale,
                "cursor_seq": row["cursor_seq"],
            })
        return views

    def raw_events_for_version(self,
                               param_version: int) -> list[Event]:
        rows = self.conn.execute(
            "SELECT * FROM raw_events WHERE param_version=? "
            "ORDER BY partition, seq", (param_version,)).fetchall()
        return [Event(partition=r["partition"], seq=r["seq"],
                      ts_ms=r["ts_ms"], value=r["value"],
                      source_line=r["source_line"]) for r in rows]

    def authoritative_replay(self, param_version: int,
                             spec: SessionSpec) -> list[dict[str, Any]]:
        """Rebuild every partition from committed raw events, cache-free."""
        engine = SessionEngine(spec, param_version)
        engine.apply_events(self.raw_events_for_version(param_version))
        engine.drain()
        return engine.views()

    def hydrate_engine(self, param_version: int,
                       spec: SessionSpec) -> tuple[SessionEngine, dict[str, int]]:
        """Rebuild an in-memory engine purely from committed records.

        Replaying the committed raw events without draining reproduces the
        exact finalized/pending/late state at the committed watermark. Read
        but uncommitted events never reached the ledger and must be reloaded
        by the caller (they surface as a cursor gap / unknown state).
        """
        engine = SessionEngine(spec, param_version)
        engine.apply_events(self.raw_events_for_version(param_version))
        cursors = {key: part.cursor for key, part in engine.parts.items()}
        return engine, cursors

    def watermark_log_rows(self, param_version: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM watermark_log WHERE param_version=? ORDER BY id",
            (param_version,)).fetchall())

    def late_log_rows(self, param_version: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM late_log WHERE param_version=? ORDER BY id",
            (param_version,)).fetchall())

    def rule_digest(self, spec: SessionSpec) -> str:
        payload = canonical_dumps(spec.digest_input())
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
