"""SQLite scheduling ledger with atomic round writes and digest verification.

Every scheduling round is persisted in a single SQLite transaction covering its
trace events, pool quota usage, preemption record and progress cursor.  Rows
are written in explicit sorted order and each round carries a SHA-256 digest
chained to the previous round.  Reopening a ledger recomputes the whole chain;
any mismatch is reported as ``E_LEDGER`` instead of silently trusting stale
state.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, List, Optional

from .errors import E_LEDGER, Problem, SchedulerError
from .events import (
    PoolUsage,
    PreemptionRecord,
    RoundRecord,
    TraceEvent,
)
from .model import Declaration, canonical_json, content_digest

LEDGER_FORMAT_VERSION = 1
META_FORMAT_VERSION = "format_version"
META_LEDGER_DIGEST = "ledger_digest"
META_POOL_VERSION = "pool_version"
META_TASK_VERSION = "task_version"
META_CONFIG_VERSION = "config_version"
META_SCALE = "fixed_point_scale"


class Ledger:
    """Thin, deterministic SQLite wrapper used by the orchestrating scheduler."""

    def __init__(self, path: str) -> None:
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def initialize(self, declaration: Declaration) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rounds (
                round_no INTEGER PRIMARY KEY,
                clock_ms INTEGER NOT NULL,
                running TEXT NOT NULL,
                waiting TEXT NOT NULL,
                global_gpu_busy INTEGER NOT NULL,
                global_shares_busy INTEGER NOT NULL,
                prev_digest TEXT NOT NULL,
                round_digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trace_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                round_no INTEGER NOT NULL,
                clock_ms INTEGER NOT NULL,
                kind TEXT NOT NULL,
                task_id TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                pool_id TEXT,
                gpu_count INTEGER NOT NULL,
                memory_mib INTEGER NOT NULL,
                detail TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS preemptions (
                round_no INTEGER PRIMARY KEY,
                clock_ms INTEGER NOT NULL,
                trigger_task_id TEXT NOT NULL,
                trigger_shard_index INTEGER NOT NULL,
                victim_task_id TEXT NOT NULL,
                victim_shard_index INTEGER NOT NULL,
                pool_id TEXT NOT NULL,
                released_gpu INTEGER NOT NULL,
                released_memory_mib INTEGER NOT NULL,
                trigger_priority INTEGER NOT NULL,
                victim_priority INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pool_usage (
                round_no INTEGER NOT NULL,
                pool_id TEXT NOT NULL,
                gpu_busy INTEGER NOT NULL,
                gpu_capacity INTEGER NOT NULL,
                memory_busy_mib INTEGER NOT NULL,
                memory_capacity_mib INTEGER NOT NULL,
                shares_busy INTEGER NOT NULL,
                shares_capacity INTEGER NOT NULL,
                concurrent_tasks INTEGER NOT NULL,
                concurrent_cap INTEGER NOT NULL,
                PRIMARY KEY (round_no, pool_id)
            );
            CREATE TABLE IF NOT EXISTS progress_cursor (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_round INTEGER NOT NULL,
                last_clock INTEGER NOT NULL,
                running_shards TEXT NOT NULL,
                ledger_digest TEXT NOT NULL
            );
            """
        )

        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS shard_states (
                round_no INTEGER NOT NULL,
                task_id TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                state TEXT NOT NULL,
                version INTEGER NOT NULL,
                output_digest TEXT,
                pool_id TEXT,
                started_ms INTEGER,
                finish_ms INTEGER,
                ended_ms INTEGER,
                wait_duration_ms INTEGER NOT NULL,
                attempts INTEGER NOT NULL,
                PRIMARY KEY (round_no, task_id, shard_index)
            );
            """
        )
        self._write_meta({
            META_FORMAT_VERSION: str(LEDGER_FORMAT_VERSION),
            META_POOL_VERSION: declaration.pool_version,
            META_TASK_VERSION: declaration.task_version,
            META_CONFIG_VERSION: declaration.config_version,
            META_SCALE: str(declaration.config.fixed_point_scale),
        })
        self.conn.commit()

    def _write_meta(self, values: Dict[str, str]) -> None:
        for key in sorted(values):
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, values[key]),
            )

    def read_meta(self) -> Dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM meta").fetchall()
        return {row["key"]: row["value"] for row in rows}

    # ------------------------------------------------------------------
    # Atomic round append
    # ------------------------------------------------------------------
    def append_round(self, record: RoundRecord,
                     prev_digest: str,
                     shard_states: Optional[List[dict]] = None) -> str:
        digest = self._round_digest(record, prev_digest)
        try:
            self.conn.execute("BEGIN")
            self.conn.execute(
                "INSERT INTO rounds(round_no, clock_ms, running, waiting, "
                "global_gpu_busy, global_shares_busy, prev_digest, round_digest) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (record.round_no, record.clock_ms,
                 canonical_json(record.running), canonical_json(record.waiting),
                 record.global_gpu_busy, record.global_shares_busy,
                 prev_digest, digest),
            )
            for event in sorted(record.events,
                                key=lambda item: (item.clock_ms, item.kind,
                                                  item.task_id,
                                                  item.shard_index)):
                self.conn.execute(
                    "INSERT INTO trace_events(round_no, clock_ms, kind, task_id,"
                    " shard_index, pool_id, gpu_count, memory_mib, detail) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (event.round_no, event.clock_ms, event.kind, event.task_id,
                     event.shard_index, event.pool_id, event.gpu_count,
                     event.memory_mib, event.detail),
                )
            if record.preemption is not None:
                pre = record.preemption
                self.conn.execute(
                    "INSERT INTO preemptions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (pre.round_no, pre.clock_ms, pre.trigger_task_id,
                     pre.trigger_shard_index, pre.victim_task_id,
                     pre.victim_shard_index, pre.pool_id, pre.released_gpu,
                     pre.released_memory_mib, pre.trigger_priority,
                     pre.victim_priority),
                )
            for usage in sorted(record.pool_usage, key=lambda item: item.pool_id):
                self.conn.execute(
                    "INSERT INTO pool_usage VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (record.round_no, usage.pool_id, usage.gpu_busy,
                     usage.gpu_capacity, usage.memory_busy_mib,
                     usage.memory_capacity_mib, usage.shares_busy,
                     usage.shares_capacity, usage.concurrent_tasks,
                     usage.concurrent_cap),
                )
            for state in sorted(shard_states or [],
                                key=lambda item: (item["task_id"],
                                                  item["shard_index"])):
                self.conn.execute(
                    "INSERT INTO shard_states(round_no, task_id, shard_index, "
                    "state, version, output_digest, pool_id, started_ms, "
                    "finish_ms, ended_ms, wait_duration_ms, attempts) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (record.round_no, state["task_id"], state["shard_index"],
                     state["state"], state["version"], state["output_digest"],
                     state["pool_id"], state["started_ms"], state["finish_ms"],
                     state["ended_ms"], state["wait_duration_ms"],
                     state["attempts"]))
            running_shards = canonical_json(record.running)
            self.conn.execute(
                "INSERT INTO progress_cursor(id, last_round, last_clock, "
                "running_shards, ledger_digest) VALUES(1,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET last_round=excluded.last_round, "
                "last_clock=excluded.last_clock, "
                "running_shards=excluded.running_shards, "
                "ledger_digest=excluded.ledger_digest",
                (record.round_no, record.clock_ms, running_shards, digest),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return digest

    def _round_digest(self, record: RoundRecord, prev_digest: str) -> str:
        # Canonicalize event order exactly as rows are persisted so the digest
        # recomputed after a SQLite round trip matches the stored digest.
        canonical_round = record.to_dict()
        canonical_round["events"] = [
            event.to_dict() for event in sorted(
                record.events,
                key=lambda item: (item.clock_ms, item.kind, item.task_id,
                                  item.shard_index))
        ]
        canonical_round["pool_usage"] = [
            usage.to_dict() for usage in
            sorted(record.pool_usage, key=lambda item: item.pool_id)
        ]
        payload = {"prev_digest": prev_digest, "round": canonical_round}
        return content_digest(payload)

    # ------------------------------------------------------------------
    # Read back and verification
    # ------------------------------------------------------------------
    def last_round_no(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(round_no), 0) AS n FROM rounds").fetchone()
        return int(row["n"])

    def last_cursor(self) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM progress_cursor WHERE id = 1").fetchone()

    def latest_shard_states(self) -> List[dict]:
        last_round = self.last_round_no()
        if last_round == 0:
            return []
        rows = self.conn.execute(
            "SELECT * FROM shard_states WHERE round_no=? "
            "ORDER BY task_id, shard_index", (last_round,)).fetchall()
        return [dict(row) for row in rows]

    def running_at_last_round(self) -> List[str]:
        cursor = self.last_cursor()
        if cursor is None:
            return []
        return json.loads(cursor["running_shards"])

    def load_rounds(self) -> List[RoundRecord]:
        rounds: List[RoundRecord] = []
        round_rows = self.conn.execute(
            "SELECT * FROM rounds ORDER BY round_no").fetchall()
        for row in round_rows:
            events = [
                TraceEvent(
                    round_no=item["round_no"], clock_ms=item["clock_ms"],
                    kind=item["kind"], task_id=item["task_id"],
                    shard_index=item["shard_index"], pool_id=item["pool_id"],
                    gpu_count=item["gpu_count"],
                    memory_mib=item["memory_mib"], detail=item["detail"])
                for item in self.conn.execute(
                    "SELECT * FROM trace_events WHERE round_no=? "
                    "ORDER BY seq", (row["round_no"],)).fetchall()
            ]
            pre_row = self.conn.execute(
                "SELECT * FROM preemptions WHERE round_no=?",
                (row["round_no"],)).fetchone()
            preemption = None
            if pre_row is not None:
                preemption = PreemptionRecord(**dict(pre_row))
            usage = []
            for item in self.conn.execute(
                    "SELECT * FROM pool_usage WHERE round_no=? ORDER BY pool_id",
                    (row["round_no"],)).fetchall():
                usage_payload = dict(item)
                usage_payload.pop("round_no", None)
                usage.append(PoolUsage(**usage_payload))
            rounds.append(RoundRecord(
                round_no=row["round_no"], clock_ms=row["clock_ms"],
                events=events, preemption=preemption, pool_usage=usage,
                running=json.loads(row["running"]),
                waiting=json.loads(row["waiting"]),
                global_gpu_busy=row["global_gpu_busy"],
                global_shares_busy=row["global_shares_busy"]))
        return rounds

    def verify(self) -> str:
        """Recompute the digest chain and meta summary; raise E_LEDGER on drift."""
        meta = self.read_meta()
        if int(meta.get(META_FORMAT_VERSION, "0")) != LEDGER_FORMAT_VERSION:
            raise SchedulerError(
                E_LEDGER, "ledger format version mismatch",
                [Problem("meta.format_version",
                         f"== {LEDGER_FORMAT_VERSION}",
                         meta.get(META_FORMAT_VERSION), E_LEDGER)])
        prev_digest = ""
        for record in self.load_rounds():
            expected = self._round_digest(record, prev_digest)
            stored = self.conn.execute(
                "SELECT round_digest FROM rounds WHERE round_no=?",
                (record.round_no,)).fetchone()["round_digest"]
            if stored != expected:
                raise SchedulerError(
                    E_LEDGER, "ledger content digest mismatch",
                    [Problem(f"rounds[{record.round_no}].round_digest",
                             expected, stored, E_LEDGER)])
            prev_digest = expected
        cursor = self.last_cursor()
        if cursor is not None and cursor["ledger_digest"] != prev_digest:
            raise SchedulerError(
                E_LEDGER, "progress cursor digest mismatch",
                [Problem("progress_cursor.ledger_digest", prev_digest,
                         cursor["ledger_digest"], E_LEDGER)])
        return prev_digest

    def close(self) -> None:
        self.conn.close()
