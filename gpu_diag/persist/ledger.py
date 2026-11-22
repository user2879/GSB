"""SQLite session ledger with atomic per-round transactions.

Each round's shard traces, quota water levels, preemption records and the
progress cursor are written in ONE transaction.  A separate ``dispatch_journal``
records dispatch intent before the round commits: after a forced termination its
rows are reconciled into ``unknown_shards`` and never used to back-fill quota.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..digest import canonical_bytes, digest_of
from ..errors import E_LEDGER, DiagError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
    round_no INTEGER PRIMARY KEY,
    clock_start_ms INTEGER NOT NULL,
    clock_end_ms INTEGER NOT NULL,
    ready_set TEXT NOT NULL,
    admitted TEXT NOT NULL,
    running_set TEXT NOT NULL,
    queued TEXT NOT NULL,
    completed TEXT NOT NULL,
    decisions TEXT NOT NULL,
    water_levels TEXT NOT NULL,
    preemptions TEXT NOT NULL,
    round_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shard_traces (
    round_no INTEGER NOT NULL,
    shard_id TEXT NOT NULL,
    state TEXT NOT NULL,
    pool_id TEXT,
    digest TEXT,
    attempts INTEGER NOT NULL,
    PRIMARY KEY (round_no, shard_id)
);
CREATE TABLE IF NOT EXISTS dispatch_journal (
    shard_id TEXT PRIMARY KEY,
    round_no INTEGER NOT NULL,
    pool_id TEXT NOT NULL,
    committed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS unknown_shards (
    shard_id TEXT PRIMARY KEY,
    round_no INTEGER NOT NULL,
    pool_id TEXT,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_committed_round INTEGER NOT NULL,
    chain_digest TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, db_path: str | Path, format_version: int,
                 declaration_digest: str):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._put_meta("format_version", str(format_version))
        self._put_meta("declaration_digest", declaration_digest)
        self.conn.commit()

    def _put_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))

    def log_dispatch_intent(self, round_no: int, running: list[dict[str, Any]]) -> None:
        for item in running:
            self.conn.execute(
                "INSERT OR REPLACE INTO dispatch_journal"
                "(shard_id, round_no, pool_id, committed) VALUES(?,?,?,0)",
                (item["shard_id"], round_no, item["pool_id"]))

    def commit_round(self, round_record: dict[str, Any],
                     shard_states: dict[str, dict[str, Any]],
                     chain_digest: str) -> None:
        """Atomically persist one full scheduling round."""
        rn = round_record["round"]
        cur = self.conn
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute(
                "INSERT INTO rounds(round_no, clock_start_ms, clock_end_ms, "
                "ready_set, admitted, running_set, queued, completed, "
                "decisions, water_levels, preemptions, round_digest) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (rn, round_record["clock_start_ms"], round_record["clock_end_ms"],
                 canonical_bytes(round_record["ready_set"]).decode(),
                 canonical_bytes(round_record["admitted"]).decode(),
                 canonical_bytes(round_record["running_set"]).decode(),
                 canonical_bytes(round_record["queued"]).decode(),
                 canonical_bytes(round_record["completed"]).decode(),
                 canonical_bytes(round_record["decisions"]).decode(),
                 canonical_bytes(round_record["water_levels"]).decode(),
                 canonical_bytes(round_record["preemptions"]).decode(),
                 round_record["round_digest"]))
            for sid in sorted(shard_states):
                state = shard_states[sid]
                cur.execute(
                    "INSERT INTO shard_traces VALUES(?,?,?,?,?,?)",
                    (rn, sid, state["state"], state.get("pool_id"),
                     state.get("digest"), state["attempts"]))
            for item in round_record["running_set"]:
                cur.execute(
                    "UPDATE dispatch_journal SET committed=1 WHERE shard_id=?",
                    (item,))
            cur.execute("DELETE FROM dispatch_journal WHERE committed=1")
            cur.execute(
                "INSERT INTO progress(id,last_committed_round,chain_digest) "
                "VALUES(1,?,?) ON CONFLICT(id) DO UPDATE SET "
                "last_committed_round=excluded.last_committed_round, "
                "chain_digest=excluded.chain_digest",
                (rn, chain_digest))
            cur.commit()
        except Exception:
            cur.rollback()
            raise

    def reconcile_crash(self) -> list[dict[str, Any]]:
        """Mark uncommitted dispatch intents as UNKNOWN; never back-fill quota."""
        rows = self.conn.execute(
            "SELECT shard_id, round_no, pool_id FROM dispatch_journal "
            "WHERE committed=0 ORDER BY shard_id").fetchall()
        unknown: list[dict[str, Any]] = []
        for row in rows:
            reason = "dispatch intent present but round never committed; rerun required"
            self.conn.execute(
                "INSERT OR REPLACE INTO unknown_shards VALUES(?,?,?,?)",
                (row["shard_id"], row["round_no"], row["pool_id"], reason))
            self.conn.execute(
                "DELETE FROM dispatch_journal WHERE shard_id=?",
                (row["shard_id"],))
            unknown.append({"shard_id": row["shard_id"],
                            "round_no": row["round_no"],
                            "pool_id": row["pool_id"], "reason": reason})
        self.conn.commit()
        return unknown

    def last_committed_round(self) -> int:
        row = self.conn.execute(
            "SELECT last_committed_round FROM progress WHERE id=1").fetchone()
        return int(row["last_committed_round"]) if row else 0

    def load_rounds(self) -> list[dict[str, Any]]:
        import json
        rows = self.conn.execute("SELECT * FROM rounds ORDER BY round_no").fetchall()
        result = []
        for row in rows:
            result.append({
                "round": row["round_no"],
                "clock_start_ms": row["clock_start_ms"],
                "clock_end_ms": row["clock_end_ms"],
                "ready_set": json.loads(row["ready_set"]),
                "admitted": json.loads(row["admitted"]),
                "running_set": json.loads(row["running_set"]),
                "queued": json.loads(row["queued"]),
                "completed": json.loads(row["completed"]),
                "decisions": json.loads(row["decisions"]),
                "water_levels": json.loads(row["water_levels"]),
                "preemptions": json.loads(row["preemptions"]),
                "round_digest": row["round_digest"],
            })
        return result

    def verify_digests(self) -> None:
        """Re-hash every stored round and compare against the chained digest."""
        from ..digest import chain_digest
        chain = "0" * 64
        for record in self.load_rounds():
            stored = record["round_digest"]
            payload = {key: value for key, value in record.items()
                       if key != "round_digest"}
            expected = chain_digest(chain, payload)
            if expected != stored:
                raise DiagError(E_LEDGER, "ledger content digest mismatch", [{
                    "round": record["round"], "expected": expected,
                    "actual": stored, "field": "round_digest",
                }])
            chain = stored
        progress = self.conn.execute(
            "SELECT chain_digest FROM progress WHERE id=1").fetchone()
        if progress and progress["chain_digest"] != chain:
            raise DiagError(E_LEDGER, "progress cursor digest mismatch", [{
                "expected": chain, "actual": progress["chain_digest"],
            }])

    def tamper_round_water(self, round_no: int) -> None:
        """Test hook: corrupt a stored water level to force E_LEDGER."""
        row = self.conn.execute(
            "SELECT water_levels FROM rounds WHERE round_no=?",
            (round_no,)).fetchone()
        import json
        levels = json.loads(row["water_levels"])
        levels["__global__"]["gpu_used"] += 1
        self.conn.execute(
            "UPDATE rounds SET water_levels=? WHERE round_no=?",
            (canonical_bytes(levels).decode(), round_no))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
