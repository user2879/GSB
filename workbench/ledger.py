"""SQLite ledger: durable observations, fingerprints, progress and events."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from .canonical import combined_hash
from .engine import ShardObservation, StageObservation
from .errors import WorkbenchError
from .spec import PipelineSpec, canonical_spec_snapshot

SCHEMA_VERSION = 1
LEDGER_FILE = "ledger.sqlite3"
EVENTS_FILE = "events.jsonl"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fingerprints (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT NOT NULL UNIQUE,
    first_run_id INTEGER,
    first_seen_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_fingerprint TEXT NOT NULL,
    fingerprint_version INTEGER NOT NULL,
    spec_snapshot TEXT NOT NULL,
    spec_path TEXT NOT NULL,
    spec_dir TEXT NOT NULL,
    samples_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    evolution_json TEXT NOT NULL,
    error_code TEXT,
    error_stage_id TEXT,
    error_field_path TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    recovery_of INTEGER
);
CREATE TABLE IF NOT EXISTS run_stages (
    run_id INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    stage_id TEXT NOT NULL,
    stage_type TEXT NOT NULL,
    version INTEGER NOT NULL,
    stage_fingerprint TEXT NOT NULL,
    upstream_json TEXT NOT NULL,
    params_json TEXT NOT NULL,
    status TEXT NOT NULL,
    rows_in INTEGER,
    rows_out INTEGER,
    output_digest TEXT,
    column_stats_json TEXT,
    extra_json TEXT,
    snapshot_path TEXT,
    error_code TEXT,
    error_stage_id TEXT,
    error_field_path TEXT,
    PRIMARY KEY (run_id, ordinal),
    UNIQUE (run_id, stage_id)
);
CREATE TABLE IF NOT EXISTS shards (
    run_id INTEGER NOT NULL,
    stage_id TEXT NOT NULL,
    shard_index INTEGER NOT NULL,
    row_start INTEGER NOT NULL,
    row_end INTEGER NOT NULL,
    row_count INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    shard_fingerprint TEXT NOT NULL,
    source_stage_digest TEXT NOT NULL,
    target_rows INTEGER NOT NULL,
    remainder_rule TEXT NOT NULL,
    file_path TEXT NOT NULL,
    PRIMARY KEY (run_id, stage_id, shard_index)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence INTEGER NOT NULL,
    run_id INTEGER,
    code TEXT NOT NULL,
    stage_id TEXT NOT NULL,
    field_path TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    run_id INTEGER PRIMARY KEY,
    next_ordinal INTEGER NOT NULL,
    total_stages INTEGER NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Ledger:
    def __init__(self, workspace: str) -> None:
        self.workspace = os.path.abspath(workspace)
        os.makedirs(self.workspace, exist_ok=True)
        self.db_path = os.path.join(self.workspace, LEDGER_FILE)
        self.events_path = os.path.join(self.workspace, EVENTS_FILE)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------- basics
    def _state_get(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM ledger_state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def _state_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO ledger_state(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))

    def record_event(self, code: str, message: str,
                     run_id: Optional[int], stage_id: str = "",
                     field_path: str = "$",
                     details: Optional[dict[str, Any]] = None) -> None:
        sequence_row = self.conn.execute(
            "SELECT COALESCE(MAX(sequence),0)+1 AS next FROM events "
            "WHERE run_id IS ?", (run_id,)).fetchone()
        sequence = sequence_row["next"]
        created = utc_now_iso()
        self.conn.execute(
            "INSERT INTO events(sequence,run_id,code,stage_id,field_path,"
            "message,details_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (sequence, run_id, code, stage_id, field_path, message,
             json.dumps(details or {}, sort_keys=True, separators=(",", ":")),
             created))
        self.conn.commit()
        record = {"code": code, "created_at": created,
                  "details": details or {}, "field_path": field_path,
                  "message": message, "run_id": run_id, "sequence": sequence,
                  "stage_id": stage_id}
        with open(self.events_path, "a", encoding="utf-8",
                  newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True,
                                    separators=(",", ":")) + "\n")

    # ------------------------------------------------------- fingerprints
    def fingerprint_version(self, fingerprint: str) -> int:
        row = self.conn.execute(
            "SELECT version FROM fingerprints WHERE fingerprint=?",
            (fingerprint,)).fetchone()
        if row:
            return row["version"]
        return 0

    def _ensure_fingerprint(self, fingerprint: str, run_id: int) -> int:
        existing = self.fingerprint_version(fingerprint)
        if existing:
            return existing
        cur = self.conn.execute(
            "INSERT INTO fingerprints(fingerprint,first_run_id,first_seen_at) "
            "VALUES(?,?,?)", (fingerprint, run_id, utc_now_iso()))
        return int(cur.lastrowid)

    # ------------------------------------------------------------ run life
    def active_run(self) -> Optional[sqlite3.Row]:
        active_id = self._state_get("active_run_id")
        if active_id is None:
            return None
        return self.conn.execute("SELECT * FROM runs WHERE id=?",
                                 (int(active_id),)).fetchone()

    def create_run(self, spec: PipelineSpec, spec_path: str,
                   evolution: dict[str, Any],
                   recovery_of: Optional[int] = None) -> int:
        created = utc_now_iso()
        cur = self.conn.execute(
            "INSERT INTO runs(spec_fingerprint,fingerprint_version,"
            "spec_snapshot,spec_path,spec_dir,samples_digest,status,"
            "evolution_json,created_at,recovery_of) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (spec.fingerprint, 0,
             canonical_spec_snapshot(spec).decode("utf-8"),
             os.path.abspath(spec_path), os.path.dirname(os.path.abspath(
                 spec_path)),
             spec.samples_digest, "RUNNING",
             json.dumps(evolution, sort_keys=True, separators=(",", ":")),
             created, recovery_of))
        run_id = int(cur.lastrowid)
        fp_version = self._ensure_fingerprint(spec.fingerprint, run_id)
        self.conn.execute("UPDATE runs SET fingerprint_version=? WHERE id=?",
                          (fp_version, run_id))
        total = len(spec.stages)
        self.conn.execute(
            "INSERT INTO progress(run_id,next_ordinal,total_stages,status,"
            "updated_at) VALUES(?,?,?,?,?)",
            (run_id, 0, total, "RUNNING", created))
        self._state_set("active_run_id", str(run_id))
        self.conn.commit()
        return run_id

    def commit_stage(self, run_id: int, obs: StageObservation) -> None:
        # Single transaction per stage: stage row and progress advance commit
        # together, so a killed process never shows a half-written stage.
        with self.conn:
            self.conn.execute(
                "INSERT INTO run_stages(run_id,ordinal,stage_id,stage_type,"
                "version,stage_fingerprint,upstream_json,params_json,status,"
                "rows_in,rows_out,output_digest,column_stats_json,extra_json,"
                "snapshot_path) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, obs.ordinal, obs.stage_id, obs.stage_type,
                 obs.version, obs.stage_fingerprint,
                 json.dumps(obs.upstream), json.dumps(obs.params,
                                                      sort_keys=True),
                 obs.status, obs.rows_in, obs.rows_out, obs.output_digest,
                 json.dumps(obs.column_stats, sort_keys=True),
                 json.dumps(obs.extra, sort_keys=True), obs.snapshot_path))
            self.conn.execute(
                "UPDATE progress SET next_ordinal=?,updated_at=? WHERE run_id=?",
                (obs.ordinal + 1, utc_now_iso(), run_id))

    def commit_shards(self, run_id: int,
                      shards: list[ShardObservation]) -> None:
        with self.conn:
            for shard in shards:
                self.conn.execute(
                    "INSERT INTO shards(run_id,stage_id,shard_index,row_start,"
                    "row_end,row_count,content_digest,shard_fingerprint,"
                    "source_stage_digest,target_rows,remainder_rule,file_path) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, shard.stage_id, shard.shard_index,
                     shard.row_start, shard.row_end, shard.row_count,
                     shard.content_digest, shard.shard_fingerprint,
                     shard.source_stage_digest, shard.target_rows,
                     shard.remainder_rule, shard.file_path))

    def mark_unknown_stage(self, run_id: int, ordinal: int,
                           stage_id: str, stage_type: str, version: int,
                           stage_fingerprint: str, upstream: list[str],
                           params: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO run_stages(run_id,ordinal,stage_id,stage_type,"
                "version,stage_fingerprint,upstream_json,params_json,status) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, ordinal, stage_id, stage_type, version,
                 stage_fingerprint, json.dumps(upstream),
                 json.dumps(params, sort_keys=True), "UNKNOWN"))

    def complete_run(self, run_id: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status='COMPLETED',completed_at=? WHERE id=?",
                (utc_now_iso(), run_id))
            self.conn.execute(
                "UPDATE progress SET status='COMPLETED',updated_at=? "
                "WHERE run_id=?", (utc_now_iso(), run_id))

    def fail_run(self, run_id: int, error: WorkbenchError) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status='FAILED',error_code=?,error_stage_id=?,"
                "error_field_path=?,completed_at=? WHERE id=?",
                (error.code, error.stage_id, error.field_path,
                 utc_now_iso(), run_id))
            self.conn.execute(
                "UPDATE progress SET status='FAILED',updated_at=? WHERE run_id=?",
                (utc_now_iso(), run_id))

    # ------------------------------------------------------------- queries
    def get_run(self, run_id: int) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM runs WHERE id=?",
                                (run_id,)).fetchone()
        if row is None:
            raise WorkbenchError("E_REPLAY", f"unknown run id {run_id}",
                                 field_path="$.runs")
        return row

    def latest_completed_run(self) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM runs WHERE status='COMPLETED' ORDER BY id DESC "
            "LIMIT 1").fetchone()

    def stage_rows(self, run_id: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM run_stages WHERE run_id=? ORDER BY ordinal",
            (run_id,)))

    def shard_rows(self, run_id: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM shards WHERE run_id=? ORDER BY stage_id,shard_index",
            (run_id,)))

    def event_rows(self, run_id: Optional[int]) -> list[sqlite3.Row]:
        if run_id is None:
            return list(self.conn.execute(
                "SELECT * FROM events ORDER BY id"))
        return list(self.conn.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY sequence", (run_id,)))

    def all_runs(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM runs ORDER BY id"))

    def verify_shard_binding(self, run_id: int) -> None:
        """Recompute every shard fingerprint and compare with the ledger."""
        stages = {row["stage_id"]: row for row in self.stage_rows(run_id)}
        for row in self.shard_rows(run_id):
            source = stages.get(row["stage_id"])
            if source is None or source["output_digest"] != \
                    row["source_stage_digest"]:
                raise WorkbenchError(
                    "E_SHARD",
                    f"shard {row['shard_index']:04d} source digest unbound",
                    row["stage_id"],
                    f"$.shards[{row['shard_index']}].source_stage_digest")
            with open(row["file_path"], "r", encoding="utf-8") as handle:
                lines = [line.strip().encode("utf-8")
                         for line in handle if line.strip()]
            from .canonical import chained_line_hash
            content_digest = chained_line_hash("shard-content-v1",
                                               iter(lines))
            expected = combined_hash("shard-fingerprint-v1", (
                ("stage_id", row["stage_id"]),
                ("shard_index", row["shard_index"]),
                ("row_start", row["row_start"]),
                ("row_end", row["row_end"]),
                ("target_rows", row["target_rows"]),
                ("remainder_rule", row["remainder_rule"]),
                ("source_stage_digest", row["source_stage_digest"]),
                ("content_digest", content_digest),
            ))
            if content_digest != row["content_digest"] or \
                    expected != row["shard_fingerprint"]:
                raise WorkbenchError(
                    "E_SHARD",
                    f"shard {row['shard_index']:04d} fingerprint mismatch",
                    row["stage_id"],
                    f"$.shards[{row['shard_index']}].shard_fingerprint",
                    {"shard_index": row["shard_index"],
                     "expected": expected,
                     "ledger": row["shard_fingerprint"]})

    def view_model(self, run_id: Optional[int] = None) -> dict[str, Any]:
        """Stable, serializable snapshot used by the Tk UI and ui-dump."""
        if run_id is None:
            run = self.latest_completed_run() or self.active_run()
            if run is None:
                return {"runs": [], "selected_run": None, "stages": [],
                        "shards": [], "events": [], "fingerprints": []}
            run_id = run["id"]
        run = self.get_run(run_id)
        stages = []
        for row in self.stage_rows(run_id):
            stages.append({
                "ordinal": row["ordinal"],
                "stage_id": row["stage_id"],
                "stage_type": row["stage_type"],
                "depends_on": ",".join(json.loads(row["upstream_json"])) or "-",
                "status": row["status"],
                "rows_in": row["rows_in"],
                "rows_out": row["rows_out"],
                "output_digest": row["output_digest"] or "",
                "column_stats": json.loads(row["column_stats_json"])
                if row["column_stats_json"] else {},
                "extra": json.loads(row["extra_json"])
                if row["extra_json"] else {},
            })
        shards = [dict(row) for row in self.shard_rows(run_id)]
        events = [dict(row) for row in self.event_rows(run_id)]
        fingerprints = [dict(row) for row in self.conn.execute(
            "SELECT * FROM fingerprints ORDER BY version")]
        return {
            "selected_run": run_id,
            "run": dict(run),
            "runs": [dict(row) for row in self.all_runs()],
            "stages": stages,
            "shards": shards,
            "events": events,
            "fingerprints": fingerprints,
        }
