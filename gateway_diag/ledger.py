from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from .engine import RuleExecution, receipt_set_digest
from .errors import E_SNAPSHOT, GatewayError
from .graph import dependency_graph_digest
from .models import Spec
from .util import canonical_json, digest_value

FORMAT_VERSION = 1
LEDGER_FILENAME = "ledger.sqlite3"


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    base_spec_json TEXT,
    target_spec_json TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    request_count INTEGER NOT NULL,
    graph_digest TEXT NOT NULL,
    declaration_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    created_mono_ns INTEGER NOT NULL,
    finished_mono_ns INTEGER
);
CREATE TABLE IF NOT EXISTS rule_state (
    run_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    rule_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    receipt_digest TEXT,
    receipt_version INTEGER,
    batch_count INTEGER NOT NULL DEFAULT 0,
    distribution_json TEXT NOT NULL DEFAULT '{}',
    cursor INTEGER NOT NULL DEFAULT 0,
    updated_mono_ns INTEGER NOT NULL,
    PRIMARY KEY (run_id, rule_id)
);
CREATE TABLE IF NOT EXISTS receipts (
    run_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    receipt_json TEXT NOT NULL,
    receipt_digest TEXT NOT NULL,
    destination TEXT NOT NULL,
    backend_id TEXT NOT NULL,
    backend_param_digest TEXT NOT NULL,
    request_ids_json TEXT NOT NULL,
    output_digest TEXT NOT NULL,
    receipt_version INTEGER NOT NULL,
    committed_mono_ns INTEGER NOT NULL,
    PRIMARY KEY (run_id, rule_id, batch_index)
);
CREATE TABLE IF NOT EXISTS invalidations (
    run_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    code TEXT NOT NULL,
    basis_json TEXT NOT NULL,
    trace_digest TEXT NOT NULL,
    PRIMARY KEY (run_id, generation, step_index)
);
CREATE TABLE IF NOT EXISTS batch_commits (
    run_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    receipt_digest TEXT NOT NULL,
    backend_param_digest TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    trace_digest TEXT NOT NULL,
    cursor INTEGER NOT NULL,
    committed_mono_ns INTEGER NOT NULL,
    PRIMARY KEY (run_id, generation, rule_id, batch_index)
);
"""


class Ledger:
    def __init__(self, workdir: str | os.PathLike[str], *, read_only: bool = False) -> None:
        self.workdir = Path(workdir)
        self.path = self.workdir / LEDGER_FILENAME
        self.read_only = read_only
        if read_only:
            if not self.path.exists():
                raise GatewayError(E_SNAPSHOT, "ledger file does not exist", field="ledger", entry=str(self.path))
            uri = self.path.resolve().as_uri() + "?mode=ro"
            try:
                self.conn = sqlite3.connect(uri, uri=True)
            except sqlite3.DatabaseError as exc:
                raise GatewayError(E_SNAPSHOT, "cannot open ledger database", field="ledger", details={"reason": str(exc)})
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA query_only=ON")
        else:
            self.workdir.mkdir(parents=True, exist_ok=True)
            try:
                self.conn = sqlite3.connect(self.path, timeout=30)
                self.conn.row_factory = sqlite3.Row
                self.conn.execute("PRAGMA journal_mode=DELETE")
                self.conn.execute("PRAGMA synchronous=FULL")
                self.conn.execute("PRAGMA foreign_keys=ON")
                self.conn.executescript(SCHEMA)
                self._ensure_meta()
            except sqlite3.DatabaseError as exc:
                raise GatewayError(E_SNAPSHOT, "cannot initialize ledger database", field="ledger", details={"reason": str(exc)})
        try:
            self.verify_integrity()
        except Exception:
            self.conn.close()
            raise

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_meta(self) -> None:
        with self.conn:
            existing = self.conn.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
            if existing is None:
                self.conn.execute("INSERT INTO meta(key,value) VALUES('format_version',?)", (str(FORMAT_VERSION),))
                self.conn.execute("INSERT INTO meta(key,value) VALUES('active_run_id','')")
                self.conn.execute("INSERT INTO meta(key,value) VALUES('last_generation','0')")
                self.conn.execute("INSERT INTO meta(key,value) VALUES('content_digest','')")
                self._refresh_digest()

    def _meta_value(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else str(row[0])

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def format_version(self) -> int:
        try:
            return int(self._meta_value("format_version", "0"))
        except ValueError:
            return 0

    def active_run_id(self) -> str | None:
        value = self._meta_value("active_run_id")
        return value or None

    def verify_integrity(self) -> None:
        if self.format_version() != FORMAT_VERSION:
            raise GatewayError(
                E_SNAPSHOT,
                "ledger format version is unsupported",
                field="format_version",
                details={"expected": FORMAT_VERSION, "actual": self.format_version()},
            )
        stored = self._meta_value("content_digest")
        if not stored:
            raise GatewayError(E_SNAPSHOT, "ledger content digest is missing", field="content_digest")
        actual = self.compute_content_digest()
        if stored != actual:
            raise GatewayError(
                E_SNAPSHOT,
                "ledger content digest mismatch",
                field="content_digest",
                details={"stored": stored, "actual": actual},
            )

    def _table_rows(self, table: str, order_by: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}").fetchall()
        return [dict(row) for row in rows]

    def compute_content_digest(self) -> str:
        payload = {
            "format_version": self.format_version(),
            "active_run_id": self._meta_value("active_run_id"),
            "last_generation": self._meta_value("last_generation", "0"),
            "runs": self._table_rows("runs", "run_id"),
            "rule_state": self._table_rows("rule_state", "run_id, rule_id"),
            "receipts": self._table_rows("receipts", "run_id, rule_id, batch_index"),
            "invalidations": self._table_rows(
                "invalidations", "run_id, generation, step_index"
            ),
            "batch_commits": self._table_rows(
                "batch_commits", "run_id, generation, rule_id, batch_index"
            ),
        }
        return digest_value(payload)

    def _refresh_digest(self) -> None:
        if self.read_only:
            raise RuntimeError("read-only ledger cannot be updated")
        self._set_meta("content_digest", self.compute_content_digest())

    def get_run(self, run_id: str | None = None) -> dict[str, Any] | None:
        actual = run_id or self.active_run_id()
        if not actual:
            return None
        row = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (actual,)).fetchone()
        return dict(row) if row else None

    def get_rule_states(self, run_id: str | None = None) -> dict[str, dict[str, Any]]:
        actual = run_id or self.active_run_id()
        if not actual:
            return {}
        rows = self.conn.execute(
            "SELECT * FROM rule_state WHERE run_id=? ORDER BY rule_id", (actual,)
        ).fetchall()
        return {str(row["rule_id"]): dict(row) for row in rows}

    def get_rule_receipts(
        self, rule_id: str, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        actual = run_id or self.active_run_id()
        if not actual:
            return []
        rows = self.conn.execute(
            "SELECT receipt_json FROM receipts WHERE run_id=? AND rule_id=? ORDER BY batch_index",
            (actual, rule_id),
        ).fetchall()
        return [json.loads(row["receipt_json"]) for row in rows]

    def get_rule_receipt_records(
        self, rule_id: str, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        actual = run_id or self.active_run_id()
        if not actual:
            return []
        rows = self.conn.execute(
            "SELECT * FROM receipts WHERE run_id=? AND rule_id=? ORDER BY batch_index",
            (actual, rule_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def all_rule_receipts(self, run_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
        actual = run_id or self.active_run_id()
        if not actual:
            return {}
        rows = self.conn.execute(
            "SELECT rule_id, receipt_json FROM receipts WHERE run_id=? ORDER BY rule_id, batch_index",
            (actual,),
        ).fetchall()
        result: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            result.setdefault(str(row["rule_id"]), []).append(json.loads(row["receipt_json"]))
        return result

    def latest_generation(self, run_id: str | None = None) -> int:
        actual = run_id or self.active_run_id()
        if not actual:
            return 0
        row = self.conn.execute(
            "SELECT MAX(generation) AS generation FROM invalidations WHERE run_id=?", (actual,)
        ).fetchone()
        value = row["generation"] if row else None
        return int(value) if value is not None else 0

    def invalidation_payload(
        self, run_id: str | None = None, generation: int | None = None
    ) -> list[dict[str, Any]]:
        actual = run_id or self.active_run_id()
        if not actual:
            return []
        gen = generation if generation is not None else self.latest_generation(actual)
        rows = self.conn.execute(
            "SELECT * FROM invalidations WHERE run_id=? AND generation=? ORDER BY step_index",
            (actual, gen),
        ).fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "step": int(row["step_index"]),
                    "rule_id": row["rule_id"],
                    "code": row["code"],
                    "basis": json.loads(row["basis_json"]),
                    "trace_digest": row["trace_digest"],
                }
            )
        return result

    def _insert_invalidation(self, run_id: str, generation: int, payload: dict[str, Any]) -> None:
        steps = payload.get("invalidated_rules", [])
        for step in steps:
            step_index = int(step["step"])
            basis = step.get("basis", {})
            trace_digest = digest_value(
                {
                    "generation": generation,
                    "step": step_index,
                    "rule_id": step["rule_id"],
                    "code": step.get("code", "E_STALE"),
                    "basis": basis,
                }
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO invalidations(run_id,generation,step_index,rule_id,code,basis_json,trace_digest) VALUES(?,?,?,?,?,?,?)",
                (
                    run_id,
                    generation,
                    step_index,
                    step["rule_id"],
                    step.get("code", "E_STALE"),
                    canonical_json(basis),
                    trace_digest,
                ),
            )

    def create_run(
        self,
        spec: Spec,
        invalidation: dict[str, Any],
        *,
        mode: str = "full",
        base_spec: Spec | None = None,
    ) -> str:
        run_id = "run-" + uuid.uuid4().hex
        now = time.monotonic_ns()
        generation = 1
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs(run_id,mode,base_spec_json,target_spec_json,request_digest,request_count,graph_digest,declaration_digest,status,created_mono_ns,finished_mono_ns) VALUES(?,?,?,?,?,?,?,?,?,?,NULL)",
                (
                    run_id,
                    mode,
                    canonical_json(base_spec.declaration_payload()) if base_spec else None,
                    canonical_json(spec.declaration_payload()),
                    spec.request_digest,
                    spec.request_count,
                    dependency_graph_digest(spec),
                    spec.declaration_digest,
                    "running",
                    now,
                ),
            )
            for rule_id in spec.sorted_rule_ids:
                rule = spec.rules[rule_id]
                self.conn.execute(
                    "INSERT INTO rule_state(run_id,rule_id,rule_version,status,receipt_digest,receipt_version,batch_count,distribution_json,cursor,updated_mono_ns) VALUES(?,?,?,?,NULL,NULL,0,'{}',0,?)",
                    (run_id, rule_id, rule.version, "stale", now),
                )
            self._insert_invalidation(run_id, generation, invalidation)
            self._set_meta("active_run_id", run_id)
            self._set_meta("last_generation", str(generation))
            self._refresh_digest()
        return run_id

    def begin_update(
        self,
        spec: Spec,
        invalidation: dict[str, Any],
        *,
        base_spec: Spec,
    ) -> int:
        run = self.get_run()
        if run is None:
            raise GatewayError(E_SNAPSHOT, "active run is missing", field="run_id")
        run_id = str(run["run_id"])
        generation = self.latest_generation(run_id) + 1
        now = time.monotonic_ns()
        replacement = list(invalidation.get("replacement_rules", []))
        with self.conn:
            self._insert_invalidation(run_id, generation, invalidation)
            for rule_id in replacement:
                self.conn.execute(
                    "DELETE FROM receipts WHERE run_id=? AND rule_id=?", (run_id, rule_id)
                )
                self.conn.execute(
                    "DELETE FROM batch_commits WHERE run_id=? AND rule_id=?", (run_id, rule_id)
                )
            for rule_id in invalidation.get("removed_rules", []):
                self.conn.execute(
                    "DELETE FROM rule_state WHERE run_id=? AND rule_id=?", (run_id, rule_id)
                )
            for rule_id in spec.sorted_rule_ids:
                rule = spec.rules[rule_id]
                existing = self.conn.execute(
                    "SELECT 1 FROM rule_state WHERE run_id=? AND rule_id=?", (run_id, rule_id)
                ).fetchone()
                if existing is None:
                    self.conn.execute(
                        "INSERT INTO rule_state(run_id,rule_id,rule_version,status,receipt_digest,receipt_version,batch_count,distribution_json,cursor,updated_mono_ns) VALUES(?,?,?,?,NULL,NULL,0,'{}',0,?)",
                        (run_id, rule_id, rule.version, "stale", now),
                    )
                elif rule_id in replacement:
                    self.conn.execute(
                        "UPDATE rule_state SET rule_version=?,status='stale',receipt_digest=NULL,receipt_version=NULL,batch_count=0,distribution_json='{}',cursor=0,updated_mono_ns=? WHERE run_id=? AND rule_id=?",
                        (rule.version, now, run_id, rule_id),
                    )
            self.conn.execute(
                "UPDATE runs SET mode='update',base_spec_json=?,target_spec_json=?,request_digest=?,request_count=?,graph_digest=?,declaration_digest=?,status='running',finished_mono_ns=NULL WHERE run_id=?",
                (
                    canonical_json(base_spec.declaration_payload()),
                    canonical_json(spec.declaration_payload()),
                    spec.request_digest,
                    spec.request_count,
                    dependency_graph_digest(spec),
                    spec.declaration_digest,
                    run_id,
                ),
            )
            self._set_meta("last_generation", str(generation))
            self._refresh_digest()
        return generation

    def mark_in_progress_rules_unknown(self, run_id: str | None = None) -> list[str]:
        actual = run_id or self.active_run_id()
        if not actual:
            return []
        rows = self.conn.execute(
            "SELECT rule_id FROM rule_state WHERE run_id=? AND status IN ('in_progress','unknown') ORDER BY rule_id",
            (actual,),
        ).fetchall()
        ids = [str(row["rule_id"]) for row in rows]
        if not ids:
            return []
        now = time.monotonic_ns()
        with self.conn:
            for rule_id in ids:
                self.conn.execute(
                    "UPDATE rule_state SET status='unknown',updated_mono_ns=? WHERE run_id=? AND rule_id=?",
                    (now, actual, rule_id),
                )
            self._refresh_digest()
        return ids

    def mark_rule_started(self, rule_id: str, *, run_id: str | None = None) -> None:
        actual = run_id or self.active_run_id()
        if not actual:
            raise GatewayError(E_SNAPSHOT, "active run is missing", field="run_id")
        now = time.monotonic_ns()
        with self.conn:
            self.conn.execute(
                "UPDATE rule_state SET status='in_progress',cursor=0,receipt_digest=NULL,receipt_version=NULL,batch_count=0,distribution_json='{}',updated_mono_ns=? WHERE run_id=? AND rule_id=?",
                (now, actual, rule_id),
            )
            self._refresh_digest()

    def commit_batch(
        self,
        rule_id: str,
        receipt: dict[str, Any],
        *,
        generation: int,
        trace_payload: list[dict[str, Any]],
        cursor: int,
        run_id: str | None = None,
    ) -> None:
        actual = run_id or self.active_run_id()
        if not actual:
            raise GatewayError(E_SNAPSHOT, "active run is missing", field="run_id")
        trace_json = canonical_json(trace_payload)
        trace_digest = digest_value(trace_payload)
        now = time.monotonic_ns()
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO receipts(run_id,rule_id,batch_index,receipt_json,receipt_digest,destination,backend_id,backend_param_digest,request_ids_json,output_digest,receipt_version,committed_mono_ns) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    actual,
                    rule_id,
                    int(receipt["batch_index"]),
                    canonical_json(receipt),
                    receipt["receipt_digest"],
                    receipt["destination"],
                    receipt["backend_id"],
                    receipt["backend_param_digest"],
                    canonical_json(receipt["request_ids"]),
                    receipt["output_digest"],
                    int(receipt["receipt_version"]),
                    now,
                ),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO batch_commits(run_id,generation,rule_id,batch_index,receipt_digest,backend_param_digest,trace_json,trace_digest,cursor,committed_mono_ns) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    actual,
                    generation,
                    rule_id,
                    int(receipt["batch_index"]),
                    receipt["receipt_digest"],
                    receipt["backend_param_digest"],
                    trace_json,
                    trace_digest,
                    cursor,
                    now,
                ),
            )
            self.conn.execute(
                "UPDATE rule_state SET status='in_progress',cursor=?,batch_count=MAX(batch_count,?),updated_mono_ns=? WHERE run_id=? AND rule_id=?",
                (cursor, cursor, now, actual, rule_id),
            )
            self._refresh_digest()

    def complete_rule(
        self,
        execution: RuleExecution,
        *,
        generation: int,
        trace_payload: list[dict[str, Any]],
        run_id: str | None = None,
    ) -> None:
        actual = run_id or self.active_run_id()
        if not actual:
            raise GatewayError(E_SNAPSHOT, "active run is missing", field="run_id")
        batch_count = len(execution.receipts)
        trace_json = canonical_json(trace_payload)
        trace_digest = digest_value(trace_payload)
        now = time.monotonic_ns()
        with self.conn:
            self.conn.execute(
                "DELETE FROM receipts WHERE run_id=? AND rule_id=? AND batch_index>=?",
                (actual, execution.rule_id, batch_count),
            )
            self.conn.execute(
                "DELETE FROM batch_commits WHERE run_id=? AND rule_id=? AND batch_index>=?",
                (actual, execution.rule_id, batch_count),
            )
            self.conn.execute(
                "UPDATE rule_state SET rule_version=?,status='complete',receipt_digest=?,receipt_version=?,batch_count=?,distribution_json=?,cursor=?,updated_mono_ns=? WHERE run_id=? AND rule_id=?",
                (
                    execution.rule_version,
                    execution.receipt_digest,
                    execution.receipt_version,
                    batch_count,
                    canonical_json(execution.distribution),
                    batch_count,
                    now,
                    actual,
                    execution.rule_id,
                ),
            )
            # Completing a zero-batch rule intentionally has no receipt row but
            # still records a trace-bearing commit marker.
            if batch_count == 0:
                self.conn.execute(
                    "INSERT OR REPLACE INTO batch_commits(run_id,generation,rule_id,batch_index,receipt_digest,backend_param_digest,trace_json,trace_digest,cursor,committed_mono_ns) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        actual,
                        generation,
                        execution.rule_id,
                        -1,
                        execution.receipt_digest,
                        "",
                        trace_json,
                        trace_digest,
                        0,
                        now,
                    ),
                )
            self._refresh_digest()

    def finish_run(self, run_id: str | None = None) -> None:
        actual = run_id or self.active_run_id()
        if not actual:
            raise GatewayError(E_SNAPSHOT, "active run is missing", field="run_id")
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status='complete',finished_mono_ns=? WHERE run_id=?",
                (time.monotonic_ns(), actual),
            )
            self._refresh_digest()

    def validate_complete_rule_receipts(self, spec: Spec) -> None:
        run = self.get_run()
        if run is None:
            raise GatewayError(E_SNAPSHOT, "active run is missing", field="run_id")
        states = self.get_rule_states()
        for rule_id in spec.sorted_rule_ids:
            state = states.get(rule_id)
            if state is None or state["status"] != "complete":
                continue
            receipts = self.get_rule_receipts(rule_id)
            if len(receipts) != int(state["batch_count"]):
                raise GatewayError(
                    E_SNAPSHOT,
                    "completed rule batch count differs from persisted receipts",
                    rule_id=rule_id,
                    field="batch_count",
                )
            digest = receipt_set_digest(receipts)
            if digest != state["receipt_digest"]:
                raise GatewayError(
                    E_SNAPSHOT,
                    "completed rule receipt digest differs from persisted receipts",
                    rule_id=rule_id,
                    field="receipt_digest",
                )

    def snapshot_payload(self, spec: Spec) -> dict[str, Any]:
        run = self.get_run()
        if run is None:
            raise GatewayError(E_SNAPSHOT, "active run is missing", field="run_id")
        states = self.get_rule_states()
        completed = []
        for rule_id in spec.sorted_rule_ids:
            state = states.get(rule_id)
            if state and state["status"] == "complete":
                completed.append(
                    {
                        "rule_id": rule_id,
                        "rule_version": int(state["rule_version"]),
                        "receipt_digest": state["receipt_digest"],
                        "receipt_version": int(state["receipt_version"]),
                        "batch_count": int(state["batch_count"]),
                        "distribution": json.loads(state["distribution_json"]),
                    }
                )
        full_receipt_digest = digest_value(completed)
        return {
            "schema": "session-snapshot.v1",
            "generated_mono_ns": time.monotonic_ns(),
            "run_id": run["run_id"],
            "run_status": run["status"],
            "generation": self.latest_generation(),
            "declaration_digest": spec.declaration_digest,
            "rule_versions": spec.rule_versions,
            "backend_versions": spec.backend_versions,
            "backend_set_digest": spec.backend_set_digest,
            "dependency_graph_digest": dependency_graph_digest(spec),
            "request_digest": spec.request_digest,
            "request_count": spec.request_count,
            "spec": spec.declaration_payload(),
            "completed_rules": completed,
            "full_receipt_digest": full_receipt_digest,
            "ledger_content_digest": self._meta_value("content_digest"),
        }

    def get_latest_batch_commits(self, rule_id: str) -> dict[int, dict[str, Any]]:
        run_id = self.active_run_id()
        if not run_id:
            return {}
        rows = self.conn.execute(
            "SELECT * FROM batch_commits WHERE run_id=? AND rule_id=? ORDER BY generation DESC, batch_index",
            (run_id, rule_id),
        ).fetchall()
        result: dict[int, dict[str, Any]] = {}
        for row in rows:
            batch_index = int(row["batch_index"])
            if batch_index < 0 or batch_index in result:
                continue
            result[batch_index] = dict(row)
        return result

    def expected_receipts_for_rule(self, rule_id: str) -> dict[int, dict[str, Any]]:
        return {
            int(row["batch_index"]): json.loads(row["receipt_json"])
            for row in self.get_rule_receipt_records(rule_id)
        }
