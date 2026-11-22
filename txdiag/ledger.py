"""SQLite ledger, whole-snapshot replacement, crash recovery, and replay checks."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, canonical_text, digest, structured_diff
from .errors import ReplayError
from .model import RuntimeConfig, Spec


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
    round_no INTEGER PRIMARY KEY,
    format_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    observed_json TEXT NOT NULL,
    committed_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    round_no INTEGER NOT NULL,
    event_seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    txid TEXT,
    row_key TEXT,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (round_no, event_seq)
);
CREATE TABLE IF NOT EXISTS inflight (
    round_no INTEGER PRIMARY KEY,
    running_json TEXT NOT NULL,
    started_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    format_version INTEGER NOT NULL,
    declaration_sha256 TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    state_sha256 TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS recoveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_ns INTEGER NOT NULL,
    inflight_round INTEGER,
    running_json TEXT NOT NULL,
    mechanism TEXT NOT NULL
);
"""


def declaration_digest(spec: Spec, config: RuntimeConfig, path: str) -> str:
    config_dict = config.as_dict()
    config_dict.pop("inject", None)
    config_dict["effective_path"] = path
    return digest({
        "format_version": 1,
        "spec_sha256": spec.spec_sha256,
        "config_sha256": digest(config_dict),
        "reference": {
            "transaction_count": len(spec.transactions),
            "row_key_count": len(spec.row_keys),
            "dependency_edge_count": len(spec.dependency_edges),
        },
    })


def config_digest(config: RuntimeConfig, path: str) -> str:
    data = config.as_dict()
    data.pop("inject", None)
    data["effective_path"] = path
    return digest(data)


class Ledger:
    def __init__(self, ledger_path: str | Path, snapshot_path: str | Path,
                 spec: Spec, config: RuntimeConfig, path: str) -> None:
        self.ledger_path = Path(ledger_path).resolve()
        self.snapshot_path = Path(snapshot_path).resolve()
        self.spec = spec
        self.config = config
        self.path = path
        self.declaration_sha256 = declaration_digest(spec, config, path)
        self.config_sha256 = config_digest(config, path)
        self.conn: sqlite3.Connection | None = None
        self.expected_rounds: dict[int, dict[str, Any]] = {}
        self.replacement: dict[str, Any] | None = None
        self.recovery: dict[str, Any] | None = None

    # ---- lifecycle ---------------------------------------------------------------
    def open(self, *, allow_replacement: bool = True) -> dict[str, Any]:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.ledger_path), isolation_level=None,
                                    timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute("INSERT OR IGNORE INTO ledger_meta(key,value) VALUES('schema_version','1')")

        snapshot = self._read_snapshot_file()
        db_session = self._meta("session_digest")
        diffs: list[dict[str, Any]] = []
        if snapshot is not None:
            expected = {
                "format_version": 1,
                "declaration_sha256": self.declaration_sha256,
                "content_sha256": self.declaration_sha256,
            }
            actual = {
                "format_version": snapshot.get("format_version"),
                "declaration_sha256": snapshot.get("declaration_sha256"),
                "content_sha256": snapshot.get("content_sha256"),
            }
            diffs.extend(structured_diff(actual, expected, "$.snapshot"))
            state = snapshot.get("state")
            if digest(state) != snapshot.get("state_sha256"):
                diffs.append({"path": "$.snapshot.state_sha256",
                              "left": snapshot.get("state_sha256"),
                              "right": digest(state)})
        if db_session is not None and db_session != self.declaration_sha256:
            diffs.append({"path": "$.ledger.session_digest",
                          "left": db_session, "right": self.declaration_sha256})
        if snapshot is None and (db_session is not None or self._round_count() > 0):
            diffs.append({"path": "$.snapshot", "left": "<missing>", "right": "required"})

        if diffs:
            if allow_replacement:
                self._replace_session(diffs)
            else:
                raise ReplayError("snapshot/declaration mismatch; replacement disabled",
                                  field=diffs[0]["path"], details={"differences": diffs})
        elif db_session is None:
            self._initialize_session()
        self.recovery = self._recover_inflight()
        self.expected_rounds = self._load_rounds()
        return {
            "ledger_path": str(self.ledger_path),
            "snapshot_path": str(self.snapshot_path),
            "declaration_sha256": self.declaration_sha256,
            "replaced": self.replacement is not None,
            "replacement": self.replacement,
            "recovery": self.recovery,
            "committed_rounds": len(self.expected_rounds),
            "complete": self._meta("complete") == "1",
        }

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _conn(self) -> sqlite3.Connection:
        if self.conn is None:
            raise RuntimeError("ledger is not open")
        return self.conn

    def _meta(self, key: str) -> str | None:
        row = self._conn().execute(
            "SELECT value FROM ledger_meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._conn().execute(
            "INSERT INTO ledger_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def _round_count(self) -> int:
        row = self._conn().execute("SELECT COUNT(*) AS n FROM rounds").fetchone()
        return int(row["n"])

    def _initialize_session(self) -> None:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._set_meta("schema_version", "1")
            self._set_meta("session_digest", self.declaration_sha256)
            self._set_meta("spec_sha256", self.spec.spec_sha256)
            self._set_meta("config_sha256", self.config_sha256)
            self._set_meta("effective_path", self.path)
            self._set_meta("config_json", canonical_text(self.config.as_dict()))
            self._set_meta("complete", "0")
            self._set_meta("rerun_required", "0")
            self._set_meta("unknown_txids", "[]")
            self._set_meta("last_committed_round", "0")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        self._write_snapshot({
            "phase": "initialized",
            "last_committed_round": 0,
            "spec_sha256": self.spec.spec_sha256,
            "path": self.path,
        })

    def _replace_session(self, diffs: list[dict[str, Any]]) -> None:
        conn = self._conn()
        before = {
            "format_version": 1,
            "session_digest": self._meta("session_digest"),
            "round_count": self._round_count(),
        }
        conn.execute("BEGIN IMMEDIATE")
        try:
            for table in ("rounds", "events", "inflight", "snapshots", "recoveries"):
                conn.execute(f"DELETE FROM {table}")
            conn.execute("DELETE FROM ledger_meta")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        self.replacement = {
            "action": "whole_snapshot_replaced",
            "stable_differences": sorted(diffs, key=lambda item: item["path"]),
            "before": before,
            "after": {
                "format_version": 1,
                "session_digest": self.declaration_sha256,
                "round_count": 0,
            },
        }
        self._initialize_session()

    # ---- snapshots ---------------------------------------------------------------
    def _read_snapshot_file(self) -> dict[str, Any] | None:
        if not self.snapshot_path.exists():
            return None
        try:
            raw = self.snapshot_path.read_text(encoding="utf-8")
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _write_snapshot(self, state: dict[str, Any], *, db_tx: bool = False) -> dict[str, Any]:
        state_json = canonical_text(state)
        document = {
            "format_version": 1,
            "declaration_sha256": self.declaration_sha256,
            "content_sha256": self.declaration_sha256,
            "state_sha256": digest(state),
            "created_ns": time.time_ns(),
            "state": state,
        }
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".session-", suffix=".snapshot.tmp", dir=str(self.snapshot_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_text(document))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.snapshot_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        conn = self._conn()
        if not db_tx:
            conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO snapshots(id,format_version,declaration_sha256,content_sha256,"
                "state_sha256,state_json,created_ns) VALUES(1,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET format_version=excluded.format_version,"
                "declaration_sha256=excluded.declaration_sha256,"
                "content_sha256=excluded.content_sha256,state_sha256=excluded.state_sha256,"
                "state_json=excluded.state_json,created_ns=excluded.created_ns",
                (1, self.declaration_sha256, self.declaration_sha256, digest(state),
                 state_json, document["created_ns"]),
            )
            if not db_tx:
                conn.execute("COMMIT")
        except BaseException:
            if not db_tx:
                conn.execute("ROLLBACK")
            raise
        return document

    def read_snapshot(self) -> dict[str, Any] | None:
        return self._read_snapshot_file()

    # ---- recovery ----------------------------------------------------------------
    def _recover_inflight(self) -> dict[str, Any]:
        rows = self._conn().execute(
            "SELECT round_no,running_json,started_ns FROM inflight ORDER BY round_no").fetchall()
        unknown: list[str] = []
        details: list[dict[str, Any]] = []
        for row in rows:
            running = json.loads(row["running_json"])
            unknown.extend(running)
            details.append({
                "round": row["round_no"],
                "running": running,
                "started_monotonic_ns": row["started_ns"],
            })
        unknown = sorted(set(unknown))
        if rows:
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                for row in rows:
                    conn.execute(
                        "INSERT INTO recoveries(detected_ns,inflight_round,running_json,mechanism) "
                        "VALUES(?,?,?,?)",
                        (time.time_ns(), row["round_no"], row["running_json"],
                         "uncommitted_inflight_marked_unknown"),
                    )
                conn.execute("DELETE FROM inflight")
                self._set_meta("rerun_required", "1")
                self._set_meta("unknown_txids", canonical_text(unknown))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        elif self._meta("complete") != "1" and self._round_count() > 0:
            last_payload = None
            last_round = 0
            for row in self._conn().execute(
                    "SELECT round_no,payload_json FROM rounds ORDER BY round_no DESC LIMIT 1"):
                last_round = int(row["round_no"])
                last_payload = json.loads(row["payload_json"])
            if last_payload is not None:
                completed = {
                    event.get("txid")
                    for event in last_payload.get("events", [])
                    if event.get("event") in ("commit", "abort") and event.get("txid")
                }
                unknown = sorted(set(last_payload.get("running", [])) - completed)
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._set_meta("rerun_required", "1")
                self._set_meta("unknown_txids", canonical_text(unknown))
                conn.execute(
                    "INSERT INTO recoveries(detected_ns,inflight_round,running_json,mechanism) "
                    "VALUES(?,?,?,?)",
                    (time.time_ns(), last_round or None, canonical_text(unknown),
                     "incomplete_at_committed_round_boundary"),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return {
            "had_inflight": bool(rows),
            "inflight": details,
            "unknown_txids": unknown,
            "rerun_required": bool(rows) or (self._meta("complete") != "1" and self._round_count() > 0),
            "exit_report_available": self._meta("complete") == "1" and not rows,
        }

    # ---- rounds ------------------------------------------------------------------
    def _load_rounds(self) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for row in self._conn().execute(
                "SELECT round_no,payload_json FROM rounds ORDER BY round_no"):
            out[int(row["round_no"])] = json.loads(row["payload_json"])
        return out

    def begin_round(self, round_no: int, running: list[str]) -> None:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO inflight(round_no,running_json,started_ns) VALUES(?,?,?) "
                "ON CONFLICT(round_no) DO UPDATE SET running_json=excluded.running_json,"
                "started_ns=excluded.started_ns",
                (round_no, canonical_text(running), time.monotonic_ns()),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def commit_round(self, payload: dict[str, Any], observed: dict[str, Any]) -> None:
        round_no = int(payload["round"])
        payload_json = canonical_text(payload)
        content_sha = digest(payload)
        existing = self.expected_rounds.get(round_no)
        if existing is not None:
            if canonical_bytes(existing) != canonical_bytes(payload):
                raise _round_replay_error(round_no, existing, payload)
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM inflight WHERE round_no=?", (round_no,))
                self._set_meta("last_committed_round", str(round_no))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            return

        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO rounds(round_no,format_version,payload_json,content_sha256,"
                "observed_json,committed_ns) VALUES(?,?,?,?,?,?)",
                (round_no, 1, payload_json, content_sha, canonical_text(observed),
                 time.time_ns()),
            )
            for event in payload.get("events", []):
                conn.execute(
                    "INSERT INTO events(round_no,event_seq,event_type,txid,row_key,payload_json) "
                    "VALUES(?,?,?,?,?,?)",
                    (round_no, int(event.get("event_seq", 0)), str(event.get("event", "unknown")),
                     event.get("txid"), event.get("row"), canonical_text(event)),
                )
            conn.execute("DELETE FROM inflight WHERE round_no=?", (round_no,))
            self._set_meta("last_committed_round", str(round_no))
            self._set_meta("rerun_required", "1")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        self.expected_rounds[round_no] = payload
        self._write_snapshot({
            "phase": "round_committed",
            "last_committed_round": round_no,
            "round_sha256": content_sha,
            "running": payload.get("running", []),
            "quota": payload.get("quota", {}),
            "wait_budget": payload.get("wait_budget", {}),
            "spec_sha256": self.spec.spec_sha256,
            "path": self.path,
        })

    def complete(self, result: Any) -> None:
        generated_rounds = {int(payload["round"]) for payload in result.rounds}
        expected_rounds = set(self.expected_rounds)
        extra = sorted(expected_rounds - generated_rounds)
        if extra:
            first = min(extra)
            raise ReplayError("replay ended before a committed ledger round",
                              field=f"$.rounds[{first}]",
                              details={"missing_rounds": extra})
        state = {
            "phase": "complete",
            "last_committed_round": max(generated_rounds, default=0),
            "status": result.status,
            "path": result.path,
            "spec_sha256": self.spec.spec_sha256,
            "result": result.result,
            "range_plans": result.range_plan_rounds,
            "observed_rounds": result.observed_rounds,
        }
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._set_meta("complete", "1")
            self._set_meta("rerun_required", "0")
            self._set_meta("unknown_txids", "[]")
            self._set_meta("last_committed_round", str(max(generated_rounds, default=0)))
            self._write_snapshot(state, db_tx=True)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def flush(self) -> None:
        self._conn().commit()

    def read_rounds(self) -> list[dict[str, Any]]:
        return [self.expected_rounds[key] for key in sorted(self.expected_rounds)]

    def session_info(self) -> dict[str, Any]:
        return {
            "session_digest": self.declaration_sha256,
            "spec_sha256": self.spec.spec_sha256,
            "config_sha256": self.config_sha256,
            "path": self.path,
            "complete": self._meta("complete") == "1",
            "rerun_required": self._meta("rerun_required") == "1",
            "unknown_txids": json.loads(self._meta("unknown_txids") or "[]"),
            "last_committed_round": int(self._meta("last_committed_round") or "0"),
            "round_count": len(self.expected_rounds),
        }


class ReplayHooks:
    def __init__(self, expected_rounds: dict[int, dict[str, Any]], *, complete: bool,
                 tamper: bool = False,
                 expected_final: dict[str, Any] | None = None) -> None:
        self.expected = dict(expected_rounds)
        self.ledger_complete = complete
        self.tamper = tamper
        self.expected_final = expected_final
        self.generated: list[dict[str, Any]] = []
        self.extra_rounds: list[int] = []
        self.prefix_checked = 0
        self.final_checked = False

    def begin_round(self, round_no: int, running: list[str]) -> None:
        return None

    def commit_round(self, payload: dict[str, Any], observed: dict[str, Any]) -> None:
        round_no = int(payload["round"])
        self.generated.append(payload)
        expected = self.expected.get(round_no)
        if expected is None:
            self.extra_rounds.append(round_no)
            if self.ledger_complete:
                raise ReplayError("ledger has no committed round for replay output",
                                  field=f"$.rounds[{round_no}]")
            return
        if self.tamper and round_no == min(self.expected):
            expected = json.loads(canonical_text(expected))
            target = next((event for event in expected.get("events", [])
                           if event.get("txid") and event.get("row")), None)
            if target is not None:
                target["at_ms"] = int(target.get("at_ms", 0)) + 1
            elif expected.get("events"):
                expected["events"][0]["at_ms"] = int(expected["events"][0].get("at_ms", 0)) + 1
            else:
                expected["injected_tamper"] = True
        if canonical_bytes(expected) != canonical_bytes(payload):
            raise _round_replay_error(round_no, expected, payload)
        self.prefix_checked += 1

    def complete(self, result: Any) -> None:
        generated = {int(payload["round"]) for payload in result.rounds}
        missing = sorted(set(self.expected) - generated)
        if missing and self.ledger_complete:
            raise ReplayError("replay is missing committed rounds",
                              field=f"$.rounds[{missing[0]}]",
                              details={"missing_rounds": missing})
        if self.ledger_complete and self.expected_final is not None:
            expected = _stable_final(self.expected_final)
            actual = _stable_final(result)
            if canonical_bytes(expected) != canonical_bytes(actual):
                diffs = structured_diff(expected, actual, "$.final")
                txid, row, field = _locate_final_diff(diffs, expected)
                raise ReplayError("replayed final version/isolation state differs",
                                  txid=txid, row=row, field=field,
                                  details={"differences": diffs[:16]})
            self.final_checked = True

    def summary(self) -> dict[str, Any]:
        return {
            "prefix_checked": self.prefix_checked,
            "expected_rounds": len(self.expected),
            "generated_rounds": len(self.generated),
            "extra_rounds": self.extra_rounds,
            "prefix_equal": self.prefix_checked == min(len(self.expected), len(self.generated)),
            "final_state_checked": self.final_checked,
        }


def _stable_final(value: Any) -> dict[str, Any]:
    if hasattr(value, "result"):
        result = value.result
        isolations = value.isolation_conclusions
    else:
        result = value.get("result", {}) if isinstance(value, dict) else {}
        isolations = result.get("isolations", []) if isinstance(result, dict) else []
    if not result:
        result = value if isinstance(value, dict) else {}
        isolations = result.get("isolations", []) if isinstance(result, dict) else []
    return {
        "status": result.get("status"),
        "spec_sha256": result.get("spec_sha256"),
        "path": result.get("path"),
        "transactions": result.get("transactions"),
        "final_versions": result.get("final_versions"),
        "final_locks": result.get("final_locks"),
        "isolations": isolations,
        "deadlocks": [
            {key: item.get(key) for key in (
                "round", "logical_ms", "cycle", "wait_graph_edges", "victim", "basis"
            )}
            for item in result.get("deadlocks", [])
        ],
        "timeouts": [
            {key: item.get(key) for key in (
                "round", "logical_ms", "txid", "row", "mode", "wait_ms", "code"
            )}
            for item in result.get("timeouts", [])
        ],
        "wait_budget": {
            "limit_ms": result.get("wait_budget", {}).get("limit_ms"),
            "canonical_used_ms": result.get("wait_budget", {}).get("canonical_used_ms"),
        },
    }


def _locate_final_diff(diffs: list[dict[str, Any]], expected: dict[str, Any]
                       ) -> tuple[str | None, str | None, str]:
    if not diffs:
        return None, None, "$"
    path = str(diffs[0]["path"])
    txid = None
    row = None
    version_match = re.search(r"final_versions\.(.+?)\[", path)
    if version_match:
        row = version_match.group(1)
        for item in expected.get("isolations", []):
            if any(read.get("row") == row for read in item.get("reads", [])):
                txid = item.get("txid")
                break
    match = re.search(r"isolations\[(\d+)\]", path)
    if match:
        idx = int(match.group(1))
        items = expected.get("isolations", [])
        if idx < len(items):
            txid = items[idx].get("txid")
            read_match = re.search(r"reads\[(\d+)\]", path)
            if read_match:
                read_idx = int(read_match.group(1))
                reads = items[idx].get("reads", [])
                if read_idx < len(reads):
                    row = reads[read_idx].get("row")
    if txid is None:
        txids = sorted((expected.get("transactions") or {}).keys())
        txid = txids[0] if txids else None
    return (str(txid) if txid is not None else None,
            str(row) if row is not None else None,
            path.rsplit(".", 1)[-1])


def read_session_metadata(ledger_path: str | Path) -> dict[str, Any]:
    path = Path(ledger_path)
    if not path.exists():
        return {}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ledger_meta'").fetchone()
        if not exists:
            return {}
        rows = conn.execute("SELECT key,value FROM ledger_meta").fetchall()
        meta = {row["key"]: row["value"] for row in rows}
        for key in ("config_json", "unknown_txids"):
            if key in meta:
                try:
                    meta[key] = json.loads(meta[key])
                except ValueError:
                    pass
        return meta
    finally:
        conn.close()


def _round_replay_error(round_no: int, expected: dict[str, Any],
                        actual: dict[str, Any]) -> ReplayError:
    diffs = structured_diff(expected, actual, f"$.rounds[{round_no}]")
    txid, row, field = _locate_event_diff(diffs, expected, actual)
    return ReplayError("canonical replay mismatch", txid=txid, row=row, field=field,
                       details={"round": round_no, "differences": diffs[:16]})


def _locate_event_diff(diffs: list[dict[str, Any]], expected: Any,
                       actual: Any) -> tuple[str | None, str | None, str]:
    if not diffs:
        return None, None, "$"
    path = str(diffs[0]["path"])
    event_match = re.search(r"\.events\[(\d+)\]", path)
    if event_match:
        idx = int(event_match.group(1))
        events = expected.get("events", []) if isinstance(expected, dict) else []
        if idx < len(events) and isinstance(events[idx], dict):
            txid = events[idx].get("txid")
            row = events[idx].get("row")
            field_text = path.rsplit(".", 1)[-1]
            return (str(txid) if txid is not None else None,
                    str(row) if row is not None else None, field_text)
    for candidate in (expected, actual):
        if isinstance(candidate, dict):
            for event in candidate.get("events", []):
                if isinstance(event, dict) and event.get("txid"):
                    return str(event["txid"]), (str(event["row"]) if event.get("row") else None), path
    return None, None, path
