"""SQLite authoritative ledger and crash-safe batch recovery."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .hashing import canonical_hash


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "+00:00")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def batch_tx_digest(row: dict[str, Any]) -> str:
    material = {key: value for key, value in row.items() if key != "tx_digest"}
    return canonical_hash(material)


def query_tx_digest(row: dict[str, Any]) -> str:
    material = {key: value for key, value in row.items() if key != "tx_digest"}
    return canonical_hash(material)


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self.connection.execute("INSERT OR IGNORE INTO ledger_meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
        self.connection.commit()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ledger_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                snapshot_version INTEGER NOT NULL,
                spec_digest TEXT NOT NULL,
                format_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS batch_intents (
                session_id TEXT NOT NULL,
                batch_key TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                block_group_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id, batch_key),
                UNIQUE (session_id, sequence),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS batches (
                session_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                batch_key TEXT NOT NULL,
                block_group_id TEXT NOT NULL,
                status TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                logical_digest TEXT NOT NULL,
                decoded_rows INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                occupancy_json TEXT NOT NULL,
                lease_json TEXT NOT NULL,
                cursor_json TEXT NOT NULL,
                row_version INTEGER NOT NULL,
                tx_digest TEXT NOT NULL,
                committed_at TEXT NOT NULL,
                PRIMARY KEY (session_id, sequence),
                UNIQUE (session_id, batch_key),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS query_results (
                session_id TEXT NOT NULL,
                query_id TEXT NOT NULL,
                result_digest TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                spec_digest TEXT NOT NULL,
                row_version INTEGER NOT NULL,
                tx_digest TEXT NOT NULL,
                committed_at TEXT NOT NULL,
                PRIMARY KEY (session_id, query_id),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def initialize_session(
        self,
        session_id: str,
        spec_digest: str,
        block_group_ids: list[str],
        *,
        snapshot_version: int = 1,
        format_version: int = 1,
        status: str = "active",
    ) -> dict[str, Any]:
        now = utc_now()
        row = self.connection.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            self.connection.execute(
                """INSERT INTO sessions(session_id,snapshot_version,spec_digest,format_version,status,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (session_id, snapshot_version, spec_digest, format_version, status, now, now),
            )
        else:
            self.connection.execute(
                """UPDATE sessions SET snapshot_version=?, spec_digest=?, format_version=?, status=?, updated_at=?
                   WHERE session_id=?""",
                (snapshot_version, spec_digest, format_version, status, now, session_id),
            )
        for sequence, block_group_id in enumerate(block_group_ids, start=1):
            batch_key = f"block_group:{block_group_id}"
            self.connection.execute(
                """INSERT OR IGNORE INTO batch_intents(session_id,batch_key,sequence,block_group_id,status,started_at,updated_at)
                   VALUES(?,?,?,?, 'running', ?, ?)""",
                (session_id, batch_key, sequence, block_group_id, now, now),
            )
        self.connection.commit()
        return dict(self.connection.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone())

    def committed_batch_ids(self, session_id: str) -> set[str]:
        rows = self.connection.execute(
            "SELECT block_group_id FROM batches WHERE session_id=? AND status='completed'",
            (session_id,),
        ).fetchall()
        return {row["block_group_id"] for row in rows}

    def commit_batch(
        self,
        session_id: str,
        sequence: int,
        block_group_id: str,
        *,
        content_digest: str,
        logical_digest: str,
        decoded_rows: int,
        payload: dict[str, Any],
        occupancy: dict[str, Any],
        lease: dict[str, Any],
        cursor: dict[str, Any],
        row_version: int = 1,
        status: str = "completed",
    ) -> dict[str, Any]:
        now = utc_now()
        batch_key = f"block_group:{block_group_id}"
        row_material = {
            "session_id": session_id,
            "sequence": sequence,
            "batch_key": batch_key,
            "block_group_id": block_group_id,
            "status": status,
            "content_digest": content_digest,
            "logical_digest": logical_digest,
            "decoded_rows": int(decoded_rows),
            "payload_json": _json_dumps(payload),
            "occupancy_json": _json_dumps(occupancy),
            "lease_json": _json_dumps(lease),
            "cursor_json": _json_dumps(cursor),
            "row_version": int(row_version),
            "committed_at": now,
        }
        tx_digest = batch_tx_digest(row_material)
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO batches(session_id,sequence,batch_key,block_group_id,status,content_digest,
                       logical_digest,decoded_rows,payload_json,occupancy_json,lease_json,cursor_json,row_version,
                       tx_digest,committed_at)
                   VALUES(:session_id,:sequence,:batch_key,:block_group_id,:status,:content_digest,:logical_digest,
                       :decoded_rows,:payload_json,:occupancy_json,:lease_json,:cursor_json,:row_version,:tx_digest,:committed_at)""",
                {**row_material, "tx_digest": tx_digest},
            )
            self.connection.execute(
                """UPDATE batch_intents SET status='completed', updated_at=?
                   WHERE session_id=? AND batch_key=?""",
                (now, session_id, batch_key),
            )
        return {"tx_digest": tx_digest, "committed_at": now}

    def commit_query_result(
        self,
        session_id: str,
        query_id: str,
        result_digest: str,
        payload: dict[str, Any],
        spec_digest: str,
        *,
        row_version: int = 1,
    ) -> dict[str, Any]:
        now = utc_now()
        material = {
            "session_id": session_id,
            "query_id": query_id,
            "result_digest": result_digest,
            "payload_json": _json_dumps(payload),
            "spec_digest": spec_digest,
            "row_version": row_version,
            "committed_at": now,
        }
        tx_digest = query_tx_digest(material)
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO query_results(session_id,query_id,result_digest,payload_json,spec_digest,
                       row_version,tx_digest,committed_at)
                   VALUES(:session_id,:query_id,:result_digest,:payload_json,:spec_digest,:row_version,:tx_digest,:committed_at)""",
                {**material, "tx_digest": tx_digest},
            )
        return {"tx_digest": tx_digest, "committed_at": now}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def unknown_batches(self, session_id: str, *, mark_unknown: bool = True) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM batch_intents WHERE session_id=? AND status IN ('running','unknown') ORDER BY sequence",
            (session_id,),
        ).fetchall()
        unknown = [dict(row) for row in rows if row["status"] != "completed"]
        if mark_unknown:
            now = utc_now()
            with self.connection:
                self.connection.execute(
                    "UPDATE batch_intents SET status='unknown', updated_at=? WHERE session_id=? AND status='running'",
                    (now, session_id),
                )
        return unknown

    def replay_batches(
        self,
        session_id: str,
        *,
        verify_batch: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        rows = self.connection.execute(
            "SELECT * FROM batches WHERE session_id=? ORDER BY sequence",
            (session_id,),
        ).fetchall()
        replayed: list[dict[str, Any]] = []
        first_inconsistency: dict[str, Any] | None = None
        for row in rows:
            raw = dict(row)
            stored_tx = raw.pop("tx_digest")
            expected_tx = batch_tx_digest({**raw, "tx_digest": stored_tx})
            tx_consistent = stored_tx == expected_tx
            cursor = json.loads(raw.pop("cursor_json"))
            payload = json.loads(raw.pop("payload_json"))
            json.loads(raw.pop("occupancy_json"))
            json.loads(raw.pop("lease_json"))
            consistent = tx_consistent
            inconsistency: dict[str, Any] | None = None
            if not tx_consistent:
                inconsistency = {"field": "tx_digest", "expected": expected_tx, "actual": stored_tx}
            elif verify_batch is not None:
                problem = verify_batch(raw, payload)
                if problem:
                    consistent = False
                    inconsistency = problem
            entry = {
                "sequence": raw["sequence"],
                "block_group_id": raw["block_group_id"],
                "status": raw["status"],
                "content_digest": raw["content_digest"],
                "logical_digest": raw["logical_digest"],
                "decoded_rows": raw["decoded_rows"],
                "row_version": raw["row_version"],
                "tx_digest": stored_tx,
                "committed_at": raw["committed_at"],
                "cursor": cursor,
                "consistent": consistent,
            }
            if inconsistency:
                entry["inconsistency"] = inconsistency
                if first_inconsistency is None:
                    first_inconsistency = {
                        "sequence": raw["sequence"],
                        "block_group_id": raw["block_group_id"],
                        **inconsistency,
                    }
            replayed.append(entry)
        return replayed, first_inconsistency

    def replay_queries(
        self,
        session_id: str,
        expected_query_ids: list[str],
        *,
        verify_query: Callable[[str, dict[str, Any]], dict[str, Any] | None] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None]:
        rows = self.connection.execute(
            "SELECT * FROM query_results WHERE session_id=? ORDER BY query_id",
            (session_id,),
        ).fetchall()
        by_id = {row["query_id"]: row for row in rows}
        replayed: list[dict[str, Any]] = []
        first_inconsistency: dict[str, Any] | None = None
        for query_id in sorted(by_id):
            row = dict(by_id[query_id])
            stored_tx = row.pop("tx_digest")
            expected_tx = query_tx_digest({**row, "tx_digest": stored_tx})
            payload = json.loads(row.pop("payload_json"))
            consistent = stored_tx == expected_tx
            problem = None if consistent else {"field": "tx_digest", "expected": expected_tx, "actual": stored_tx}
            digest_consistent = row["result_digest"] == canonical_hash(payload)
            if consistent and not digest_consistent:
                consistent = False
                problem = {
                    "field": "result_digest",
                    "expected": canonical_hash(payload),
                    "actual": row["result_digest"],
                }
            if consistent and verify_query is not None:
                problem = verify_query(query_id, payload)
                if problem:
                    consistent = False
            entry = {
                "query_id": query_id,
                "result_digest": row["result_digest"],
                "row_version": row["row_version"],
                "tx_digest": stored_tx,
                "consistent": consistent,
            }
            if problem:
                entry["inconsistency"] = problem
                if first_inconsistency is None:
                    first_inconsistency = {"query_id": query_id, **problem}
            replayed.append(entry)
        missing = [query_id for query_id in expected_query_ids if query_id not in by_id]
        return replayed, missing, first_inconsistency

    def recover(
        self,
        session_id: str,
        expected_query_ids: list[str] | None = None,
        *,
        verify_batch: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None] | None = None,
        verify_query: Callable[[str, dict[str, Any]], dict[str, Any] | None] | None = None,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            return {
                "session_id": session_id,
                "status": "no_session",
                "requires_rerun": True,
                "unknown_batches": [],
                "committed_batches_replayed": [],
                "query_results_replayed": [],
                "missing_query_results": list(expected_query_ids or []),
                "first_inconsistency": None,
            }
        unknown_intents = self.unknown_batches(session_id, mark_unknown=True)
        unknown_ids = [item["block_group_id"] for item in unknown_intents]
        batches, batch_problem = self.replay_batches(session_id, verify_batch=verify_batch)
        queries, missing_queries, query_problem = self.replay_queries(
            session_id,
            expected_query_ids or [],
            verify_query=verify_query,
        )
        first_problem = batch_problem or query_problem
        requires_rerun = bool(unknown_ids or missing_queries or first_problem)
        if first_problem:
            status = "inconsistent"
        elif requires_rerun:
            status = "requires_rerun"
        else:
            status = "consistent"
        return {
            "session_id": session_id,
            "snapshot_version": session["snapshot_version"],
            "spec_digest": session["spec_digest"],
            "format_version": session["format_version"],
            "status": status,
            "requires_rerun": requires_rerun,
            "unknown_batches": unknown_ids,
            "committed_batches_replayed": batches,
            "query_results_replayed": queries,
            "missing_query_results": missing_queries,
            "first_inconsistency": first_problem,
        }
