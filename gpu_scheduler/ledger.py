from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION
from .errors import SnapshotDigestError
from .hashing import canonical_text, digest_obj, sha256_hex
from .spec import Spec


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS ledger_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rounds (
        round_index INTEGER PRIMARY KEY,
        started_monotonic_ms INTEGER NOT NULL,
        ended_monotonic_ms INTEGER NOT NULL,
        cursor_json TEXT NOT NULL,
        watermarks_json TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        chain_digest TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS shard_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        round_index INTEGER NOT NULL,
        shard_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        phase TEXT NOT NULL,
        sequence_no INTEGER NOT NULL,
        monotonic_ms INTEGER NOT NULL,
        queue_start_ms INTEGER,
        queue_end_ms INTEGER,
        payload_json TEXT NOT NULL,
        receipt_digest TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS occupancy_rows (
        round_index INTEGER NOT NULL,
        scope TEXT NOT NULL,
        pool_id TEXT NOT NULL,
        used_gpu INTEGER NOT NULL,
        used_memory_mib INTEGER NOT NULL,
        gpu_capacity INTEGER NOT NULL,
        memory_capacity_mib INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (round_index, scope, pool_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS preemption_rows (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        round_index INTEGER NOT NULL,
        high_task_id TEXT NOT NULL,
        victim_task_id TEXT NOT NULL,
        victim_set_json TEXT NOT NULL,
        selection_rule TEXT NOT NULL,
        reason TEXT NOT NULL,
        released_gpu INTEGER NOT NULL,
        released_memory_mib INTEGER NOT NULL,
        monotonic_ms INTEGER NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS suspension_rows (
        task_id TEXT PRIMARY KEY,
        round_index INTEGER NOT NULL,
        pool_id TEXT NOT NULL,
        state TEXT NOT NULL,
        completed_shards_json TEXT NOT NULL,
        cursor_json TEXT NOT NULL,
        occupied_gpu INTEGER NOT NULL,
        occupied_memory_mib INTEGER NOT NULL,
        reason TEXT NOT NULL,
        lease_id TEXT NOT NULL,
        lease_expires_wall_ms INTEGER NOT NULL,
        renew_count INTEGER NOT NULL,
        recovery_path TEXT NOT NULL,
        updated_monotonic_ms INTEGER NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS suspension_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        round_index INTEGER NOT NULL,
        task_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        lease_id TEXT NOT NULL,
        lease_expires_wall_ms INTEGER NOT NULL,
        renew_count INTEGER NOT NULL,
        monotonic_ms INTEGER NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cursor_rows (
        task_id TEXT PRIMARY KEY,
        round_index INTEGER NOT NULL,
        next_shard_id TEXT,
        completed_shards_json TEXT NOT NULL,
        pending_shards_json TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS attempts (
        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        round_index INTEGER NOT NULL,
        state TEXT NOT NULL,
        shard_ids_json TEXT NOT NULL,
        pid INTEGER NOT NULL,
        created_wall_ms INTEGER NOT NULL,
        updated_wall_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS receipts (
        receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        shard_id TEXT NOT NULL,
        round_index INTEGER NOT NULL,
        attempt_id INTEGER,
        canonical_receipt_json TEXT NOT NULL,
        receipt_digest TEXT NOT NULL,
        outcome_status TEXT NOT NULL,
        resource_json TEXT NOT NULL,
        created_monotonic_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS propagation_rows (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        round_index INTEGER NOT NULL,
        trigger_shards_json TEXT NOT NULL,
        affected_json TEXT NOT NULL,
        trace_json TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
]


class Ledger:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        for statement in SCHEMA_STATEMENTS:
            self.conn.execute(statement)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def initialize_meta(self, spec: Spec) -> None:
        now = int(time.time() * 1000)
        genesis = sha256_hex(f"genesis:{LEDGER_FORMAT_VERSION}:{spec.spec_digest}")
        values = {
            "format_version": str(LEDGER_FORMAT_VERSION),
            "spec_digest": spec.spec_digest,
            "pool_version_json": canonical_text(spec.pool_version),
            "task_version_json": canonical_text(spec.task_version),
            "scale_version_json": canonical_text(spec.scale_version),
            "head_round": "0",
            "head_chain_digest": genesis,
            "genesis_digest": genesis,
            "created_wall_ms": str(now),
        }
        for key, value in values.items():
            self.conn.execute(
                "INSERT OR IGNORE INTO ledger_meta(key, value) VALUES(?, ?)",
                (key, value),
            )

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM ledger_meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def meta_dict(self) -> dict[str, str]:
        return {row["key"]: row["value"] for row in self.conn.execute("SELECT key,value FROM ledger_meta")}

    @property
    def head_round(self) -> int:
        return int(self.get_meta("head_round", "0") or 0)

    @property
    def head_chain_digest(self) -> str:
        return self.get_meta("head_chain_digest", "") or ""

    def begin_attempt(self, round_index: int, shard_ids: list[str]) -> int:
        now = int(time.time() * 1000)
        cursor = self.conn.execute(
            """
            INSERT INTO attempts(round_index,state,shard_ids_json,pid,created_wall_ms,updated_wall_ms)
            VALUES(?, 'open', ?, ?, ?, ?)
            """,
            (round_index, canonical_text(sorted(shard_ids)), __import__("os").getpid(), now, now),
        )
        return int(cursor.lastrowid)

    def open_attempts(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM attempts WHERE state='open' ORDER BY attempt_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def commit_round(self, bundle: dict[str, Any], *, fault_before_commit: bool = False) -> dict[str, Any]:
        round_index = int(bundle["round_index"])
        payload = {
            "format_version": LEDGER_FORMAT_VERSION,
            "round_index": round_index,
            "started_monotonic_ms": bundle["started_monotonic_ms"],
            "ended_monotonic_ms": bundle["ended_monotonic_ms"],
            "cursor": bundle.get("cursor", {"head_round": round_index}),
            "watermarks": bundle["watermarks"],
            "events": bundle.get("events", []),
            "occupancy_rows": bundle.get("occupancy_rows", []),
            "preemptions": bundle.get("preemptions", []),
            "suspensions": bundle.get("suspensions", []),
            "cursors": bundle.get("cursors", []),
            "propagation": bundle.get("propagation"),
            "propagations": bundle.get("propagations", []),
            "receipts": [
                {k: v for k, v in receipt.items() if k not in ("receipt_id",)}
                for receipt in bundle.get("receipts", [])
            ],
        }
        content_digest = digest_obj(payload)
        previous = self.head_chain_digest
        chain_digest = sha256_hex(previous + content_digest)
        payload_with_digests = {**payload, "content_digest": content_digest, "chain_digest": chain_digest}
        payload_json = canonical_text(payload_with_digests)

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                """
                INSERT INTO rounds(
                    round_index,started_monotonic_ms,ended_monotonic_ms,cursor_json,
                    watermarks_json,payload_json,content_digest,chain_digest
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    round_index,
                    bundle["started_monotonic_ms"],
                    bundle["ended_monotonic_ms"],
                    canonical_text(bundle.get("cursor", {"head_round": round_index})),
                    canonical_text(bundle["watermarks"]),
                    payload_json,
                    content_digest,
                    chain_digest,
                ),
            )
            for event in bundle.get("events", []):
                self.conn.execute(
                    """
                    INSERT INTO shard_events(
                        round_index,shard_id,task_id,phase,sequence_no,monotonic_ms,
                        queue_start_ms,queue_end_ms,payload_json,receipt_digest
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        round_index,
                        event["shard_id"],
                        event.get("task_id", event["shard_id"].split("#", 1)[0]),
                        event["phase"],
                        int(event.get("sequence_no", 0)),
                        int(event["monotonic_ms"]),
                        event.get("queue_start_ms"),
                        event.get("queue_end_ms"),
                        canonical_text(event.get("payload", {})),
                        event.get("receipt_digest"),
                    ),
                )
            for row in bundle.get("occupancy_rows", []):
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO occupancy_rows(
                        round_index,scope,pool_id,used_gpu,used_memory_mib,gpu_capacity,
                        memory_capacity_mib,payload_json
                    ) VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        round_index,
                        row["scope"],
                        row["pool_id"],
                        int(row["used_gpu"]),
                        int(row["used_memory_mib"]),
                        int(row["gpu_capacity"]),
                        int(row["memory_capacity_mib"]),
                        canonical_text(row),
                    ),
                )
            for preemption in bundle.get("preemptions", []):
                self.conn.execute(
                    """
                    INSERT INTO preemption_rows(
                        round_index,high_task_id,victim_task_id,victim_set_json,selection_rule,
                        reason,released_gpu,released_memory_mib,monotonic_ms,payload_json
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        round_index,
                        preemption["high_task_id"],
                        preemption["victim_task_id"],
                        canonical_text(preemption.get("victim_set", [preemption["victim_task_id"]])),
                        preemption.get("selection_rule", ""),
                        preemption.get("reason", ""),
                        int(preemption.get("released_gpu", 0)),
                        int(preemption.get("released_memory_mib", 0)),
                        int(preemption["monotonic_ms"]),
                        canonical_text(preemption),
                    ),
                )
            for suspension in bundle.get("suspensions", []):
                self.conn.execute(
                    """
                    INSERT INTO suspension_events(
                        round_index,task_id,event_type,lease_id,lease_expires_wall_ms,
                        renew_count,monotonic_ms,payload_json
                    ) VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        round_index,
                        suspension["task_id"],
                        suspension.get("event_type", "state"),
                        suspension["lease_id"],
                        int(suspension["lease_expires_wall_ms"]),
                        int(suspension.get("renew_count", 0)),
                        int(suspension["monotonic_ms"]),
                        canonical_text(suspension),
                    ),
                )
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO suspension_rows(
                        task_id,round_index,pool_id,state,completed_shards_json,cursor_json,
                        occupied_gpu,occupied_memory_mib,reason,lease_id,lease_expires_wall_ms,
                        renew_count,recovery_path,updated_monotonic_ms,payload_json
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        suspension["task_id"],
                        round_index,
                        suspension["pool_id"],
                        suspension["state"],
                        canonical_text(suspension.get("completed_shards", [])),
                        canonical_text(suspension.get("cursor", {})),
                        int(suspension.get("occupied_gpu", 0)),
                        int(suspension.get("occupied_memory_mib", 0)),
                        suspension.get("reason", ""),
                        suspension["lease_id"],
                        int(suspension["lease_expires_wall_ms"]),
                        int(suspension.get("renew_count", 0)),
                        suspension.get("recovery_path", f"sqlite://{self.path}#cursor_rows"),
                        int(suspension["monotonic_ms"]),
                        canonical_text(suspension),
                    ),
                )
            for cursor in bundle.get("cursors", []):
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO cursor_rows(
                        task_id,round_index,next_shard_id,completed_shards_json,
                        pending_shards_json,payload_json
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        cursor["task_id"],
                        round_index,
                        cursor.get("next_shard_id"),
                        canonical_text(cursor.get("completed_shards", [])),
                        canonical_text(cursor.get("pending_shards", [])),
                        canonical_text(cursor),
                    ),
                )
            for receipt in bundle.get("receipts", []):
                canonical_receipt = receipt["canonical_receipt_json"]
                self.conn.execute(
                    """
                    INSERT INTO receipts(
                        shard_id,round_index,attempt_id,canonical_receipt_json,receipt_digest,
                        outcome_status,resource_json,created_monotonic_ms
                    ) VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        receipt["shard_id"],
                        round_index,
                        receipt.get("attempt_id", bundle.get("attempt_id")),
                        canonical_receipt,
                        receipt["receipt_digest"],
                        receipt["outcome_status"],
                        canonical_text(receipt.get("resource_usage", {})),
                        int(receipt["created_monotonic_ms"]),
                    ),
                )
            if bundle.get("propagation"):
                propagation = bundle["propagation"]
                self.conn.execute(
                    """
                    INSERT INTO propagation_rows(
                        round_index,trigger_shards_json,affected_json,trace_json,payload_json
                    ) VALUES(?,?,?,?,?)
                    """,
                    (
                        round_index,
                        canonical_text(propagation.get("trigger_shards", [])),
                        canonical_text(propagation.get("affected", [])),
                        canonical_text(propagation.get("trace", [])),
                        canonical_text(propagation),
                    ),
                )
            for propagation in bundle.get("propagations", []):
                self.conn.execute(
                    """
                    INSERT INTO propagation_rows(
                        round_index,trigger_shards_json,affected_json,trace_json,payload_json
                    ) VALUES(?,?,?,?,?)
                    """,
                    (
                        round_index,
                        canonical_text(propagation.get("trigger_shards", [])),
                        canonical_text(propagation.get("affected", [])),
                        canonical_text(propagation.get("trace", [])),
                        canonical_text(propagation),
                    ),
                )
            if bundle.get("attempt_id") is not None:
                self.conn.execute(
                    "UPDATE attempts SET state='committed', updated_wall_ms=? WHERE attempt_id=?",
                    (int(time.time() * 1000), int(bundle["attempt_id"])),
                )
            for attempt_id in bundle.get("commit_attempt_ids", []):
                self.conn.execute(
                    "UPDATE attempts SET state='committed', updated_wall_ms=? WHERE attempt_id=?",
                    (int(time.time() * 1000), int(attempt_id)),
                )
            for attempt_id in bundle.get("interrupt_attempt_ids", []):
                self.conn.execute(
                    "UPDATE attempts SET state='interrupted', updated_wall_ms=? WHERE attempt_id=?",
                    (int(time.time() * 1000), int(attempt_id)),
                )
            self.conn.execute(
                "INSERT OR REPLACE INTO ledger_meta(key,value) VALUES('head_round',?)",
                (str(round_index),),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO ledger_meta(key,value) VALUES('head_chain_digest',?)",
                (chain_digest,),
            )
            if fault_before_commit:
                import os
                import signal

                os.kill(os.getpid(), signal.SIGKILL)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return {"content_digest": content_digest, "chain_digest": chain_digest, "payload_json": payload_json}

    def list_rounds(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM rounds ORDER BY round_index").fetchall()
        return [dict(row) for row in rows]

    def verify_digests(self) -> dict[str, Any]:
        meta = self.meta_dict()
        expected_previous = meta.get("genesis_digest", "")
        rows = self.list_rounds()
        for row in rows:
            payload = json.loads(row["payload_json"])
            stored_content = payload.pop("content_digest", None)
            stored_chain = payload.pop("chain_digest", None)
            recomputed_content = digest_obj(payload)
            if stored_content != row["content_digest"]:
                raise SnapshotDigestError(
                    "round payload content_digest column does not match payload",
                    locator={"round_index": row["round_index"], "field": "rounds.content_digest"},
                )
            if recomputed_content != row["content_digest"]:
                raise SnapshotDigestError(
                    "round payload content digest mismatch",
                    locator={
                        "round_index": row["round_index"],
                        "field": "payload.content_digest",
                        "expected": recomputed_content,
                        "actual": row["content_digest"],
                    },
                )
            expected_chain = sha256_hex(expected_previous + row["content_digest"])
            if row["chain_digest"] != expected_chain or stored_chain != expected_chain:
                raise SnapshotDigestError(
                    "ledger chain digest mismatch",
                    locator={"round_index": row["round_index"], "field": "rounds.chain_digest"},
                )
            expected_previous = row["chain_digest"]
        if rows and meta.get("head_chain_digest") != expected_previous:
            raise SnapshotDigestError(
                "ledger head digest mismatch",
                locator={"field": "ledger_meta.head_chain_digest"},
            )
        return {"ok": True, "rounds": len(rows), "head_round": self.head_round, "head_chain_digest": self.head_chain_digest}

    def latest_receipts(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT r.* FROM receipts r
            JOIN (SELECT shard_id, MAX(receipt_id) AS max_id FROM receipts GROUP BY shard_id) m
              ON r.receipt_id = m.max_id
            ORDER BY r.shard_id
            """
        ).fetchall()
        return {row["shard_id"]: dict(row) for row in rows}

    def all_receipts(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for row in self.conn.execute("SELECT * FROM receipts ORDER BY receipt_id"):
            result.setdefault(row["shard_id"], []).append(dict(row))
        return result

    def export_receipt_snapshot(self) -> dict[str, Any]:
        exported: dict[str, Any] = {}
        for shard_id, row in self.latest_receipts().items():
            exported[shard_id] = {
                "receipt_id": row["receipt_id"],
                "round_index": row["round_index"],
                "canonical_receipt_json": row["canonical_receipt_json"],
                "receipt_digest": row["receipt_digest"],
                "outcome_status": row["outcome_status"],
                "resource": json.loads(row["resource_json"]),
            }
        return {"format_version": 1, "receipts": exported}

    def tamper_latest_content_digest(self) -> int:
        row = self.conn.execute("SELECT MAX(round_index) AS idx FROM rounds").fetchone()
        if not row or row["idx"] is None:
            self.conn.execute(
                "INSERT OR REPLACE INTO ledger_meta(key,value) VALUES('head_chain_digest', ?)",
                ("0" * 64,),
            )
            return 0
        idx = int(row["idx"])
        self.conn.execute("UPDATE rounds SET content_digest=? WHERE round_index=?", ("0" * 64, idx))
        return idx
