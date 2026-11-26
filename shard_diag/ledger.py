from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_bytes, content_digest, sha256_hex
from .errors import LeaseConflictError, ReplayConsistencyError


LEDGER_FORMAT_VERSION = 1


def now_wall_ms() -> int:
    return time.time_ns() // 1_000_000


def now_monotonic_ns() -> int:
    return time.monotonic_ns()


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              int_value INTEGER,
              text_value TEXT,
              format_version INTEGER NOT NULL DEFAULT 1,
              content_digest TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS shard_state (
              shard_id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              current_lease_id TEXT,
              format_version INTEGER NOT NULL,
              content_digest TEXT NOT NULL,
              updated_wall_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS shard_outcomes (
              shard_id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              node_id TEXT NOT NULL,
              quota_version INTEGER NOT NULL,
              loss_digest TEXT NOT NULL,
              outcome_blob BLOB NOT NULL,
              format_version INTEGER NOT NULL,
              content_digest TEXT NOT NULL,
              committed_seq INTEGER,
              updated_wall_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS leases (
              lease_id TEXT PRIMARY KEY,
              shard_id TEXT NOT NULL,
              node_id TEXT NOT NULL,
              grant_number INTEGER NOT NULL,
              lease_version INTEGER NOT NULL,
              quota_version INTEGER NOT NULL,
              status TEXT NOT NULL,
              granted_wall_ms INTEGER NOT NULL,
              granted_monotonic_ns INTEGER NOT NULL,
              expires_wall_ms INTEGER NOT NULL,
              expires_monotonic_ns INTEGER NOT NULL,
              lease_blob BLOB NOT NULL,
              format_version INTEGER NOT NULL,
              content_digest TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_lease_per_shard
              ON leases(shard_id) WHERE status='active';
            CREATE INDEX IF NOT EXISTS leases_node_status ON leases(node_id, status);
            CREATE TABLE IF NOT EXISTS resource_holdings (
              lease_id TEXT PRIMARY KEY,
              shard_id TEXT NOT NULL,
              node_id TEXT NOT NULL,
              resource_units INTEGER NOT NULL,
              status TEXT NOT NULL,
              holding_blob BLOB NOT NULL,
              format_version INTEGER NOT NULL,
              content_digest TEXT NOT NULL,
              updated_wall_ms INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS holdings_status_node ON resource_holdings(status, node_id);
            CREATE TABLE IF NOT EXISTS water_records (
              record_key TEXT PRIMARY KEY,
              scope TEXT NOT NULL,
              shard_id TEXT,
              node_id TEXT,
              quota_version INTEGER NOT NULL,
              record_blob BLOB NOT NULL,
              format_version INTEGER NOT NULL,
              content_digest TEXT NOT NULL,
              updated_wall_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS progress_cursor (
              id INTEGER PRIMARY KEY CHECK (id = 1),
              last_topology_index INTEGER NOT NULL,
              last_shard_id TEXT NOT NULL,
              completed_count INTEGER NOT NULL,
              snapshot_version INTEGER NOT NULL,
              cursor_blob BLOB NOT NULL,
              format_version INTEGER NOT NULL,
              content_digest TEXT NOT NULL,
              updated_wall_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS commit_events (
              seq INTEGER PRIMARY KEY AUTOINCREMENT,
              event_type TEXT NOT NULL,
              shard_id TEXT,
              payload_blob BLOB NOT NULL,
              format_version INTEGER NOT NULL,
              content_digest TEXT NOT NULL,
              created_wall_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots (
              snapshot_version INTEGER PRIMARY KEY,
              format_version INTEGER NOT NULL,
              content_digest TEXT NOT NULL,
              snapshot_blob BLOB NOT NULL,
              created_wall_ms INTEGER NOT NULL
            );
            """
        )
        self.conn.commit()
        meta = self.get_meta("schema_format_version")
        if meta is None:
            self.set_meta("schema_format_version", LEDGER_FORMAT_VERSION)

    def get_meta(self, key: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM meta WHERE key=?", (key,)).fetchone()
        return dict(row) if row else None

    def set_meta(self, key: str, int_value: int | None = None, text_value: str | None = None) -> None:
        self._upsert_meta(key, int_value, text_value)
        self.conn.commit()

    def _upsert_meta(self, key: str, int_value: int | None = None, text_value: str | None = None) -> None:
        payload = {"key": key, "int_value": int_value, "text_value": text_value, "format_version": LEDGER_FORMAT_VERSION}
        digest = content_digest(payload)
        self.conn.execute(
            """
            INSERT INTO meta(key,int_value,text_value,format_version,content_digest)
            VALUES(?,?,?,?,?)
            ON CONFLICT(key) DO UPDATE SET
              int_value=excluded.int_value,
              text_value=excluded.text_value,
              format_version=excluded.format_version,
              content_digest=excluded.content_digest
            """,
            (key, int_value, text_value, LEDGER_FORMAT_VERSION, digest),
        )

    def active_usage(self) -> dict[str, dict[str, int]]:
        rows = self.conn.execute(
            """
            SELECT node_id,
                   COALESCE(SUM(resource_units), 0) AS units,
                   COUNT(*) AS leases
            FROM resource_holdings
            WHERE status='active'
            GROUP BY node_id
            """
        ).fetchall()
        usage = {row["node_id"]: {"resource_units": int(row["units"]), "leases": int(row["leases"]), "executions": int(row["leases"])} for row in rows}
        return usage

    def active_lease_count_by_node(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT node_id, COUNT(*) AS n FROM leases WHERE status='active' GROUP BY node_id"
        ).fetchall()
        return {row["node_id"]: int(row["n"]) for row in rows}

    def grant_lease(
        self,
        *,
        shard: dict[str, Any],
        quota_version: int,
        ttl_ms: int,
        limits: dict[str, int],
        global_budget_units: int,
        now_ns: int | None = None,
    ) -> dict[str, Any]:
        now_ns = now_ns if now_ns is not None else now_monotonic_ns()
        wall_ms = now_wall_ms()
        expires_ns = now_ns + int(ttl_ms) * 1_000_000
        shard_id = shard["shard_id"]
        node_id = shard["node_id"]
        units = int(shard["resource_units"])
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            active = self.conn.execute(
                "SELECT lease_id FROM leases WHERE shard_id=? AND status='active'", (shard_id,)
            ).fetchone()
            if active:
                raise LeaseConflictError("分片已有活动租约，禁止重复调度", entry=shard_id, field="status")

            global_used = self.conn.execute(
                "SELECT COALESCE(SUM(resource_units),0) AS u FROM resource_holdings WHERE status='active'"
            ).fetchone()["u"]
            node_row = self.conn.execute(
                "SELECT COALESCE(SUM(resource_units),0) AS u, COUNT(*) AS n FROM resource_holdings WHERE status='active' AND node_id=?",
                (node_id,),
            ).fetchone()
            node_used = int(node_row["u"])
            node_leases = int(node_row["n"])
            if int(global_used) + units > int(global_budget_units):
                raise LeaseConflictError("全局资源单位预算不足", entry=shard_id, field="global_budget_units")
            if node_used + units > int(limits["resource_units_cap"]):
                raise LeaseConflictError("节点资源配额不足", entry=node_id, field="resource_units_cap")
            if node_leases + 1 > int(limits["lease_limit"]):
                raise LeaseConflictError("每节点租约上限不足", entry=node_id, field="lease_limit")
            if node_leases + 1 > int(limits["concurrency"]):
                raise LeaseConflictError("每节点并发配额不足", entry=node_id, field="concurrency")

            grant_number_row = self.conn.execute(
                "SELECT COALESCE(MAX(grant_number),0)+1 AS n FROM leases WHERE shard_id=?", (shard_id,)
            ).fetchone()
            grant_number = int(grant_number_row["n"])
            lease_id = f"lease:{shard_id}:g{grant_number}"
            lease = {
                "lease_id": lease_id,
                "shard_id": shard_id,
                "node_id": node_id,
                "grant_number": grant_number,
                "lease_version": 1,
                "quota_version": int(quota_version),
                "status": "active",
                "granted_wall_ms": wall_ms,
                "granted_monotonic_ns": now_ns,
                "expires_wall_ms": wall_ms + int(ttl_ms),
                "expires_monotonic_ns": expires_ns,
                "ttl_ms": int(ttl_ms),
                "resource_units": units,
            }
            lease_blob = canonical_bytes(lease)
            self.conn.execute(
                """INSERT INTO leases VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    lease_id, shard_id, node_id, grant_number, 1, int(quota_version), "active",
                    wall_ms, now_ns, wall_ms + int(ttl_ms), expires_ns, lease_blob,
                    LEDGER_FORMAT_VERSION, sha256_hex(lease_blob),
                ),
            )
            holding = {
                "lease_id": lease_id,
                "shard_id": shard_id,
                "node_id": node_id,
                "resource_units": units,
                "status": "active",
                "quota_version": int(quota_version),
            }
            holding_blob = canonical_bytes(holding)
            self.conn.execute(
                "INSERT INTO resource_holdings VALUES(?,?,?,?,?,?,?,?,?)",
                (lease_id, shard_id, node_id, units, "active", holding_blob,
                 LEDGER_FORMAT_VERSION, sha256_hex(holding_blob), wall_ms),
            )
            state_digest = content_digest({"shard_id": shard_id, "status": "running", "lease_id": lease_id})
            self.conn.execute(
                """INSERT INTO shard_state VALUES(?,?,?,?,?,?)
                   ON CONFLICT(shard_id) DO UPDATE SET
                     status='running', current_lease_id=excluded.current_lease_id,
                     format_version=excluded.format_version, content_digest=excluded.content_digest,
                     updated_wall_ms=excluded.updated_wall_ms""",
                (shard_id, "running", lease_id, LEDGER_FORMAT_VERSION, state_digest, wall_ms),
            )
            self._append_event("lease_granted", shard_id, lease)
            self.conn.commit()
            return lease
        except Exception:
            self.conn.rollback()
            raise

    def _append_event(self, event_type: str, shard_id: str | None, payload: Any) -> int:
        blob = canonical_bytes(payload)
        cursor = self.conn.execute(
            "INSERT INTO commit_events(event_type,shard_id,payload_blob,format_version,content_digest,created_wall_ms) VALUES(?,?,?,?,?,?)",
            (event_type, shard_id, blob, LEDGER_FORMAT_VERSION, sha256_hex(blob), now_wall_ms()),
        )
        return int(cursor.lastrowid)

    def renew_lease(
        self,
        *,
        shard_id: str,
        expected_version: int,
        ttl_ms: int,
        now_ns: int | None = None,
        node_new_version: int | None = None,
    ) -> dict[str, Any]:
        now_ns = now_ns if now_ns is not None else now_monotonic_ns()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT * FROM leases WHERE shard_id=? AND status='active'", (shard_id,)
            ).fetchone()
            if row is None or int(row["lease_version"]) != int(expected_version):
                raise LeaseConflictError(
                    "账本租约版本不符，拒绝续约且不延长到期",
                    entry=shard_id,
                    field="lease_version",
                    details={
                        "expected": int(expected_version),
                        "current": None if row is None else int(row["lease_version"]),
                    },
                )
            new_version = int(node_new_version) if node_new_version is not None else int(expected_version) + 1
            if new_version != int(expected_version) + 1:
                raise LeaseConflictError("节点返回的新租约版本不连续", entry=shard_id, field="lease_version")
            expires_ns = now_ns + int(ttl_ms) * 1_000_000
            wall_ms = now_wall_ms()
            self.conn.execute(
                """UPDATE leases
                   SET lease_version=?, expires_monotonic_ns=?, expires_wall_ms=?
                   WHERE lease_id=? AND lease_version=? AND status='active'""",
                (new_version, expires_ns, wall_ms + ttl_ms, row["lease_id"], expected_version),
            )
            self._append_event(
                "lease_renewed",
                shard_id,
                {"shard_id": shard_id, "old_version": expected_version, "new_version": new_version, "expires_monotonic_ns": expires_ns},
            )
            self.conn.commit()
            return {"shard_id": shard_id, "lease_version": new_version, "expires_monotonic_ns": expires_ns}
        except Exception:
            self.conn.rollback()
            raise

    def reap_expired_and_generation(
        self,
        *,
        now_ns: int | None = None,
        current_quota_version: int | None = None,
    ) -> dict[str, list[str]]:
        now_ns = now_ns if now_ns is not None else now_monotonic_ns()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            expired_rows = self.conn.execute(
                "SELECT lease_id,shard_id FROM leases WHERE status='active' AND expires_monotonic_ns<=? ORDER BY shard_id",
                (now_ns,),
            ).fetchall()
            expired_ids = [row["lease_id"] for row in expired_rows]
            expired_shards = [row["shard_id"] for row in expired_rows]
            if expired_ids:
                self.conn.executemany(
                    "UPDATE leases SET status='reaped_expired' WHERE lease_id=? AND status='active'",
                    [(lease_id,) for lease_id in expired_ids],
                )
                self.conn.executemany(
                    "UPDATE resource_holdings SET status='reaped_expired', updated_wall_ms=? WHERE lease_id=? AND status='active'",
                    [(now_wall_ms(), lease_id) for lease_id in expired_ids],
                )

            generation_shards: list[str] = []
            generation_ids: list[str] = []
            if current_quota_version is not None:
                generation_rows = self.conn.execute(
                    """SELECT lease_id,shard_id FROM leases
                       WHERE status='active' AND quota_version<>?
                       ORDER BY shard_id""",
                    (int(current_quota_version),),
                ).fetchall()
                generation_ids = [row["lease_id"] for row in generation_rows]
                generation_shards = [row["shard_id"] for row in generation_rows]
                if generation_ids:
                    self.conn.executemany(
                        "UPDATE leases SET status='reaped_generation' WHERE lease_id=? AND status='active'",
                        [(lease_id,) for lease_id in generation_ids],
                    )
                    self.conn.executemany(
                        "UPDATE resource_holdings SET status='reaped_generation', updated_wall_ms=? WHERE lease_id=? AND status='active'",
                        [(now_wall_ms(), lease_id) for lease_id in generation_ids],
                    )
            for shard_id in expired_shards + generation_shards:
                self.conn.execute(
                    "UPDATE shard_state SET status='frozen_reclaimed', updated_wall_ms=? WHERE shard_id=? AND status='running'",
                    (now_wall_ms(), shard_id),
                )
            if expired_shards or generation_shards:
                self._append_event(
                    "leases_reaped",
                    None,
                    {"expired": expired_shards, "generation": generation_shards, "mutex_rule": "expiry_first_then_generation_disjoint"},
                )
            self.conn.commit()
            return {"expired": expired_shards, "generation": generation_shards}
        except Exception:
            self.conn.rollback()
            raise

    def recover_interrupted(self) -> dict[str, Any]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                "SELECT lease_id,shard_id FROM leases WHERE status='active' ORDER BY shard_id"
            ).fetchall()
            unknown = [row["shard_id"] for row in rows]
            lease_ids = [row["lease_id"] for row in rows]
            if lease_ids:
                self.conn.executemany(
                    "UPDATE leases SET status='unknown_reclaimed' WHERE lease_id=? AND status='active'",
                    [(lease_id,) for lease_id in lease_ids],
                )
                self.conn.executemany(
                    "UPDATE resource_holdings SET status='reclaimed_unknown', updated_wall_ms=? WHERE lease_id=? AND status='active'",
                    [(now_wall_ms(), lease_id) for lease_id in lease_ids],
                )
                self.conn.executemany(
                    "UPDATE shard_state SET status='unknown', content_digest=?, updated_wall_ms=? WHERE shard_id=?",
                    [(content_digest({"status": "unknown", "shard_id": shard_id}), now_wall_ms(), shard_id) for shard_id in unknown],
                )
            if unknown:
                self._append_event("recovery_replay", None, {"unknown_shards": unknown, "reclaimed_leases": lease_ids})
            self.conn.commit()
            return {"unknown_shards": unknown, "reclaimed_leases": lease_ids}
        except Exception:
            self.conn.rollback()
            raise

    def commit_shard_completion(
        self,
        *,
        receipt: dict[str, Any],
        lease: dict[str, Any],
        water_records: Iterable[dict[str, Any]],
        topology_index: int,
        completed_count: int,
        snapshot_version: int,
    ) -> dict[str, Any]:
        shard_id = receipt["shard_id"]
        node_id = receipt["node_id"]
        quota_version = int(receipt["quota_version"])
        loss = receipt["loss_summary"]
        completion = {
            "format_version": LEDGER_FORMAT_VERSION,
            "receipt": receipt,
            "lease_attribution": {
                "lease_id": lease["lease_id"],
                "shard_id": shard_id,
                "node_id": node_id,
                "lease_version": lease["lease_version"],
                "granted_monotonic_ns": lease["granted_monotonic_ns"],
                "expires_monotonic_ns": lease["expires_monotonic_ns"],
            },
            "resource_occupancy": {
                "lease_id": lease["lease_id"],
                "shard_id": shard_id,
                "resource_units": lease.get("resource_units"),
                "status_at_commit": "committed_then_released",
            },
            "quota_version": quota_version,
            "progress_cursor": {
                "last_topology_index": int(topology_index),
                "last_shard_id": shard_id,
                "completed_count": int(completed_count),
                "snapshot_version": int(snapshot_version),
            },
        }
        outcome_blob = canonical_bytes(completion)
        digest = sha256_hex(outcome_blob)
        wall_ms = now_wall_ms()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            event_seq = self._append_event("shard_committed", shard_id, completion)
            self.conn.execute(
                   """INSERT INTO shard_outcomes VALUES(?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(shard_id) DO UPDATE SET
                     status=excluded.status, node_id=excluded.node_id,
                     quota_version=excluded.quota_version, loss_digest=excluded.loss_digest,
                     outcome_blob=excluded.outcome_blob, format_version=excluded.format_version,
                     content_digest=excluded.content_digest, committed_seq=excluded.committed_seq,
                     updated_wall_ms=excluded.updated_wall_ms""",
                (shard_id, "committed", node_id, quota_version, loss["loss_digest"], outcome_blob,
                 LEDGER_FORMAT_VERSION, digest, event_seq, wall_ms),
            )
            self.conn.execute(
                "UPDATE leases SET status='completed' WHERE lease_id=? AND status='active'",
                (lease["lease_id"],),
            )
            self.conn.execute(
                "UPDATE resource_holdings SET status='released_committed', updated_wall_ms=? WHERE lease_id=? AND status='active'",
                (wall_ms, lease["lease_id"]),
            )
            state_digest = content_digest({"shard_id": shard_id, "status": "committed", "event_seq": event_seq})
            self.conn.execute(
                """INSERT INTO shard_state VALUES(?,?,?,?,?,?)
                   ON CONFLICT(shard_id) DO UPDATE SET status='committed', current_lease_id=?,
                     content_digest=?, updated_wall_ms=?""",
                (shard_id, "committed", lease["lease_id"], LEDGER_FORMAT_VERSION, state_digest, wall_ms,
                 lease["lease_id"], state_digest, wall_ms),
            )
            for record in water_records:
                record_key = record["record_key"]
                scope = record.get("scope", "shard")
                record_blob = canonical_bytes(record)
                self.conn.execute(
                    """INSERT INTO water_records(record_key,scope,shard_id,node_id,quota_version,record_blob,format_version,content_digest,updated_wall_ms)
                       VALUES(?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(record_key) DO UPDATE SET
                         record_blob=excluded.record_blob, quota_version=excluded.quota_version,
                         content_digest=excluded.content_digest, updated_wall_ms=excluded.updated_wall_ms""",
                    (record_key, scope, record.get("shard_id", shard_id), record.get("node_id"),
                     quota_version, record_blob, LEDGER_FORMAT_VERSION, sha256_hex(record_blob), wall_ms),
                )
            cursor_blob = canonical_bytes(completion["progress_cursor"])
            self.conn.execute(
                """INSERT INTO progress_cursor(id,last_topology_index,last_shard_id,completed_count,snapshot_version,cursor_blob,format_version,content_digest,updated_wall_ms)
                   VALUES(1,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     last_topology_index=excluded.last_topology_index,
                     last_shard_id=excluded.last_shard_id,
                     completed_count=excluded.completed_count,
                     snapshot_version=excluded.snapshot_version,
                     cursor_blob=excluded.cursor_blob,
                     content_digest=excluded.content_digest,
                     updated_wall_ms=excluded.updated_wall_ms""",
                (int(topology_index), shard_id, int(completed_count), int(snapshot_version),
                 cursor_blob, LEDGER_FORMAT_VERSION, sha256_hex(cursor_blob), wall_ms),
            )
            self._upsert_meta("last_committed_shard", text_value=shard_id)
            self.conn.commit()
            return {"event_seq": event_seq, "content_digest": digest, "completion": completion}
        except Exception:
            self.conn.rollback()
            raise

    def committed_outcomes(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute("SELECT shard_id, outcome_blob, content_digest FROM shard_outcomes WHERE status='committed' ORDER BY shard_id").fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            outcome = json.loads(bytes(row["outcome_blob"]).decode("utf-8"))
            result[row["shard_id"]] = outcome
        return result

    def canonical_blob(self, table: str, key_column: str, key: str, blob_column: str) -> bytes | None:
        allowed_tables = {
            "shard_outcomes": ("shard_id", "outcome_blob"),
            "leases": ("lease_id", "lease_blob"),
            "water_records": ("record_key", "record_blob"),
        }
        if table not in allowed_tables or (key_column, blob_column) != allowed_tables[table]:
            raise ValueError("unsupported canonical blob query")
        row = self.conn.execute(f"SELECT {blob_column} AS b FROM {table} WHERE {key_column}=?", (key,)).fetchone()
        return None if row is None else bytes(row["b"])

    def replay_and_verify(self) -> dict[str, Any]:
        events = self.conn.execute(
            "SELECT seq,event_type,shard_id,payload_blob,content_digest FROM commit_events ORDER BY seq"
        ).fetchall()
        checked: list[str] = []
        expected_outcomes: dict[str, dict[str, Any]] = {}
        last_commit: dict[str, Any] | None = None
        commit_event_count = 0
        for event in events:
            payload = json.loads(bytes(event["payload_blob"]).decode("utf-8"))
            if event["event_type"] == "invalidation":
                for shard_id in payload.get("shard_ids", []):
                    expected_outcomes.pop(shard_id, None)
                continue
            if event["event_type"] != "shard_committed":
                continue
            shard_id = event["shard_id"]
            expected_outcomes[shard_id] = {
                "payload": payload,
                "payload_blob": bytes(event["payload_blob"]),
                "content_digest": event["content_digest"],
            }
            last_commit = payload
            commit_event_count += 1

        for shard_id in sorted(expected_outcomes):
            expected_entry = expected_outcomes[shard_id]
            expected = expected_entry["payload"]
            row = self.conn.execute("SELECT * FROM shard_outcomes WHERE shard_id=?", (shard_id,)).fetchone()
            if row is None:
                raise ReplayConsistencyError("重放事件缺少账本结局行", entry=shard_id, field="shard_outcomes.row")
            fields = [
                ("status", "committed", row["status"]),
                ("node_id", expected["receipt"]["node_id"], row["node_id"]),
                ("quota_version", expected["quota_version"], row["quota_version"]),
                ("loss_digest", expected["receipt"]["loss_summary"]["loss_digest"], row["loss_digest"]),
                ("outcome_blob", expected_entry["payload_blob"], bytes(row["outcome_blob"])),
                ("content_digest", expected_entry["content_digest"], row["content_digest"]),
            ]
            for field_name, expected_value, actual_value in fields:
                if expected_value != actual_value:
                    raise ReplayConsistencyError(
                        "重放发现首个不一致分片字段",
                        entry=shard_id,
                        field=field_name,
                        details={"expected": str(expected_value)[:200], "actual": str(actual_value)[:200]},
                    )
            loss_fields = [
                ("loss_summary.loss_digest", expected["receipt"]["loss_summary"]["loss_digest"],
                 json.loads(bytes(row["outcome_blob"]).decode())["receipt"]["loss_summary"]["loss_digest"]),
                ("loss_summary.participating_rows", expected["receipt"]["loss_summary"]["participating_rows"],
                 json.loads(bytes(row["outcome_blob"]).decode())["receipt"]["loss_summary"]["participating_rows"]),
            ]
            for field_name, expected_value, actual_value in loss_fields:
                if expected_value != actual_value:
                    raise ReplayConsistencyError("重放损失字段不一致", entry=shard_id, field=field_name)
            checked.append(shard_id)

        committed_rows = self.conn.execute(
            "SELECT shard_id FROM shard_outcomes WHERE status='committed' ORDER BY shard_id"
        ).fetchall()
        for row in committed_rows:
            if row["shard_id"] not in expected_outcomes:
                raise ReplayConsistencyError(
                    "账本存在事件流无法重建的已提交分片",
                    entry=row["shard_id"],
                    field="shard_outcomes.status",
                )

        cursor = self.conn.execute("SELECT * FROM progress_cursor WHERE id=1").fetchone()
        if last_commit is not None and cursor is None:
            first_shard = last_commit["progress_cursor"]["last_shard_id"]
            raise ReplayConsistencyError("缺少进度游标", entry=first_shard, field="progress_cursor")
        if cursor is not None and last_commit is not None:
            last_expected = last_commit["progress_cursor"]
            for field_name in ("last_shard_id", "completed_count", "snapshot_version"):
                actual = cursor[field_name] if field_name in cursor.keys() else None
                expected_value = last_expected[field_name]
                if actual != expected_value:
                    raise ReplayConsistencyError(
                        "进度游标与重放不一致",
                        entry=last_expected["last_shard_id"],
                        field=f"progress_cursor.{field_name}",
                    )
        return {"ok": True, "replayed_shards": checked, "event_count": len(events), "commit_event_count": commit_event_count}

    def build_snapshot_state(self) -> dict[str, Any]:
        outcomes = self.committed_outcomes()
        states = {
            row["shard_id"]: dict(row)
            for row in self.conn.execute("SELECT shard_id,status,current_lease_id,content_digest,updated_wall_ms FROM shard_state ORDER BY shard_id")
        }
        leases = [
            {k: row[k] for k in row.keys() if k not in {"lease_blob"}}
            for row in self.conn.execute("SELECT * FROM leases ORDER BY shard_id,grant_number")
        ]
        holdings = [dict(row) for row in self.conn.execute("SELECT lease_id,shard_id,node_id,resource_units,status,content_digest FROM resource_holdings ORDER BY lease_id")]
        water = [
            {"record_key": row["record_key"], "scope": row["scope"], "shard_id": row["shard_id"], "quota_version": row["quota_version"], "content_digest": row["content_digest"]}
            for row in self.conn.execute("SELECT * FROM water_records ORDER BY record_key")
        ]
        cursor_row = self.conn.execute("SELECT * FROM progress_cursor WHERE id=1").fetchone()
        cursor = None if cursor_row is None else {k: cursor_row[k] for k in cursor_row.keys() if k != "cursor_blob"}
        version_row = self.get_meta("snapshot_version")
        snapshot_version = int(version_row["int_value"]) if version_row and version_row["int_value"] is not None else 1
        return {
            "format_version": LEDGER_FORMAT_VERSION,
            "snapshot_version": snapshot_version,
            "shard_states": states,
            "outcomes": outcomes,
            "leases": leases,
            "resource_holdings": holdings,
            "water_records": water,
            "progress_cursor": cursor,
        }

    def save_snapshot_file(self, path: str | Path, snapshot_version: int | None = None) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        current = self.get_meta("snapshot_version")
        version = int(snapshot_version) if snapshot_version is not None else ((int(current["int_value"]) + 1) if current and current["int_value"] is not None else 1)
        self.set_meta("snapshot_version", int_value=version)
        state = self.build_snapshot_state()
        state["snapshot_version"] = version
        state["format_version"] = LEDGER_FORMAT_VERSION
        digest = content_digest(state)
        envelope = {"format_version": LEDGER_FORMAT_VERSION, "snapshot_version": version, "content_digest": digest, "state": state}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        self.conn.execute(
            "INSERT OR REPLACE INTO snapshots(snapshot_version,format_version,content_digest,snapshot_blob,created_wall_ms) VALUES(?,?,?,?,?)",
            (version, LEDGER_FORMAT_VERSION, digest, canonical_bytes(envelope), now_wall_ms()),
        )
        self.conn.commit()
        return envelope


def load_or_replace_snapshot(path: str | Path, ledger: Ledger) -> tuple[dict[str, Any], bool]:
    path = Path(path)
    should_replace = not path.exists()
    reason = "missing"
    envelope: dict[str, Any] | None = None
    if path.exists():
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            state = envelope["state"]
            expected_digest = envelope["content_digest"]
            if envelope.get("format_version") != LEDGER_FORMAT_VERSION:
                should_replace = True
                reason = "format_version"
            elif content_digest(state) != expected_digest:
                should_replace = True
                reason = "content_digest"
            elif not isinstance(envelope.get("snapshot_version"), int):
                should_replace = True
                reason = "snapshot_version"
        except Exception:
            should_replace = True
            reason = "parse_error"

    if should_replace:
        envelope = ledger.save_snapshot_file(path, snapshot_version=1)
        return {"replaced": True, "reason": reason, "snapshot": envelope}, True

    assert envelope is not None
    ledger_version_row = ledger.get_meta("snapshot_version")
    ledger_version = int(ledger_version_row["int_value"]) if ledger_version_row and ledger_version_row["int_value"] is not None else None
    if ledger_version is not None and ledger_version != int(envelope["snapshot_version"]):
        envelope = ledger.save_snapshot_file(path, snapshot_version=ledger_version)
        return {"replaced": True, "reason": "snapshot_version_mismatch", "snapshot": envelope}, True
    return {"replaced": False, "reason": "valid", "snapshot": envelope}, False
