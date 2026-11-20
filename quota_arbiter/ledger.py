"""SQLite 账本：批次原子提交、内容摘要链、强制终止恢复。"""
from __future__ import annotations

import json
import os
import sqlite3

from . import LEDGER_FORMAT_VERSION
from .errors import LedgerError
from .util import digest_of, rolling_digest

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tenant_state (
    tid TEXT PRIMARY KEY,
    quota_remaining INTEGER NOT NULL,
    burst_remaining INTEGER NOT NULL,
    inflight INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pool_state (
    pid TEXT PRIMARY KEY,
    capacity_remaining INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    rid TEXT PRIMARY KEY,
    tid TEXT NOT NULL,
    pid TEXT NOT NULL,
    arrival INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    reason TEXT,
    basis TEXT NOT NULL,
    decided_at INTEGER NOT NULL,
    granted_at INTEGER,
    completed_at INTEGER,
    wait_ms INTEGER NOT NULL,
    tenant_quota_remaining INTEGER NOT NULL,
    tenant_burst_remaining INTEGER NOT NULL,
    inflight INTEGER NOT NULL,
    pool_capacity_remaining INTEGER NOT NULL,
    demand INTEGER NOT NULL,
    enqueued INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS queue_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rid TEXT NOT NULL,
    tid TEXT NOT NULL,
    pid TEXT NOT NULL,
    kind TEXT NOT NULL,
    at_ms INTEGER NOT NULL,
    wait_ms INTEGER NOT NULL,
    basis TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_index INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    request_count INTEGER NOT NULL,
    skipped_count INTEGER NOT NULL,
    rids_json TEXT NOT NULL,
    state_digest TEXT NOT NULL,
    head_digest TEXT NOT NULL,
    outcome_digests_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS unknown_requests (
    rid TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    detected_batch INTEGER
);
"""


class Ledger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------------ #
    # 初始化与会话标识
    # ------------------------------------------------------------------ #
    def initialize_session(self, tenants, pools, path_name: str, spec_digest: str):
        head = self.get_meta("head_digest")
        if head is None:
            self.conn.execute("BEGIN")
            self.set_meta("format_version", str(LEDGER_FORMAT_VERSION))
            self.set_meta("path", path_name)
            self.set_meta("spec_digest", spec_digest)
            self.set_meta("head_digest", "GENESIS")
            for tenant in tenants:
                self.conn.execute(
                    "INSERT INTO tenant_state(tid, quota_remaining, burst_remaining, inflight) "
                    "VALUES (?,?,?,?)",
                    (tenant.tid, tenant.quota, tenant.burst, 0),
                )
            for pool in pools:
                self.conn.execute(
                    "INSERT INTO pool_state(pid, capacity_remaining) VALUES (?,?)",
                    (pool.pid, pool.capacity),
                )
            self.conn.commit()

    def get_meta(self, key: str):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str):
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def verify_format(self):
        version = self.get_meta("format_version")
        if version is None:
            raise LedgerError("账本缺少格式版本", field="format_version")
        if int(version) != LEDGER_FORMAT_VERSION:
            raise LedgerError(
                "账本格式版本不符",
                field="format_version",
                detail={"found": int(version), "expected": LEDGER_FORMAT_VERSION},
            )

    def verify_spec_digest(self, spec_digest: str):
        stored = self.get_meta("spec_digest")
        if stored != spec_digest:
            raise LedgerError(
                "账本内容摘要不符：声明摘要与账本记录不一致",
                field="spec_digest",
                detail={"stored": stored, "current": spec_digest},
            )

    # ------------------------------------------------------------------ #
    # 批次原子提交
    # ------------------------------------------------------------------ #
    def append_batch(self, batch_index, path_name, batch_meta, outcomes, queue_events,
                     tenant_state, pool_state, *, crash_in_batch=False):
        state_block = {
            "tenants": {k: tenant_state[k] for k in sorted(tenant_state)},
            "pools": {k: pool_state[k] for k in sorted(pool_state)},
            "rids": [o.rid for o in outcomes],
            # 结局与排队轨迹纳入摘要，任何账本内容篡改都会破坏链式摘要
            "outcomes": [list(o.canonical_tuple()) for o in outcomes],
            "queue_events": [
                [e.rid, e.tid, e.pid, e.kind, e.at_ms, e.wait_ms, e.basis, e.detail]
                for e in queue_events
            ],
        }
        state_digest = digest_of(state_block)
        outcome_digests = {
            o.rid: digest_of(list(o.canonical_tuple())) for o in outcomes
        }
        head = self.get_meta("head_digest") or "GENESIS"
        head_digest = rolling_digest(head, {"batch_index": batch_index, "state_digest": state_digest})

        self.conn.execute("BEGIN")
        for outcome in outcomes:
            self._upsert_outcome(outcome)
        for event in queue_events:
            self.conn.execute(
                "INSERT INTO queue_events(rid, tid, pid, kind, at_ms, wait_ms, basis, detail) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (event.rid, event.tid, event.pid, event.kind, event.at_ms,
                 event.wait_ms, event.basis, event.detail),
            )
        for tid, values in tenant_state.items():
            self.conn.execute(
                "UPDATE tenant_state SET quota_remaining=?, burst_remaining=?, inflight=? WHERE tid=?",
                (values[0], values[1], values[2], tid),
            )
        for pid, remaining in pool_state.items():
            self.conn.execute(
                "UPDATE pool_state SET capacity_remaining=? WHERE pid=?",
                (remaining, pid),
            )
        self.conn.execute(
            "INSERT INTO batches(batch_index, path, request_count, skipped_count, rids_json, "
            "state_digest, head_digest, outcome_digests_json) VALUES (?,?,?,?,?,?,?,?)",
            (
                batch_index,
                path_name,
                batch_meta.get("request_count", len(outcomes)),
                batch_meta.get("skipped_count", 0),
                json.dumps(batch_meta.get("rids", []), ensure_ascii=False),
                state_digest,
                head_digest,
                json.dumps(outcome_digests, ensure_ascii=False),
            ),
        )
        self.set_meta("head_digest", head_digest)
        if crash_in_batch:
            # 模拟强制终止：不提交即关闭，已写入内容随事务回滚丢弃。
            self.conn.close()
            raise LedgerError(
                "注入强制终止：批次事务未提交即中断",
                ref_id=outcomes[0].rid if outcomes else None,
                field="transaction",
            )
        self.conn.commit()
        return {"batch_index": batch_index, "state_digest": state_digest, "head_digest": head_digest}

    def _upsert_outcome(self, outcome):
        # 一条请求在全生命周期只保留一条结局记录：入队时先落 QUEUE，
        # 唤醒为 GRANT 后在同一 rid 上更新为最终态，绝不产生第二条记录。
        existing = self.conn.execute(
            "SELECT rid FROM outcomes WHERE rid=?", (outcome.rid,)
        ).fetchone()
        if existing is not None:
            self.conn.execute(
                "UPDATE outcomes SET tid=?, pid=?, arrival=?, verdict=?, reason=?, basis=?, "
                "decided_at=?, granted_at=?, completed_at=?, wait_ms=?, "
                "tenant_quota_remaining=?, tenant_burst_remaining=?, inflight=?, "
                "pool_capacity_remaining=?, demand=?, enqueued=? WHERE rid=?",
                (
                    outcome.tid, outcome.pid, outcome.arrival, outcome.verdict,
                    outcome.reason, outcome.basis, outcome.decided_at, outcome.granted_at,
                    outcome.completed_at, outcome.wait_ms, outcome.tenant_quota_remaining,
                    outcome.tenant_burst_remaining, outcome.inflight,
                    outcome.pool_capacity_remaining, outcome.demand,
                    1 if outcome.enqueued else 0, outcome.rid,
                ),
            )
            return
        self.conn.execute(
            "INSERT INTO outcomes(rid, tid, pid, arrival, verdict, reason, basis, decided_at, "
            "granted_at, completed_at, wait_ms, tenant_quota_remaining, tenant_burst_remaining, "
            "inflight, pool_capacity_remaining, demand, enqueued) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                outcome.rid, outcome.tid, outcome.pid, outcome.arrival, outcome.verdict,
                outcome.reason, outcome.basis, outcome.decided_at, outcome.granted_at,
                outcome.completed_at, outcome.wait_ms, outcome.tenant_quota_remaining,
                outcome.tenant_burst_remaining, outcome.inflight,
                outcome.pool_capacity_remaining, outcome.demand, 1 if outcome.enqueued else 0,
            ),
        )

    # ------------------------------------------------------------------ #
    # 摘要链校验与恢复
    # ------------------------------------------------------------------ #
    def _outcome_canonical(self, row):
        return [
            row["rid"], row["tid"], row["pid"], row["arrival"], row["verdict"],
            row["reason"] if row["reason"] is not None else "",
            row["basis"], row["decided_at"],
            row["granted_at"] if row["granted_at"] is not None else -1,
            row["completed_at"] if row["completed_at"] is not None else -1,
            row["wait_ms"], row["tenant_quota_remaining"], row["tenant_burst_remaining"],
            row["inflight"], row["pool_capacity_remaining"], row["demand"], row["enqueued"],
        ]

    def _recompute_batch_digest(self, rids, batch_row):
        """从实际表内容重算批次摘要，任何篡改都会改变结果。

        返回 (摘要, 首个内容不一致的请求标识)。不一致请求通过与批次写入时
        的结局摘要比对定位；若状态表被改则标识为状态维度。
        """
        tenant_rows = self.conn.execute("SELECT * FROM tenant_state").fetchall()
        pool_rows = self.conn.execute("SELECT * FROM pool_state").fetchall()
        outcomes = []
        mismatch_rid = None
        placeholders = ",".join("?" for _ in rids)
        stored_outcome_digests = json.loads(batch_row["outcome_digests_json"])
        for o_row in self.conn.execute(
            f"SELECT * FROM outcomes WHERE rid IN ({placeholders}) ORDER BY rid", rids
        ).fetchall():
            canonical = self._outcome_canonical(o_row)
            outcomes.append(canonical)
            if (
                mismatch_rid is None
                and stored_outcome_digests.get(o_row["rid"])
                != digest_of(canonical)
            ):
                mismatch_rid = o_row["rid"]
        events = []
        for row in self.conn.execute(
            f"SELECT rid, tid, pid, kind, at_ms, wait_ms, basis, detail "
            f"FROM queue_events WHERE rid IN ({placeholders}) ORDER BY id", rids
        ).fetchall():
            events.append([row["rid"], row["tid"], row["pid"], row["kind"], row["at_ms"],
                           row["wait_ms"], row["basis"], row["detail"]])
        block = {
            "tenants": {
                row["tid"]: [row["quota_remaining"], row["burst_remaining"], row["inflight"]]
                for row in sorted(tenant_rows, key=lambda r: r["tid"])
            },
            "pools": {
                row["pid"]: row["capacity_remaining"]
                for row in sorted(pool_rows, key=lambda r: r["pid"])
            },
            "rids": rids,
            "outcomes": outcomes,
            "queue_events": events,
        }
        return digest_of(block), mismatch_rid

    def verify_digest_chain(self):
        head = "GENESIS"
        rows = self.conn.execute(
            "SELECT batch_index, state_digest, head_digest, rids_json, "
            "outcome_digests_json FROM batches ORDER BY batch_index"
        ).fetchall()
        for row in rows:
            rids = json.loads(row["rids_json"])
            # 先从实际表内容重算内容摘要，再校验链式摘要
            recomputed, mismatch_rid = self._recompute_batch_digest(rids, row)
            if recomputed != row["state_digest"]:
                raise LedgerError(
                    "账本内容摘要不符：批次内容被篡改",
                    ref_id=mismatch_rid or str(row["batch_index"]),
                    field="state_digest",
                    detail={
                        "batch_index": row["batch_index"],
                        "stored": row["state_digest"],
                        "recomputed": recomputed,
                    },
                )
            expected = rolling_digest(
                head, {"batch_index": row["batch_index"], "state_digest": row["state_digest"]}
            )
            if expected != row["head_digest"]:
                raise LedgerError(
                    "账本内容摘要不符：批次摘要链断裂",
                    ref_id=str(row["batch_index"]),
                    field="head_digest",
                    detail={"expected": expected, "stored": row["head_digest"]},
                )
            head = row["head_digest"]
        stored_head = self.get_meta("head_digest")
        if stored_head != head:
            raise LedgerError(
                "账本内容摘要不符：头部摘要与批次链不一致",
                field="head_digest",
                detail={"stored": stored_head, "computed": head},
            )
        return head

    def tamper_outcome(self, rid: str, new_verdict: str):
        """异常注入：直接改写一条结局，制造摘要不符（仅改数据文件，不改源码）。"""
        self.conn.execute(
            "UPDATE outcomes SET verdict=? WHERE rid=?", (new_verdict, rid)
        )
        self.conn.commit()

    def recovery_point(self, all_rids: list[str]):
        """只依据已完整提交的批次恢复，未提交请求标注为未知。"""
        self.verify_format()
        committed_rows = self.conn.execute(
            "SELECT rids_json FROM batches ORDER BY batch_index"
        ).fetchall()
        committed: list[str] = []
        for row in committed_rows:
            committed.extend(json.loads(row["rids_json"]))
        committed_set = set(committed)
        unknown = [rid for rid in all_rids if rid not in committed_set]
        for rid in unknown:
            self.conn.execute(
                "INSERT OR IGNORE INTO unknown_requests(rid, reason, detected_batch) "
                "VALUES (?,?,?)",
                (rid, "IN_FLIGHT_NOT_COMMITTED", None),
            )
        self.conn.commit()
        state_rows = self.conn.execute("SELECT * FROM tenant_state").fetchall()
        pool_rows = self.conn.execute("SELECT * FROM pool_state").fetchall()
        return {
            "last_committed_rid": committed[-1] if committed else None,
            "committed_count": len(committed),
            "unknown_count": len(unknown),
            "unknown_rids": unknown,
            "tenant_state": {
                row["tid"]: (
                    row["quota_remaining"], row["burst_remaining"], row["inflight"]
                )
                for row in state_rows
            },
            "pool_state": {row["pid"]: row["capacity_remaining"] for row in pool_rows},
            "head_digest": self.get_meta("head_digest"),
        }
