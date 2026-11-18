"""SQLite 持久化账本与强制终止恢复。

设计原则：
- 所有业务记录先写入 pending_operations（状态 OPEN），完整提交后改写 COMMITTED；
- 重启恢复时，OPEN 记录对应规则一律标注 UNKNOWN，要求重新编译/重跑；
- 只依据已 COMMITTED 的记录恢复计划版本与依赖图状态；
- 不生成强制终止时刻的“退出报告”，只输出恢复报告与未知集合。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS rulesets (
    ruleset_version TEXT PRIMARY KEY,
    rules_json      TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graphs (
    graph_version   TEXT PRIMARY KEY,
    ruleset_version TEXT NOT NULL,
    topo_json       TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    plan_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_version       INTEGER NOT NULL UNIQUE,
    graph_version      TEXT NOT NULL,
    ruleset_version    TEXT NOT NULL,
    plan_hash          TEXT NOT NULL,
    payload_json       TEXT NOT NULL,
    artifact_versions  TEXT NOT NULL,
    recompiled_json    TEXT NOT NULL,
    reused_json        TEXT NOT NULL,
    created_at         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS invalidations (
    invalidation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_version    INTEGER NOT NULL,
    sequence_json   TEXT NOT NULL,
    trace_json      TEXT NOT NULL,
    summary_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    observation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_version    INTEGER NOT NULL,
    digest_json     TEXT NOT NULL,
    records_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_path   TEXT NOT NULL UNIQUE,
    ruleset_version TEXT NOT NULL,
    graph_version   TEXT NOT NULL,
    plan_version    INTEGER NOT NULL,
    content_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS compiler_cache (
    cache_id    INTEGER PRIMARY KEY CHECK (cache_id = 1),
    cache_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_operations (
    op_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    op_type     TEXT NOT NULL,           -- compile / shape / snapshot
    scope_json  TEXT NOT NULL,           -- 受影响规则标识数组等
    status      TEXT NOT NULL,           -- OPEN / COMMITTED
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RecoveryReport:
    recovered_plan_version: int | None
    recovered_graph_version: str | None
    recovered_ruleset_version: str | None
    unknown_rules: list[str]
    open_operations: list[dict[str, Any]]
    ledger_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "recovered_plan_version": self.recovered_plan_version,
            "recovered_graph_version": self.recovered_graph_version,
            "recovered_ruleset_version": self.recovered_ruleset_version,
            "unknown_rules": self.unknown_rules,
            "open_operations": self.open_operations,
            "ledger_version": self.ledger_version,
            "note": "OPEN 状态操作在强制终止时未提交，相关规则标注为 UNKNOWN，"
                    "须重新编译或重跑；不生成终止时刻退出报告。",
        }


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=FULL;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def execute(self, sql: str, params: tuple = ()):
        return self.conn.execute(sql, params)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---- pending 操作 ---------------------------------------------------
    def begin_operation(self, op_type: str, scope: list[str]) -> int:
        now = _utcnow()
        cur = self.conn.execute(
            "INSERT INTO pending_operations(op_type, scope_json, status, "
            "created_at, updated_at) VALUES (?, ?, 'OPEN', ?, ?)",
            (op_type, canonical_text(scope), now, now))
        self.conn.commit()
        return int(cur.lastrowid)

    def commit_operation(self, op_id: int) -> None:
        self.conn.execute(
            "UPDATE pending_operations SET status='COMMITTED', updated_at=? "
            "WHERE op_id=?", (_utcnow(), op_id))
        self.conn.commit()

    def crash_without_commit(self, op_id: int) -> None:
        """测试辅助：保留 OPEN 状态模拟强制终止（故意不 commit）。"""
        assert self.conn.execute(
            "SELECT status FROM pending_operations WHERE op_id=?",
            (op_id,)).fetchone()["status"] == "OPEN"

    # ---- 规则集 / 图 ----------------------------------------------------
    def save_ruleset(self, ruleset_version: str, rules: list[Any]) -> None:
        payload = [r.to_dict() for r in rules]
        self.conn.execute(
            "INSERT OR REPLACE INTO rulesets(ruleset_version, rules_json, created_at) "
            "VALUES (?, ?, ?)",
            (ruleset_version, canonical_text(payload), _utcnow()))
        self.conn.commit()

    def save_graph(self, graph_version: str, ruleset_version: str,
                   topo: list[str]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO graphs(graph_version, ruleset_version, "
            "topo_json, created_at) VALUES (?, ?, ?, ?)",
            (graph_version, ruleset_version, canonical_text(topo), _utcnow()))
        self.conn.commit()

    # ---- 计划 -----------------------------------------------------------
    def next_plan_version(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(plan_version), 0) AS v FROM plans").fetchone()
        return int(row["v"]) + 1

    def save_plan(self, plan: Any, graph_version: str,
                  ruleset_version: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO plans(plan_version, graph_version, ruleset_version, "
            "plan_hash, payload_json, artifact_versions, recompiled_json, "
            "reused_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (plan.plan_version, graph_version, ruleset_version, plan.plan_hash,
             canonical_text(plan.payload()),
             canonical_text(plan.artifact_versions),
             canonical_text(plan.recompiled), canonical_text(plan.reused),
             _utcnow()))
        self.conn.commit()

    def save_invalidation(self, plan_version: int, sequence: list[str],
                          trace: list[Any], summary: dict[str, list[str]]) -> None:
        self.conn.execute(
            "INSERT INTO invalidations(plan_version, sequence_json, trace_json, "
            "summary_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (plan_version, canonical_text(sequence),
             canonical_text([t.to_dict() for t in trace]),
             canonical_text(summary), _utcnow()))
        self.conn.commit()

    def latest_plan(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM plans ORDER BY plan_version DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    # ---- 观测 -----------------------------------------------------------
    def save_observation(self, plan_version: int, digest: dict[str, Any],
                         records: list[dict[str, Any]]) -> None:
        self.conn.execute(
            "INSERT INTO observations(plan_version, digest_json, records_json, "
            "created_at) VALUES (?, ?, ?, ?)",
            (plan_version, canonical_text(digest), canonical_text(records),
             _utcnow()))
        self.conn.commit()

    def latest_observation(self, plan_version: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM observations WHERE plan_version=? "
            "ORDER BY observation_id DESC LIMIT 1", (plan_version,)).fetchone()
        return dict(row) if row else None

    # ---- 快照登记 -------------------------------------------------------
    def register_snapshot(self, path: str, content: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO snapshots(snapshot_path, ruleset_version, "
            "graph_version, plan_version, content_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (path, content["ruleset_version"], content["graph_version"],
             content["plan_version"], canonical_text(content), _utcnow()))
        self.conn.commit()

    # ---- 编译器缓存 -----------------------------------------------------
    def save_compiler_cache(self, cache: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO compiler_cache(cache_id, cache_json, updated_at) "
            "VALUES (1, ?, ?) ON CONFLICT(cache_id) DO UPDATE SET "
            "cache_json=excluded.cache_json, updated_at=excluded.updated_at",
            (canonical_text(cache), _utcnow()))
        self.conn.commit()

    def load_compiler_cache(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT cache_json FROM compiler_cache WHERE cache_id=1").fetchone()
        return json.loads(row["cache_json"]) if row else None

    # ---- 恢复 -----------------------------------------------------------
    def recover(self) -> RecoveryReport:
        latest = self.latest_plan()
        graph_row = self.conn.execute(
            "SELECT * FROM graphs ORDER BY rowid DESC LIMIT 1").fetchone()
        open_rows = self.conn.execute(
            "SELECT * FROM pending_operations WHERE status='OPEN' ORDER BY op_id"
        ).fetchall()
        unknown: set[str] = set()
        open_ops: list[dict[str, Any]] = []
        for row in open_rows:
            scope = json.loads(row["scope_json"])
            unknown.update(scope)
            open_ops.append({
                "op_id": row["op_id"], "op_type": row["op_type"],
                "scope": scope, "created_at": row["created_at"],
            })
        return RecoveryReport(
            recovered_plan_version=int(latest["plan_version"]) if latest else None,
            recovered_graph_version=graph_row["graph_version"] if graph_row else None,
            recovered_ruleset_version=latest["ruleset_version"] if latest else None,
            unknown_rules=sorted(unknown),
            open_operations=open_ops,
            ledger_version=self.ledger_version(),
        )

    def ledger_version(self) -> str:
        row = self.conn.execute("PRAGMA data_version").fetchone()
        plan_count = self.conn.execute("SELECT COUNT(*) AS c FROM plans").fetchone()["c"]
        return f"sqlite:{sqlite3.sqlite_version}:plans={plan_count}:data_version={row[0]}"

    def mark_open_operations_resolved(self) -> int:
        cur = self.conn.execute(
            "UPDATE pending_operations SET status='COMMITTED', updated_at=? "
            "WHERE status='OPEN'", (_utcnow(),))
        self.conn.commit()
        return cur.rowcount
