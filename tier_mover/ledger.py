"""SQLite 元数据账本。

设计要点：
- 整数格式版本 LEDGER_FORMAT_VERSION 与账本内容摘要链
- 每个迁移批次在单事务内写入迁移记录、层占用快照与进度游标
- 策略版本单调递增；旧版本的计划/轨迹/结论整体标记为 stale，不参与当前视图
- 所有层占用、余量、对象数、成本均为整数
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION
from .errors import VersionConflictError
from .manifest import Manifest, canonical_json, manifest_from_validated


@dataclass(frozen=True)
class TierState:
    tier_id: str
    role: str
    capacity_bytes: int
    used_bytes: int
    free_bytes: int
    object_count: int
    unit_cost: int
    rw_weight: int
    eviction_priority: int
    cost: int


class Ledger:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def close(self) -> None:
        self.conn.close()

    # ---------- schema ----------
    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS spec_versions (
                spec_version INTEGER PRIMARY KEY,
                registered_ms INTEGER NOT NULL,
                active INTEGER NOT NULL,
                spec_blob TEXT NOT NULL,
                spec_digest TEXT NOT NULL,
                tier_digest TEXT NOT NULL,
                object_digest TEXT NOT NULL,
                rules_digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tier_state (
                spec_version INTEGER NOT NULL,
                tier_id TEXT NOT NULL,
                role TEXT NOT NULL,
                capacity_bytes INTEGER NOT NULL,
                used_bytes INTEGER NOT NULL,
                object_count INTEGER NOT NULL,
                unit_cost INTEGER NOT NULL,
                rw_weight INTEGER NOT NULL,
                eviction_priority INTEGER NOT NULL,
                cost INTEGER NOT NULL,
                PRIMARY KEY (spec_version, tier_id)
            );
            CREATE TABLE IF NOT EXISTS object_state (
                key TEXT PRIMARY KEY,
                size_bytes INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                tier_id TEXT NOT NULL,
                spec_version INTEGER NOT NULL,
                last_event_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS plans (
                plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                spec_version INTEGER NOT NULL,
                plan_digest TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                created_ms INTEGER NOT NULL,
                superseded_ms INTEGER
            );
            CREATE TABLE IF NOT EXISTS plan_steps (
                plan_id INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                key TEXT NOT NULL,
                direction TEXT NOT NULL,
                source_tier TEXT NOT NULL,
                target_tier TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                heat INTEGER NOT NULL,
                recency_rank INTEGER NOT NULL,
                ranking INTEGER NOT NULL,
                PRIMARY KEY (plan_id, seq)
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id INTEGER NOT NULL,
                spec_version INTEGER NOT NULL,
                batch_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                committed_ms INTEGER,
                digest TEXT,
                payload TEXT
            );
            CREATE TABLE IF NOT EXISTS migrations (
                migration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL,
                plan_id INTEGER NOT NULL,
                spec_version INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                key TEXT NOT NULL,
                direction TEXT NOT NULL,
                source_tier TEXT NOT NULL,
                target_tier TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                move_cost INTEGER NOT NULL,
                committed_ms INTEGER NOT NULL,
                UNIQUE (plan_id, seq)
            );
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                active_spec_version INTEGER NOT NULL,
                active_plan_id INTEGER,
                next_batch_index INTEGER NOT NULL,
                last_committed_seq INTEGER NOT NULL,
                digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conclusions (
                conclusion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                spec_version INTEGER NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                created_ms INTEGER NOT NULL,
                superseded_ms INTEGER
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                event TEXT NOT NULL,
                code TEXT,
                details TEXT
            );
            """
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='format_version'").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),))
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('chain_digest', ?)",
                (hashlib.sha256(b"tier-mover-genesis").hexdigest(),))
            self.conn.execute(
                "INSERT INTO progress(id, active_spec_version, active_plan_id, "
                "next_batch_index, last_committed_seq, digest) "
                "VALUES(1, 0, NULL, 0, 0, ?)",
                (hashlib.sha256(b"progress-genesis").hexdigest(),))
            self.conn.commit()
            cols = {r[1] for r in self.conn.execute(
                "PRAGMA table_info(batches)").fetchall()}
            if "payload" not in cols:
                self.conn.execute(
                    "ALTER TABLE batches ADD COLUMN payload TEXT")
                self.conn.commit()
        else:
            fmt = int(row[0])
            if fmt != LEDGER_FORMAT_VERSION:
                raise VersionConflictError(
                    f"账本格式版本冲突: 文件版本 {fmt}，软件支持版本 "
                    f"{LEDGER_FORMAT_VERSION}",
                    field="format_version",
                    details={"file_version": fmt,
                             "supported_version": LEDGER_FORMAT_VERSION})

    # ---------- low level ----------
    @contextmanager
    def transaction(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def audit(self, ts_ms: int, event: str, code: str | None = None,
              details: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO audit_log(ts_ms, event, code, details) VALUES(?,?,?,?)",
            (ts_ms, event, code,
             json.dumps(details or {}, ensure_ascii=False, sort_keys=True)))

    def chain_digest(self) -> str:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='chain_digest'").fetchone()
        return row[0]

    def _advance_chain(self, payload: dict[str, Any]) -> str:
        base = self.chain_digest()
        new_digest = hashlib.sha256(
            base.encode("ascii") + canonical_json(payload)).hexdigest()
        self.conn.execute(
            "UPDATE meta SET value=? WHERE key='chain_digest'", (new_digest,))
        return new_digest

    def verify_chain(self) -> None:
        """按每个已提交批次存储的规范载荷重放链式摘要，检测账本篡改。"""
        rows = self.conn.execute(
            "SELECT batch_id, digest, payload FROM batches "
            "WHERE status='committed' ORDER BY batch_id").fetchall()
        digest = hashlib.sha256(b"tier-mover-genesis").hexdigest()
        for row in rows:
            if not row["payload"]:
                raise VersionConflictError(
                    f"批次 {row['batch_id']} 缺少提交载荷，无法校验摘要链",
                    field="payload", details={"batch_id": row["batch_id"]})
            digest = hashlib.sha256(
                digest.encode("ascii")
                + canonical_json(json.loads(row["payload"]))).hexdigest()
            if digest != row["digest"]:
                raise VersionConflictError(
                    f"批次 {row['batch_id']} 链式摘要校验失败，账本可能被篡改",
                    field="chain_digest",
                    details={"batch_id": row["batch_id"],
                             "stored": row["digest"], "recomputed": digest})
        if digest != self.chain_digest():
            raise VersionConflictError(
                "账本内容摘要链校验失败，账本可能被篡改或损坏",
                field="chain_digest",
                details={"stored": self.chain_digest(), "recomputed": digest})

    # ---------- spec registration / versioning ----------
    def register_spec(self, manifest: Manifest, ts_ms: int) -> int:
        """注册策略版本。同摘要复用原版本；参数变更则版本号单调递增。"""
        row = self.conn.execute(
            "SELECT spec_version FROM spec_versions WHERE spec_digest=?",
            (manifest.spec_digest,)).fetchone()
        if row is not None:
            return int(row[0])
        prev = self.conn.execute(
            "SELECT COALESCE(MAX(spec_version), 0) FROM spec_versions").fetchone()[0]
        version = max(int(prev) + 1, manifest.spec_version)
        blob = json.dumps(manifest_from_validated(manifest),
                          ensure_ascii=False, sort_keys=True)
        with self.transaction():
            self.conn.execute(
                "UPDATE spec_versions SET active=0")
            self.conn.execute(
                "INSERT INTO spec_versions(spec_version, registered_ms, active, "
                "spec_blob, spec_digest, tier_digest, object_digest, rules_digest) "
                "VALUES(?,?,1,?,?,?,?,?)",
                (version, ts_ms, blob, manifest.spec_digest,
                 manifest.tier_digest, manifest.object_digest,
                 manifest.rules_digest))
            for tier in manifest.tiers:
                existing = self.conn.execute(
                    "SELECT used_bytes, object_count FROM tier_state "
                    "WHERE spec_version=? AND tier_id=?",
                    (version - 1, tier.tier_id)).fetchone()
                used = int(existing["used_bytes"]) if existing else 0
                count = int(existing["object_count"]) if existing else 0
                self.conn.execute(
                    "INSERT INTO tier_state(spec_version, tier_id, role, "
                    "capacity_bytes, used_bytes, object_count, unit_cost, "
                    "rw_weight, eviction_priority, cost) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (version, tier.tier_id, tier.role, tier.capacity_bytes,
                     used, count, tier.unit_cost, tier.rw_weight,
                     tier.eviction_priority, used * tier.unit_cost))
            self._invalidate_stale(version, ts_ms)
            self.conn.execute(
                "UPDATE progress SET active_spec_version=?, active_plan_id=NULL, "
                "next_batch_index=0, last_committed_seq=0", (version,))
            self.audit(ts_ms, "spec_registered", details={"spec_version": version})
        return version

    def _invalidate_stale(self, new_version: int, ts_ms: int) -> list[int]:
        """旧版本的计划/结论整体标记 stale，返回被作废的计划 id 列表。"""
        rows = self.conn.execute(
            "SELECT plan_id FROM plans WHERE status='pending'").fetchall()
        plan_ids = [int(r["plan_id"]) for r in rows]
        for plan_id in plan_ids:
            self.conn.execute(
                "UPDATE plans SET status='stale', superseded_ms=? WHERE plan_id=?",
                (ts_ms, plan_id))
        self.conn.execute(
            "UPDATE conclusions SET status='stale', superseded_ms=? "
            "WHERE status='active'", (ts_ms,))
        self.conn.execute(
            "UPDATE batches SET status='stale' WHERE status='open'")
        return plan_ids

    def superseded_plan_fragments(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT plan_id, spec_version, plan_digest, superseded_ms "
            "FROM plans WHERE status='stale' ORDER BY plan_id").fetchall()
        return [dict(r) for r in rows]

    def active_spec_version(self) -> int:
        return int(self.conn.execute(
            "SELECT active_spec_version FROM progress WHERE id=1"
        ).fetchone()[0])

    def spec_blob(self, spec_version: int) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT spec_blob FROM spec_versions WHERE spec_version=?",
            (spec_version,)).fetchone()
        if row is None:
            raise VersionConflictError(
                f"账本中不存在策略版本 {spec_version}",
                field="spec_version", details={"spec_version": spec_version})
        return json.loads(row[0])

    # ---------- initial placement ----------
    def initialize_objects(self, manifest: Manifest, ts_ms: int) -> None:
        """首次装载：对象全部登记到热层，占用按清单累计。"""
        version = self.active_spec_version()
        with self.transaction():
            for obj in manifest.objects:
                self.conn.execute(
                    "INSERT INTO object_state(key, size_bytes, content_digest, "
                    "tier_id, spec_version, last_event_ms) VALUES(?,?,?,?,?,?)",
                    (obj.key, obj.size_bytes, obj.content_digest,
                     manifest.hot_tier.tier_id, version, ts_ms))
            hot = manifest.hot_tier
            total = sum(o.size_bytes for o in manifest.objects)
            self.conn.execute(
                "UPDATE tier_state SET used_bytes=?, object_count=?, cost=? "
                "WHERE spec_version=? AND tier_id=?",
                (total, len(manifest.objects), total * hot.unit_cost,
                 version, hot.tier_id))
            self.audit(ts_ms, "objects_initialized",
                       details={"spec_version": version,
                                "object_count": len(manifest.objects)})

    def object_count(self) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM object_state").fetchone()[0])

    def placement(self) -> dict[str, str]:
        return {r["key"]: r["tier_id"] for r in self.conn.execute(
            "SELECT key, tier_id FROM object_state ORDER BY key")}

    # ---------- plans ----------
    def save_plan(self, plan, ts_ms: int) -> int:
        with self.transaction():
            cur = self.conn.execute(
                "INSERT INTO plans(spec_version, plan_digest, payload, status, "
                "created_ms) VALUES(?,?,?,'pending',?)",
                (plan.spec_version, plan.plan_digest, plan.to_json(), ts_ms))
            plan_id = int(cur.lastrowid)
            self.conn.executemany(
                "INSERT INTO plan_steps(plan_id, seq, key, direction, "
                "source_tier, target_tier, size_bytes, heat, recency_rank, "
                "ranking) VALUES(?,?,?,?,?,?,?,?,?,?)",
                [(plan_id, s.seq, s.key, s.direction, s.source_tier,
                  s.target_tier, s.size_bytes, s.heat, s.recency_rank,
                  s.ranking) for s in plan.steps])
            self.conn.execute(
                "UPDATE progress SET active_plan_id=? WHERE id=1", (plan_id,))
            self.audit(ts_ms, "plan_saved",
                       details={"plan_id": plan_id, "steps": len(plan.steps)})
        return plan_id

    def pending_plan(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM plans WHERE status='pending' ORDER BY plan_id DESC "
            "LIMIT 1").fetchone()

    def plan_steps(self, plan_id: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM plan_steps WHERE plan_id=? ORDER BY seq", (plan_id,)))

    def mark_plan_done(self, plan_id: int, ts_ms: int) -> None:
        with self.transaction():
            self.conn.execute(
                "UPDATE plans SET status='committed', superseded_ms=? "
                "WHERE plan_id=?", (ts_ms, plan_id))
            self.conn.execute(
                "UPDATE progress SET active_plan_id=NULL WHERE id=1")

    # ---------- batches / migrations ----------
    def open_batch(self, plan_id: int, spec_version: int,
                   batch_index: int, ts_ms: int) -> int:
        # open 状态立即落盘：强制终止后恢复才能识别未完成批次
        cur = self.conn.execute(
            "INSERT INTO batches(plan_id, spec_version, batch_index, status, "
            "committed_ms) VALUES(?,?,?,'open',?)",
            (plan_id, spec_version, batch_index, ts_ms))
        self.conn.commit()
        return int(cur.lastrowid)

    def commit_batch(self, batch_id: int, migrations: list[dict[str, Any]],
                     tier_states: list[TierState], progress_cursor: dict[str, int],
                     ts_ms: int) -> None:
        """单事务提交：迁移记录 + 各层占用 + 进度游标 + 链式摘要。"""
        with self.transaction():
            for mig in migrations:
                self.conn.execute(
                    "INSERT INTO migrations(batch_id, plan_id, spec_version, seq, "
                    "key, direction, source_tier, target_tier, size_bytes, "
                    "content_digest, move_cost, committed_ms) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (batch_id, mig["plan_id"], mig["spec_version"], mig["seq"],
                     mig["key"], mig["direction"], mig["source_tier"],
                     mig["target_tier"], mig["size_bytes"],
                     mig["content_digest"], mig["move_cost"], ts_ms))
                self.conn.execute(
                    "UPDATE object_state SET tier_id=?, spec_version=?, "
                    "last_event_ms=? WHERE key=?",
                    (mig["target_tier"], mig["spec_version"], ts_ms, mig["key"]))
            version = self.active_spec_version()
            for state in tier_states:
                self.conn.execute(
                    "UPDATE tier_state SET used_bytes=?, object_count=?, cost=? "
                    "WHERE spec_version=? AND tier_id=?",
                    (state.used_bytes, state.object_count, state.cost,
                     version, state.tier_id))
            payload = {"event": "commit_batch", "batch_id": batch_id,
                       "migrations": sorted(
                           (m["key"], m["direction"], m["size_bytes"],
                            m["content_digest"]) for m in migrations),
                       "tier_state": sorted(
                           (s.tier_id, s.used_bytes, s.object_count, s.cost)
                           for s in tier_states),
                       "cursor": progress_cursor}
            digest = self._advance_chain(payload)
            self.conn.execute(
                "UPDATE batches SET status='committed', committed_ms=?, "
                "digest=?, payload=? WHERE batch_id=?",
                (ts_ms, digest,
                 json.dumps(payload, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")), batch_id))
            self.conn.execute(
                "UPDATE progress SET next_batch_index=?, last_committed_seq=? "
                "WHERE id=1",
                (progress_cursor["next_batch_index"],
                 progress_cursor["last_committed_seq"]))
            self.audit(ts_ms, "batch_committed",
                       details={"batch_id": batch_id,
                                "migrations": len(migrations)})

    def rollback_open_batch(self, batch_id: int, ts_ms: int) -> None:
        with self.transaction():
            self.conn.execute(
                "UPDATE batches SET status='aborted', committed_ms=? "
                "WHERE batch_id=? AND status='open'", (ts_ms, batch_id))

    def open_batches(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM batches WHERE status='open' ORDER BY batch_id"))

    def committed_migrations(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM migrations ORDER BY migration_id"))

    def progress(self) -> sqlite3.Row:
        return self.conn.execute(
            "SELECT * FROM progress WHERE id=1").fetchone()

    # ---------- views ----------
    def tier_states(self, spec_version: int | None = None) -> list[TierState]:
        version = spec_version or self.active_spec_version()
        rows = self.conn.execute(
            "SELECT * FROM tier_state WHERE spec_version=? ORDER BY tier_id",
            (version,)).fetchall()
        return [TierState(
            tier_id=r["tier_id"], role=r["role"],
            capacity_bytes=r["capacity_bytes"], used_bytes=r["used_bytes"],
            free_bytes=r["capacity_bytes"] - r["used_bytes"],
            object_count=r["object_count"], unit_cost=r["unit_cost"],
            rw_weight=r["rw_weight"], eviction_priority=r["eviction_priority"],
            cost=r["cost"]) for r in rows]

    def current_view(self) -> dict[str, Any]:
        """当前视图：只包含 active 版本数据，stale 记录被排除。"""
        version = self.active_spec_version()
        return {
            "active_spec_version": version,
            "tiers": [s.__dict__ for s in self.tier_states(version)],
            "placement": self.placement(),
            "pending_plan": (lambda r: None if r is None else
                             {"plan_id": r["plan_id"],
                              "plan_digest": r["plan_digest"],
                              "status": r["status"]})(self.pending_plan()),
            "progress": dict(self.progress()),
        }

    def save_conclusion(self, payload: dict[str, Any], ts_ms: int) -> int:
        with self.transaction():
            self.conn.execute(
                "UPDATE conclusions SET status='stale', superseded_ms=? "
                "WHERE status='active'", (ts_ms,))
            cur = self.conn.execute(
                "INSERT INTO conclusions(spec_version, payload, status, "
                "created_ms) VALUES(?,?,'active',?)",
                (self.active_spec_version(),
                 json.dumps(payload, ensure_ascii=False, sort_keys=True), ts_ms))
            return int(cur.lastrowid)

    def active_conclusion(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM conclusions WHERE status='active' ORDER BY "
            "conclusion_id DESC LIMIT 1").fetchone()
        return None if row is None else {
            "conclusion_id": row["conclusion_id"],
            "spec_version": row["spec_version"],
            "payload": json.loads(row["payload"])}

    def snapshot_digest(self) -> str:
        rows = self.conn.execute(
            "SELECT key, tier_id, content_digest, size_bytes FROM object_state "
            "ORDER BY key").fetchall()
        return hashlib.sha256(canonical_json([dict(r) for r in rows])).hexdigest()
