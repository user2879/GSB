"""键值引擎：过期语义、增量二级索引、派生计数视图与一致性校验。

设计要点：

* 逻辑时钟由调用方（会话/轨迹）注入，引擎绝不读取系统时钟做可见性判定；
  ``expire_ms == 0`` 约定为永不过期。
* 不变式：索引条目 ``(prefix, ref_key)`` 存在当且仅当 ``kv_records`` 中存在该键。
* ``prefix_counts`` 是结构计数（索引条目数），随每次变更同事务增量维护；
  面向用户的“可见计数”在查询时按注入时钟过滤过期键后给出。
* 每次变更在单事务内完成 kv + 索引 + 计数更新，避免半写状态。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from . import codec
from .errors import ExpiredError, IndexError as KvIndexError
from .storage import Storage


class Engine:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        # 重建期间由 RebuildWorker 注入：delta_listener(kind, key)
        self.delta_listener: Optional[Callable[[str, str], None]] = None
        if self.storage.get_meta("keyspace_version") is None:
            self.storage.set_meta("keyspace_version", "0")
        if self.storage.get_meta("trace_cursor") is None:
            self.storage.set_meta("trace_cursor", "0")

    # ---------------- 基础属性 ----------------
    @property
    def keyspace_version(self) -> int:
        return int(self.storage.get_meta("keyspace_version", "0"))

    def _bump_keyspace_version(self) -> None:
        self.storage.set_meta("keyspace_version", str(self.keyspace_version + 1))

    @property
    def trace_cursor(self) -> int:
        return int(self.storage.get_meta("trace_cursor", "0"))

    def set_trace_cursor(self, cursor: int) -> None:
        self.storage.set_meta("trace_cursor", str(cursor))

    # ---------------- 可见性（纯逻辑时钟） ----------------
    @staticmethod
    def is_visible(expire_ms: int, clock: int) -> bool:
        return expire_ms == 0 or clock < expire_ms

    def visibility_basis(self, key: str, clock: int) -> dict[str, Any]:
        """给出单次判定的完整依据步骤，不依赖系统时钟。"""
        row = self.storage.get_record(key)
        steps: list[dict[str, Any]] = [
            {"step": 1, "rule": "注入逻辑时钟", "clock": clock, "source": "session/trace"},
        ]
        if row is None:
            steps.append({"step": 2, "rule": "键不存在于 kv_records", "conclusion": "MISSING"})
            return {"key": key, "clock": clock, "conclusion": "MISSING", "steps": steps}
        expire_ms = row["expire_ms"]
        steps.append(
            {"step": 2, "rule": "读取记录毫秒过期戳", "expire_ms": expire_ms, "system_clock_used": False}
        )
        if expire_ms == 0:
            steps.append({"step": 3, "rule": "expire_ms=0 约定永不过期", "conclusion": "VISIBLE"})
            conclusion = "VISIBLE"
        elif clock < expire_ms:
            steps.append(
                {
                    "step": 3,
                    "rule": "clock < expire_ms => 未过期",
                    "compare": f"{clock} < {expire_ms}",
                    "conclusion": "VISIBLE",
                }
            )
            conclusion = "VISIBLE"
        else:
            steps.append(
                {
                    "step": 3,
                    "rule": "clock >= expire_ms => 已过期",
                    "compare": f"{clock} >= {expire_ms}",
                    "conclusion": "EXPIRED",
                }
            )
            conclusion = "EXPIRED"
        return {"key": key, "clock": clock, "conclusion": conclusion, "steps": steps}

    # ---------------- 初始记录批量加载（整体拒绝） ----------------
    def load_initial_records(self, records: list[dict[str, Any]]) -> None:
        # 先全部校验，任一非法整体拒绝，不写入任何记录
        normalized: list[dict[str, Any]] = []
        last_seq: dict[str, int] = {}
        for raw in records:
            record = codec.validate_record(raw)
            codec.check_write_seq_monotonic(
                last_seq.get(record["key"]), record["write_seq"], record["key"]
            )
            last_seq[record["key"]] = record["write_seq"]
            normalized.append(record)
        conn = self.storage.conn
        with self.storage.lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for record in normalized:
                    self._apply_put(record)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    # ---------------- PUT / DEL / PURGE ----------------
    def put(
        self,
        key: str,
        value: str,
        expire_ms: int,
        write_seq: int,
        clock: int,
    ) -> dict[str, Any]:
        record = codec.validate_record(
            {
                "key": key,
                "value": value,
                "expire_ms": expire_ms,
                "write_seq": write_seq,
                "format_version": codec.FORMAT_VERSION,
            }
        )
        existing = self.storage.get_record(key)
        codec.check_write_seq_monotonic(
            existing["write_seq"] if existing else None, record["write_seq"], key
        )
        conn = self.storage.conn
        with self.storage.lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._apply_put(record)
                self._bump_keyspace_version()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        if self.delta_listener is not None:
            self.delta_listener("PUT", key)
        return {
            "op": "PUT",
            "key": key,
            "clock": clock,
            "write_seq": record["write_seq"],
            "keyspace_version": self.keyspace_version,
        }

    def _apply_put(self, record: dict[str, Any]) -> None:
        key = record["key"]
        index_value = codec.canonical_json(
            [record["value"], record["expire_ms"]]
        ).decode("utf-8")
        self.storage.upsert_record(record)
        for prefix in codec.prefixes_of(key):
            old_version = self.storage.get_index_version(prefix, key)
            new_version = 1 if old_version is None else old_version + 1
            self.storage.upsert_index(prefix, key, index_value, new_version)
            self._recompute_count(prefix)

    def delete(self, key: str, clock: int) -> dict[str, Any]:
        existing = self.storage.get_record(key)
        conn = self.storage.conn
        with self.storage.lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if existing is not None:
                    for prefix in codec.prefixes_of(key):
                        self.storage.delete_index(prefix, key)
                        self._recompute_count(prefix)
                    self.storage.delete_record(key)
                    self._bump_keyspace_version()
                removed = existing is not None
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        if removed and self.delta_listener is not None:
            self.delta_listener("DEL", key)
        return {"op": "DEL", "key": key, "clock": clock, "removed": removed,
                "keyspace_version": self.keyspace_version}

    def purge(self, clock: int) -> dict[str, Any]:
        basis: list[dict[str, Any]] = []
        expired_keys = [
            row["key"]
            for row in self.storage.all_records()
            if not self.is_visible(row["expire_ms"], clock)
        ]
        conn = self.storage.conn
        with self.storage.lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for key in expired_keys:
                    row = self.storage.get_record(key)
                    basis.append(
                        {
                            "key": key,
                            "clock": clock,
                            "expire_ms": row["expire_ms"],
                            "compare": f"{clock} >= {row['expire_ms']}",
                            "rule": "clock >= expire_ms => 物理移除键与索引条目",
                        }
                    )
                    for prefix in codec.prefixes_of(key):
                        self.storage.delete_index(prefix, key)
                        self._recompute_count(prefix)
                    self.storage.delete_record(key)
                if expired_keys:
                    self._bump_keyspace_version()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        if self.delta_listener is not None:
            for key in expired_keys:
                self.delta_listener("PURGE", key)
        return {
            "op": "PURGE",
            "clock": clock,
            "purged_keys": sorted(expired_keys),
            "keyspace_version": self.keyspace_version,
            "basis": basis,
        }

    # ---------------- GET 与前缀查询 ----------------
    def get(self, key: str, clock: int) -> dict[str, Any]:
        codec.validate_key(key)
        judgment = self.visibility_basis(key, clock)
        if judgment["conclusion"] == "MISSING":
            result = {"op": "GET", "key": key, "clock": clock, "status": "MISSING",
                      "basis": judgment["steps"]}
        elif judgment["conclusion"] == "EXPIRED":
            result = {"op": "GET", "key": key, "clock": clock, "status": "E_EXPIRED",
                      "basis": judgment["steps"]}
        else:
            row = self.storage.get_record(key)
            result = {
                "op": "GET",
                "key": key,
                "clock": clock,
                "status": "OK",
                "value": row["value"],
                "expire_ms": row["expire_ms"],
                "basis": judgment["steps"],
            }
        return result

    def keyset(self, prefix: str, clock: int) -> list[str]:
        codec.validate_prefix(prefix)
        conn = self.storage.conn
        with self.storage.lock:
            rows = conn.execute(
                "SELECT i.ref_key AS ref_key, k.expire_ms AS expire_ms "
                "FROM index_entries i JOIN kv_records k ON k.key = i.ref_key "
                "WHERE i.prefix = ? ORDER BY i.ref_key",
                (prefix,),
            ).fetchall()
        return [r["ref_key"] for r in rows if self.is_visible(r["expire_ms"], clock)]

    def visible_count(self, prefix: str, clock: int) -> int:
        return len(self.keyset(prefix, clock))

    def query_prefix(self, prefix: str, clock: int) -> dict[str, Any]:
        """快速查询：键集合与计数走增量缓存（索引 + 派生计数视图）。

        ``keys`` 来自增量二级索引按注入时钟过滤；``visible_count`` 读取派生
        计数视图后扣除当前时钟下的过期条目，保证过期键不计入计数。
        权威重放独立从初始状态重算，两者分歧时由 E_VISIBLE 捕获。
        """
        codec.validate_prefix(prefix)
        keys = self.keyset(prefix, clock)
        structural = self.storage.get_count(prefix)
        conn = self.storage.conn
        with self.storage.lock:
            expired_row = conn.execute(
                "SELECT COUNT(*) AS c FROM index_entries i "
                "JOIN kv_records k ON k.key = i.ref_key "
                "WHERE i.prefix=? AND k.expire_ms<>0 AND ? >= k.expire_ms",
                (prefix, clock),
            ).fetchone()
        expired_in_prefix = expired_row["c"]
        visible_count = max(0, structural - expired_in_prefix)
        return {
            "prefix": prefix,
            "clock": clock,
            "keys": keys,
            "visible_count": visible_count,
            "cached_structural_count": structural,
            "keyspace_version": self.keyspace_version,
        }

    # ---------------- 计数派生与一致性 ----------------
    def _recompute_count(self, prefix: str) -> None:
        row = self.storage.conn.execute(
            "SELECT COUNT(*) AS c FROM index_entries WHERE prefix=?", (prefix,)
        ).fetchone()
        cnt = row["c"]
        if cnt == 0:
            self.storage.delete_count(prefix)
        else:
            self.storage.set_count(prefix, cnt)

    def verify_index_consistency(self, clock: Optional[int] = None) -> dict[str, Any]:
        """结构计数 vs 键集合重算；可选再核对可见计数。首错抛 E_INDEX。"""
        records = self.storage.all_records()
        recomputed: dict[str, int] = {}
        for row in records:
            for prefix in codec.prefixes_of(row["key"]):
                recomputed[prefix] = recomputed.get(prefix, 0) + 1
        cached = self.storage.all_counts()
        all_prefixes = sorted(set(recomputed) | set(cached))
        for prefix in all_prefixes:
            expected = recomputed.get(prefix, 0)
            actual = cached.get(prefix, 0)
            if expected != actual:
                raise KvIndexError(
                    "派生计数与键集合重算不一致",
                    details={
                        "prefix": prefix,
                        "expected_from_keyset": expected,
                        "actual_cached": actual,
                    },
                )
        # 索引条目存在性不变式
        index_rows = self.storage.all_index()
        index_pairs = {(r["prefix"], r["ref_key"]) for r in index_rows}
        expected_pairs = {
            (prefix, row["key"])
            for row in records
            for prefix in codec.prefixes_of(row["key"])
        }
        if index_pairs != expected_pairs:
            sample = sorted(index_pairs ^ expected_pairs)[0]
            raise KvIndexError(
                "索引条目集合与键集合不变式被破坏",
                details={"prefix": sample[0], "ref_key": sample[1]},
            )
        checked = len(all_prefixes)
        if clock is not None:
            for prefix in all_prefixes:
                quick = self.visible_count(prefix, clock)
                expected_visible = sum(
                    1
                    for row in records
                    if prefix in codec.prefixes_of(row["key"])
                    and self.is_visible(row["expire_ms"], clock)
                )
                if quick != expected_visible:
                    raise KvIndexError(
                        "可见计数与按时钟重算不一致",
                        details={"prefix": prefix, "expected": expected_visible,
                                 "actual": quick, "clock": clock},
                    )
        return {"ok": True, "prefixes_checked": checked,
                "index_entries": len(index_pairs), "kv_records": len(records)}

    # ---------------- 状态快照（供快照模块使用） ----------------
    def export_state(self) -> dict[str, Any]:
        return {
            "keyspace_version": self.keyspace_version,
            "trace_cursor": self.trace_cursor,
            "kv_records": [dict(r) for r in self.storage.all_records()],
            "index_entries": [dict(r) for r in self.storage.all_index()],
            "prefix_counts": [
                {"prefix": p, "cnt": c}
                for p, c in sorted(self.storage.all_counts().items())
            ],
            "meta": self.storage.all_meta(),
        }

    def replace_all(self, state: dict[str, Any]) -> None:
        """整体替换键空间与索引状态（绝不增量合并）。"""
        conn = self.storage.conn
        with self.storage.lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self.storage.clear_kv()
                self.storage.clear_index()
                self.storage.clear_counts()
                for record in state["kv_records"]:
                    self.storage.upsert_record(record)
                for entry in state["index_entries"]:
                    self.storage.upsert_index(
                        entry["prefix"], entry["ref_key"],
                        entry["index_value"], entry["entry_version"],
                    )
                for item in state["prefix_counts"]:
                    self.storage.set_count(item["prefix"], item["cnt"])
                self.storage.set_meta(
                    "keyspace_version", str(state["keyspace_version"])
                )
                self.storage.set_meta("trace_cursor", str(state["trace_cursor"]))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
