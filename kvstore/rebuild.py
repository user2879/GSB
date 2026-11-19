"""REBUILD_INDEX：工作线程内按键快照重建索引，与前台读写并发。

一致性策略：

1. 在工作线程内先对当前已提交键空间取快照（短时持锁），据此在 ``*_stage``
   暂存表重建；前台查询始终读正式表，看不到任何半写条目。
2. 重建期间前台 PUT/DEL/PURGE 通过 ``delta_listener`` 记录受影响键，并在
   每次引擎变更后把这些键的最新状态同步进暂存表（与引擎同锁）。
3. 换表前再 drain 一次 delta 并核对暂存表 == 正式表应有的派生结果，随后在
   单事务内原子 RENAME 换表，保证任何查询都等于该时刻已提交状态的重放结果。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from . import codec
from .storage import Storage


class RebuildWorker:
    def __init__(self, storage: Storage, logger=None) -> None:
        self.storage = storage
        self.logger = logger
        self.thread: Optional[threading.Thread] = None
        self.phase = "IDLE"
        self.phase_lock = threading.Lock()
        self.delta_keys: set[str] = set()
        self._barrier: Optional[threading.Event] = None
        self.result: dict[str, Any] = {}
        self.error: Optional[Exception] = None

    # 由引擎在每次变更后回调（已持有 storage.lock 时调用）
    def on_delta(self, kind: str, key: str) -> None:
        with self.phase_lock:
            if self.phase in ("SNAPSHOTTING", "BUILDING", "TRACKING"):
                self.delta_keys.add(key)
                self._sync_key_locked(key)

    def _sync_key_locked(self, key: str) -> None:
        """把单个键的最新状态同步进暂存表（调用方持 storage.lock）。"""
        for prefix in codec.prefixes_of(key):
            self.storage.stage_delete_index(prefix, key)
            self.storage.stage_delete_count(prefix)
        row = self.storage.get_record(key)
        if row is not None:
            index_value = codec.canonical_json(
                [row["value"], row["expire_ms"]]
            ).decode("utf-8")
            # 暂存条目版本继承正式表版本，保持条目版本语义不回退
            for prefix in codec.prefixes_of(key):
                version = self.storage.get_index_version(prefix, key) or 1
                self.storage.stage_upsert_index(prefix, key, index_value, version)
        # 计数为派生视图，换表前统一从暂存索引全量重算（见 _rebuild_stage_counts）

    def start(self, clock: int, barrier: Optional[threading.Event] = None) -> None:
        self.phase = "SNAPSHOTTING"
        self.delta_keys = set()
        self.error = None
        self.result = {}
        self._barrier = barrier
        self.thread = threading.Thread(
            target=self._run, args=(clock,), name="rebuild-worker", daemon=True
        )
        self.thread.start()

    def _run(self, clock: int) -> None:
        start_wall_ms = int(time.time() * 1000)
        start_logical_clock = clock
        if self.logger:
            self.logger.log(
                "rebuild_start",
                start_wall_ms=start_wall_ms,
                logical_clock=start_logical_clock,
            )
        try:
            storage = self.storage
            with storage.lock:
                storage.create_stage()
                snapshot = [dict(r) for r in storage.all_records()]
                with self.phase_lock:
                    self.phase = "BUILDING"

            # 确定性按键排序重建，保证多次重建结果逐字节一致
            for record in sorted(snapshot, key=lambda r: r["key"]):
                if self._barrier is not None:
                    # 演示用屏障：在重建中途挂起，供并发查询测试观察
                    self._barrier.wait()
                key = record["key"]
                index_value = codec.canonical_json(
                    [record["value"], record["expire_ms"]]
                ).decode("utf-8")
                with storage.lock:
                    # 若该键在快照后已变更，则以正式表当前状态为准，跳过陈旧快照值
                    if key in self.delta_keys:
                        continue
                    for prefix in codec.prefixes_of(key):
                        version = storage.get_index_version(prefix, key) or 1
                        storage.stage_upsert_index(prefix, key, index_value, version)

            with storage.lock:
                with self.phase_lock:
                    self.phase = "TRACKING"
                # drain：把快照后变更过的键全部以最新状态同步
                for key in sorted(self.delta_keys):
                    self._sync_key_locked(key)
                self._verify_stage_locked()
                storage.atomic_swap_stage()
                storage.drop_stage()
                finish_wall_ms = int(time.time() * 1000)
                self.phase = "DONE"
                self.result = {
                    "start_wall_ms": start_wall_ms,
                    "finish_wall_ms": finish_wall_ms,
                    "start_logical_clock": start_logical_clock,
                    "finish_logical_clock": clock,
                    "rebuilt_keys": len(snapshot),
                    "delta_keys": sorted(self.delta_keys),
                }
            if self.logger:
                self.logger.log("rebuild_finish", **self.result)
        except Exception as exc:  # 保留现场，交由 join 方抛出
            self.error = exc
            with self.phase_lock:
                self.phase = "FAILED"
            if self.logger:
                self.logger.log("rebuild_failed", error=repr(exc), level="ERROR")

    def _rebuild_stage_counts(self) -> None:
        """从暂存索引全量重算派生计数，避免增量同步遗漏归零前缀。"""
        rows = self.storage.conn.execute(
            "SELECT prefix, COUNT(*) AS c FROM idx_stage GROUP BY prefix"
        ).fetchall()
        self.storage.conn.execute("DELETE FROM cnt_stage")
        for row in rows:
            self.storage.stage_set_count(row["prefix"], row["c"])

    def _verify_stage_locked(self) -> None:
        self._rebuild_stage_counts()
        idx_rows, cnt_rows = self.storage.stage_rows()
        stage_pairs = {(r["prefix"], r["ref_key"]) for r in idx_rows}
        expected_pairs = {
            (prefix, row["key"])
            for row in self.storage.all_records()
            for prefix in codec.prefixes_of(row["key"])
        }
        if stage_pairs != expected_pairs:
            sample = sorted(stage_pairs ^ expected_pairs)[0]
            raise RuntimeError(f"重建暂存表与已提交键空间不一致: {sample}")
        stage_counts = {r["prefix"]: r["cnt"] for r in cnt_rows}
        recomputed: dict[str, int] = {}
        for prefix, _ref in stage_pairs:
            recomputed[prefix] = recomputed.get(prefix, 0) + 1
        if stage_counts != recomputed:
            raise RuntimeError("重建暂存计数与索引重算不一致")

    def join(self, timeout: Optional[float] = None) -> dict[str, Any]:
        if self.thread is not None:
            self.thread.join(timeout)
        if self.error is not None:
            raise self.error
        return self.result

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()
