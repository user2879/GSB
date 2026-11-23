"""三重约束批次调度器（需求 3）。

约束（任一时刻）：
  - 运行批次总数 <= 全局修复并发上限；
  - 单节点运行批次数 <= 全局修复并发上限；
  - 单节点已被占用带宽之和 <= 该节点配额（KB/s）；
  - 累计构造字节数 <= 全局修复字节预算。

带宽占用模型（确定性）：批次占用带宽 = max(1, ceil(批字节数 / 1024)) KB/s，
且不超过节点剩余配额；模拟传输时长 = 批字节数 / (占用带宽 * 1024) 秒 * time_scale。
配额或预算不足时批次排队，记录等待开始/结束的单调时钟毫秒；
预算为消耗型资源，剩余批次在预算永久不足时标记 DEFERRED_BUDGET（仍记录等待结束）。
调度真实投射到 ThreadPoolExecutor 工作线程池。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .content import BlockRegistry, generate_content, segment_digest
from .errors import E_INTEGRITY, StructuredLog, WorkbenchError
from .fixedpoint import ErrorLedger, FixedPoint
from .ledger import Ledger
from .spec import GroupSpec

QUEUED = "QUEUED"
RUNNING = "RUNNING"
COMMITTED = "COMMITTED"
DEFERRED_BUDGET = "DEFERRED_BUDGET"
FAILED_INTEGRITY = "FAILED_INTEGRITY"


def _mono_ms() -> int:
    return int(time.monotonic() * 1000)


class _CrashSimulated(Exception):
    pass


@dataclass
class Batch:
    batch_id: str
    block_id: str
    node_id: str
    start_byte: int
    end_byte: int
    state: str = QUEUED
    bandwidth_kbps: int = 0
    wait_start_ms: int = 0
    wait_end_ms: int = 0
    digest: str = ""
    err_bound_quanta: int = 0

    @property
    def byte_count(self) -> int:
        return self.end_byte - self.start_byte

    @property
    def wait_ms(self) -> int:
        if self.wait_end_ms == 0:
            return 0
        return max(0, self.wait_end_ms - self.wait_start_ms)

    def ledger_record(self, spec_version: int) -> dict:
        return {
            "batch_id": self.batch_id,
            "block_id": self.block_id,
            "node_id": self.node_id,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "digest": self.digest,
            "bandwidth_kbps": self.bandwidth_kbps,
            "wait_ms": self.wait_ms,
            "err_bound_quanta": self.err_bound_quanta,
            "state": self.state,
            "spec_version": spec_version,
        }


class Scheduler:
    """单副本组的三重约束批次调度器。"""

    def __init__(
        self,
        group: GroupSpec,
        registry: BlockRegistry,
        err_ledger: ErrorLedger,
        ledger: Ledger | None = None,
        log: StructuredLog | None = None,
        spec_version: int = 1,
        ranked_order: list | None = None,
        crash_after_commits: int | None = None,
    ):
        self._group = group
        self._registry = registry
        self._err = err_ledger
        self._ledger = ledger
        self._log = log
        self._spec_version = spec_version
        self._fp: FixedPoint = err_ledger.fp
        self._ranked_order = ranked_order or []
        self._crash_after = crash_after_commits

        self._cond = threading.Condition()
        self._running_total = 0
        self._running_per_node: dict[str, int] = {}
        self._bw_used: dict[str, int] = {}
        self._consumed_bytes = 0
        self._reserved_bytes = 0
        self._committed = 0
        self._crashed = False
        self._first_error: WorkbenchError | None = None
        self.batches: list[Batch] = []

    # ---- 批次构造 ----
    def plan(self) -> list[Batch]:
        """把每个块切分为整数批次字节段；按紧迫度排序结果确定构造顺序。"""
        params = self._group.params
        seg = params.segment_bytes
        block_rank = {
            item["block_id"]: idx for idx, item in enumerate(self._ranked_order)
        }
        ordered_blocks = sorted(
            self._group.blocks,
            key=lambda b: (block_rank.get(b.block_id, len(block_rank)), b.block_id),
        )
        batches: list[Batch] = []
        for block in ordered_blocks:
            # 误差核算：块字节量化截断 1 量子；批次切分取整（末段向上取整）。
            self._err.add_block_quantization(block.block_id, block.node_id, 1)
            n_full, rem = divmod(block.byte_count, seg)
            n_batches = n_full + (1 if rem else 0)
            rounding_bytes = n_batches * seg - block.byte_count
            split_quanta = self._fp.quantize_bytes(rounding_bytes)
            self._err.add_batch_split(
                block.block_id, block.node_id, split_quanta
            )
            for i in range(n_batches):
                start = i * seg
                end = min(start + seg, block.byte_count)
                # 每批次确定性误差界：块量化(首批 1) + 消耗量化(1) + 切分取整(末批)。
                err_quanta = 1 + (1 if i == 0 else 0) + (split_quanta if i == n_batches - 1 else 0)
                batches.append(
                    Batch(
                        batch_id=f"{self._group.group_id}:{block.block_id}:{i}",
                        block_id=block.block_id,
                        node_id=block.node_id,
                        start_byte=start,
                        end_byte=end,
                        wait_start_ms=_mono_ms(),
                        err_bound_quanta=err_quanta,
                    )
                )
        self.batches = batches
        return batches

    # ---- 约束判定 ----
    def _batch_demand_kbps(self, batch: Batch) -> int:
        return max(1, -((-batch.byte_count) // 1024))

    def _dispatchable(self, batch: Batch) -> tuple[bool, int]:
        params = self._group.params
        if self._running_total >= params.concurrency_limit:
            return False, 0
        if self._running_per_node.get(batch.node_id, 0) >= params.concurrency_limit:
            return False, 0
        node = self._group.node(batch.node_id)
        used = self._bw_used.get(batch.node_id, 0)
        demand = min(self._batch_demand_kbps(batch), node.bandwidth_kbps)
        if used + demand > node.bandwidth_kbps:
            return False, 0
        if demand <= 0:
            return False, 0
        if self._reserved_bytes + batch.byte_count > params.global_budget_bytes:
            return False, 0
        return True, demand

    # ---- 工作线程体 ----
    def _execute(self, batch: Batch, bandwidth_kbps: int) -> None:
        params = self._group.params
        try:
            duration = batch.byte_count / (bandwidth_kbps * 1024) * params.time_scale
            if duration > 0:
                time.sleep(duration)
            block = next(b for b in self._group.blocks if b.block_id == batch.block_id)
            expected = generate_content(block.seed, block.block_id, block.byte_count)[
                batch.start_byte:batch.end_byte
            ]
            actual = self._registry.read_segment(
                batch.block_id, batch.start_byte, batch.end_byte
            )
            if actual != expected:
                raise WorkbenchError(
                    E_INTEGRITY,
                    f"批次内容核对失败: {batch.batch_id}",
                    batch_id=batch.batch_id,
                    block_id=batch.block_id,
                    field="segment_digest",
                )
            batch.digest = segment_digest(
                batch.block_id, batch.start_byte, batch.end_byte, actual
            )
            batch.state = COMMITTED
            # 误差核算：消耗量化截断已计入批次误差界；
            # 排队延迟折算（若有等待）1 量子，仅作会话级归因（墙钟相关，非确定段）。
            self._err.add_block_quantization(batch.block_id, batch.node_id, 1)
            if batch.wait_ms > 0:
                self._err.add_queue_delay(batch.block_id, batch.node_id, 1)
            with self._cond:
                if self._crash_after is not None and self._committed >= self._crash_after:
                    self._crashed = True
                    raise _CrashSimulated()
                self._consumed_bytes += batch.byte_count
                if self._ledger is not None:
                    self._ledger.commit_batch(
                        batch.ledger_record(self._spec_version), self._consumed_bytes
                    )
                self._committed += 1
        except _CrashSimulated:
            pass
        except WorkbenchError as err:
            batch.state = FAILED_INTEGRITY
            with self._cond:
                if self._first_error is None:
                    self._first_error = err
            if self._log is not None:
                self._log.emit_error(err)
        finally:
            with self._cond:
                self._running_total -= 1
                self._running_per_node[batch.node_id] -= 1
                self._bw_used[batch.node_id] -= bandwidth_kbps
                self._cond.notify_all()

    # ---- 主调度循环 ----
    def run(self) -> dict:
        if not self.batches:
            self.plan()
        params = self._group.params
        queue = list(self.batches)
        max_workers = max(1, params.concurrency_limit)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while queue and not self._crashed and self._first_error is None:
                progressed = False
                with self._cond:
                    still_queued = []
                    for batch in queue:
                        ok, demand = self._dispatchable(batch)
                        if ok:
                            batch.state = RUNNING
                            batch.wait_end_ms = _mono_ms()
                            batch.bandwidth_kbps = demand
                            self._running_total += 1
                            self._running_per_node[batch.node_id] = (
                                self._running_per_node.get(batch.node_id, 0) + 1
                            )
                            self._bw_used[batch.node_id] = (
                                self._bw_used.get(batch.node_id, 0) + demand
                            )
                            self._reserved_bytes += batch.byte_count
                            if self._ledger is not None:
                                self._ledger.begin_batch(
                                    batch.ledger_record(self._spec_version)
                                )
                            pool.submit(self._execute, batch, demand)
                            progressed = True
                        else:
                            still_queued.append(batch)
                    queue = still_queued
                    if not queue:
                        break
                    if not progressed:
                        if self._running_total == 0:
                            # 预算为消耗型资源：剩余批次永久不可调度。
                            now = _mono_ms()
                            for batch in queue:
                                batch.state = DEFERRED_BUDGET
                                batch.wait_end_ms = now
                                if self._ledger is not None:
                                    self._ledger.commit_batch(
                                        batch.ledger_record(self._spec_version),
                                        self._consumed_bytes,
                                    )
                            queue = []
                            break
                        self._cond.wait(timeout=0.05)
        if self._crashed and self._log is not None:
            self._log.emit("CRASH", "模拟强制终止：存在未提交批次", committed=self._committed)
        if self._first_error is not None:
            raise self._first_error
        return self.summary()

    def summary(self) -> dict:
        states: dict[str, int] = {}
        for batch in self.batches:
            states[batch.state] = states.get(batch.state, 0) + 1
        return {
            "group_id": self._group.group_id,
            "spec_version": self._spec_version,
            "committed": self._committed,
            "consumed_bytes": self._consumed_bytes,
            "crashed": self._crashed,
            "states": states,
            "batches": [
                {
                    "batch_id": b.batch_id,
                    "block_id": b.block_id,
                    "node_id": b.node_id,
                    "range": [b.start_byte, b.end_byte],
                    "state": b.state,
                    "bandwidth_kbps": b.bandwidth_kbps,
                    "wait_ms": b.wait_ms,
                    "err_bound_quanta": b.err_bound_quanta,
                }
                for b in self.batches
            ],
        }
