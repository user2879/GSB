"""双重并发约束下的并行判定调度（业务要求 4）。

约束：
* 全局并发上限 ``global_concurrency``；
* 每读点并发配额 ``per_readpoint_quota``。

调度真实投射到 :class:`concurrent.futures.ThreadPoolExecutor`，每个判定
任务必须先获取读点配额信号量、再获取全局信号量，任何任务都不得绕过配额
直接调用执行器。等待区间以本地单调时钟（毫秒）记录。
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass


def now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class TaskTrace:
    query_id: str
    read_point_id: str
    row_key: str
    batch_id: str
    enqueue_ms: int
    rp_wait_start_ms: int
    rp_wait_end_ms: int
    global_wait_start_ms: int
    global_wait_end_ms: int
    run_start_ms: int
    run_end_ms: int
    rp_held: int
    global_held: int

    def to_dict(self) -> dict:
        return asdict(self)


class Scheduler:
    def __init__(self, spec, batch_size: int = 2):
        self.spec = spec
        self.batch_size = max(1, batch_size)
        self.global_limit = spec.global_concurrency
        self.rp_quota = spec.per_readpoint_quota
        self._global = threading.BoundedSemaphore(self.global_limit)
        self._rp_semaphores = {
            rp.read_point_id: threading.BoundedSemaphore(self.rp_quota)
            for rp in spec.read_points
        }
        self._rp_held = {rp.read_point_id: 0 for rp in spec.read_points}
        self._global_held = 0
        self._rp_peak = {rp.read_point_id: 0 for rp in spec.read_points}
        self._global_peak = 0
        self._lock = threading.Lock()
        self.traces: list[TaskTrace] = []

    def verify_within_limits(self) -> None:
        """断言整个调度过程没有任何越界放行。"""
        with self._lock:
            if self._global_peak > self.global_limit:
                raise AssertionError(
                    f"全局并发越界: 峰值 {self._global_peak} > "
                    f"{self.global_limit}"
                )
            for rp_id, peak in self._rp_peak.items():
                if peak > self.rp_quota:
                    raise AssertionError(
                        f"读点 {rp_id} 配额越界: 峰值 {peak} > "
                        f"{self.rp_quota}"
                    )

    def _plan_batches(self, queries) -> list[list]:
        # 按读点标识序、行键序、查询标识序稳定分组，再按批次大小切分。
        ordered = sorted(
            queries,
            key=lambda q: (q.read_point_id, q.row_key, q.query_id),
        )
        batches: list[list] = []
        current: list = []
        current_rp = None
        for query in ordered:
            if current_rp is None:
                current_rp = query.read_point_id
            if (len(current) >= self.batch_size or
                    query.read_point_id != current_rp):
                batches.append(current)
                current = []
                current_rp = query.read_point_id
            current.append(query)
        if current:
            batches.append(current)
        return batches

    def run(self, executor) -> list[dict]:
        """通过执行器按批次并行判定，返回逐查询回执摘要列表。"""
        batches = self._plan_batches(self.spec.queries)
        results: list[dict] = []
        # 工作线程池真实存在；其容量与全局上限一致，信号量再做硬约束。
        pool = ThreadPoolExecutor(max_workers=max(1, self.global_limit))
        futures = []
        enqueue = now_ms()
        for index, batch in enumerate(batches):
            batch_id = f"batch-{index:04d}"
            futures.append(pool.submit(self._run_batch, executor, batch_id,
                                       batch, enqueue))
        for future in as_completed(futures):
            results.extend(future.result())
        pool.shutdown(wait=True)
        # 稳定输出：读点 -> 行键 -> 查询。
        results.sort(key=lambda s: (s["read_point_id"], s["row_key"],
                                    s["query_id"]))
        self.traces.sort(key=lambda t: (t.read_point_id, t.row_key,
                                        t.query_id))
        return results

    def _run_batch(self, executor, batch_id: str, batch, enqueue_ms: int):
        read_point = self.spec.read_point(batch[0].read_point_id)
        rp_id = read_point.read_point_id
        traces: list[TaskTrace] = []
        summaries: list[dict] = []
        # 每个任务独立成对申请（先读点配额、再全局上限），执行后立即释放，
        # 从而不存在“持有一方等待另一方到批次末尾”的循环等待。
        for query in batch:
            rp_wait_start = now_ms()
            self._rp_semaphores[rp_id].acquire()
            rp_wait_end = now_ms()
            try:
                with self._lock:
                    self._rp_held[rp_id] += 1
                    rp_held = self._rp_held[rp_id]
                    self._rp_peak[rp_id] = max(self._rp_peak[rp_id], rp_held)
                global_wait_start = now_ms()
                self._global.acquire()
                global_wait_end = now_ms()
                try:
                    with self._lock:
                        self._global_held += 1
                        global_held = self._global_held
                        self._global_peak = max(self._global_peak,
                                                 global_held)
                    run_start = now_ms()
                    receipt = executor.send_batch(
                        batch_id=f"{batch_id}-q{query.query_id}",
                        snapshot_seq=read_point.snapshot_seq,
                        reader_txn_id=read_point.reader_txn_id,
                        query_ids=[query.query_id],
                        row_keys=[query.row_key],
                    )
                    run_end = now_ms()
                finally:
                    self._global.release()
                    with self._lock:
                        self._global_held = max(0, self._global_held - 1)
            finally:
                self._rp_semaphores[rp_id].release()
                with self._lock:
                    self._rp_held[rp_id] = max(0, self._rp_held[rp_id] - 1)
            traces.append(TaskTrace(
                query_id=query.query_id,
                read_point_id=query.read_point_id,
                row_key=query.row_key,
                batch_id=batch_id,
                enqueue_ms=enqueue_ms,
                rp_wait_start_ms=rp_wait_start,
                rp_wait_end_ms=rp_wait_end,
                global_wait_start_ms=global_wait_start,
                global_wait_end_ms=global_wait_end,
                run_start_ms=run_start,
                run_end_ms=run_end,
                rp_held=rp_held,
                global_held=global_held,
            ))
            summaries.extend(receipt["summaries"])
        self.traces.extend(traces)
        return summaries

    def occupancy_trace(self) -> list[dict]:
        return [trace.to_dict() for trace in self.traces]

    def peaks(self) -> dict:
        with self._lock:
            return {
                "global_peak": self._global_peak,
                "rp_peak": dict(self._rp_peak),
                "global_limit": self.global_limit,
                "rp_quota": self.rp_quota,
            }
