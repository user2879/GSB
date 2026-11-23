"""受全局并发上限与每后端配额约束的并行批次调度。

- 任一时刻运行批次数 <= global_limit，单后端运行批次数 <= per_backend_limit；
- 配额耗尽时批次排队，记录等待开始/结束的单调时钟毫秒；
- 任务投射到工作线程池，执行函数发起真实套接字超时；
- 超时按策略重试：释放配额后重新排队获取，不绕过配额。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


@dataclass
class WaitRecord:
    batch_id: str
    backend_id: str
    attempt: int
    wait_start_ms: int
    wait_end_ms: int

    @property
    def waited_ms(self) -> int:
        return self.wait_end_ms - self.wait_start_ms


@dataclass
class BatchJob:
    batch_id: str
    backend_id: str
    payload: dict = field(default_factory=dict)
    timeout_ms: int = 2000
    max_retries: int = 2


@dataclass
class BatchOutcome:
    job: BatchJob
    response: dict
    attempts: int
    waits: list[WaitRecord] = field(default_factory=list)


class RetryableError(Exception):
    """可重试的执行错误（如套接字超时）。"""


class Scheduler:
    def __init__(
        self,
        global_limit: int,
        per_backend_limit: int,
        backend_ids,
        max_workers: int | None = None,
    ):
        if global_limit <= 0 or per_backend_limit <= 0:
            raise ValueError("并发上限必须为正整数")
        self._global_sem = threading.BoundedSemaphore(global_limit)
        self._backend_sems = {
            bid: threading.BoundedSemaphore(per_backend_limit) for bid in backend_ids
        }
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers or (global_limit * 2),
            thread_name_prefix="gsb-worker",
        )
        self._running = 0
        self._running_per_backend: dict[str, int] = {bid: 0 for bid in backend_ids}
        self._peak_running = 0
        self._peak_lock = threading.Lock()

    @property
    def peak_running(self) -> int:
        return self._peak_running

    def _observe_running(self, backend_id: str, delta: int) -> None:
        with self._peak_lock:
            self._running += delta
            self._running_per_backend[backend_id] += delta
            self._peak_running = max(self._peak_running, self._running)

    def submit(self, job: BatchJob, execute) -> Future:
        """execute(payload, timeout_ms) -> dict；抛 RetryableError 时按策略重试。"""
        return self._pool.submit(self._run, job, execute)

    def _run(self, job: BatchJob, execute) -> BatchOutcome:
        backend_sem = self._backend_sems[job.backend_id]
        waits: list[WaitRecord] = []
        attempt = 0
        while True:
            wait_start = monotonic_ms()
            self._global_sem.acquire()
            backend_sem.acquire()
            wait_end = monotonic_ms()
            waits.append(
                WaitRecord(
                    batch_id=job.batch_id,
                    backend_id=job.backend_id,
                    attempt=attempt,
                    wait_start_ms=wait_start,
                    wait_end_ms=wait_end,
                )
            )
            try:
                self._observe_running(job.backend_id, +1)
                try:
                    response = execute(job.payload, job.timeout_ms)
                finally:
                    self._observe_running(job.backend_id, -1)
                return BatchOutcome(
                    job=job, response=response, attempts=attempt + 1, waits=waits
                )
            except RetryableError:
                attempt += 1
                if attempt > job.max_retries:
                    raise
                # 重试：配额在 finally 中释放，下一轮循环重新排队获取。
            finally:
                backend_sem.release()
                self._global_sem.release()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
