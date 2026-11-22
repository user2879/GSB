"""租约下发与拉取调度：全局并发上限 + 每节点并发配额。

配额耗尽时任务排队，记录等待开始/结束的真实单调时钟毫秒；
单次拉取带真实套接字超时，超时按策略重试且不改变单次拉取语义
（同节点、同路径、同租约，仅尝试序号递增）。
"""

import threading
import time
from dataclasses import dataclass, field


def _now_mono_ms():
    return int(time.monotonic() * 1000)


@dataclass
class PullTask:
    seq: int
    node_id: str
    path: str
    execute: callable  # attempt_no -> (outcome, detail)


@dataclass
class AttemptRecord:
    task_seq: int
    node_id: str
    path: str
    lease_version: int
    attempt_no: int
    outcome: str
    detail: str = ""
    wait_start_mono_ms: int = 0
    wait_end_mono_ms: int = 0


class PullScheduler:
    def __init__(self, global_inflight, per_node_inflight, max_retries):
        self._global_sem = threading.Semaphore(global_inflight)
        self._node_sems = {}
        self._node_limit = per_node_inflight
        self._max_retries = max_retries
        self.records = []
        self._lock = threading.Lock()

    def _node_sem(self, node_id):
        with self._lock:
            if node_id not in self._node_sems:
                self._node_sems[node_id] = threading.Semaphore(self._node_limit)
            return self._node_sems[node_id]

    def run(self, tasks):
        """按提交顺序调度，全部完成后返回按 (task_seq, attempt_no) 排序的尝试记录。"""
        self.records = []
        threads = []
        for task in tasks:
            th = threading.Thread(target=self._run_task, args=(task,), daemon=True)
            threads.append(th)
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        return sorted(self.records, key=lambda r: (r.task_seq, r.attempt_no))

    def _run_task(self, task):
        wait_start = _now_mono_ms()
        self._global_sem.acquire()
        node_sem = self._node_sem(task.node_id)
        node_sem.acquire()
        wait_end = _now_mono_ms()
        try:
            attempt = 0
            while True:
                attempt += 1
                lease_version = -1
                try:
                    outcome, detail, lease_version = task.execute(attempt)
                except (TimeoutError, OSError) as exc:
                    outcome, detail = "timeout", f"{type(exc).__name__}: {exc}"
                record = AttemptRecord(
                    task_seq=task.seq,
                    node_id=task.node_id,
                    path=task.path,
                    lease_version=lease_version,
                    attempt_no=attempt,
                    outcome=outcome,
                    detail=detail,
                    wait_start_mono_ms=wait_start,
                    wait_end_mono_ms=wait_end,
                )
                with self._lock:
                    self.records.append(record)
                if outcome != "timeout" or attempt > self._max_retries:
                    break
        finally:
            node_sem.release()
            self._global_sem.release()
