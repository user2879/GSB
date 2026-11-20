"""受全局并发上限与每节点配额约束的优先级并行下发调度器。

排队顺序：priority 降序；优先级相同按 (key, node) 稳定键决胜。
任一时刻运行中下发数 <= global_limit，每节点运行中 <= node_quota。
超时按策略重试（仅 E_TIMEOUT），其余错误不重试。
"""

import threading
import time
from dataclasses import dataclass, field

from .errors import TimeoutError_


@dataclass
class SliceTask:
    priority: int
    key: str
    node: str
    payload: dict
    timeout_ms: int
    max_retries: int


@dataclass
class SliceResult:
    task: SliceTask
    ok: bool
    receipt: dict = None
    error: dict = None
    attempts: int = 0
    started_ms: int = 0
    finished_ms: int = 0


class Scheduler:
    def __init__(self, global_limit, node_quota):
        self.global_limit = max(1, global_limit)
        self.node_quota = max(1, node_quota)
        self._cond = threading.Condition()
        self._pending = []
        self._in_flight_global = 0
        self._in_flight_node = {}
        self._node_peak = {}
        self._global_peak = 0
        self._done = 0
        self._total = 0
        self._results = []
        self._start_order = []

    def snapshot(self):
        """线程安全的进度/配额占用快照，供界面轮询。"""
        with self._cond:
            return {
                "total": self._total,
                "done": self._done,
                "in_flight_global": self._in_flight_global,
                "in_flight_node": dict(self._in_flight_node),
                "global_peak": self._global_peak,
                "node_peak": dict(self._node_peak),
                "queued": len(self._pending),
            }

    def quota_stats(self):
        with self._cond:
            return {"global_peak": self._global_peak,
                    "node_peak": dict(self._node_peak)}

    def start_order(self):
        """实际开始下发的 (key, node) 顺序，用于验证优先级调度。"""
        with self._cond:
            return list(self._start_order)

    def execute(self, tasks, dispatch_fn):
        ordered = sorted(tasks, key=lambda t: (-t.priority, t.key, t.node))
        with self._cond:
            self._pending = list(ordered)
            self._total = len(ordered)
            self._done = 0
            self._results = []
            self._in_flight_global = 0
            self._in_flight_node = {}
            self._node_peak = {}
            self._global_peak = 0
            self._start_order = []
        workers = [threading.Thread(target=self._worker, args=(dispatch_fn,),
                                    daemon=True)
                   for _ in range(min(self.global_limit, len(ordered)) or 1)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        return list(self._results)

    def _pick(self):
        for i, task in enumerate(self._pending):
            if self._in_flight_node.get(task.node, 0) < self.node_quota:
                return self._pending.pop(i)
        return None

    def _worker(self, dispatch_fn):
        while True:
            with self._cond:
                while True:
                    task = self._pick()
                    if task is not None:
                        self._in_flight_global += 1
                        n = self._in_flight_node.get(task.node, 0) + 1
                        self._in_flight_node[task.node] = n
                        self._node_peak[task.node] = max(
                            self._node_peak.get(task.node, 0), n)
                        self._global_peak = max(self._global_peak,
                                                self._in_flight_global)
                        self._start_order.append((task.key, task.node))
                        break
                    if not self._pending:
                        return
                    self._cond.wait()
            result = self._run_task(task, dispatch_fn)
            with self._cond:
                self._in_flight_global -= 1
                self._in_flight_node[task.node] -= 1
                self._done += 1
                self._results.append(result)
                self._cond.notify_all()

    def _run_task(self, task, dispatch_fn):
        started = time.monotonic_ns() // 1_000_000
        attempts = 0
        receipt = None
        error = None
        while attempts <= task.max_retries:
            attempts += 1
            try:
                receipt = dispatch_fn(task)
                error = None
                break
            except TimeoutError_ as exc:
                error = exc.to_dict()  # 仅超时按策略重试
                continue
            except Exception as exc:
                error = exc.to_dict() if hasattr(exc, "to_dict") else {
                    "code": "E_UNKNOWN", "message": str(exc)}
                break
        finished = time.monotonic_ns() // 1_000_000
        return SliceResult(task=task, ok=receipt is not None,
                           receipt=receipt, error=error, attempts=attempts,
                           started_ms=started, finished_ms=finished)
