"""受全局并发上限与每节点配额约束的线程池调度。

约束：任意时刻在运行的任务数 <= max_global；同一节点在运行任务数
<= per_node_quota。配额耗尽时任务排队，记录单调时钟的等待开始/结束。
单次下发有真实套接字超时，按策略重试；每次尝试都输出序号与结局分类，
重试不改变单次下发语义（同 task_id、同切片，节点侧聚合幂等）。
"""

import threading
import time
from dataclasses import dataclass, field


def mono_ms():
    return int(time.monotonic() * 1000)


# 结局分类：
# SUCCESS 成功；TIMEOUT 真实超时；CONN_REFUSED 连接被拒（端口未监听）；
# HANDSHAKE 握手失败；FIELD 回执字段缺失；EXHAUSTED 重试耗尽；
# UNKNOWN 强制终止等未完整提交状态（恢复时使用，不在线上调度产生）。
OUTCOMES = ("SUCCESS", "TIMEOUT", "CONN_REFUSED", "HANDSHAKE",
            "FIELD", "EXHAUSTED", "UNKNOWN")
RETRYABLE = {"TIMEOUT", "CONN_REFUSED"}


@dataclass
class Task:
    task_id: str
    node_id: str
    attempt_fn: object
    retries: int


@dataclass
class TaskResult:
    task_id: str
    node_id: str
    outcome: str
    attempts: list = field(default_factory=list)
    wait_queued_ms: int = 0
    wait_started_ms: int = 0
    wait_ended_ms: int = 0
    total_elapsed_ms: int = 0
    receipt: object = None


class Scheduler:
    def __init__(self, max_global, per_node_quota):
        if max_global < 1 or per_node_quota < 1:
            raise ValueError("并发配额必须 >= 1")
        self.max_global = max_global
        self.per_node_quota = per_node_quota
        self._global_slots = threading.Semaphore(max_global)
        self._cond = threading.Condition()
        self._node_inflight = {}
        self._peak_global = 0
        self._peak_node = {}
        self._lock = threading.Lock()

    def _acquire(self, node_id):
        queued_at = mono_ms()
        self._global_slots.acquire()
        with self._cond:
            while self._node_inflight.get(node_id, 0) >= self.per_node_quota:
                self._cond.wait()
            started_at = mono_ms()
            self._node_inflight[node_id] = self._node_inflight.get(node_id, 0) + 1
        with self._lock:
            inflight = sum(self._node_inflight.values())
            self._peak_global = max(self._peak_global, inflight)
            self._peak_node[node_id] = max(self._peak_node.get(node_id, 0),
                                           self._node_inflight[node_id])
        return queued_at, started_at

    def _release(self, node_id):
        with self._cond:
            self._node_inflight[node_id] -= 1
            self._cond.notify_all()
        self._global_slots.release()

    def _run_task(self, task):
        q_at, s_at = self._acquire(task.node_id)
        result = TaskResult(
            task_id=task.task_id, node_id=task.node_id, outcome="EXHAUSTED",
            wait_queued_ms=s_at - q_at, wait_started_ms=q_at, wait_ended_ms=s_at)
        begin = mono_ms()
        try:
            last_outcome = "EXHAUSTED"
            for attempt_no in range(task.retries + 1):
                t0 = mono_ms()
                try:
                    receipt = task.attempt_fn()
                    last_outcome = "SUCCESS"
                    result.attempts.append({
                        "attempt_no": attempt_no + 1,
                        "outcome": "SUCCESS",
                        "elapsed_ms": mono_ms() - t0,
                    })
                    result.receipt = receipt
                    break
                except Exception as exc:  # 单次尝试失败
                    code = getattr(exc, "code", type(exc).__name__)
                    outcome = self._classify(code)
                    last_outcome = outcome
                    result.attempts.append({
                        "attempt_no": attempt_no + 1,
                        "outcome": outcome,
                        "elapsed_ms": mono_ms() - t0,
                        "error_code": code,
                    })
                    if outcome not in RETRYABLE or attempt_no == task.retries:
                        break
            result.outcome = last_outcome
            result.total_elapsed_ms = mono_ms() - begin
            return result
        finally:
            self._release(task.node_id)

    @staticmethod
    def _classify(code):
        mapping = {
            "E_TIMEOUT": "TIMEOUT",
            "E_HANDSHAKE": "HANDSHAKE",
            "E_FIELD": "FIELD",
            "E_PORT": "CONN_REFUSED",
        }
        if code in mapping:
            return mapping[code]
        if code in ("ConnectionRefusedError", "ConnectionResetError",
                    "BrokenPipeError", "ConnectionError", "EOFError"):
            return "CONN_REFUSED"
        if code in ("TimeoutError", "socket.timeout"):
            return "TIMEOUT"
        return "EXHAUSTED"

    def run(self, tasks):
        threads = []
        results = []
        rlock = threading.Lock()

        def worker(task):
            res = self._run_task(task)
            with rlock:
                results.append(res)

        for task in tasks:
            t = threading.Thread(target=worker, args=(task,), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        order = {task.task_id: i for i, task in enumerate(tasks)}
        results.sort(key=lambda r: order[r.task_id])
        return results

    @property
    def peaks(self):
        return {"peak_global": self._peak_global, "peak_node": dict(self._peak_node)}
