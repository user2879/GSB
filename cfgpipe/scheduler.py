"""并行下发调度：全局并发上限 + 每节点并发配额，真实套接字超时与退避重试。

- 任一时刻在途下发数 <= global_limit，单节点在途数 <= per_node_limit。
- 配额耗尽时任务排队，记录等待起止的单调时钟毫秒。
- 单次下发使用真实 socket 超时；超时/连接失败按声明退避序列重试，
  重试发送的线路字节逐字节不变（配置版本与切片内容不变）。
"""

import json
import socket
import threading
import time
from dataclasses import dataclass, field


def mono_ms():
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class DispatchTask:
    node_id: str
    host: str
    port: int
    payload: bytes       # 线路字节（不含换行），重试时不变
    batch_id: str


@dataclass
class DispatchResult:
    node_id: str
    ok: bool
    receipt: dict = None
    attempts: int = 0
    wait_start_ms: int = 0   # 开始排队等待配额的单调时钟毫秒
    wait_end_ms: int = 0     # 获得配额（开始下发）的单调时钟毫秒
    elapsed_ms: int = 0
    error: str = ""
    retries: list = field(default_factory=list)  # 每次失败的原因


class Dispatcher:
    def __init__(self, global_limit, per_node_limit, timeout_ms, backoff_ms):
        self.global_sem = threading.Semaphore(global_limit)
        self.per_node_limit = per_node_limit
        self.timeout_ms = timeout_ms
        self.backoff_ms = tuple(backoff_ms)
        self._node_sems = {}
        self._lock = threading.Lock()

    def _node_sem(self, node_id):
        with self._lock:
            sem = self._node_sems.get(node_id)
            if sem is None:
                sem = threading.Semaphore(self.per_node_limit)
                self._node_sems[node_id] = sem
            return sem

    def _roundtrip(self, task):
        timeout_s = self.timeout_ms / 1000.0
        with socket.create_connection((task.host, task.port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            sock.sendall(task.payload + b"\n")
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("connection closed by peer")
                buf += chunk
            return json.loads(buf.decode("utf-8"))

    def _run_task(self, task):
        result = DispatchResult(node_id=task.node_id, ok=False)
        result.wait_start_ms = mono_ms()
        node_sem = self._node_sem(task.node_id)
        self.global_sem.acquire()
        node_sem.acquire()
        result.wait_end_ms = mono_ms()
        start = mono_ms()
        try:
            max_attempts = 1 + len(self.backoff_ms)
            for attempt in range(max_attempts):
                result.attempts = attempt + 1
                try:
                    receipt = self._roundtrip(task)
                    if receipt.get("type") == "receipt":
                        result.ok = True
                        result.receipt = receipt
                        return result
                    result.retries.append("unexpected reply: %r" % (receipt,))
                except (socket.timeout, TimeoutError) as exc:
                    result.retries.append("timeout: %s" % exc)
                except (ConnectionError, OSError) as exc:
                    result.retries.append("transport: %s" % exc)
                if attempt < len(self.backoff_ms):
                    time.sleep(self.backoff_ms[attempt] / 1000.0)
            result.error = result.retries[-1] if result.retries else "unknown"
            return result
        finally:
            result.elapsed_ms = mono_ms() - start
            node_sem.release()
            self.global_sem.release()

    def run(self, tasks):
        """并行执行全部任务，返回与 tasks 同序的 DispatchResult 列表。"""
        results = [None] * len(tasks)
        threads = []
        for idx, task in enumerate(tasks):
            def worker(i=idx, t=task):
                results[i] = self._run_task(t)
            th = threading.Thread(target=worker, daemon=True)
            threads.append(th)
            th.start()
        for th in threads:
            th.join()
        return results
