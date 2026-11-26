"""并行批处理调度：全局并发上限 + 每工作进程配额，真实回环套接字派发。

- 工作进程以子进程托管在 127.0.0.1 回环端口；
- 握手与读取均使用真实套接字超时（socket.settimeout），不以人为延迟代替；
- 超时报 E_TIMEOUT，端口占用/不可连接报 E_PORT；
- 排队记录单调时钟毫秒的等待起止（enqueue 到获得双重配额）。
"""

import os
import queue
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

from .core import monotonic_ms
from .errors import PortError, SocketTimeoutError
from . import receipts as receipts_mod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class BatchTask:
    batch_id: str
    worker_id: str
    param_version: int
    request_ids: tuple
    enqueue_ms: int = 0


@dataclass
class BatchResult:
    batch_id: str
    worker_id: str
    receipts: list = field(default_factory=list)
    wait_start_ms: int = 0
    wait_end_ms: int = 0
    finish_ms: int = 0
    error: object = None


class Scheduler:
    def __init__(self, cfg, ports, compute_delay_ms=0, spawn_workers=True):
        sched = cfg["scheduler"]
        self.cfg = cfg
        self.ports = ports
        self.timeout_s = sched["timeout_ms"] / 1000.0
        self.global_limit = int(sched["global_limit"])
        self.per_worker_limit = int(sched["per_worker_limit"])
        self.compute_delay_ms = compute_delay_ms
        self._global_sem = threading.Semaphore(self.global_limit)
        self._worker_sems = {
            wid: threading.Semaphore(self.per_worker_limit) for wid in cfg["workers_by_id"]
        }
        self._stats_lock = threading.Lock()
        self._in_flight = {wid: 0 for wid in cfg["workers_by_id"]}
        self._global_in_flight = 0
        self._procs = []
        if spawn_workers:
            self.start_workers()

    def start_workers(self):
        host = self.ports["host"]
        for wid in sorted(self.cfg["workers_by_id"]):
            port = int(self.ports["workers"][wid])
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError as exc:
                probe.close()
                raise PortError(f"端口被占用: {host}:{port}", port=port, detail=str(exc))
            probe.close()
        for wid in sorted(self.cfg["workers_by_id"]):
            port = int(self.ports["workers"][wid])
            proc = subprocess.Popen(
                [sys.executable, "-m", "src.worker",
                 "--config", self.cfg["__config_path__"],
                 "--worker-id", wid, "--host", host, "--port", str(port),
                 "--compute-delay-ms", str(self.compute_delay_ms)],
                cwd=REPO_ROOT, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._procs.append(proc)
            self._wait_ready(wid, host, port, proc)

    def _wait_ready(self, wid, host, port, proc):
        deadline = time.monotonic() + max(5.0, self.timeout_s)
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise PortError(f"工作进程 {wid} 启动失败（退出码 {proc.returncode}）",
                                port=port)
            try:
                with socket.create_connection((host, port), timeout=self.timeout_s) as sock:
                    sock.settimeout(self.timeout_s)
                    sock.sendall(f"HELLO {wid} 0\nQUIT\n".encode("utf-8"))
                    sock.makefile("r").readline()
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(0.02)
        raise SocketTimeoutError(f"工作进程 {wid} 就绪等待超时", port=port)

    def shutdown(self):
        host = self.ports["host"]
        for wid in sorted(self.cfg["workers_by_id"]):
            port = int(self.ports["workers"][wid])
            try:
                with socket.create_connection((host, port), timeout=1.0) as sock:
                    sock.sendall(b"SHUTDOWN\n")
            except OSError:
                pass
        for proc in self._procs:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._procs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()

    def stats_snapshot(self):
        with self._stats_lock:
            return {
                "global": {"in_flight": self._global_in_flight,
                           "remaining": self.global_limit - self._global_in_flight},
                "workers": {
                    wid: {"in_flight": self._in_flight[wid],
                          "remaining": self.per_worker_limit - self._in_flight[wid]}
                    for wid in sorted(self._in_flight)
                },
            }

    def run(self, tasks):
        return sorted(self.run_iter(tasks), key=lambda r: r.batch_id)

    def run_iter(self, tasks):
        """流式派发：批完成一个就产出一个结果（用于逐批原子提交）。"""
        for task in tasks:
            task.enqueue_ms = monotonic_ms()
        results = [None] * len(tasks)
        done_queue = queue.Queue()
        threads = []
        for idx, task in enumerate(tasks):
            th = threading.Thread(target=self._run_task,
                                  args=(task, results, idx, done_queue), daemon=True)
            th.start()
            threads.append(th)
        for _ in threads:
            yield done_queue.get()
        for th in threads:
            th.join()

    def _run_task(self, task, results, idx, done_queue=None):
        res = BatchResult(batch_id=task.batch_id, worker_id=task.worker_id,
                          wait_start_ms=task.enqueue_ms)
        with self._global_sem:
            with self._worker_sems[task.worker_id]:
                res.wait_end_ms = monotonic_ms()
                with self._stats_lock:
                    self._in_flight[task.worker_id] += 1
                    self._global_in_flight += 1
                try:
                    res.receipts = self._exchange(task)
                except (SocketTimeoutError, PortError) as exc:
                    res.error = exc
                finally:
                    with self._stats_lock:
                        self._in_flight[task.worker_id] -= 1
                        self._global_in_flight -= 1
                    res.finish_ms = monotonic_ms()
        results[idx] = res
        if done_queue is not None:
            done_queue.put(res)

    def _exchange(self, task):
        host = self.ports["host"]
        port = int(self.ports["workers"][task.worker_id])
        try:
            sock = socket.create_connection((host, port), timeout=self.timeout_s)
        except socket.timeout as exc:
            raise SocketTimeoutError(f"连接超时: {host}:{port}", port=port) from exc
        except OSError as exc:
            raise PortError(f"端口不可连接: {host}:{port}", port=port, detail=str(exc)) from exc
        try:
            sock.settimeout(self.timeout_s)
            reader = sock.makefile("r", encoding="utf-8", newline="\n")
            sock.sendall(f"HELLO {task.worker_id} {task.param_version}\n".encode("utf-8"))
            ready = reader.readline()
            if not ready.startswith("READY"):
                raise PortError(f"握手失败: {ready!r}", port=port)
            rids = ",".join(task.request_ids)
            sock.sendall(f"BATCH {task.batch_id} {rids} {task.param_version}\n".encode("utf-8"))
            lines = []
            while True:
                line = reader.readline()
                if not line:
                    raise PortError("连接被工作进程提前关闭", port=port)
                line = line.rstrip("\n")
                if line == "END":
                    break
                lines.append(line)
            sock.sendall(b"QUIT\n")
            src = f"worker://{task.worker_id}:{port}"
            return [receipts_mod.parse_line(line, line_no=i + 1, src_file=src)
                    for i, line in enumerate(lines)]
        except socket.timeout as exc:
            raise SocketTimeoutError(
                f"批 {task.batch_id} 套接字读写超时（{self.timeout_s}s）",
                port=port) from exc
        finally:
            sock.close()
