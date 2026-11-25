"""工作进程子进程编排（真实子进程 + 真实回环端口）。"""

import json
import subprocess
import sys

from .errors import PortError, TimeoutError_, WorkerError
from .protocol import WorkerClient


class ManagedWorker:
    def __init__(self, worker_spec, spec_dict, ready_timeout_ms=5000):
        self.worker_spec = worker_spec
        self.spec_dict = dict(spec_dict)
        self.spec_dict["_this_worker"] = worker_spec.worker_id
        self.ready_timeout_ms = ready_timeout_ms
        self.process = None

    def start(self):
        command = [sys.executable, "-m", "batch_receipt_ledger.worker"]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            close_fds=True,
        )
        try:
            self.process.stdin.write(json.dumps(self.spec_dict))
            self.process.stdin.close()
        except BrokenPipeError as exc:
            self._raise_start_failure(exc)

        if self.process.stdout is None:
            raise WorkerError("子进程标准输出不可用")
        import threading

        ready = {}

        def wait_ready():
            line = self.process.stdout.readline()
            if not line:
                ready["err"] = "子进程未输出就绪报文即退出"
                return
            ready["msg"] = json.loads(line)

        waiter = threading.Thread(target=wait_ready, daemon=True)
        waiter.start()
        waiter.join(self.ready_timeout_ms / 1000.0)
        if waiter.is_alive():
            self.stop()
            raise TimeoutError_(
                f"等待工作进程 {self.worker_spec.worker_id} 就绪超时",
                location=f"127.0.0.1:{self.worker_spec.port}", field="ready")
        if "err" in ready:
            self._raise_start_failure(RuntimeError(ready["err"]))
        msg = ready["msg"]
        if msg.get("type") == "fatal" and msg.get("code") == "E_PORT":
            self.stop()
            raise PortError(
                f"工作进程 {self.worker_spec.worker_id} 绑定端口 "
                f"127.0.0.1:{self.worker_spec.port} 失败: {msg.get('message')}",
                location=f"127.0.0.1:{self.worker_spec.port}", field="port")
        if msg.get("type") != "ready":
            self._raise_start_failure(RuntimeError(f"非法就绪报文 {msg}"))
        return self

    def __enter__(self):
        if self.process is None:
            self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    def _raise_start_failure(self, cause):
        stderr_text = ""
        if self.process is not None and self.process.stderr is not None:
            try:
                stderr_text = self.process.stderr.read()
            except Exception:
                stderr_text = ""
        self.stop()
        raise WorkerError(
            f"工作进程 {self.worker_spec.worker_id} 启动失败: {cause} {stderr_text}"
        ) from cause

    def client(self, connect_timeout_ms, read_timeout_ms):
        return WorkerClient(self.worker_spec.port, connect_timeout_ms, read_timeout_ms)

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        for stream in (self.process.stdin, self.process.stdout,
                       self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self.process = None


class WorkerPool:
    def __init__(self, spec, spec_dict, ready_timeout_ms=5000):
        self.spec = spec
        self.spec_dict = spec_dict
        self.workers = {}

    def __enter__(self):
        for worker_spec in self.spec.workers:
            managed = ManagedWorker(worker_spec, self.spec_dict).start()
            self.workers[worker_spec.worker_id] = managed
        return self

    def get(self, worker_id):
        return self.workers[worker_id]

    def __exit__(self, exc_type, exc, tb):
        for managed in self.workers.values():
            managed.stop()
        return False
