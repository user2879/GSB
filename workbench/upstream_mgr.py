"""Launch and manage upstream service child processes on loopback ports."""

import socket
import subprocess
import sys
import time


class UpstreamManager:
    def __init__(self, spec):
        self.spec = spec
        self.procs = {}  # service_id -> Popen

    def start(self, service_ids=None, ready_timeout_s=5.0):
        ids = service_ids if service_ids is not None else list(
            self.spec.services)
        for sid in ids:
            if sid in self.procs:
                continue
            svc = self.spec.services[sid]
            proc = subprocess.Popen(
                [sys.executable, "-m", "workbench.upstream",
                 "--port", str(svc.port), "--service", sid],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.procs[sid] = proc
            self._wait_ready(svc.host, svc.port, ready_timeout_s, proc)

    @staticmethod
    def _wait_ready(host, port, timeout_s, proc):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"upstream process exited early rc={proc.returncode}")
            try:
                conn = socket.create_connection((host, port), timeout=0.2)
                conn.close()
                return
            except OSError:
                time.sleep(0.02)
        raise RuntimeError(f"upstream on {host}:{port} not ready in time")

    def stop_all(self):
        for proc in self.procs.values():
            try:
                proc.terminate()
            except OSError:
                pass
        for proc in self.procs.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.procs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop_all()
