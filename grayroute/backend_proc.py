"""Subprocess lifecycle management and loopback client for inference backends."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass

from .constants import LOOPBACK_HOST, PORT_MAX, PORT_MIN
from .errors import SpecError


def is_port_free(host: str, port: int) -> bool:
    if not (PORT_MIN <= port <= PORT_MAX) or host != LOOPBACK_HOST:
        return False
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.25)
    try:
        probe.connect((host, port))
    except OSError:
        return True
    else:
        return False
    finally:
        probe.close()


def assert_ports_free(backends) -> None:
    seen: set[int] = set()
    for backend in backends.values():
        if backend.port in seen:
            raise SpecError(f"duplicate port {backend.port}",
                            object_id=backend.backend_id, field="port")
        seen.add(backend.port)
        if not is_port_free(backend.host, backend.port):
            raise SpecError(
                f"loopback port {backend.port} already in use; pick another "
                f"free port in [{PORT_MIN},{PORT_MAX}]",
                object_id=backend.backend_id, field="port",
            )


@dataclass
class ManagedBackend:
    backend_id: str
    host: str
    port: int
    process: subprocess.Popen

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


class BackendPool:
    """Starts one subprocess per backend and offers line-protocol calls."""

    def __init__(self, backends):
        self.backends = backends
        self._managed: dict[str, ManagedBackend] = {}

    def __enter__(self) -> "BackendPool":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        assert_ports_free(self.backends)
        for backend in self.backends.values():
            proc = subprocess.Popen(
                [sys.executable, "-m", "grayroute.backend_server",
                 "--host", backend.host, "--port", str(backend.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 10
            ready = False
            while time.monotonic() < deadline:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        err = proc.stderr.read()
                        raise RuntimeError(
                            f"backend {backend.backend_id} exited early: {err}")
                    break
                if line.startswith("READY"):
                    ready = True
                    break
            if not ready:
                proc.kill()
                raise RuntimeError(
                    f"backend {backend.backend_id} did not become ready")
            self._managed[backend.backend_id] = ManagedBackend(
                backend.backend_id, backend.host, backend.port, proc)

    def stop(self) -> None:
        for managed in list(self._managed.values()):
            managed.stop()
        self._managed.clear()

    def infer(self, backend_id: str, request_id: str, vector,
              model_dim: int, hidden_dim: int,
              param_summary: str) -> dict:
        managed = self._managed[backend_id]
        payload = {
            "id": request_id,
            "vector": list(vector),
            "param_summary": param_summary,
            "model_dim": model_dim,
            "hidden_dim": hidden_dim,
        }
        with socket.create_connection((managed.host, managed.port),
                                      timeout=10) as conn:
            conn.sendall(
                (json.dumps(payload, ensure_ascii=False) + "\n")
                .encode("utf-8"))
            with conn.makefile("r", encoding="utf-8") as fileobj:
                response = json.loads(fileobj.readline())
        if not response.get("ok"):
            raise RuntimeError(
                f"backend {backend_id} error: {response.get('error')}")
        return response
