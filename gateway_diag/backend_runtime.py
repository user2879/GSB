from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from .errors import E_BACKEND, GatewayError
from .logging_utils import StructuredLogger
from .models import Backend
from .util import atomic_write_json


def _port_is_free(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _ping(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(b'{"op":"ping"}\n')
            data = sock.recv(4096)
            payload = json.loads(data.decode("utf-8").split("\n", 1)[0])
            return payload.get("service") == "deterministic-mock-inference"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


class BackendRuntime:
    """Optionally launches the shipped mock script for declared loopback ports."""

    def __init__(
        self,
        backends: Iterable[Backend],
        *,
        workdir: str | Path,
        logger: StructuredLogger,
        timeout: float = 15.0,
    ) -> None:
        by_port: dict[int, Backend] = {}
        for backend in sorted(backends, key=lambda item: (item.port, item.id)):
            by_port.setdefault(backend.port, backend)
        self.backends = [by_port[port] for port in sorted(by_port)]
        self.workdir = Path(workdir)
        self.logger = logger
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self.config_path = self.workdir / "backend-runtime.json"
        self.stderr_handle = None

    def __enter__(self) -> "BackendRuntime":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> "BackendRuntime":
        if not self.backends:
            raise GatewayError(E_BACKEND, "no loopback backends were declared", field="backends")
        self.workdir.mkdir(parents=True, exist_ok=True)
        occupy_deadline = time.monotonic() + min(2.5, self.timeout)
        while True:
            occupied = [
                {"backend_id": backend.id, "port": backend.port}
                for backend in self.backends
                if not _port_is_free(backend.host, backend.port)
            ]
            if not occupied:
                break
            if time.monotonic() >= occupy_deadline:
                raise GatewayError(
                    E_BACKEND,
                    "one or more declared loopback ports are already occupied",
                    field="port",
                    details={"occupied": occupied},
                )
            # A crashed --spawn-backends parent leaves a short-lived watchdog
            # window in which the previous child releases its listeners.
            time.sleep(0.1)
        payload = {
            "backends": [
                {
                    "id": f"listen-{backend.port}",
                    "host": backend.host,
                    "port": backend.port,
                    "model_dim": backend.model_dim,
                    "hidden_dim": backend.hidden_dim,
                    "seed": backend.seed,
                    "quantization": backend.quantization,
                }
                for backend in self.backends
            ]
        }
        atomic_write_json(self.config_path, payload)
        wrapper = Path(__file__).resolve().parent.parent / "mock_backend.py"
        if wrapper.exists():
            entry = [sys.executable, str(wrapper)]
            child_cwd = str(wrapper.parent)
        else:
            entry = [sys.executable, "-m", "gateway_diag.backend_server"]
            child_cwd = str(Path.cwd())
        stderr_path = self.workdir / "backend-runtime.stderr.log"
        self.stderr_handle = stderr_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                *entry,
                "--config",
                str(self.config_path),
                "--allow-dynamic",
                "--parent-pid",
                str(os.getpid()),
            ],
            cwd=child_cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self.stderr_handle,
        )
        deadline = time.monotonic() + self.timeout
        pending = {backend.port for backend in self.backends}
        while pending and time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise GatewayError(
                    E_BACKEND,
                    "mock backend process exited during startup",
                    field="backend",
                    details={
                        "exit_code": self.process.returncode,
                        "stderr_log": str(stderr_path),
                    },
                )
            for port in sorted(list(pending)):
                backend = next(item for item in self.backends if item.port == port)
                if _ping(backend.host, port, timeout=0.25):
                    pending.remove(port)
            if pending:
                time.sleep(0.05)
        if pending:
            self.stop()
            raise GatewayError(
                E_BACKEND,
                "mock backend did not become ready before the timeout",
                field="port",
                details={"ports": sorted(pending)},
            )
        self.logger.emit(
            "backend_runtime_ready",
            ports=[backend.port for backend in self.backends],
            pid=self.process.pid if self.process else None,
        )
        return self

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self.process = None
        if self.stderr_handle is not None:
            try:
                self.stderr_handle.flush()
                self.stderr_handle.close()
            except Exception:
                pass
            self.stderr_handle = None
