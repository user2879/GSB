"""Orchestration of real local node subprocesses."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import signal
import ctypes
from pathlib import Path
from types import TracebackType
from typing import Any

from .errors import DiagnosticError, PortError
from .protocol import JsonLineClient
from .spec import Spec


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parent_death_signal_preexec() -> None:
    """Deliver SIGTERM to a node if the WSL/Linux parent is force-killed."""

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass


def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def check_declared_ports(spec: Spec) -> list[dict[str, Any]]:
    results = []
    for node_id in sorted(spec.nodes):
        node = spec.nodes[node_id]
        free = is_port_free(node["host"], node["port"])
        results.append(
            {
                "node_id": node_id,
                "host": node["host"],
                "port": node["port"],
                "available_before_start": free,
                "code": None if free else "E_PORT",
            }
        )
    return results


class NodeProcess:
    def __init__(self, node: dict[str, Any], *, max_renewals: int, log_path: Path) -> None:
        self.node = node
        self.node_id = node["node_id"]
        self.host = node["host"]
        self.port = int(node["port"])
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None
        self.max_renewals = max_renewals

    def start(self, *, handshake_timeout_ms: int, startup_timeout_ms: int) -> None:
        if not is_port_free(self.host, self.port):
            raise PortError(
                f"declared loopback port {self.port} is already occupied",
                node_id=self.node_id,
                host=self.host,
                port=self.port,
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab")
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        command = [
            sys.executable,
            "-m",
            "sharddiag.node",
            "--node-id",
            self.node_id,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--seed",
            str(self.node["seed"]),
            "--fixed-point-decimals",
            str(self.node["fixed_point_decimals"]),
            "--capacity-version",
            str(self.node["capacity_version"]),
            "--concurrency-quota",
            str(self.node["concurrency_quota"]),
            "--lease-limit",
            str(self.node["lease_limit"]),
            "--max-renewals",
            str(self.max_renewals),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=_parent_death_signal_preexec if sys.platform.startswith("linux") else None,
        )
        deadline = time.monotonic() + max(1, startup_timeout_ms) / 1000.0
        last_error: DiagnosticError | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = self._read_log_tail()
                code = "E_PORT" if "E_PORT" in tail or "cannot bind" in tail else "E_PROTO"
                raise PortError(
                    f"node subprocess {self.node_id!r} exited before handshake",
                    node_id=self.node_id,
                    port=self.port,
                    exit_code=self.process.returncode,
                    log_tail=tail[-1000:],
                    error_code=code,
                )
            client = JsonLineClient(
                self.host,
                self.port,
                expected_node_id=self.node_id,
                timeout_ms=handshake_timeout_ms,
            )
            try:
                client.connect()
                client.handshake(timeout_ms=handshake_timeout_ms)
                client.close()
                return
            except DiagnosticError as exc:
                last_error = exc
                client.close()
                if exc.code == "E_TIMEOUT":
                    # A real socket timeout is fatal for this handshake attempt
                    # only until the overall subprocess startup deadline.
                    if time.monotonic() + handshake_timeout_ms / 1000.0 >= deadline:
                        raise
                time.sleep(0.01)
        if last_error is not None:
            raise last_error
        raise PortError("timed out waiting for node subprocess", node_id=self.node_id, port=self.port)

    def _read_log_tail(self) -> str:
        try:
            with self.log_path.open("rb") as handle:
                handle.seek(max(0, self.log_path.stat().st_size - 4000))
                return handle.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def client(self, timeout_ms: int) -> JsonLineClient:
        return JsonLineClient(
            self.host,
            self.port,
            expected_node_id=self.node_id,
            timeout_ms=timeout_ms,
        )

    def stop(self) -> None:
        if self.process is None:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


class NodeCluster:
    def __init__(self, spec: Spec, *, log_dir: str | Path) -> None:
        self.spec = spec
        self.log_dir = Path(log_dir)
        self.processes: dict[str, NodeProcess] = {}

    def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for node_id in sorted(self.spec.nodes):
            node = self.spec.nodes[node_id]
            handle = NodeProcess(
                node,
                max_renewals=int(self.spec.raw["max_renewals"]),
                log_path=self.log_dir / f"{node_id}.log",
            )
            handle.start(
                handshake_timeout_ms=int(self.spec.raw["handshake_timeout_ms"]),
                startup_timeout_ms=int(self.spec.raw.get("startup_timeout_ms", 5000)),
            )
            self.processes[node_id] = handle

    def client(self, node_id: str, timeout_ms: int | None = None) -> JsonLineClient:
        if node_id not in self.processes:
            raise PortError(f"node {node_id!r} is not started", node_id=node_id)
        return self.processes[node_id].client(
            timeout_ms or int(self.spec.raw["io_timeout_ms"])
        )

    def stop(self) -> None:
        for handle in self.processes.values():
            handle.stop()
        self.processes.clear()

    def __enter__(self) -> "NodeCluster":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


def occupy_port_for_test(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    return sock
