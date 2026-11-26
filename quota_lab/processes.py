from __future__ import annotations

import ctypes
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import PortError, TimeoutError as LabTimeoutError
from .protocol import request_jsonline
from .spec import NodeSpec, SessionSpec


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def parent_death_signal_preexec() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


@dataclass
class NodeProcess:
    node: NodeSpec
    spec_path: Path
    log_path: Path
    process: subprocess.Popen[bytes] | None = None
    log_handle: Any = None

    def start(self) -> "NodeProcess":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("ab", buffering=0)
        cmd = [
            sys.executable,
            "-m",
            "quota_lab.node_server",
            "--spec",
            str(self.spec_path),
            "--node-id",
            self.node.node_id,
        ]
        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=parent_death_signal_preexec if sys.platform.startswith("linux") else None,
        )
        deadline = monotonic_ms() + 5000
        last_error: Exception | None = None
        while monotonic_ms() < deadline:
            if self.process.poll() is not None:
                tail = self._log_tail()
                if "E_PORT" in tail or "cannot bind" in tail:
                    raise PortError(
                        f"node {self.node.node_id} port {self.node.port} is occupied",
                        field=f"nodes.{self.node.node_id}.port",
                        details={"port": self.node.port, "log_tail": tail[-1000:]},
                    )
                raise PortError(
                    f"node {self.node.node_id} exited before TCP handshake",
                    field=f"nodes.{self.node.node_id}",
                    details={"returncode": self.process.returncode, "log_tail": tail[-1000:]},
                )
            try:
                response = self.request({"op": "hello"}, timeout_ms=250)
                if response.get("node_id") == self.node.node_id:
                    return self
            except (LabTimeoutError, PortError, OSError) as exc:
                last_error = exc
                time.sleep(0.04)
        raise LabTimeoutError(
            f"node {self.node.node_id} did not become ready",
            field=f"nodes.{self.node.node_id}.ready_timeout",
            details={"last_error": str(last_error)},
        )

    def _log_tail(self) -> str:
        try:
            self.log_handle.flush()
            data = self.log_path.read_bytes()[-4000:]
            return data.decode("utf-8", errors="replace")
        except OSError:
            return ""

    def request(self, payload: dict[str, Any], *, timeout_ms: int) -> dict[str, Any]:
        return request_jsonline(self.node.host, self.node.port, payload, timeout_ms=timeout_ms)

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.5)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


class Cluster:
    def __init__(self, spec: SessionSpec, spec_path: str | Path, log_dir: str | Path) -> None:
        self.spec = spec
        self.spec_path = Path(spec_path)
        self.log_dir = Path(log_dir)
        self.nodes: dict[str, NodeProcess] = {}

    def __enter__(self) -> "Cluster":
        for node in self.spec.nodes:
            runner = NodeProcess(
                node=node,
                spec_path=self.spec_path,
                log_path=self.log_dir / f"{node.node_id}.log",
            )
            self.nodes[node.node_id] = runner.start()
        return self

    def request(self, node_id: str, payload: dict[str, Any], *, timeout_ms: int | None = None) -> dict[str, Any]:
        if timeout_ms is None:
            timeout_ms = self.spec.socket_timeout_ms
        return self.nodes[node_id].request(payload, timeout_ms=timeout_ms)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def stop(self) -> None:
        for runner in list(self.nodes.values()):
            runner.stop()


def bind_release_probe(port: int = 0) -> dict[str, Any]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    bound_port = sock.getsockname()[1]
    sock.close()

    second = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    second.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    second.bind(("127.0.0.1", bound_port))
    second.close()
    return {"requested_port": port, "bound_released_and_rebound": bound_port, "ok": True}


def hold_port(port: int = 0) -> tuple[socket.socket, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(1)
    return sock, int(sock.getsockname()[1])
