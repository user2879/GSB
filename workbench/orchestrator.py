from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from .errors import WorkbenchError
from .protocol import wait_for_handshake
from .spec import Spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NodeHandle:
    def __init__(self, node_id: str, port: int, process: subprocess.Popen[bytes], log_dir: Path):
        self.node_id = node_id
        self.port = port
        self.process = process
        self.log_dir = log_dir

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    @property
    def returncode(self) -> int | None:
        return self.process.returncode


class NodeOrchestrator:
    def __init__(self, spec: Spec, workdir: str | Path) -> None:
        if spec.path is None:
            raise WorkbenchError("orchestrator requires a Spec loaded from a path")
        self.spec = spec
        self.workdir = Path(workdir)
        self.log_dir = self.workdir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.handles: dict[str, NodeHandle] = {}
        self._lock = threading.RLock()

    def _node(self, node_id: str) -> dict[str, Any]:
        return self.spec.node_by_id[node_id]

    def start_node(self, node_id: str) -> NodeHandle:
        with self._lock:
            existing = self.handles.get(node_id)
            if existing and existing.alive:
                return existing
            node = self._node(node_id)
            out_path = self.log_dir / f"{node_id}.stdout.log"
            err_path = self.log_dir / f"{node_id}.stderr.log"
            stdout_handle = out_path.open("ab", buffering=0)
            stderr_handle = err_path.open("ab", buffering=0)
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "workbench.node_process",
                        "--spec",
                        str(self.spec.path),
                        "--node-id",
                        node_id,
                    ],
                    cwd=str(PROJECT_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT)),
                )
            finally:
                stdout_handle.close()
                stderr_handle.close()
            process.stderr_log_path = str(err_path)  # type: ignore[attr-defined]
            handle = NodeHandle(node_id, int(node["port"]), process, self.log_dir)
            self.handles[node_id] = handle
            try:
                wait_for_handshake(
                    node,
                    startup_timeout_ms=max(3_000, int(self.spec.handshake_timeout_ms) * 5),
                    socket_timeout_ms=int(self.spec.handshake_timeout_ms),
                    process=process,
                )
            except Exception:
                self._kill_handle(handle)
                raise
            return handle

    def start_all(self) -> None:
        for node in self.spec.nodes:
            self.start_node(str(node["node_id"]))

    def _kill_handle(self, handle: NodeHandle, timeout: float = 5.0) -> int:
        process = handle.process
        if process.poll() is None:
            process.kill()
        try:
            return int(process.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            os.kill(process.pid, signal.SIGKILL)
            return int(process.wait(timeout=timeout))

    def kill_node(self, node_id: str) -> int:
        with self._lock:
            handle = self.handles.get(node_id)
            if handle is None:
                raise WorkbenchError(f"no such node handle: {node_id}", code="E_PORT")
            return self._kill_handle(handle)

    def stop_node_gracefully(self, node_id: str, timeout: float = 2.0) -> int:
        with self._lock:
            handle = self.handles[node_id]
            process = handle.process
            if process.poll() is None:
                process.terminate()
                try:
                    return int(process.wait(timeout=timeout))
                except subprocess.TimeoutExpired:
                    return self._kill_handle(handle)
            return int(process.returncode)

    def restart_node(self, node_id: str) -> NodeHandle:
        with self._lock:
            handle = self.handles.get(node_id)
            if handle is not None:
                self._kill_handle(handle)
            return self.start_node(node_id)

    def stop_all(self) -> None:
        with self._lock:
            for node_id in list(self.handles):
                try:
                    self.stop_node_gracefully(node_id)
                except Exception:  # noqa: BLE001 - shutdown best effort
                    try:
                        self.kill_node(node_id)
                    except Exception:  # noqa: BLE001
                        pass

    def alive_node_ids(self) -> set[str]:
        with self._lock:
            return {node_id for node_id, handle in self.handles.items() if handle.alive}


def allocate_free_port() -> int:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()
