from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import PortInUseError, RpcTimeoutError, WaterLeaseError
from .protocol import rpc
from .spec import NodeDecl


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class NodeHandle:
    decl: NodeDecl
    process: subprocess.Popen[str] | None
    adopted: bool = False

    @property
    def node_id(self) -> str:
        return self.decl.node_id

    def client_message(self, message_type: str, **payload: Any) -> dict[str, Any]:
        message = {"type": message_type, "node_id": self.decl.node_id, **payload}
        return rpc(
            self.decl.host,
            self.decl.port,
            message,
            timeout_ms=max(2000, self.decl.processing_delay_ms + 1000),
            node_id=self.decl.node_id,
        )


class ProcessManager:
    def __init__(
        self,
        nodes: tuple[NodeDecl, ...] | list[NodeDecl],
        *,
        project_root: Path | str = PROJECT_ROOT,
        startup_timeout_s: float = 8.0,
    ) -> None:
        self.nodes = list(nodes)
        self.project_root = Path(project_root)
        self.startup_timeout_s = startup_timeout_s
        self.handles: dict[str, NodeHandle] = {}

    def __enter__(self) -> "ProcessManager":
        self.start_all()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_all()

    def start_all(self) -> None:
        started: list[NodeHandle] = []
        try:
            for node in self.nodes:
                handle = self._start_node(node)
                self.handles[node.node_id] = handle
                started.append(handle)
        except Exception:
            for handle in started:
                self._stop_handle(handle)
            raise

    def _start_node(self, node: NodeDecl) -> NodeHandle:
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.project_root) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        args = [
            sys.executable,
            "-m",
            "waterlease.node_worker",
            "--node-id",
            node.node_id,
            "--host",
            node.host,
            "--port",
            str(node.port),
            "--seed",
            str(node.seed),
            "--fixed-point-scale",
            str(node.fixed_point_scale),
            "--processing-delay-ms",
            str(node.processing_delay_ms),
        ]
        process = subprocess.Popen(
            args,
            cwd=self.project_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=1)
                if "E_PORT" in stdout + stderr:
                    raise PortInUseError(
                        f"port {node.port} is already occupied",
                        path=f"$.nodes[{node.node_id}].port",
                        details={
                            "node_id": node.node_id,
                            "port": node.port,
                            "stdout": stdout[-1000:],
                            "stderr": stderr[-1000:],
                        },
                    )
                raise WaterLeaseError(
                    f"node process {node.node_id} exited with code {process.returncode}",
                    code_override="E_NODE_EXIT",
                    details={"stdout": stdout[-1000:], "stderr": stderr[-1000:]},
                )
            try:
                response = rpc(
                    node.host,
                    node.port,
                    {"type": "ping", "node_id": node.node_id},
                    timeout_ms=150,
                    node_id=node.node_id,
                )
                if response.get("ok") is True:
                    return NodeHandle(node, process)
            except (RpcTimeoutError, PortInUseError, WaterLeaseError) as exc:
                last_error = exc
                time.sleep(0.03)

        process.kill()
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = "", ""
        if isinstance(last_error, PortInUseError):
            raise last_error
        raise RpcTimeoutError(
            f"node {node.node_id!r} did not complete handshake within "
            f"{self.startup_timeout_s:.1f}s",
            path=f"$.nodes[{node.node_id}].port",
            details={
                "port": node.port,
                "last_error": str(last_error),
                "stdout": (stdout or "")[-1000:],
                "stderr": (stderr or "")[-1000:],
            },
        )

    def stop_all(self) -> None:
        for handle in list(self.handles.values()):
            self._stop_handle(handle)
        self.handles.clear()

    @staticmethod
    def _stop_handle(handle: NodeHandle) -> None:
        process = handle.process
        if process is None or handle.adopted:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
