"""节点子进程编排：真实 ``Popen`` 启动、端口预检、就绪探测与强制终止。"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .client import NodeClient
from .errors import PortError, TimeoutError
from .errors import TimeoutError as DispatchTimeoutError
from .spec import NodeSpec


def _port_in_use(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.25)
    try:
        probe.connect((host, port))
    except OSError:
        return False
    else:
        return True
    finally:
        probe.close()


class NodeProcess:
    def __init__(self, node: NodeSpec, scale: int) -> None:
        self.node = node
        self.scale = scale
        self.process: subprocess.Popen | None = None
        self.client: NodeClient | None = None

    def start(self, *, ready_timeout: float = 8.0) -> NodeClient:
        if _port_in_use(self.node.host, self.node.port):
            raise PortError(
                f"port {self.node.port} already occupied before spawn",
                location=f"nodes[{self.node.node_id}]", field="port")
        command = [
            sys.executable, "-m", "dispatch_tokens.child",
            json.dumps(asdict(self.node), sort_keys=True), str(self.scale),
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        deadline = time.monotonic() + ready_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stderr = self.process.stderr.read().decode("utf-8", "replace") \
                    if self.process.stderr else ""
                raise PortError(
                    f"worker {self.node.node_id} exited early: {stderr.strip()}",
                    location=f"nodes[{self.node.node_id}]", field="process")
            try:
                client = NodeClient(self.node)
                client.connect()
            except (PortError, TimeoutError, OSError) as exc:
                last_error = exc
                time.sleep(0.05)
                continue
            self.client = client
            return client
        raise DispatchTimeoutError(
            f"worker {self.node.node_id} not ready within {ready_timeout}s "
            f"({last_error!r})",
            location=f"nodes[{self.node.node_id}]", field="ready")

    def hard_kill(self) -> int | None:
        """强制终止（SIGKILL 语义），不生成退出报告，仅返回退出码。"""
        if self.process is None:
            return None
        self.process.kill()
        return self.process.wait()

    def stop(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.process is not None:
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None


class Supervisor:
    def __init__(self, nodes: list[NodeSpec], scale: int) -> None:
        self.processes = {node.node_id: NodeProcess(node, scale)
                          for node in nodes}
        self.clients: dict[str, NodeClient] = {}

    def start_all(self) -> dict[str, NodeClient]:
        try:
            for node_id, process in self.processes.items():
                self.clients[node_id] = process.start()
        except Exception:
            self.stop_all()
            raise
        return self.clients

    def stop_all(self) -> None:
        for process in self.processes.values():
            process.stop()
        self.clients.clear()

    def hard_kill(self, node_id: str) -> int | None:
        return self.processes[node_id].hard_kill()
