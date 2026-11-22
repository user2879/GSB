"""节点子进程生命周期管理与单次 JSON 行采集客户端。"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import NodeSpec, Registry


class NodeProtocolError(RuntimeError):
    """回执缺失、JSON 非法或响应与请求不匹配。"""


@dataclass
class RunningNode:
    spec: NodeSpec
    process: subprocess.Popen

    def is_alive(self) -> bool:
        return self.process.poll() is None


class NodePool:
    """按注册表拉起节点子进程；Windows 下用 CREATE_NEW_PROCESS_GROUP 隔离。"""

    def __init__(self, registry: Registry, seed: int, log_dir: str | Path | None = None):
        self.registry = registry
        self.seed = seed
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._nodes: dict[str, RunningNode] = {}
        self._log_handles: list[object] = []

    def start_all(self) -> None:
        for node_id in self.registry.node_ids():
            self.start(node_id)

    def start(self, node_id: str) -> RunningNode:
        spec = self.registry.nodes[node_id]
        command = [
            sys.executable,
            "-u",
            "-m",
            "configreg.node",
            "--node-id",
            spec.node_id,
            "--host",
            spec.host,
            "--port",
            str(spec.port),
            "--seed",
            str(self.seed),
            "--fail-rate",
            str(spec.fail_rate),
        ]
        creation_flags = 0
        if sys.platform.startswith("win"):
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        log_handle = None
        if self.log_dir:
            log_handle = (self.log_dir / f"{node_id}.log").open("wb")
            self._log_handles.append(log_handle)
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=log_handle,
            creationflags=creation_flags,
        )
        running = RunningNode(spec, process)
        self._nodes[node_id] = running
        self._wait_until_listening(spec)
        return running

    def _wait_until_listening(self, spec: NodeSpec, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if spec.node_id in self._nodes and not self._nodes[spec.node_id].is_alive():
                raise NodeProtocolError(f"节点 {spec.node_id} 启动后立即退出")
            try:
                with socket.create_connection((spec.host, spec.port), timeout=0.3):
                    return
            except OSError as exc:
                last_error = exc
                time.sleep(0.05)
        raise NodeProtocolError(f"节点 {spec.node_id} 在 {timeout}s 内未开始监听: {last_error}")

    def stop_all(self, timeout: float = 5.0) -> None:
        for node_id in sorted(self._nodes):
            self._shutdown_one(node_id)
        deadline = time.monotonic() + timeout
        for running in self._nodes.values():
            remaining = max(0.0, deadline - time.monotonic())
            if running.is_alive():
                running.process.wait(timeout=remaining)
        self._nodes.clear()
        for handle in self._log_handles:
            handle.close()
        self._log_handles.clear()

    def _shutdown_one(self, node_id: str) -> None:
        running = self._nodes.get(node_id)
        if running is None:
            return
        try:
            with socket.create_connection((running.spec.host, running.spec.port), timeout=1.0) as sock:
                sock.sendall(b'{"type":"shutdown"}\n')
                sock.recv(64)
        except OSError:
            pass

    def terminate_all(self) -> None:
        """强制杀掉全部节点（恢复测试使用）。"""
        for running in self._nodes.values():
            if running.is_alive():
                running.process.terminate()
        for running in self._nodes.values():
            if running.is_alive():
                try:
                    running.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    running.process.kill()
        self._nodes.clear()
        for handle in self._log_handles:
            handle.close()
        self._log_handles.clear()


def collect_once(
    node: NodeSpec,
    request: dict[str, Any],
    socket_timeout: float,
) -> dict[str, Any]:
    """执行一次真实采集：连接、发送一行、按超时读取一行回执并核对请求号。"""
    with socket.create_connection((node.host, node.port), timeout=socket_timeout) as sock:
        sock.settimeout(socket_timeout)
        sock.sendall(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
        buffer = b""
        while not buffer.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                raise NodeProtocolError(f"节点 {node.node_id} 回执缺失：连接被关闭")
            buffer += chunk
    try:
        response = json.loads(buffer.decode("utf-8").strip())
    except json.JSONDecodeError as exc:
        raise NodeProtocolError(f"节点 {node.node_id} 回执不是合法 JSON 行") from exc
    if response.get("request_id") != request.get("request_id"):
        raise NodeProtocolError(
            f"节点 {node.node_id} 回执 request_id 不匹配: "
            f"{response.get('request_id')} != {request.get('request_id')}"
        )
    if not response.get("ok"):
        raise NodeProtocolError(f"节点 {node.node_id} 回执 ok=false")
    return response
