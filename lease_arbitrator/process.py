from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import PortError
from .protocol import port_is_open, request_json_line, wait_for_node
from .specs import NodeSpec


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NodeCluster:
    def __init__(
        self,
        nodes: list[NodeSpec] | tuple[NodeSpec, ...],
        *,
        run_dir: str | Path,
        quota_version: int = 1,
        shard_bytes: int = 64,
        handshake_timeout_ms: int = 2000,
        request_timeout_ms: int = 3000,
    ) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self.run_dir = Path(run_dir)
        self.artifact_dir = self.run_dir / "artifacts"
        self.log_dir = self.run_dir / "logs"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.quota_version = quota_version
        self.shard_bytes = shard_bytes
        self.handshake_timeout_ms = handshake_timeout_ms
        self.request_timeout_ms = request_timeout_ms
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_handles: dict[str, Any] = {}

    def start_node(self, node_id: str) -> dict[str, Any]:
        if node_id in self.processes and self.processes[node_id].poll() is None:
            return {"node_id": node_id, "already_started": True}
        node = self.nodes[node_id]
        if port_is_open(node.host, node.port):
            deadline = time.monotonic() + 3.0
            while port_is_open(node.host, node.port) and time.monotonic() < deadline:
                time.sleep(0.03)
            if port_is_open(node.host, node.port):
                raise PortError(
                    f"loopback port {node.port} for node {node_id} is already occupied",
                    entry_id=node_id,
                    field="port",
                )

        log_path = self.log_dir / f"{node_id}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        env = dict(os.environ)
        python_path = str(PROJECT_ROOT)
        env["PYTHONPATH"] = python_path + os.pathsep + env.get("PYTHONPATH", "")
        command = [
            sys.executable,
            "-u",
            "-m",
            "lease_arbitrator.node_worker",
            "--node-id",
            node.node_id,
            "--host",
            node.host,
            "--port",
            str(node.port),
            "--seed",
            str(node.seed),
            "--artifact-dir",
            str(self.artifact_dir),
            "--shard-bytes",
            str(self.shard_bytes),
            "--quota-version",
            str(self.quota_version),
            "--parent-pid",
            str(os.getpid()),
        ]
        creationflags = 0
        if sys.platform == "win32":
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        self.processes[node_id] = process
        self.log_handles[node_id] = log_handle
        try:
            receipt = wait_for_node(
                host=node.host,
                port=node.port,
                node_id=node.node_id,
                process=process,
                handshake_timeout_ms=self.handshake_timeout_ms,
                request_timeout_ms=self.request_timeout_ms,
                log_path=log_path,
            )
        except Exception:
            self.stop_node(node_id)
            raise
        return receipt

    def start_all(self, node_ids: list[str] | tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
        selected = list(node_ids) if node_ids is not None else list(self.nodes)
        return {node_id: self.start_node(node_id) for node_id in selected}

    def call(
        self,
        node_id: str,
        payload: dict[str, Any],
        *,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        node = self.nodes[node_id]
        return request_json_line(
            node.host,
            node.port,
            payload,
            timeout_ms=timeout_ms or self.request_timeout_ms,
        )

    def configure_quota(self, version: int, node_ids: list[str] | tuple[str, ...] | None = None) -> None:
        self.quota_version = version
        selected = list(node_ids) if node_ids is not None else list(self.processes)
        for node_id in selected:
            if node_id in self.processes and self.processes[node_id].poll() is None:
                self.call(
                    node_id,
                    {"action": "configure_quota", "quota_version": version, "request_seq": f"quota-{version}"},
                )

    def stop_node(self, node_id: str) -> None:
        process = self.processes.pop(node_id, None)
        handle = self.log_handles.pop(node_id, None)
        if process is not None and process.poll() is None:
            try:
                node = self.nodes[node_id]
                request_json_line(
                    node.host,
                    node.port,
                    {"action": "shutdown", "request_seq": "shutdown"},
                    timeout_ms=700,
                )
            except Exception:
                pass
            try:
                process.wait(timeout=0.8)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=0.8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.8)
        if process is not None and process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if handle is not None:
            handle.close()

    def stop_all(self) -> None:
        for node_id in list(self.processes):
            self.stop_node(node_id)

    def __enter__(self) -> "NodeCluster":
        self.start_all()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_all()
