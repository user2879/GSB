"""Lifecycle management for real loopback node child processes."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .client import NodeClient
from .errors import EdgeError, KilledError, PortError
from .spec import EdgeSpec, NodeSpec


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NodeProcessManager:
    def __init__(self, spec: EdgeSpec, log_dir: str | Path):
        self.spec = spec
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.startup_errors: dict[str, EdgeError] = {}
        self.blocked_listeners: dict[str, socket.socket] = {}
        self.injection = "none"
        self.target_node_id: str | None = None
        self.killed_node_ids: set[str] = set()

    def start(self, *, injection: str, target_node_id: str | None = None) -> None:
        self.injection = injection
        ordered_nodes = sorted(self.spec.nodes, key=lambda node: node.node_id)
        self.target_node_id = target_node_id or (ordered_nodes[0].node_id if ordered_nodes else None)
        if target_node_id is not None and target_node_id not in {node.node_id for node in ordered_nodes}:
            raise EdgeError(
                "E_SPEC",
                f"unknown injection target node {target_node_id}",
                details={"node_id": target_node_id},
                exit_code=3,
            )
        for node in ordered_nodes:
            if injection == "port" and node.node_id == self.target_node_id:
                holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                holder.bind((node.host, node.port))
                holder.listen(1)
                self.blocked_listeners[node.node_id] = holder
            self._spawn(node)
        for node in ordered_nodes:
            if node.node_id in self.startup_errors:
                continue
            if not self._wait_ready(node):
                continue
        if injection == "killed" and self.target_node_id is not None:
            process = self.processes.get(self.target_node_id)
            if process is not None and process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                self.killed_node_ids.add(self.target_node_id)

    def _spawn(self, node: NodeSpec) -> None:
        command = [
            sys.executable,
            "-m",
            "edgecache.node_server",
            "--spec",
            str(self.spec.path.resolve()),
            "--node-id",
            node.node_id,
            "--log",
            str((self.log_dir / f"node-{node.node_id}.jsonl").resolve()),
        ]
        if self.injection == "timeout" and node.node_id == self.target_node_id:
            command.extend(["--delay-ms", str(self.spec.global_spec.socket_timeout_ms + 600)])
        if self.injection == "receipt-missing" and node.node_id == self.target_node_id:
            command.extend(["--missing-receipt-field", "sampling_counts"])
        if self.injection == "rate-skew" and node.node_id == self.target_node_id:
            command.extend(["--rate-skew-ppm", "137000"])
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
        try:
            self.processes[node.node_id] = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            self.startup_errors[node.node_id] = PortError(
                node.node_id,
                node.host,
                node.port,
                os_error=str(exc),
            )

    def _wait_ready(self, node: NodeSpec) -> bool:
        process = self.processes[node.node_id]
        probe = NodeClient(node, timeout_ms=100)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = ""
                if process.stderr is not None:
                    try:
                        stderr = process.stderr.read()
                    except OSError:
                        stderr = ""
                self.startup_errors[node.node_id] = PortError(
                    node.node_id,
                    node.host,
                    node.port,
                    child_exit_code=process.returncode,
                    child_stderr=stderr.strip()[:2000],
                )
                return False
            try:
                response = probe.ping()
                if response.get("node_id") == node.node_id:
                    return True
            except (EdgeError, OSError):
                time.sleep(0.03)
        self.startup_errors[node.node_id] = PortError(
            node.node_id,
            node.host,
            node.port,
            reason="node did not become ready",
        )
        return False

    def is_alive(self, node_id: str) -> bool:
        process = self.processes.get(node_id)
        return process is not None and process.poll() is None

    def death_error(self, node: NodeSpec) -> KilledError:
        process = self.processes.get(node.node_id)
        return KilledError(
            node.node_id,
            exit_code=None if process is None else process.poll(),
            injected=self.injection == "killed",
        )

    def startup_error_for(self, node_id: str) -> EdgeError | None:
        return self.startup_errors.get(node_id)

    def cleanup(self) -> None:
        for node in sorted(self.spec.nodes, key=lambda item: item.node_id):
            process = self.processes.get(node.node_id)
            if process is None or process.poll() is not None:
                continue
            try:
                NodeClient(node, timeout_ms=100).request({"op": "shutdown"})
            except Exception:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        for holder in self.blocked_listeners.values():
            holder.close()
        self.blocked_listeners.clear()

    def process_snapshot(self) -> dict[str, Any]:
        return {
            node_id: {
                "pid": process.pid,
                "returncode": process.poll(),
                "alive": process.poll() is None,
            }
            for node_id, process in sorted(self.processes.items())
        }

