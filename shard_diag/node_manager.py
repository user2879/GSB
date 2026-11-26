from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import PortError
from .protocol import ensure_port_available, request_json
from .spec import NodeSpec, Spec

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NodeManager:
    def __init__(self, spec: Spec, runtime_dir: str | Path = ".runtime") -> None:
        self.spec = spec
        self.runtime_dir = Path(runtime_dir)
        self.log_dir = self.runtime_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_handles: dict[str, Any] = {}

    def preflight_ports(self) -> None:
        for node in self.spec.nodes:
            ensure_port_available(node.port, node.node_id)

    def start_all(self) -> None:
        self.preflight_ports()
        for node in self.spec.nodes:
            self.start_node(node.node_id)

    def start_node(self, node_id: str) -> None:
        node = self.spec.node_map[node_id]
        if node_id in self.processes and self.processes[node_id].poll() is None:
            return
        ensure_port_available(node.port, node.node_id)
        log_path = self.log_dir / f"{node.node_id}.log"
        log_handle = log_path.open("ab")
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        cmd = [
            sys.executable,
            "-m",
            "shard_diag.node_worker",
            "--node-id",
            node.node_id,
            "--port",
            str(node.port),
            "--seed",
            str(node.seed),
            "--quota-version",
            str(node.quota_version),
            "--scale",
            str(self.spec.runtime.fixed_point_scale),
            "--io-timeout-ms",
            str(self.spec.runtime.io_timeout_ms),
        ]
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.processes[node_id] = process
        self.log_handles[node_id] = log_handle
        self._wait_ready(node, process, log_path)

    def _wait_ready(self, node: NodeSpec, process: subprocess.Popen[bytes], log_path: Path) -> None:
        deadline = time.monotonic() + self.spec.runtime.ready_timeout_ms / 1000.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = self._log_tail(log_path)
                raise PortError(
                    f"node {node.node_id} exited before binding 127.0.0.1:{node.port}; {tail}",
                    field="nodes.port",
                    object_id=node.node_id,
                    details={"returncode": process.returncode, "log_tail": tail},
                )
            try:
                response = request_json(
                    node.port,
                    {"type": "ping"},
                    timeout_ms=min(300, self.spec.runtime.io_timeout_ms),
                )
                if response.get("type") == "pong" and response.get("node_id") == node.node_id:
                    return
            except Exception as exc:  # startup race: retry until deadline
                last_error = exc
                time.sleep(0.03)
        raise PortError(
            f"node {node.node_id} did not become ready on 127.0.0.1:{node.port}: {last_error}",
            field="nodes.port",
            object_id=node.node_id,
        )

    @staticmethod
    def _log_tail(log_path: Path, limit: int = 800) -> str:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            return text[-limit:]
        except OSError:
            return ""

    def call(self, node_id: str, payload: dict[str, Any], *, timeout_ms: int | None = None) -> dict[str, Any]:
        node = self.spec.node_map[node_id]
        return request_json(
            node.port,
            payload,
            timeout_ms=timeout_ms or self.spec.runtime.io_timeout_ms,
        )

    def stop_node(self, node_id: str, *, force: bool = False) -> None:
        process = self.processes.get(node_id)
        if process is None:
            return
        if process.poll() is None and not force:
            try:
                self.call(node_id, {"type": "shutdown"}, timeout_ms=500)
            except Exception:
                process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                force = True
        if force and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                pass
        handle = self.log_handles.pop(node_id, None)
        if handle is not None:
            handle.close()
        self.processes.pop(node_id, None)

    def force_kill_node(self, node_id: str) -> None:
        self.stop_node(node_id, force=True)
        ensure_port_available(self.spec.node_map[node_id].port, node_id)

    def stop_all(self, *, force: bool = False) -> None:
        for node_id in list(self.processes):
            self.stop_node(node_id, force=force)

    def __enter__(self) -> "NodeManager":
        self.start_all()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_all(force=True)
