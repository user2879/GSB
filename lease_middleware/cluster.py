"""Management of real loopback TCP node subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import PortError
from .protocol import LeaseNodeClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NodeCluster:
    def __init__(
        self,
        nodes: list[dict[str, Any]],
        *,
        log_dir: str | Path,
        python_executable: str | None = None,
        handshake_timeout: float = 0.5,
        start_deadline_s: float = 6.0,
    ) -> None:
        self.nodes = nodes
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.python_executable = python_executable or sys.executable
        self.handshake_timeout = handshake_timeout
        self.start_deadline_s = start_deadline_s
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_files: dict[str, Any] = {}
        self.clients: dict[str, LeaseNodeClient] = {}

    def start_all(self, order: list[str] | None = None) -> dict[str, dict[str, Any]]:
        node_order = order or [node["node_id"] for node in self.nodes]
        by_id = {node["node_id"]: node for node in self.nodes}
        handshake_results: dict[str, dict[str, Any]] = {}
        try:
            for node_id in node_order:
                node = by_id[node_id]
                handshake_results[node_id] = self._start_one(node)
            return handshake_results
        except Exception:
            self.stop_all()
            raise

    def _start_one(self, node: dict[str, Any]) -> dict[str, Any]:
        node_id = node["node_id"]
        log_path = self.log_dir / f"{node_id}.log"
        log_handle = log_path.open("ab", buffering=0)
        self.log_files[node_id] = log_handle
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        cmd = [
            self.python_executable,
            "-m",
            "lease_middleware.node_server",
            "--node-id",
            node_id,
            "--host",
            node["host"],
            "--port",
            str(node["port"]),
        ]
        if int(node.get("processing_delay_ms", 0)):
            cmd.extend(
                ["--process-delay-ms", str(int(node["processing_delay_ms"]))]
            )
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self.processes[node_id] = process
        client = LeaseNodeClient(
            node["host"],
            int(node["port"]),
            node_id=node_id,
            timeout=self.handshake_timeout,
        )
        self.clients[node_id] = client
        deadline = time.monotonic() + self.start_deadline_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise self._port_error_from_log(node, log_path, process.returncode)
            try:
                return client.handshake()
            except Exception as exc:  # startup race: retry until deadline
                last_error = exc
                time.sleep(0.05)
        if process.poll() is not None:
            raise self._port_error_from_log(node, log_path, process.returncode)
        raise PortError(
            "node did not complete handshake before deadline",
            field="handshake",
            details={
                "node_id": node_id,
                "host": node["host"],
                "port": node["port"],
                "last_error": str(last_error),
            },
        )

    def _port_error_from_log(
        self, node: dict[str, Any], log_path: Path, returncode: int | None
    ) -> PortError:
        code = "E_PORT"
        message = "node subprocess exited before handshake"
        details: dict[str, Any] = {
            "node_id": node["node_id"],
            "port": node["port"],
            "returncode": returncode,
        }
        field = "port"
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            lines = [line for line in text.splitlines() if line.strip()]
            if lines:
                try:
                    payload = json.loads(lines[-1])
                    code = payload.get("code", code)
                    message = payload.get("message", message)
                    field = payload.get("field", field)
                    details["node_error"] = payload.get("details")
                except json.JSONDecodeError:
                    details["log_tail"] = lines[-1][-500:]
        except OSError:
            pass
        return PortError(message, field=field, details=details)

    def stop_all(self) -> None:
        for node_id, process in list(self.processes.items()):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.5)
        for handle in self.log_files.values():
            try:
                handle.close()
            except OSError:
                pass
        self.processes.clear()
        self.log_files.clear()

    def __enter__(self) -> "NodeCluster":
        self.start_all()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop_all()
