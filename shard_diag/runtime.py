from __future__ import annotations

import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Any

from .errors import DiagnosticError, PortInUseError
from .protocol import assert_loopback_port_free, wait_for_tcp_port


class NodeCluster:
    def __init__(self, root: str | Path, nodes: list[dict[str, Any]], log_dir: str | Path | None = None) -> None:
        self.root = Path(root)
        self.nodes = {node["node_id"]: node for node in nodes}
        self.log_dir = Path(log_dir) if log_dir is not None else self.root / "ledger" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_handles: dict[str, Any] = {}

    def start_node(self, node_id: str) -> dict[str, Any]:
        if node_id in self.processes and self.processes[node_id].poll() is None:
            return {"started": False, "already_running": True, "node_id": node_id}
        node = self.nodes[node_id]
        if node_id not in self.processes:
            assert_loopback_port_free(node["port"], node["host"])

        log_path = self.log_dir / f"{node_id}.log"
        log_handle = log_path.open("ab")
        self.log_handles[node_id] = log_handle
        command = [
            sys.executable,
            "-m",
            "shard_diag.node_worker",
            "--root",
            str(self.root),
            "--node-id",
            node_id,
        ]
        source_root = Path(__file__).resolve().parent.parent
        python_path = os.pathsep.join(
            part for part in [str(source_root), str(self.root), os.environ.get("PYTHONPATH", "")] if part
        )
        process = subprocess.Popen(
            command,
            cwd=str(self.root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONPATH": python_path, "PYTHONUNBUFFERED": "1"},
        )
        self.processes[node_id] = process
        try:
            wait_for_tcp_port(node["host"], node["port"], int(node["startup_timeout_ms"]))
        except Exception:
            code = process.poll()
            tail = self.read_log_tail(node_id)
            if code is not None:
                if "E_PORT" in tail:
                    raise PortInUseError(
                        f"节点 {node_id} 报告端口占用",
                        entry=node_id,
                        field="port",
                        details={"exit_code": code, "log_tail": tail[-1000:]},
                    )
                raise DiagnosticError(
                    f"节点 {node_id} 子进程退出 code={code}",
                    entry=node_id,
                    field="subprocess",
                    details={"exit_code": code, "log_tail": tail[-1000:]},
                )
            raise
        return {"started": True, "node_id": node_id, "pid": process.pid, "port": node["port"]}

    def start_all(self) -> list[dict[str, Any]]:
        return [self.start_node(node_id) for node_id in sorted(self.nodes)]

    def read_log_tail(self, node_id: str, max_bytes: int = 4000) -> str:
        path = self.log_dir / f"{node_id}.log"
        if not path.exists():
            return ""
        data = path.read_bytes()[-max_bytes:]
        return data.decode("utf-8", errors="replace")

    def is_alive(self, node_id: str) -> bool:
        process = self.processes.get(node_id)
        return process is not None and process.poll() is None

    def kill_node(self, node_id: str) -> dict[str, Any]:
        process = self.processes.get(node_id)
        if process is None:
            return {"killed": False, "node_id": node_id, "reason": "not_started"}
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        return {"killed": True, "node_id": node_id, "returncode": process.returncode, "signal": "SIGKILL"}

    def terminate_node(self, node_id: str, timeout: float = 3.0) -> dict[str, Any]:
        process = self.processes.get(node_id)
        if process is None or process.poll() is not None:
            return {"terminated": False, "node_id": node_id}
        process.terminate()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.05)
        if process.poll() is None:
            return self.kill_node(node_id)
        return {"terminated": True, "node_id": node_id, "returncode": process.returncode}

    def stop_all(self) -> None:
        for node_id in sorted(self.processes):
            try:
                self.terminate_node(node_id)
            finally:
                handle = self.log_handles.get(node_id)
                if handle is not None:
                    handle.close()

    def __enter__(self) -> "NodeCluster":
        self.start_all()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop_all()
