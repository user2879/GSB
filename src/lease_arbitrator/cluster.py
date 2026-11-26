from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Any

from .errors import NodeUnavailableError
from .protocol import request_json_line, wait_for_port


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


class NodeCluster:
    def __init__(
        self,
        nodes: list[dict[str, Any]],
        *,
        timeout: float = 2.0,
        start_timeout: float = 6.0,
        log_dir: str | Path | None = None,
    ) -> None:
        self.nodes = {node["node_id"]: node for node in nodes}
        self.timeout = timeout
        self.start_timeout = start_timeout
        self.log_dir = Path(log_dir) if log_dir else PROJECT_ROOT / "reports" / "node-logs"
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_files: dict[str, Any] = {}

    def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH")
        src = str(SRC_ROOT)
        env["PYTHONPATH"] = src + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        for node_id, node in self.nodes.items():
            log_path = self.log_dir / f"{node_id}.log"
            log_handle = log_path.open("wb")
            self.log_files[node_id] = log_handle
            cmd = [
                sys.executable,
                "-m",
                "lease_arbitrator.node_server",
                "--node-id",
                node_id,
                "--host",
                node["host"],
                "--port",
                str(node["port"]),
                "--seed",
                str(node["seed"]),
                "--timeout",
                str(max(1.0, self.timeout * 2)),
            ]
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            except OSError as exc:
                log_handle.close()
                raise NodeUnavailableError(f"failed to spawn node {node_id}: {exc}") from exc
            self.processes[node_id] = process

        deadline = time.monotonic() + self.start_timeout
        for node_id, node in self.nodes.items():
            remaining = max(0.1, deadline - time.monotonic())
            process = self.processes[node_id]
            try:
                wait_for_port(
                    node["host"],
                    int(node["port"]),
                    timeout=remaining,
                    request_timeout=min(0.3, self.timeout),
                )
            except NodeUnavailableError as exc:
                if process.poll() is not None:
                    log_text = self._read_log_tail(node_id)
                    raise NodeUnavailableError(
                        f"node {node_id} exited with code {process.returncode}; log tail: {log_text}",
                        path=f"$.nodes[?node_id={node_id}]",
                    ) from exc
                raise exc

    def _read_log_tail(self, node_id: str) -> str:
        path = self.log_dir / f"{node_id}.log"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[-1000:]
        except OSError:
            return "<no log>"

    def call(self, node_id: str, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        node = self.nodes[node_id]
        return request_json_line(
            node["host"],
            int(node["port"]),
            payload,
            timeout=timeout or self.timeout,
        )

    def stop(self) -> None:
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 2.0
        for node_id, process in self.processes.items():
            remaining = max(0.05, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        for handle in self.log_files.values():
            try:
                handle.close()
            except OSError:
                pass
        self.processes.clear()
        self.log_files.clear()

    def __enter__(self) -> "NodeCluster":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
