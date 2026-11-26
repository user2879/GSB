from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BusinessError, E_WORKER_PROTOCOL


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def send_json_line(
    host: str,
    port: int,
    request: dict[str, Any],
    *,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    payload = (json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout_seconds) as sock:
        sock.settimeout(timeout_seconds)
        sock.sendall(payload)
        buffer = bytearray()
        while not buffer.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            buffer.extend(chunk)
            if len(buffer) > 64 * 1024 * 1024:
                raise BusinessError(E_WORKER_PROTOCOL, "worker response exceeded size limit")
    if not buffer:
        raise BusinessError(E_WORKER_PROTOCOL, "worker closed connection without a response")
    response = json.loads(buffer.decode("utf-8").splitlines()[0])
    if not isinstance(response, dict):
        raise BusinessError(E_WORKER_PROTOCOL, "worker response must be an object")
    return response


@dataclass
class NodeProcess:
    config: dict[str, Any]
    process: subprocess.Popen[str]
    log_path: Path

    @property
    def node_id(self) -> str:
        return str(self.config["node_id"])


class SubprocessCluster:
    def __init__(self, nodes: dict[str, NodeProcess]) -> None:
        self.nodes = nodes

    @classmethod
    def start(
        cls,
        topology: dict[str, Any],
        state_dir: str | Path,
        *,
        startup_timeout_seconds: float = 15.0,
        override_free_ports: bool = False,
    ) -> "SubprocessCluster":
        state_dir = Path(state_dir)
        log_dir = state_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handles: dict[str, NodeProcess] = {}
        try:
            for raw_node in topology["nodes"]:
                node = dict(raw_node)
                if override_free_ports:
                    node["port"] = find_free_port()
                log_path = log_dir / f"{node['node_id']}.log"
                log_handle = log_path.open("w", encoding="utf-8")
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "topic_ledger.worker",
                    "--node-id",
                    node["node_id"],
                    "--host",
                    node["host"],
                    "--port",
                    str(node["port"]),
                    "--concurrency-quota",
                    str(node["concurrency_quota"]),
                    "--lease-limit",
                    str(node["lease_limit"]),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=Path.cwd(),
                    stdout=subprocess.PIPE,
                    stderr=log_handle,
                    text=True,
                    bufsize=1,
                )
                log_handle.close()
                handle = NodeProcess(config=node, process=process, log_path=log_path)
                handles[handle.node_id] = handle
                cls._wait_ready(handle, startup_timeout_seconds)
            return cls(handles)
        except Exception:
            cluster = cls(handles)
            cluster.close()
            raise

    @staticmethod
    def _wait_ready(handle: NodeProcess, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        assert handle.process.stdout is not None
        while time.monotonic() < deadline:
            if handle.process.poll() is not None:
                log_text = handle.log_path.read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(
                    f"worker {handle.node_id} exited early with code {handle.process.returncode}: {log_text}"
                )
            line = handle.process.stdout.readline()
            if line:
                try:
                    ready = json.loads(line)
                    if ready.get("event") == "ready":
                        return
                except json.JSONDecodeError:
                    pass
            time.sleep(0.02)
        raise TimeoutError(f"worker {handle.node_id} did not become ready within {timeout_seconds}s")

    def request(self, node_id: str, request: dict[str, Any], *, timeout_seconds: float = 30.0) -> dict[str, Any]:
        handle = self.nodes[node_id]
        return send_json_line(
            handle.config["host"],
            int(handle.config["port"]),
            request,
            timeout_seconds=timeout_seconds,
        )

    def sample(self, node_id: str, request: dict[str, Any]) -> dict[str, Any]:
        response = self.request(node_id, {"op": "sample", **request})
        if not response.get("ok"):
            error = BusinessError(
                str(response.get("code", E_WORKER_PROTOCOL)),
                str(response.get("message", "worker request failed")),
                batch_id=response.get("batch_id"),
                details=response.get("details"),
            )
            raise error
        return response

    def worker_rejections(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for node_id in sorted(self.nodes):
            response = self.request(node_id, {"op": "rejections"}, timeout_seconds=5.0)
            result[node_id] = response.get("rejections", []) if response.get("ok") else []
        return result

    def close(self) -> None:
        for node_id in sorted(self.nodes):
            handle = self.nodes.get(node_id)
            if handle is None:
                continue
            if handle.process.poll() is None:
                try:
                    self.request(node_id, {"op": "shutdown"}, timeout_seconds=1.0)
                except Exception:
                    handle.process.terminate()
            try:
                handle.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                handle.process.kill()
                handle.process.wait(timeout=3.0)

    def __enter__(self) -> "SubprocessCluster":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
