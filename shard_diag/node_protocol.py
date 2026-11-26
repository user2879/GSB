from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import CapacityStaleError, LeaseError, NodeProtocolError
from .models import NodeSpec


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NodeClient:
    def __init__(self, node: NodeSpec, timeout_s: float = 10.0) -> None:
        self.node = node
        self.timeout_s = timeout_s

    def request(self, payload: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                with socket.create_connection(
                    (self.node.host, self.node.port), timeout=timeout_s or self.timeout_s
                ) as sock:
                    sock.settimeout(timeout_s or self.timeout_s)
                    sock.sendall(data)
                    with sock.makefile("rb") as buffered:
                        line = buffered.readline()
                if not line:
                    raise NodeProtocolError("node closed connection without a response")
                response = json.loads(line.decode("utf-8"))
                if not response.get("ok"):
                    error = response.get("error") or {}
                    code = error.get("code", "E_NODE")
                    message = error.get("message", "node request failed")
                    details = error.get("details") or {}
                    if code == "E_LEASE":
                        field = error.get("field")
                        if not field and "current_version" in details:
                            field = "lease_version"
                        raise LeaseError(
                            message,
                            object=payload.get("lease_id") or payload.get("shard_id"),
                            field=field,
                            details=details,
                        )
                    if code == "E_CAPACITY":
                        raise CapacityStaleError(message, details=details)
                    raise NodeProtocolError(message, details=details)
                return response
            except (OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == 0 and payload.get("type") == "ping":
                    time.sleep(0.05)
                    continue
                break
        raise NodeProtocolError(
            f"node {self.node.node_id} request {payload.get('type')!r} failed: {last_error}",
            object=self.node.node_id,
            field="port",
        ) from last_error

    def ping(self) -> dict[str, Any]:
        return self.request({"type": "ping"}, timeout_s=1.0)

    def grant_lease(
        self, lease_id: str, lease_version: int, ttl_ms: int, capacity_version: int
    ) -> dict[str, Any]:
        return self.request(
            {
                "type": "grant_lease",
                "lease_id": lease_id,
                "lease_version": lease_version,
                "ttl_ms": ttl_ms,
                "capacity_version": capacity_version,
            }
        )

    def renew_lease(self, lease_id: str, expected_version: int) -> dict[str, Any]:
        return self.request(
            {"type": "renew_lease", "lease_id": lease_id, "expected_version": expected_version}
        )

    def release_lease(self, lease_id: str) -> dict[str, Any]:
        return self.request({"type": "release_lease", "lease_id": lease_id})

    def fence_capacity(self, capacity_version: int) -> dict[str, Any]:
        return self.request({"type": "fence_capacity", "capacity_version": capacity_version})

    def aggregate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request({"type": "aggregate", **payload})


class NodeHandle:
    def __init__(self, node: NodeSpec) -> None:
        self.node = node
        self.process: subprocess.Popen[bytes] | None = None
        self.client = NodeClient(node)

    def start(self) -> "NodeHandle":
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "shard_diag.node_worker",
                "--host",
                self.node.host,
                "--port",
                str(self.node.port),
                "--node-id",
                self.node.node_id,
                "--seed",
                str(self.node.seed),
                "--capacity-version",
                str(self.node.capacity_version),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 8.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stderr = b""
                if self.process.stderr is not None:
                    stderr = self.process.stderr.read()
                raise NodeProtocolError(
                    f"node subprocess exited early: {stderr.decode('utf-8', 'replace')}",
                    object=self.node.node_id,
                    field="port",
                )
            try:
                self.client.ping()
                return self
            except NodeProtocolError as exc:
                last_error = exc
                time.sleep(0.05)
        raise NodeProtocolError(
            f"node {self.node.node_id} did not become ready on {self.node.host}:{self.node.port}",
            object=self.node.node_id,
            field="port",
        ) from last_error

    def kill(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5.0)
        self._close_pipes()
        self.process = None

    def _close_pipes(self) -> None:
        if self.process is None:
            return
        for pipe in (self.process.stdout, self.process.stderr):
            if pipe is not None:
                pipe.close()

    def restart(self) -> "NodeHandle":
        self.kill()
        return self.start()

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.kill()
        self._close_pipes()
        self.process = None


class NodeCluster:
    def __init__(self, nodes: list[NodeSpec] | tuple[NodeSpec, ...]) -> None:
        self.handles = {node.node_id: NodeHandle(node) for node in nodes}
        self.clients = {node_id: handle.client for node_id, handle in self.handles.items()}

    def start(self) -> "NodeCluster":
        for handle in self.handles.values():
            handle.start()
        return self

    def restart_node(self, node_id: str) -> NodeClient:
        self.handles[node_id].restart()
        return self.clients[node_id]

    def kill_node(self, node_id: str) -> None:
        self.handles[node_id].kill()

    def stop(self) -> None:
        for handle in self.handles.values():
            handle.stop()

    def __enter__(self) -> "NodeCluster":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()
