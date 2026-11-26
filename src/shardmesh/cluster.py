"""Management of real loopback node subprocesses."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import ShardMeshError
from .protocol import assert_port_available, handshake, rpc
from .spec import NodeSpec
from .timeutil import now_ns


class NodeProcess:
    def __init__(
        self,
        node: NodeSpec,
        *,
        scale: int,
        log_dir: str | Path = "artifacts/logs",
        request_timeout_ms: int = 3000,
    ) -> None:
        self.node = node
        self.scale = scale
        self.request_timeout_ms = request_timeout_ms
        self.log_dir = Path(log_dir)
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path = self.log_dir / f"{node.id}.log"
        self._log_handle = None

    @property
    def node_id(self) -> str:
        return self.node.id

    def _python_executable(self) -> str:
        return os.environ.get("SHARDMESH_PYTHON", sys.executable)

    def start(self) -> dict[str, Any]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        assert_port_available(self.node.host, self.node.port)
        self._log_handle = self.log_path.open("ab")
        cmd = [
            self._python_executable(),
            "-m",
            "shardmesh.node_worker",
            "--node-id",
            self.node.id,
            "--host",
            self.node.host,
            "--port",
            str(self.node.port),
            "--seed",
            str(self.node.seed),
            "--scale",
            str(self.scale),
            "--capacity-version",
            str(self.node.capacity_version),
            "--request-timeout-ms",
            str(max(1000, self.request_timeout_ms + 1000)),
        ]
        env = os.environ.copy()
        src_path = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return self.wait_ready(timeout_ms=max(2000, self.request_timeout_ms))

    def wait_ready(self, timeout_ms: int) -> dict[str, Any]:
        deadline_ns = now_ns() + timeout_ms * 1_000_000
        last_error: Exception | None = None
        while now_ns() < deadline_ns:
            if self.process is not None and self.process.poll() is not None:
                tail = self.read_log_tail()
                code = "E_PORT" if "E_PORT" in tail or "Address already in use" in tail else "E_NODE"
                raise ShardMeshError(
                    code,
                    f"node {self.node.id} exited during startup with code {self.process.returncode}",
                    category="node",
                    index=None,
                    field_name="port",
                    details={"port": self.node.port, "log_tail": tail[-1000:]},
                )
            try:
                return handshake(self.node.host, self.node.port, self.node.id, min(500, timeout_ms))
            except ShardMeshError as exc:
                last_error = exc
                if exc.code in {"E_FIELD", "E_TIMEOUT"}:
                    time.sleep(0.02)
                    continue
                time.sleep(0.02)
        if isinstance(last_error, ShardMeshError):
            raise last_error
        raise ShardMeshError("E_TIMEOUT", f"node {self.node.id} did not become ready", category="node", field_name="handshake")

    def read_log_tail(self, max_bytes: int = 4000) -> str:
        try:
            data = self.log_path.read_bytes()[-max_bytes:]
            return data.decode("utf-8", errors="replace")
        except OSError:
            return ""

    def rpc(self, message: dict[str, Any], *, required_fields: tuple[str, ...] = ()) -> dict[str, Any]:
        return rpc(
            self.node.host,
            self.node.port,
            message,
            self.request_timeout_ms,
            required_fields=required_fields,
        )

    def aggregate(self, message: dict[str, Any]) -> dict[str, Any]:
        return self.rpc(
            {"type": "aggregate", **message},
            required_fields=("node_id", "shard_id", "capacity_version", "summary", "elapsed_ms"),
        )

    def lease_open(self, lease: dict[str, Any], ttl_ms: int) -> dict[str, Any]:
        return self.rpc(
            {
                "type": "lease_open",
                "lease_id": lease["lease_id"],
                "shard_id": lease["shard_id"],
                "ttl_ms": ttl_ms,
                "capacity_version": lease["capacity_version"],
            },
            required_fields=("lease_id", "version"),
        )

    def lease_renew(self, lease_id: str, version: int, ttl_ms: int, max_renewals: int) -> dict[str, Any]:
        return self.rpc(
            {
                "type": "lease_renew",
                "lease_id": lease_id,
                "version": version,
                "ttl_ms": ttl_ms,
                "max_renewals": max_renewals,
            },
            required_fields=("lease_id", "version", "renew_count"),
        )

    def lease_release(self, lease_id: str) -> None:
        try:
            self.rpc({"type": "lease_release", "lease_id": lease_id})
        except ShardMeshError:
            pass

    def stop(self) -> None:
        if self.process is None:
            self._close_log()
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        self._close_log()

    def kill_hard(self) -> int | None:
        if self.process is None:
            return None
        if self.process.poll() is None:
            self.process.kill()
            return self.process.wait(timeout=3.0)
        return self.process.returncode

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None


class Cluster:
    def __init__(self, nodes: tuple[NodeSpec, ...] | list[NodeSpec], *, scale: int, log_dir: str | Path, timeout_ms: int) -> None:
        self.node_specs = {node.id: node for node in nodes}
        self.processes: dict[str, NodeProcess] = {}
        self.scale = scale
        self.log_dir = Path(log_dir)
        self.timeout_ms = timeout_ms

    def start_all(self) -> None:
        for node in self.node_specs.values():
            process = NodeProcess(node, scale=self.scale, log_dir=self.log_dir, request_timeout_ms=self.timeout_ms)
            process.start()
            self.processes[node.id] = process

    def get(self, node_id: str) -> NodeProcess:
        return self.processes[node_id]

    def restart_node(self, node_id: str, capacity_version: int | None = None) -> NodeProcess:
        old = self.processes.pop(node_id, None)
        if old is not None:
            old.stop()
        base = self.node_specs[node_id]
        if capacity_version is not None:
            base = NodeSpec(
                id=base.id,
                host=base.host,
                port=base.port,
                concurrency=base.concurrency,
                lease_limit=base.lease_limit,
                seed=base.seed,
                capacity_version=int(capacity_version),
            )
            self.node_specs[node_id] = base
        process = NodeProcess(base, scale=self.scale, log_dir=self.log_dir, request_timeout_ms=self.timeout_ms)
        process.start()
        self.processes[node_id] = process
        return process

    def stop_all(self) -> None:
        for process in list(self.processes.values()):
            process.stop()
        self.processes.clear()

    def __enter__(self) -> "Cluster":
        self.start_all()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_all()
