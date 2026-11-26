from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import GSBError, PortError, ProtocolError, ProtocolTimeout
from .protocol import call_jsonline, check_port_available, wait_for_tcp


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NodeCluster:
    def __init__(self, spec: dict[str, Any], *, log_dir: str | Path | None = None) -> None:
        self.spec = spec
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_handles: dict[str, Any] = {}

    def start(self) -> None:
        for node in self.spec["nodes"]:
            check_port_available(node["host"], int(node["port"]))

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        spec_path = self.spec.get("source_path")
        if not spec_path:
            raise GSBError("spec must be loaded from a path before starting subprocesses", field="source_path")

        try:
            for node in self.spec["nodes"]:
                node_id = node["node_id"]
                log_handle = None
                if self.log_dir:
                    log_handle = (self.log_dir / f"node-{node_id}.log").open("ab", buffering=0)
                    self.log_handles[node_id] = log_handle
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "gsb.node",
                        "--spec",
                        str(spec_path),
                        "--node-id",
                        node_id,
                    ],
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=log_handle if log_handle else subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self.processes[node_id] = process
                if self.log_dir:
                    (self.log_dir / f"node-{node_id}.pid").write_text(str(process.pid), encoding="utf-8")

            timeout_ms = int(self.spec["scheduling"]["handshake_timeout_ms"])
            for node in self.spec["nodes"]:
                self._wait_node(node, timeout_ms)
        except Exception:
            self.stop()
            raise

    def _wait_node(self, node: dict[str, Any], timeout_ms: int) -> None:
        node_id = node["node_id"]
        process = self.processes[node_id]
        deadline = time.monotonic_ns() + timeout_ms * 1_000_000
        while time.monotonic_ns() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                detail = self._read_log_tail(node_id)
                if '"E_PORT"' in detail or "E_PORT" in detail:
                    raise PortError(
                        f"node {node_id} reported E_PORT while binding port {node['port']}",
                        field="nodes.port",
                        position=node["port"],
                        node_log=detail,
                    )
                raise GSBError(
                    f"node {node_id} exited during startup with code {exit_code}",
                    field="nodes",
                    position=node_id,
                    node_log=detail,
                )
            try:
                wait_for_tcp(node["host"], int(node["port"]), timeout_ms=100)
                response = self.call(
                    node_id,
                    {"type": "hello", "client": "gsb-parent"},
                    timeout_ms=min(1000, timeout_ms),
                )
                if not response.get("ok") or response.get("node_id") != node_id:
                    raise GSBError(f"bad handshake from node {node_id}", remote=response)
                return
            except (ProtocolTimeout, ProtocolError, OSError):
                continue
        raise ProtocolTimeout(
            f"handshake with node {node_id} timed out",
            field="nodes.port",
            position=node["port"],
            node_log=self._read_log_tail(node_id),
        )

    def _read_log_tail(self, node_id: str, max_bytes: int = 4000) -> str:
        if not self.log_dir:
            return ""
        path = self.log_dir / f"node-{node_id}.log"
        try:
            data = path.read_bytes()[-max_bytes:]
            return data.decode("utf-8", errors="replace")
        except OSError:
            return ""

    def node(self, node_id: str) -> dict[str, Any]:
        return self.spec["node_by_id"][node_id]

    def call(self, node_id: str, message: dict[str, Any], *, timeout_ms: int | None = None) -> dict[str, Any]:
        node = self.node(node_id)
        if timeout_ms is None:
            timeout_ms = int(self.spec["scheduling"]["request_timeout_ms"])
        return call_jsonline(node["host"], int(node["port"]), message, timeout_ms=timeout_ms)

    def grant_lease_on_node(self, lease: dict[str, Any]) -> dict[str, Any]:
        return self.call(lease["node_id"], {"type": "lease_grant", "lease": lease})

    def renew_lease_on_node(
        self,
        *,
        node_id: str,
        lease_id: str,
        expected_version: int,
        ttl_ms: int | None = None,
    ) -> dict[str, Any]:
        return self.call(
            node_id,
            {
                "type": "lease_renew",
                "lease_id": lease_id,
                "expected_version": expected_version,
                "ttl_ms": ttl_ms,
            },
        )

    def stop(self) -> None:
        for node_id, process in list(self.processes.items()):
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), 15)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(process.pid), 9)
                    except (ProcessLookupError, PermissionError):
                        process.kill()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            handle = self.log_handles.get(node_id)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
        self.processes.clear()
        self.log_handles.clear()

    def __enter__(self) -> "NodeCluster":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()
