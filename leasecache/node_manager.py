from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import LeaseError
from .protocol import NodeClient
from .spec import NodeSpec, PipelineSpec
from .util import StructuredLogger, mono_ms


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _probe_bind(host: str, port: int) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name != "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Deliberately avoid SO_REUSEADDR: a successful bind is the local
        # preflight proof that this loopback endpoint is currently free.  On
        # POSIX the flag only bypasses TIME_WAIT; an active listener still
        # makes bind fail.
        probe.bind((host, port))
    finally:
        probe.close()


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            process.kill()
    else:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


class NodeManager:
    def __init__(
        self,
        spec: PipelineSpec,
        workspace: str | Path,
        runtime_dir: str | Path,
        logger: StructuredLogger,
        *,
        start_timeout_ms: int = 5000,
        request_timeout_ms: int = 5000,
    ) -> None:
        self.spec = spec
        self.workspace = Path(workspace).resolve()
        self.runtime_dir = Path(runtime_dir).resolve()
        self.log_dir = self.runtime_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.start_timeout_ms = start_timeout_ms
        self.request_timeout_ms = request_timeout_ms
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_handles: dict[str, Any] = {}
        self.clients: dict[str, NodeClient] = {}

    def _client(self, node: NodeSpec) -> NodeClient:
        return NodeClient(
            node.node_id,
            node.host,
            node.port,
            timeout_ms=self.request_timeout_ms,
        )

    def _spawn(self, node: NodeSpec) -> None:
        node_log = self.log_dir / f"node-{node.node_id}.jsonl"
        stdout_log = self.log_dir / f"node-{node.node_id}.stdout.log"
        command = [
            sys.executable,
            "-m",
            "leasecache.node_server",
            "--spec",
            str(self.spec.path),
            "--node-id",
            node.node_id,
            "--workspace",
            str(self.workspace),
            "--log",
            str(node_log),
            "--parent-pid",
            str(os.getpid()),
        ]
        environment = dict(os.environ)
        prior = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(PROJECT_ROOT) + (os.pathsep + prior if prior else "")
        )
        output_handle = stdout_log.open("ab")
        kwargs: dict[str, Any] = {
            "cwd": str(PROJECT_ROOT),
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "stdout": output_handle,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)
        self.processes[node.node_id] = process
        self.log_handles[node.node_id] = output_handle
        self.clients[node.node_id] = self._client(node)
        self.logger.info(
            "node subprocess spawned",
            node_id=node.node_id,
            pid=process.pid,
            port=node.port,
            log=str(node_log),
        )

    def _node_log_tail(self, node_id: str) -> str:
        path = self.log_dir / f"node-{node_id}.jsonl"
        try:
            data = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return data[-1] if data else ""

    def _wait_ready(self, node: NodeSpec, deadline_ms: int) -> None:
        client = self.clients[node.node_id]
        last_error: Exception | None = None
        process = self.processes[node.node_id]
        while mono_ms() < deadline_ms:
            if process.poll() is not None:
                tail = self._node_log_tail(node.node_id)
                if '"E_PORT"' in tail:
                    raise LeaseError(
                        "E_PORT",
                        f"node {node.node_id} could not bind its loopback port",
                        field="port",
                        expected=f"free 127.0.0.1:{node.port}",
                        actual=tail,
                        node_id=node.node_id,
                    )
                raise LeaseError(
                    "E_HANDSHAKE",
                    f"node {node.node_id} exited before handshake",
                    field="handshake",
                    expected="running cache node child process",
                    actual=f"exit code {process.returncode}",
                    node_id=node.node_id,
                )
            remaining = max(1, deadline_ms - mono_ms())
            try:
                client.handshake(timeout_ms=min(remaining, 500))
                self.logger.info(
                    "node handshake succeeded",
                    node_id=node.node_id,
                    port=node.port,
                    handshake_timeout_ms=min(remaining, 500),
                )
                return
            except LeaseError as exc:
                last_error = exc
                if exc.code == "E_ACK":
                    self.logger.error(
                        "E_ACK",
                        "node handshake acknowledgement is invalid",
                        node_id=node.node_id,
                        field=exc.field,
                        expected=exc.expected,
                        actual=exc.actual,
                    )
                    raise
                time.sleep(0.03)
        tail = self._node_log_tail(node.node_id)
        self.logger.error(
            "E_HANDSHAKE",
            "real socket handshake timed out",
            node_id=node.node_id,
            port=node.port,
            timeout_ms=self.start_timeout_ms,
            last_error=str(last_error or "no reply"),
        )
        raise LeaseError(
            "E_HANDSHAKE",
            f"real socket handshake timed out for node {node.node_id}",
            field="handshake",
            expected=f"reply within {self.start_timeout_ms} ms",
            actual=str(last_error or "no reply") + (f"; last log: {tail}" if tail else ""),
            node_id=node.node_id,
        )

    def start_all(self) -> None:
        nodes = sorted(self.spec.nodes, key=lambda node: node.node_id)
        for node in nodes:
            port_deadline = mono_ms() + 1200
            last_error: OSError | None = None
            while True:
                try:
                    _probe_bind(node.host, node.port)
                    last_error = None
                    break
                except OSError as exc:
                    last_error = exc
                    if mono_ms() >= port_deadline:
                        break
                    time.sleep(0.03)
            if last_error is not None:
                self.logger.error(
                    "E_PORT",
                    "loopback port is already occupied",
                    node_id=node.node_id,
                    host=node.host,
                    port=node.port,
                    error=str(last_error),
                )
                raise LeaseError(
                    "E_PORT",
                    "loopback port is already occupied",
                    field="port",
                    expected=f"free 127.0.0.1:{node.port}",
                    actual=str(last_error),
                    node_id=node.node_id,
                ) from last_error
        try:
            for node in nodes:
                self._spawn(node)
            deadline = mono_ms() + self.start_timeout_ms
            for node in nodes:
                self._wait_ready(node, deadline)
        except BaseException:
            self.stop_all()
            raise

    def client(self, node_id: str) -> NodeClient:
        return self.clients[node_id]

    def digest_snapshot(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for node_id in sorted(self.clients):
            reply = self.clients[node_id].digests()
            result[node_id] = {
                "digests": reply["digests"],
                "refs": {
                    key: sorted(reply["refs"][key])
                    for key in sorted(reply["refs"])
                },
                "mono_ms": reply["mono_ms"],
            }
        return result

    def stop_all(self) -> None:
        for node_id in sorted(self.clients):
            try:
                self.clients[node_id].shutdown()
            except LeaseError:
                pass
            except Exception:
                pass
        for node_id in sorted(self.processes):
            process = self.processes[node_id]
            try:
                process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                _terminate_process(process)
            handle = self.log_handles.get(node_id)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            self.logger.info(
                "node subprocess stopped",
                node_id=node_id,
                returncode=process.poll(),
            )
        self.clients.clear()
        self.processes.clear()
        self.log_handles.clear()

    def restart_node(self, node_id: str) -> None:
        node = self.spec.node_by_id[node_id]
        client = self.clients.pop(node_id, None)
        if client is not None:
            try:
                client.shutdown()
            except LeaseError:
                pass
        process = self.processes.pop(node_id, None)
        if process is not None:
            try:
                process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                _terminate_process(process)
        handle = self.log_handles.pop(node_id, None)
        if handle is not None:
            handle.close()
        deadline = mono_ms() + self.start_timeout_ms
        while mono_ms() < deadline:
            try:
                _probe_bind(node.host, node.port)
                break
            except OSError:
                time.sleep(0.03)
        else:
            raise LeaseError(
                "E_PORT",
                "node port did not become free for restart",
                field="port",
                expected=f"free 127.0.0.1:{node.port}",
                node_id=node_id,
            )
        self._spawn(node)
        self._wait_ready(node, mono_ms() + self.start_timeout_ms)

    def __enter__(self) -> "NodeManager":
        self.start_all()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop_all()
