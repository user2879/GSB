from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .errors import NodeLifecycleError, PortInUseError
from .node import ERROR_PREFIX, READY_PREFIX
from .protocol import JsonLineClient, wait_for_port
from .spec import NodeSpec


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NodeHandle:
    def __init__(self, spec: NodeSpec, *, ledger_path: str | Path | None = None, startup_timeout: float = 5.0) -> None:
        self.spec = spec
        self.ledger_path = None if ledger_path is None else Path(ledger_path)
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen[str] | None = None
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self._threads: list[threading.Thread] = []

    def _drain(self, pipe: Any, sink: list[str]) -> None:
        if pipe is None:
            return
        for line in pipe:
            sink.append(line.rstrip("\n"))

    def start(self) -> "NodeHandle":
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        command = [
            sys.executable,
            "-m",
            "cacheflow.node",
            "--node-id",
            self.spec.node_id,
            "--host",
            self.spec.host,
            "--port",
            str(self.spec.port),
            "--seed",
            self.spec.seed,
        ]
        if self.ledger_path is not None:
            command.extend(["--ledger", str(self.ledger_path)])
        self.proc = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert self.proc.stdout is not None and self.proc.stderr is not None
        out_thread = threading.Thread(target=self._drain, args=(self.proc.stdout, self.stdout_lines), daemon=True)
        err_thread = threading.Thread(target=self._drain, args=(self.proc.stderr, self.stderr_lines), daemon=True)
        out_thread.start()
        err_thread.start()
        self._threads = [out_thread, err_thread]

        deadline = time.monotonic() + self.startup_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self._raise_start_failure()
            try:
                client = wait_for_port(
                    self.spec.host,
                    self.spec.port,
                    timeout_seconds=0.4,
                    node_id=self.spec.node_id,
                )
                client.close()
                return self
            except Exception as exc:  # startup polling; exit is checked above
                last_error = exc
                time.sleep(0.03)
        self.kill()
        raise NodeLifecycleError(
            f"node {self.spec.node_id} did not become ready: {last_error!r}",
            entry=self.spec.node_id,
            field="handshake",
        )

    def _raise_start_failure(self) -> None:
        assert self.proc is not None
        error_payload: dict[str, Any] | None = None
        for line in self.stderr_lines + self.stdout_lines:
            if ERROR_PREFIX in line:
                raw = line.split(ERROR_PREFIX, 1)[1]
                try:
                    import json

                    error_payload = json.loads(raw)
                except Exception:
                    error_payload = {"code": "E_NODE", "message": raw}
        if error_payload and error_payload.get("code") == "E_PORT":
            raise PortInUseError(
                error_payload.get("message", "port already in use"),
                entry=self.spec.node_id,
                field="port",
                details={"port": self.spec.port},
            )
        raise NodeLifecycleError(
            f"node {self.spec.node_id} exited with code {self.proc.returncode}",
            entry=self.spec.node_id,
            details={"stderr": "\n".join(self.stderr_lines[-20:])},
        )

    def client(self, *, timeout_seconds: float = 3.0) -> JsonLineClient:
        client = JsonLineClient(
            self.spec.host,
            self.spec.port,
            node_id=self.spec.node_id,
            timeout_seconds=timeout_seconds,
        )
        client.connect()
        client.handshake()
        return client

    def poll(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def send_signal(self, sig: int) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                pass

    def kill(self) -> int:
        if self.proc is None:
            return 0
        returncode = self.proc.poll()
        if returncode is None:
            self.send_signal(signal.SIGKILL)
            try:
                returncode = self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.send_signal(signal.SIGKILL)
                returncode = self.proc.wait(timeout=3)
        return int(returncode)

    def terminate(self) -> int:
        if self.proc is None:
            return 0
        if self.proc.poll() is None:
            self.send_signal(signal.SIGTERM)
            try:
                return int(self.proc.wait(timeout=2))
            except subprocess.TimeoutExpired:
                return self.kill()
        return int(self.proc.returncode)


class Cluster:
    def __init__(self, nodes: list[NodeSpec] | tuple[NodeSpec, ...], *, ledger_path: str | Path | None = None) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self.ledger_path = ledger_path
        self.handles: dict[str, NodeHandle] = {}

    def start_all(self) -> "Cluster":
        for node_id in sorted(self.nodes):
            self.start_node(node_id)
        return self

    def start_node(self, node_id: str) -> NodeHandle:
        spec = self.nodes[node_id]
        handle = NodeHandle(spec, ledger_path=self.ledger_path).start()
        self.handles[node_id] = handle
        return handle

    def client(self, node_id: str, *, timeout_seconds: float = 3.0) -> JsonLineClient:
        if node_id not in self.handles or self.handles[node_id].poll() is not None:
            self.start_node(node_id)
        return self.handles[node_id].client(timeout_seconds=timeout_seconds)

    def kill_node(self, node_id: str) -> int:
        handle = self.handles[node_id]
        return handle.kill()

    def restart_node(self, node_id: str) -> NodeHandle:
        old = self.handles.get(node_id)
        if old is not None and old.poll() is None:
            old.kill()
        handle = NodeHandle(self.nodes[node_id], ledger_path=self.ledger_path).start()
        self.handles[node_id] = handle
        return handle

    def close(self) -> None:
        for node_id in sorted(self.handles, reverse=True):
            self.handles[node_id].terminate()

    def __enter__(self) -> "Cluster":
        return self.start_all()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
