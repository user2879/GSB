from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .errors import TransportError
from .spec import NodeSpec, Spec


MAX_LINE_BYTES = 1_048_576


def _read_json_line(stream: Any, timeout_s: float) -> str | None:
    ready, _, _ = select.select([stream], [], [], timeout_s)
    if not ready:
        return None
    return stream.readline()


class NodeProcess:
    def __init__(self, node: NodeSpec, process: subprocess.Popen[str]) -> None:
        self.node = node
        self.process = process

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass

    def force_stop(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.kill()
        except OSError:
            pass
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


class LiveNodeCluster:
    def __init__(self, spec: Spec) -> None:
        self.spec = spec
        self.processes: dict[str, NodeProcess] = {}
        self._root = Path(__file__).resolve().parent.parent

    def __enter__(self) -> "LiveNodeCluster":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def start(self) -> None:
        started: list[NodeProcess] = []
        try:
            for node in self.spec.nodes:
                process = self._spawn(node)
                started.append(process)
                self.processes[node.node_id] = process
        except Exception:
            for process in reversed(started):
                process.force_stop()
            raise

    def stop(self) -> None:
        for node_id in sorted(self.processes, reverse=True):
            self.processes[node_id].stop()
        self.processes.clear()

    def force_stop(self) -> None:
        for node_id in sorted(self.processes, reverse=True):
            self.processes[node_id].force_stop()
        self.processes.clear()

    def _spawn(self, node: NodeSpec) -> NodeProcess:
        self._assert_port_bindable(node)
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self._root) + (os.pathsep + existing if existing else "")
        command = [
            sys.executable,
            "-m",
            "lease_arbiter.node_server",
            "--node-id",
            node.node_id,
            "--host",
            node.host,
            "--port",
            str(node.port),
            "--read-timeout-ms",
            str(self.spec.read_timeout_ms),
        ]
        process = subprocess.Popen(
            command,
            cwd=str(self._root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        startup_timeout = max(2.0, (self.spec.connect_timeout_ms + self.spec.read_timeout_ms) / 1000.0)
        line = _read_json_line(process.stdout, startup_timeout)
        if line is None:
            stderr = ""
            if process.poll() is not None and process.stderr is not None:
                stderr = process.stderr.read()
            process.kill()
            process.wait(timeout=2.0)
            if "E_PORT" in stderr:
                raise TransportError("E_PORT", "node failed to bind loopback port", node_id=node.node_id, port=node.port)
            raise TransportError(
                "E_TIMEOUT",
                "node did not report readiness before timeout",
                node_id=node.node_id,
                port=node.port,
                stderr=stderr,
            )
        try:
            status = json.loads(line)
        except json.JSONDecodeError as exc:
            process.kill()
            process.wait(timeout=2.0)
            raise TransportError(
                "E_PROTOCOL",
                "node startup emitted invalid JSON",
                node_id=node.node_id,
                raw=line.rstrip("\n"),
            ) from exc
        if not status.get("ok") or status.get("code") != "READY":
            code = status.get("code", "E_PROTOCOL")
            process.kill()
            process.wait(timeout=2.0)
            if code == "E_PORT":
                raise TransportError("E_PORT", status.get("message", "port occupied"), node_id=node.node_id, port=node.port)
            raise TransportError("E_PROTOCOL", status.get("message", "node startup failed"), node_id=node.node_id)
        return NodeProcess(node, process)

    @staticmethod
    def _assert_port_bindable(node: NodeSpec) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((node.host, node.port))
        except OSError as exc:
            raise TransportError(
                "E_PORT",
                f"loopback port is already occupied: {node.host}:{node.port}",
                node_id=node.node_id,
                port=node.port,
                error=str(exc),
            ) from exc
        finally:
            probe.close()


class JsonLineNodeClient:
    def __init__(self, connect_timeout_ms: int, read_timeout_ms: int) -> None:
        self.connect_timeout_s = connect_timeout_ms / 1000.0
        self.read_timeout_s = read_timeout_ms / 1000.0

    def request(self, node: NodeSpec, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id")
        request = dict(request)
        request.setdefault("request_id", request_id)
        conn: socket.socket | None = None
        try:
            conn = socket.create_connection((node.host, node.port), timeout=self.connect_timeout_s)
            conn.settimeout(self.read_timeout_s)
            hello = {
                "op": "hello",
                "protocol": 1,
                "request_id": request_id,
                "client": "lease-arbiter-central",
            }
            conn.sendall((canonical_json(hello) + "\n").encode("utf-8"))
            hello_response = self._read_frame(conn, node, "hello")
            if hello_response.get("ok") is not True or hello_response.get("protocol") != 1:
                raise TransportError(
                    "E_PROTOCOL",
                    "invalid node hello response",
                    node_id=node.node_id,
                    response=hello_response,
                )
            conn.sendall((canonical_json(request) + "\n").encode("utf-8"))
            return self._read_frame(conn, node, str(request.get("op")))
        except socket.timeout as exc:
            raise TransportError(
                "E_TIMEOUT",
                "node socket timeout",
                node_id=node.node_id,
                port=node.port,
                op=request.get("op"),
            ) from exc
        except ConnectionRefusedError as exc:
            raise TransportError(
                "E_PORT",
                "node loopback endpoint refused the connection",
                node_id=node.node_id,
                port=node.port,
            ) from exc
        except OSError as exc:
            if isinstance(exc, socket.timeout):
                raise TransportError("E_TIMEOUT", "node socket timeout", node_id=node.node_id, port=node.port) from exc
            raise TransportError(
                "E_PORT",
                "node loopback endpoint is unavailable",
                node_id=node.node_id,
                port=node.port,
                error=str(exc),
            ) from exc
        finally:
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass

    def _read_frame(self, conn: socket.socket, node: NodeSpec, phase: str) -> dict[str, Any]:
        buffer = bytearray()
        deadline = time.monotonic() + self.read_timeout_s
        while True:
            if len(buffer) > MAX_LINE_BYTES:
                raise TransportError("E_PROTOCOL", "node response line is too large", node_id=node.node_id, phase=phase)
            newline = buffer.find(b"\n")
            if newline >= 0:
                payload = bytes(buffer[:newline])
                try:
                    value = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise TransportError(
                        "E_PROTOCOL",
                        "node returned invalid JSON",
                        node_id=node.node_id,
                        phase=phase,
                    ) from exc
                if not isinstance(value, dict):
                    raise TransportError("E_PROTOCOL", "node response must be an object", node_id=node.node_id, phase=phase)
                return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise socket.timeout()
            conn.settimeout(remaining)
            chunk = conn.recv(65536)
            if not chunk:
                raise TransportError(
                    "E_PROTOCOL",
                    "node closed the connection before a complete JSON line",
                    node_id=node.node_id,
                    phase=phase,
                )
            buffer.extend(chunk)
