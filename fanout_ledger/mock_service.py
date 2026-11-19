"""Deterministic loopback mock target service.

Every node runs its own ``http.server`` based daemon on 127.0.0.1.  The
outcome of request *n* for a node is a pure function of the node's ``seed``
and ``n`` (plus the configured rates), so replaying the same attempt sequence
against freshly started services reproduces the same outcome sequence.

Deterministic rule (all rates are integer basis points, 0..10000):

* ``bucket = (seed * 7919 + attempt_seq * 104729) % 10000``
* ``bucket < failure_rate``            -> HTTP 500 (failure)
* otherwise ``bucket < failure_rate + slow_rate`` -> slow response, add
  ``slow_extra_ms`` sleeping inside the handler (may cross the client socket
  timeout and then surface as a real ``socket.timeout``)
* otherwise                            -> HTTP 200 (success)

A slow response is still served to completion; whether it becomes a timeout
depends purely on the client's real socket timeout.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import NodeSpec
from .errors import PortError


def deterministic_bucket(seed: int, attempt_seq: int) -> int:
    return (seed * 7919 + attempt_seq * 104729) % 10000


def classify_request(node: NodeSpec, attempt_seq: int) -> dict[str, Any]:
    bucket = deterministic_bucket(node.seed, attempt_seq)
    if bucket < node.failure_rate:
        return {"bucket": bucket, "kind": "failure"}
    if bucket < node.failure_rate + node.slow_rate:
        return {"bucket": bucket, "kind": "slow", "delay_ms": node.slow_extra_ms}
    return {"bucket": bucket, "kind": "success"}


class _Handler(BaseHTTPRequestHandler):
    server_version = "FanoutMock/1.0"

    @property
    def node(self) -> NodeSpec:
        return self.server.node_spec  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # silence stderr
        return

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            request = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._write_json(400, {"error": "bad json"})
            return
        attempt_seq = int(request.get("attempt_seq", 0))
        verdict = classify_request(self.node, attempt_seq)
        # Deterministic per-request processing cost: a fixed base delay plus,
        # for the deterministic "slow" bucket, an additional delay.  Both are
        # real sleeps inside the handler; a client whose socket timeout is
        # shorter observes a genuine socket.timeout.
        delay_ms = self.node.process_ms
        if verdict["kind"] == "slow":
            delay_ms += verdict["delay_ms"]
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        if verdict["kind"] == "failure":
            self._write_json(500, {"node_id": self.node.node_id, "attempt_seq": attempt_seq,
                                   "bucket": verdict["bucket"], "result": "failure"})
            return
        self._write_json(200, {"node_id": self.node.node_id, "attempt_seq": attempt_seq,
                               "bucket": verdict["bucket"], "result": verdict["kind"]})

    do_GET = do_POST  # health check reuses the same logic


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], node: NodeSpec):
        self.node_spec = node
        super().__init__(address, _Handler)


@dataclass
class MockServerHandle:
    node: NodeSpec
    server: _Server
    thread: threading.Thread

    @property
    def endpoint(self) -> str:
        return f"http://{self.node.host}:{self.node.port}/"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        _wait_until_free(self.node.host, self.node.port)


def _wait_until_free(host: str, port: int, attempts: int = 50) -> None:
    """Block until the loopback port is actually released after server_close."""
    for _ in range(attempts):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((host, port))
            return
        except OSError:
            time.sleep(0.02)
        finally:
            probe.close()


def ensure_port_free(host: str, port: int) -> None:
    """Raise E_PORT if the target loopback port is already occupied."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    except OSError as exc:
        raise PortError(
            f"port {host}:{port} is already in use",
            {"host": host, "port": port, "os_error": str(exc)},
        ) from exc
    finally:
        probe.close()


def start_mock_server(node: NodeSpec) -> MockServerHandle:
    ensure_port_free(node.host, node.port)
    try:
        server = _Server((node.host, node.port), node)
    except OSError as exc:  # race with another process between probe and bind
        raise PortError(
            f"cannot bind mock service on {node.host}:{node.port}",
            {"host": node.host, "port": node.port, "os_error": str(exc)},
        ) from exc
    thread = threading.Thread(target=server.serve_forever, name=f"mock-{node.node_id}",
                              daemon=True)
    thread.start()
    return MockServerHandle(node, server, thread)


def start_all(nodes: tuple[NodeSpec, ...]) -> list[MockServerHandle]:
    handles: list[MockServerHandle] = []
    try:
        for node in nodes:  # stable scene order, no dict/hash iteration
            handles.append(start_mock_server(node))
    except BaseException:
        for handle in handles:
            handle.stop()
        raise
    return handles


def stop_all(handles: list[MockServerHandle]) -> None:
    for handle in handles:
        handle.stop()
