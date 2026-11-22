"""Loopback TCP client and local node subprocess lifecycle management."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import ExhaustError
from .model import NodeEntry

_READY_TIMEOUT_S = 10.0


class NodeProcess:
    """Spawn and supervise one node subprocess on 127.0.0.1."""

    def __init__(self, entry: NodeEntry):
        self.entry = entry
        self.process: subprocess.Popen | None = None
        self.host = entry.host
        self.port = entry.port

    def start(self) -> None:
        cmd = [
            sys.executable, "-m", "config_regression.node",
            "--node-id", self.entry.node_id,
            "--host", self.entry.host,
            "--port", str(self.entry.port),
            "--work-ms-min", str(self.entry.work_ms_min),
            "--work-ms-max", str(self.entry.work_ms_max),
            "--fault-mode", self.entry.fault_mode,
            "--fault-arg", str(self.entry.fault_arg),
        ]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert self.process.stdout is not None
        deadline = time.monotonic() + _READY_TIMEOUT_S
        line = self.process.stdout.readline()
        if not line:
            err = self.process.stderr.read() if self.process.stderr else ""
            raise ExhaustError(
                f"node {self.entry.node_id} did not become ready",
                [(self.entry.node_id, err.strip() or "no readiness line")])
        ready = json.loads(line)
        if ready.get("event") != "ready":
            raise ExhaustError(
                f"node {self.entry.node_id} sent unexpected readiness payload",
                [(self.entry.node_id, line.strip())])
        self.host = ready["host"]
        self.port = int(ready["port"])

    def stop(self) -> None:
        if self.process is None:
            return
        process = self.process
        self.process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()


@dataclass
class AttemptRecord:
    attempt: int
    started_monotonic_ms: int
    ended_monotonic_ms: int
    outcome: str
    error: str | None


def send_request(host: str, port: int, payload: dict,
                 timeout_ms: int) -> dict:
    """Send one JSON-line request with a real socket timeout.

    Raises OSError / socket.timeout / ConnectionError on connection or read
    failure so the scheduler can apply its retry policy.
    """

    timeout_s = max(1, timeout_ms) / 1000.0
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall((json.dumps(payload, sort_keys=True) + "\n")
                     .encode("utf-8"))
        buffer = b""
        while b"\n" not in buffer:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("peer closed connection without reply")
            buffer += chunk
    line, _, _ = buffer.partition(b"\n")
    response = json.loads(line.decode("utf-8"))
    if response.get("status") != "ok":
        raise ConnectionError(f"node error response: {response.get('error')}")
    return response
