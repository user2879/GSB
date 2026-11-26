from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .errors import make_error
from .models import NodeSpec


class ProtocolError(Exception):
    def __init__(self, error: dict[str, Any]):
        super().__init__(canonical_json(error))
        self.error = error


class JsonLineClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout: float = 1.0,
        response_timeout: float = 5.0,
    ):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.response_timeout = response_timeout

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout
        ) as sock:
            sock.settimeout(self.response_timeout)
            sock.sendall(canonical_json(payload).encode("utf-8") + b"\n")
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
            data = b"".join(chunks).split(b"\n", 1)[0]
            if not data:
                raise ConnectionError("node closed connection without a response")
            return json.loads(data.decode("utf-8"))


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class NodeProcessManager:
    def __init__(self, specs: list[NodeSpec], workdir: Path):
        self.specs = {spec.node_id: spec for spec in specs}
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.clients: dict[str, JsonLineClient] = {}
        self.ready: dict[str, dict[str, Any]] = {}
        self.execution_errors: list[dict[str, Any]] = []

    def start_all(self) -> None:
        for node_id in sorted(self.specs):
            self._start(self.specs[node_id])

    def _start(self, spec: NodeSpec) -> None:
        spec_file = self.workdir / f"node-{spec.node_id}.json"
        spec_file.write_text(
            json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = source_root + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "reco_shard_diag.node",
                "--spec-file",
                str(spec_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(self.workdir),
            text=False,
        )
        self.processes[spec.node_id] = process
        deadline = time.monotonic() + 5.0
        ready: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
                raise RuntimeError(
                    f"node {spec.node_id} exited early with {process.returncode}: {stderr}"
                )
            line = process.stdout.readline() if process.stdout else b""
            if line:
                try:
                    candidate = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if candidate.get("event") == "READY":
                    ready = candidate
                    break
        if ready is None:
            self.stop_all(force=True)
            raise RuntimeError(f"node {spec.node_id} did not become ready")
        self.clients[spec.node_id] = JsonLineClient(spec.host, spec.port)
        self.ready[spec.node_id] = ready
        for _ in range(20):
            try:
                response = self.clients[spec.node_id].request({"operation": "ping"})
                if response.get("ok"):
                    break
            except (ConnectionError, OSError, json.JSONDecodeError):
                time.sleep(0.01)
        else:
            self.stop_all(force=True)
            raise RuntimeError(f"node {spec.node_id} is not reachable")

    def client(self, node_id: str) -> JsonLineClient:
        return self.clients[node_id]

    def request(self, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.client(node_id).request(payload)
        except (ConnectionError, OSError, json.JSONDecodeError) as exc:
            error = make_error(
                "E_NODE_UNAVAILABLE",
                "node",
                node_id,
                "socket",
                str(exc),
            ).to_dict()
            self.execution_errors.append(error)
            raise ProtocolError(error) from exc

    def statuses(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for node_id in sorted(self.specs):
            try:
                result[node_id] = self.request(node_id, {"operation": "status"})
            except ProtocolError as exc:
                result[node_id] = {"ok": False, "error": exc.error}
        return result

    def sweep_all(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for node_id in sorted(self.specs):
            try:
                result[node_id] = self.request(node_id, {"operation": "sweep"})
            except ProtocolError as exc:
                result[node_id] = {"ok": False, "error": exc.error}
        return result

    def ledgers(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for node_id in sorted(self.specs):
            try:
                result[node_id] = self.request(node_id, {"operation": "ledger"})
            except ProtocolError as exc:
                result[node_id] = {"ok": False, "error": exc.error}
        return result

    def stop_all(self, *, force: bool = False) -> None:
        for node_id, process in sorted(self.processes.items()):
            if process.poll() is not None:
                continue
            process.terminate()
        deadline = time.monotonic() + 2.0
        for node_id, process in sorted(self.processes.items()):
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
            if force and process.stderr:
                try:
                    process.stderr.close()
                except OSError:
                    pass
        self.processes.clear()
        self.clients.clear()

    def __enter__(self) -> "NodeProcessManager":
        self.start_all()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_all()


def reserve_ports(count: int) -> list[int]:
    ports: list[int] = []
    sockets: list[socket.socket] = []
    try:
        while len(ports) < count:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
            if port not in ports:
                ports.append(port)
                sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close()
    return ports
