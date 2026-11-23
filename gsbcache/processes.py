"""Windows-friendly subprocess lifecycle management for local cache nodes."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import HandshakeError, PortError
from .protocol import NodeClient
from .spec import NodeSpec, Spec


@dataclass
class NodeProcess:
    spec_node: NodeSpec
    data_dir: Path
    process: subprocess.Popen[str]
    stdout_queue: "queue.Queue[str]"
    stderr_lines: list[str]
    client: NodeClient


class NodeReader(threading.Thread):
    def __init__(self, stream, output: "queue.Queue[str]") -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.output = output

    def run(self) -> None:
        try:
            for line in self.stream:
                self.output.put(line.rstrip("\r\n"))
        finally:
            self.output.put("__EOF__")


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


class NodeCluster:
    def __init__(
        self,
        spec: Spec,
        spec_path: str | Path,
        nodes_root: str | Path,
        startup_timeout_s: float = 5.0,
        handshake_timeout_s: float = 2.0,
    ) -> None:
        self.spec = spec
        self.spec_path = str(Path(spec_path).resolve())
        self.nodes_root = Path(nodes_root)
        self.startup_timeout_s = startup_timeout_s
        self.handshake_timeout_s = handshake_timeout_s
        self.nodes: dict[str, NodeProcess] = {}
        self._stopped = False

    def start_all(self) -> dict[str, NodeProcess]:
        try:
            for node in self.spec.nodes:
                self.nodes[node.node_id] = self.start_one(node)
            return self.nodes
        except Exception:
            self.stop_all()
            raise

    def start_one(self, node: NodeSpec) -> NodeProcess:
        data_dir = self.nodes_root / node.node_id
        data_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "gsbcache.node",
            "--spec",
            self.spec_path,
            "--node-id",
            node.node_id,
            "--data-dir",
            str(data_dir),
            "--parent-pid",
            str(os.getpid()),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
        stdout_queue: "queue.Queue[str]" = queue.Queue()
        stderr_queue: "queue.Queue[str]" = queue.Queue()
        NodeReader(process.stdout, stdout_queue).start()
        NodeReader(process.stderr, stderr_queue).start()
        deadline = time.monotonic() + self.startup_timeout_s
        ready: dict[str, Any] | None = None
        stderr_lines: list[str] = []
        while time.monotonic() < deadline:
            if process.poll() is not None and stdout_queue.empty():
                break
            try:
                line = stdout_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if line == "__EOF__":
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "ready":
                ready = payload
                break
            if payload.get("ok") is False:
                code = payload.get("code")
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
                _close_process_pipes(process)
                if code == "E_PORT":
                    raise PortError(
                        "node failed to bind port",
                        node_id=node.node_id,
                        host=node.host,
                        port=node.port,
                        remote=payload,
                    )
                raise HandshakeError(
                    "node failed before readiness",
                    node_id=node.node_id,
                    remote=payload,
                )
        while True:
            try:
                stderr_lines.append(stderr_queue.get_nowait())
            except queue.Empty:
                break
        if ready is None:
            exit_code = process.poll()
            details = {
                "node_id": node.node_id,
                "host": node.host,
                "port": node.port,
                "exit_code": exit_code,
                "stderr": stderr_lines[-20:],
            }
            encoded = "\n".join(stderr_lines).lower()
            if exit_code is not None or "address already in use" in encoded or "e_port" in encoded:
                process.kill()
                process.wait(timeout=2)
                _close_process_pipes(process)
                raise PortError("node did not become ready on its loopback port", **details)
            process.kill()
            process.wait(timeout=2)
            _close_process_pipes(process)
            raise HandshakeError("node readiness handshake failed", **details)
        client = NodeClient(node.host, node.port, timeout_s=max(1.0, self.handshake_timeout_s))
        try:
            hello = client.hello(node.node_id)
        except HandshakeError:
            try:
                process.kill()
                process.wait(timeout=2)
            except Exception:
                pass
            _close_process_pipes(process)
            raise
        if hello.get("node_id") != node.node_id:
            process.kill()
            process.wait(timeout=2)
            _close_process_pipes(process)
            raise HandshakeError(
                "ready node id mismatch",
                node_id=node.node_id,
                actual=hello.get("node_id"),
            )
        return NodeProcess(
            spec_node=node,
            data_dir=data_dir,
            process=process,
            stdout_queue=stdout_queue,
            stderr_lines=stderr_lines,
            client=client,
        )

    def client(self, node_id: str) -> NodeClient:
        return self.nodes[node_id].client

    def stop_all(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for node_id, item in list(self.nodes.items()):
            try:
                item.client.shutdown(node_id)
            except Exception:
                pass
        for node_id, item in list(self.nodes.items()):
            process = item.process
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
            _close_process_pipes(process)
        self.nodes.clear()

    def kill_all(self) -> None:
        for item in list(self.nodes.values()):
            item.process.kill()
        for item in list(self.nodes.values()):
            try:
                item.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
            _close_process_pipes(item.process)
        self.nodes.clear()
        self._stopped = True

    def __enter__(self) -> "NodeCluster":
        self.start_all()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_all()
