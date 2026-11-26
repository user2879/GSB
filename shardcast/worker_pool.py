"""Orchestrate node worker subprocesses on real 127.0.0.1 TCP ports."""

from __future__ import annotations
import os
import socket
import subprocess
import sys
import time

from .errors import PortError
from .protocol import HANDSHAKE_HELLO, HANDSHAKE_READY, read_line, write_line
from .spec import NodeSpec, Spec


def port_in_use(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        probe.connect((host, port))
    except OSError:
        return False
    else:
        return True
    finally:
        probe.close()


class WorkerHandle:
    def __init__(self, node: NodeSpec, process: subprocess.Popen):
        self.node = node
        self.process = process

    def is_alive(self) -> bool:
        return self.process.poll() is None


class WorkerPool:
    def __init__(self, spec: Spec, readiness_timeout: float = 5.0):
        self.spec = spec
        self.readiness_timeout = readiness_timeout
        self._handles: dict[str, WorkerHandle] = {}

    def start_all(self) -> None:
        for node in self.spec.nodes:
            if port_in_use(node.host, node.port):
                raise PortError(
                    f"port already occupied before launch: {node.host}:{node.port}",
                    path=f"nodes.{node.node_id}.port",
                )
        for node in self.spec.nodes:
            self._start_one(node)

    def _start_one(self, node: NodeSpec) -> WorkerHandle:
        command = [
            sys.executable,
            "-m",
            "shardcast.node",
            "--node-id",
            node.node_id,
            "--host",
            node.host,
            "--port",
            str(node.port),
            "--seed",
            str(node.seed),
            "--ttl-ms",
            str(self.spec.lease_ttl_ms),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        handle = WorkerHandle(node, process)
        self._handles[node.node_id] = handle
        self._wait_ready(node)
        return handle

    def _wait_ready(self, node: NodeSpec) -> None:
        deadline = time.monotonic() + self.readiness_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            handle = self._handles[node.node_id]
            if not handle.is_alive():
                stderr = handle.process.stderr.read().decode("utf-8", "replace") if handle.process.stderr else ""
                if "bind_error" in stderr or "Address already in use" in stderr:
                    raise PortError(
                        f"port bind failed for {node.node_id}: {stderr.strip()}",
                        path=f"nodes.{node.node_id}.port",
                    )
                raise PortError(
                    f"node {node.node_id} exited during startup: {stderr.strip()}",
                    path=f"nodes.{node.node_id}.port",
                )
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            try:
                sock.connect((node.host, node.port))
                write_line(sock, HANDSHAKE_HELLO)
                ready = read_line(sock)
                if ready == HANDSHAKE_READY:
                    sock.close()
                    return
            except OSError as exc:
                last_error = exc
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
            time.sleep(0.02)
        raise PortError(
            f"node {node.node_id} did not become ready: {last_error}",
            path=f"nodes.{node.node_id}.port",
        )

    def restart_node(self, node_id: str) -> WorkerHandle:
        """Force-kill and restart one worker; lease state is rebuilt by caller."""
        handle = self._handles.get(node_id)
        node = self.spec.node(node_id)
        if handle is not None and handle.is_alive():
            handle.process.kill()
            handle.process.wait(timeout=5)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and port_in_use(node.host, node.port):
            time.sleep(0.02)
        return self._start_one(node)

    def kill_node_hard(self, node_id: str) -> None:
        handle = self._handles[node_id]
        handle.process.kill()
        handle.process.wait(timeout=5)

    def install_lease(self, node_id: str, lease: dict) -> None:
        """Push an authoritative lease into a fresh worker via an internal socket."""
        node = self.spec.node(node_id)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((node.host, node.port))
        try:
            write_line(sock, HANDSHAKE_HELLO)
            ready = read_line(sock)
            if ready != HANDSHAKE_READY:
                raise PortError("worker rejected handshake during lease install", path=node.node_id)
            write_line(sock, {"type": "install_lease", "lease": lease})
            response = read_line(sock)
            if response.get("type") != "lease_installed":
                raise PortError("worker rejected lease install", path=node.node_id)
        finally:
            sock.close()

    def stop_all(self) -> None:
        for handle in list(self._handles.values()):
            if handle.is_alive():
                handle.process.terminate()
        for handle in list(self._handles.values()):
            try:
                handle.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handle.process.kill()
                handle.process.wait(timeout=5)
            for stream in (handle.process.stdout, handle.process.stderr):
                if stream is not None:
                    stream.close()
        self._handles.clear()

    def handles(self) -> dict[str, WorkerHandle]:
        return dict(self._handles)
