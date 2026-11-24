"""Node subprocess orchestration and the JSON-line TCP client protocol."""

import json
import socket
import subprocess
import sys
import time

from .errors import FieldError, PortError, TimeoutError
from .node import REQUIRED_REPLY_FIELDS
from .spec import NodeSpec


class NodeProcess:
    def __init__(self, manifest_path: str, node: NodeSpec):
        self.node = node
        self.manifest_path = manifest_path
        self.process = None

    def start(self, startup_timeout: float = 8.0):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((self.node.host, self.node.port))
        except OSError as exc:
            probe.close()
            raise PortError(
                "loopback port already in use",
                entity=self.node.node_id,
                field="port",
                port=self.node.port,
                detail=str(exc),
            ) from exc
        else:
            probe.close()

        command = [sys.executable, "-m", "shard_ledger.node", self.manifest_path, self.node.node_id]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if line.startswith("NODE_READY"):
                return self
            if self.process.poll() is not None:
                stderr = self.process.stderr.read()
                raise PortError(
                    "node process exited before becoming ready",
                    entity=self.node.node_id,
                    field="process",
                    stderr=stderr.strip(),
                )
        self.stop()
        raise TimeoutError("node handshake startup timed out", entity=self.node.node_id, field="handshake")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)
        if self.process is not None:
            for stream in (self.process.stdout, self.process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()


def start_all(manifest_path: str, nodes: dict) -> dict:
    processes = {}
    try:
        for node_id in sorted(nodes):
            process = NodeProcess(manifest_path, nodes[node_id]).start()
            processes[node_id] = process
    except Exception:
        for process in processes.values():
            process.stop()
        raise
    return processes


def stop_all(processes: dict) -> None:
    for process in processes.values():
        process.stop()


def kill_orphans(manifest_path: str, manifest) -> None:
    """Best-effort cleanup of node processes orphaned by a hard crash."""
    import os
    import signal
    import subprocess

    pattern = f"shard_ledger.node {manifest_path}"
    try:
        listing = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, check=False)
    except OSError:
        return
    for line in listing.stdout.splitlines():
        if pattern in line and "grep" not in line:
            try:
                pid = int(line.strip().split()[0])
                if pid != os.getpid():
                    os.kill(pid, signal.SIGTERM)
            except (ValueError, ProcessLookupError, PermissionError):
                continue


def send_batch(
    node: NodeSpec,
    shard_id: str,
    config_version: int,
    timeout_ms: int,
    handshake: bool = True,
) -> dict:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_ms / 1000.0)
    try:
        try:
            sock.connect((node.host, node.port))
        except ConnectionRefusedError as exc:
            raise PortError("connection refused on loopback port", entity=node.node_id, field="port") from exc
        except socket.timeout as exc:
            raise TimeoutError("real socket connect timeout", entity=node.node_id, field="connect") from exc

        if handshake:
            try:
                sock.sendall((json.dumps({"op": "handshake"}) + "\n").encode())
                ack = _read_json_line(sock)
            except socket.timeout as exc:
                raise TimeoutError("real socket handshake timeout", entity=node.node_id, field="handshake") from exc
            if ack.get("op") != "handshake_ack":
                raise FieldError("invalid handshake acknowledgement", entity=node.node_id, field="handshake")

        request = {"op": "batch", "shard_id": shard_id, "config_version": config_version}
        try:
            sock.sendall((json.dumps(request) + "\n").encode())
            reply = _read_json_line(sock)
        except socket.timeout as exc:
            raise TimeoutError("real socket batch timeout", entity=node.node_id, field="batch") from exc

        for required in REQUIRED_REPLY_FIELDS:
            if required not in reply:
                raise FieldError("receipt missing required field", entity=shard_id, field=required)
        if reply["node_id"] != node.node_id:
            raise FieldError("receipt node_id mismatch", entity=shard_id, field="node_id")
        if reply["shard_id"] != shard_id:
            raise FieldError("receipt shard_id mismatch", entity=shard_id, field="shard_id")
        return reply
    finally:
        sock.close()


def _read_json_line(sock: socket.socket) -> dict:
    buffer = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise FieldError("peer closed connection without a complete receipt", entity="socket", field="eof")
        buffer.extend(chunk)
        if b"\n" in buffer:
            line, _, _ = buffer.partition(b"\n")
            return json.loads(line.decode("utf-8"))
