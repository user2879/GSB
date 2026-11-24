"""Node process orchestration and JSON-line client with real socket timeouts."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .spec import NodeSpec
from .util import canonical_json, is_int, monotonic_ms

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NodeClient:
    def __init__(self, node: NodeSpec, *, timeout_s: float = 3.0):
        self.node = node
        self.timeout_s = timeout_s

    @property
    def position(self) -> str:
        return f"node:{self.node.id}@127.0.0.1:{self.node.port}"

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = dict(payload)
        request.setdefault("node_id", self.node.id)
        request.setdefault("request_id", f"{self.node.id}:{monotonic_ms()}")
        line = (canonical_json(request) + "\n").encode("utf-8")
        try:
            with socket.create_connection(("127.0.0.1", self.node.port), timeout=self.timeout_s) as sock:
                sock.settimeout(self.timeout_s)
                sock.sendall(line)
                chunks = bytearray()
                while not chunks.endswith(b"\n"):
                    part = sock.recv(65536)
                    if not part:
                        break
                    chunks.extend(part)
                    if len(chunks) > 2_000_000:
                        raise BusinessError(
                            "E_REPLY",
                            "node reply exceeded size limit",
                            position=self.position,
                            field="reply",
                        )
        except socket.timeout as exc:
            raise BusinessError(
                "E_TIMEOUT",
                f"real socket timeout after {self.timeout_s:.3f}s",
                position=self.position,
                field="timeout_s",
                detail={"timeout_s": self.timeout_s},
            ) from exc
        except OSError as exc:
            raise BusinessError(
                "E_PORT",
                f"cannot communicate with node socket: {exc}",
                position=self.position,
                field="port",
                detail={"port": self.node.port},
            ) from exc
        if not chunks:
            raise BusinessError("E_REPLY", "empty node reply", position=self.position, field="reply")
        try:
            reply = json.loads(bytes(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BusinessError("E_REPLY", f"invalid JSON reply: {exc}", position=self.position, field="reply") from exc
        if not isinstance(reply, dict):
            raise BusinessError("E_REPLY", "node reply must be an object", position=self.position, field="reply")
        if reply.get("ok") is not True:
            code = reply.get("code", "E_REPLY")
            if code not in {"E_LEASE", "E_REPLY", "E_TIMEOUT", "E_PORT"}:
                code = "E_REPLY"
            raise BusinessError(
                code,
                str(reply.get("message", "node request failed")),
                position=self.position,
                field=reply.get("field"),
                detail={key: value for key, value in reply.items() if key not in {"ok", "code", "message", "field"}},
            )
        return reply

    def ping(self) -> dict[str, Any]:
        reply = self._request({"op": "ping"})
        for field_name in ("node_id", "ready", "pid"):
            if field_name not in reply:
                raise BusinessError("E_REPLY", f"reply missing field: {field_name}", position=self.position, field=field_name)
        if reply["node_id"] != self.node.id:
            raise BusinessError("E_REPLY", "reply node_id mismatch", position=self.position, field="node_id")
        if reply["ready"] is not True:
            raise BusinessError("E_REPLY", "node is not ready", position=self.position, field="ready")
        return reply

    def snapshot(self) -> dict[str, Any]:
        return self._request({"op": "snapshot"})

    def configure(self, *, concurrency: int, lease_limit: int, capacity_version: int) -> dict[str, Any]:
        return self._request(
            {
                "op": "configure",
                "concurrency": concurrency,
                "lease_limit": lease_limit,
                "capacity_version": capacity_version,
            }
        )

    def acquire_lease(self, *, lease_id: str, shard_id: str, ttl_ms: int) -> dict[str, Any]:
        reply = self._request({"op": "lease_acquire", "lease_id": lease_id, "shard_id": shard_id, "ttl_ms": ttl_ms})
        required = ("node_id", "lease_id", "shard_id", "granted_ms", "expires_ms", "version")
        for field_name in required:
            if field_name not in reply:
                raise BusinessError("E_REPLY", f"reply missing field: {field_name}", position=self.position, field=field_name)
        if reply["node_id"] != self.node.id or reply["lease_id"] != lease_id or reply["shard_id"] != shard_id:
            raise BusinessError("E_REPLY", "lease reply identity mismatch", position=self.position, field="lease_id")
        return reply

    def renew_lease(self, *, lease_id: str, expected_version: int, ttl_ms: int) -> dict[str, Any]:
        reply = self._request(
            {
                "op": "lease_renew",
                "lease_id": lease_id,
                "expected_version": expected_version,
                "ttl_ms": ttl_ms,
            }
        )
        for field_name in ("node_id", "lease_id", "version", "expires_ms"):
            if field_name not in reply:
                raise BusinessError("E_REPLY", f"reply missing field: {field_name}", position=self.position, field=field_name)
        if reply["node_id"] != self.node.id or reply["lease_id"] != lease_id:
            raise BusinessError("E_REPLY", "lease renewal identity mismatch", position=self.position, field="lease_id")
        return reply

    def release_lease(self, *, lease_id: str) -> None:
        self._request({"op": "lease_release", "lease_id": lease_id})

    def compute(
        self,
        *,
        shard_id: str,
        config_version: int,
        lease_id: str,
        lease_version: int,
        samples: list[list[int]],
        scale: int,
        group_modulus: int,
    ) -> dict[str, Any]:
        reply = self._request(
            {
                "op": "compute",
                "shard_id": shard_id,
                "config_version": config_version,
                "lease_id": lease_id,
                "lease_version": lease_version,
                "samples": samples,
                "scale": scale,
                "group_modulus": group_modulus,
            }
        )
        required = (
            "node_id",
            "shard_id",
            "config_version",
            "lease_id",
            "lease_version",
            "summary",
            "elapsed_monotonic_ms",
        )
        for field_name in required:
            if field_name not in reply:
                raise BusinessError(
                    "E_REPLY",
                    f"reply missing field: {field_name}",
                    position=self.position,
                    field=field_name,
                    shard_id=shard_id,
                )
        if reply["node_id"] != self.node.id or reply["shard_id"] != shard_id:
            raise BusinessError("E_REPLY", "compute reply identity mismatch", position=self.position, field="shard_id", shard_id=shard_id)
        if reply["lease_id"] != lease_id or reply["lease_version"] != lease_version:
            raise BusinessError("E_REPLY", "compute reply lease identity mismatch", position=self.position, field="lease_id", shard_id=shard_id)
        if reply["config_version"] != config_version:
            raise BusinessError("E_REPLY", "compute reply config version mismatch", position=self.position, field="config_version", shard_id=shard_id)
        if not is_int(reply["elapsed_monotonic_ms"]) or reply["elapsed_monotonic_ms"] < 0:
            raise BusinessError("E_REPLY", "invalid elapsed_monotonic_ms", position=self.position, field="elapsed_monotonic_ms", shard_id=shard_id)
        summary = reply["summary"]
        if not isinstance(summary, dict):
            raise BusinessError("E_REPLY", "summary must be an object", position=self.position, field="summary", shard_id=shard_id)
        for field_name in ("groups", "sample_count", "digest"):
            if field_name not in summary:
                raise BusinessError("E_REPLY", f"summary missing field: {field_name}", position=self.position, field=field_name, shard_id=shard_id)
        if not isinstance(summary["groups"], list) or not is_int(summary["sample_count"]):
            raise BusinessError("E_REPLY", "invalid summary structure", position=self.position, field="summary", shard_id=shard_id)
        if not isinstance(summary["digest"], str) or len(summary["digest"]) != 64:
            raise BusinessError("E_REPLY", "invalid summary digest", position=self.position, field="digest", shard_id=shard_id)
        return reply


class NodeProcess:
    def __init__(
        self,
        node: NodeSpec,
        default_ttl_ms: int,
        *,
        handshake_timeout_s: float = 5.0,
        command_override: list[str] | None = None,
    ):
        self.node = node
        self.default_ttl_ms = default_ttl_ms
        self.handshake_timeout_s = handshake_timeout_s
        self.command_override = command_override
        self.process: subprocess.Popen[bytes] | None = None
        self.client = NodeClient(node, timeout_s=min(3.0, handshake_timeout_s))

    def start(self) -> None:
        preflight = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                preflight.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            elif hasattr(socket, "SO_REUSEADDR"):
                preflight.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            preflight.settimeout(0.5)
            preflight.bind(("127.0.0.1", self.node.port))
        except OSError as exc:
            raise BusinessError(
                "E_PORT",
                f"loopback port is already occupied: {exc}",
                position=f"node:{self.node.id}@127.0.0.1:{self.node.port}",
                field="port",
                detail={"port": self.node.port},
            ) from exc
        finally:
            preflight.close()
        cmd = list(self.command_override) if self.command_override is not None else [
            sys.executable,
            "-m",
            "quota_ledger.node_server",
            "--node-id",
            self.node.id,
            "--port",
            str(self.node.port),
            "--seed",
            str(self.node.seed),
            "--concurrency",
            str(self.node.concurrency),
            "--lease-limit",
            str(self.node.lease_limit),
            "--capacity-version",
            str(self.node.capacity_version),
            "--default-ttl-ms",
            str(self.default_ttl_ms),
            "--parent-pid",
            str(os.getpid()),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        kwargs: dict[str, Any] = {
            "cwd": str(PROJECT_ROOT),
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(cmd, **kwargs)
        deadline = time.monotonic() + self.handshake_timeout_s
        last_error: BusinessError | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stderr = b""
                try:
                    _, stderr = self.process.communicate(timeout=0.5)
                except Exception:
                    pass
                message = stderr.decode("utf-8", errors="replace").strip() or "node process exited before handshake"
                raise BusinessError(
                    "E_PORT",
                    f"node startup failed: {message}",
                    position=f"node:{self.node.id}@127.0.0.1:{self.node.port}",
                    field="port",
                    detail={"exit_code": self.process.returncode, "port": self.node.port},
                )
            try:
                self.client.ping()
                return
            except BusinessError as exc:
                if exc.code == "E_TIMEOUT":
                    last_error = exc
                    time.sleep(0.03)
                    continue
                if exc.code == "E_PORT":
                    last_error = exc
                    time.sleep(0.03)
                    continue
                raise
        self.stop()
        if last_error is not None:
            raise BusinessError(
                "E_TIMEOUT",
                f"node handshake timed out after {self.handshake_timeout_s:.3f}s",
                position=f"node:{self.node.id}@127.0.0.1:{self.node.port}",
                field="handshake_timeout_s",
                detail={"last_error": last_error.code},
            )
        raise BusinessError(
            "E_TIMEOUT",
            f"node handshake timed out after {self.handshake_timeout_s:.3f}s",
            position=f"node:{self.node.id}@127.0.0.1:{self.node.port}",
            field="handshake_timeout_s",
        )

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                NodeClient(self.node, timeout_s=0.3)._request({"op": "shutdown"})
            except Exception:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        try:
            process.communicate(timeout=0.2)
        except Exception:
            pass
        self.process = None


class NodePool:
    def __init__(self, nodes: dict[str, NodeSpec], default_ttl_ms: int, *, handshake_timeout_s: float = 5.0):
        self.nodes = nodes
        self.default_ttl_ms = default_ttl_ms
        self.handshake_timeout_s = handshake_timeout_s
        self.processes: dict[str, NodeProcess] = {}
        self.clients: dict[str, NodeClient] = {}

    def start_all(self) -> None:
        try:
            for node_id in sorted(self.nodes):
                process = NodeProcess(
                    self.nodes[node_id],
                    self.default_ttl_ms,
                    handshake_timeout_s=self.handshake_timeout_s,
                )
                process.start()
                self.processes[node_id] = process
                self.clients[node_id] = process.client
        except Exception:
            self.stop_all()
            raise

    def stop_all(self) -> None:
        for node_id in sorted(self.processes, reverse=True):
            try:
                self.processes[node_id].stop()
            except Exception:
                pass
        self.processes.clear()
        self.clients.clear()

    def snapshot(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for node_id in sorted(self.clients):
            try:
                rows.append(self.clients[node_id].snapshot())
            except BusinessError as exc:
                rows.append({"node_id": node_id, "error": exc.to_dict()})
        return {"nodes": rows}
