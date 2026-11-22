from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .errors import AppError
from .models import ClusterSpec, Spec


@dataclass
class NodeHandle:
    cluster: ClusterSpec
    revision_version: int
    process: subprocess.Popen[str]


def _recv_json_line(sock: socket.socket) -> dict[str, Any]:
    chunks = bytearray()
    while True:
        piece = sock.recv(4096)
        if not piece:
            raise ConnectionError("connection closed before JSON line")
        chunks.extend(piece)
        newline = chunks.find(b"\n")
        if newline >= 0:
            return json.loads(chunks[:newline].decode("utf-8"))


def _send_json_line(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(canonical_json(payload).encode("utf-8") + b"\n")


class NodeManager:
    """Own one real Python subprocess per targeted cluster."""

    def __init__(self, spec: Spec, handshake_timeout_ms: int = 3000, dispatch_timeout_ms: int = 8000) -> None:
        self.spec = spec
        self.handshake_timeout_ms = handshake_timeout_ms
        self.dispatch_timeout_ms = dispatch_timeout_ms
        self.handles: dict[str, NodeHandle] = {}
        self._project_root = Path(__file__).resolve().parents[1]

    def __enter__(self) -> "NodeManager":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def _preflight_port(self, cluster: ClusterSpec) -> None:
        active_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        active_probe.settimeout(0.15)
        try:
            if active_probe.connect_ex((cluster.host, cluster.port)) == 0:
                raise AppError(
                    "E_PORT",
                    f"集群 {cluster.id} 的端口 {cluster.port} 已被占用",
                    "port",
                    f"clusters[{cluster.id}].port",
                    {"cluster_id": cluster.id, "port": cluster.port, "reason": "active listener detected"},
                )
        finally:
            active_probe.close()
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((cluster.host, cluster.port))
        except OSError as exc:
            raise AppError(
                "E_PORT",
                f"集群 {cluster.id} 的端口 {cluster.port} 已被占用",
                "port",
                f"clusters[{cluster.id}].port",
                {"cluster_id": cluster.id, "port": cluster.port, "reason": str(exc)},
            ) from None
        finally:
            probe.close()

    def start(self, cluster_ids: set[str] | None = None) -> None:
        clusters = self.spec.cluster_map()
        selected = sorted(cluster_ids if cluster_ids is not None else set(clusters))
        current = self.spec.current_revisions()
        for cluster_id in selected:
            if cluster_id in self.handles:
                continue
            cluster = clusters[cluster_id]
            revision = current[cluster_id]
            self._preflight_port(cluster)
            env = os.environ.copy()
            existing_path = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                str(self._project_root) if not existing_path else str(self._project_root) + os.pathsep + existing_path
            )
            command = [
                sys.executable,
                "-m",
                "configdist.node",
                "--cluster-id",
                cluster.id,
                "--host",
                cluster.host,
                "--port",
                str(cluster.port),
                "--revision-version",
                str(revision.version),
                "--config-json",
                canonical_json(revision.config),
                "--simulation-seed",
                str(self.spec.simulation_seed),
            ]
            process = subprocess.Popen(
                command,
                cwd=str(self._project_root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            handle = NodeHandle(cluster=cluster, revision_version=revision.version, process=process)
            try:
                self._wait_for_handshake(handle)
            except AppError:
                self._terminate(handle)
                raise
            self.handles[cluster_id] = handle

    def _wait_for_handshake(self, handle: NodeHandle) -> None:
        deadline = time.perf_counter() + self.handshake_timeout_ms / 1000.0
        cluster = handle.cluster
        last_error = "connection not established"
        while time.perf_counter() < deadline:
            if handle.process.poll() is not None:
                stderr = handle.process.stderr.read() if handle.process.stderr else ""
                if "E_PORT" in stderr:
                    raise AppError(
                        "E_PORT",
                        f"集群 {cluster.id} 的节点无法绑定端口 {cluster.port}",
                        "port",
                        f"clusters[{cluster.id}].port",
                        {"cluster_id": cluster.id, "port": cluster.port, "node_stderr": stderr.strip()},
                    )
                raise AppError(
                    "E_HANDSHAKE",
                    f"集群 {cluster.id} 的节点在握手前退出",
                    "handshake",
                    f"clusters[{cluster.id}].handshake",
                    {"cluster_id": cluster.id, "exit_code": handle.process.returncode, "node_stderr": stderr.strip()},
                )
            try:
                with socket.create_connection(
                    (cluster.host, cluster.port), timeout=min(0.25, max(0.02, deadline - time.perf_counter()))
                ) as sock:
                    sock.settimeout(max(0.05, deadline - time.perf_counter()))
                    _send_json_line(
                        sock,
                        {"op": "handshake", "cluster_id": cluster.id, "revision_version": handle.revision_version},
                    )
                    response = _recv_json_line(sock)
                    if response.get("ok") and response.get("cluster_id") == cluster.id:
                        return
                    last_error = str(response)
            except (OSError, ConnectionError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                time.sleep(0.02)
        raise AppError(
            "E_HANDSHAKE",
            f"集群 {cluster.id} 的节点握手超时",
            "handshake",
            f"clusters[{cluster.id}].handshake",
            {
                "cluster_id": cluster.id,
                "port": cluster.port,
                "timeout_ms": self.handshake_timeout_ms,
                "last_error": last_error,
            },
        )

    def dispatch_shard(self, shard: dict[str, Any]) -> dict[str, Any]:
        handle = self.handles.get(shard["cluster_id"])
        if handle is None:
            raise AppError("E_HANDSHAKE", "目标集群节点未启动", "cluster_id", str(shard["cluster_id"]))
        cluster = handle.cluster
        request = {
            "op": "dispatch",
            "cluster_id": shard["cluster_id"],
            "task_id": shard["task_id"],
            "shard_id": shard["shard_id"],
            "config_keys": shard["config_keys"],
            "resource_units": shard["resource_units"],
            "revision_version": shard["revision_version"],
        }
        started = time.perf_counter_ns()
        try:
            with socket.create_connection((cluster.host, cluster.port), timeout=self.dispatch_timeout_ms / 1000.0) as sock:
                sock.settimeout(self.dispatch_timeout_ms / 1000.0)
                _send_json_line(
                    sock,
                    {"op": "handshake", "cluster_id": cluster.id, "revision_version": handle.revision_version},
                )
                handshake = _recv_json_line(sock)
                if not handshake.get("ok"):
                    raise AppError("E_HANDSHAKE", "节点分片连接握手失败", "handshake", shard["shard_id"], handshake)
                _send_json_line(sock, request)
                response = _recv_json_line(sock)
        except AppError:
            raise
        except (OSError, ConnectionError, ValueError, json.JSONDecodeError) as exc:
            elapsed_ms = int((time.perf_counter_ns() - started) // 1_000_000)
            raise AppError(
                "E_HANDSHAKE",
                "分片下发握手或回传失败",
                "handshake",
                shard["shard_id"],
                {"cluster_id": shard["cluster_id"], "shard_id": shard["shard_id"], "elapsed_ms": elapsed_ms, "reason": str(exc)},
            ) from None
        if not response.get("ok"):
            raise AppError(
                str(response.get("code", "E_PROTOCOL")),
                str(response.get("message", "node rejected shard")),
                "dispatch",
                shard["shard_id"],
                response,
            )
        response["pid"] = handshake.get("pid")
        return response

    def _terminate(self, handle: NodeHandle) -> None:
        process = handle.process
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=0.8)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=0.8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.8)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def stop(self) -> None:
        for handle in list(self.handles.values()):
            cluster = handle.cluster
            if handle.process.poll() is None:
                try:
                    with socket.create_connection((cluster.host, cluster.port), timeout=0.15) as sock:
                        sock.settimeout(0.15)
                        _send_json_line(sock, {"op": "stop"})
                        _recv_json_line(sock)
                except (OSError, ConnectionError, ValueError, json.JSONDecodeError):
                    pass
            self._terminate(handle)
        self.handles.clear()
