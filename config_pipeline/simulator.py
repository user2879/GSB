"""Parent-side orchestration and JSONL client for cluster simulators."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import E_HANDSHAKE, E_PORT, E_REPLY, PipelineError
from .manifest import Cluster
from .planner import ShardPlan
from .util import canonical_json, digest_json, monotonic_ms


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_line(connection: socket.socket, limit: int = 2 * 1024 * 1024) -> bytes:
    chunks = bytearray()
    while len(chunks) <= limit:
        block = connection.recv(min(65536, limit + 1 - len(chunks)))
        if not block:
            break
        chunks.extend(block)
        newline = chunks.find(b"\n")
        if newline >= 0:
            return bytes(chunks[: newline + 1])
    if len(chunks) > limit:
        raise ValueError("response line too large")
    return bytes(chunks)


def _request(
    host: str,
    port: int,
    request: dict[str, Any],
    *,
    timeout: float,
) -> tuple[dict[str, Any], int]:
    started = monotonic_ms()
    connection = socket.create_connection((host, port), timeout=timeout)
    try:
        connection.settimeout(timeout)
        connection.sendall(canonical_json(request).encode("utf-8") + b"\n")
        raw = _read_line(connection)
    finally:
        connection.close()
    if not raw:
        raise ConnectionError("simulator closed the connection without a reply")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON reply: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("reply must be a JSON object")
    return decoded, monotonic_ms() - started


def _validate_handshake(
    response: dict[str, Any], cluster: Cluster
) -> None:
    required: dict[str, type | tuple[type, ...]] = {
        "ok": bool,
        "service": str,
        "protocol": int,
        "cluster_id": str,
        "pid": int,
        "port": int,
    }
    for field, expected_type in required.items():
        if field not in response:
            raise PipelineError(
                E_REPLY,
                "handshake reply is missing a required field",
                field=f"handshake.{field}",
                details={"cluster_id": cluster.id},
            )
        value = response[field]
        if isinstance(expected_type, tuple):
            valid = isinstance(value, expected_type)
        else:
            valid = isinstance(value, expected_type) and not (
                expected_type is int and isinstance(value, bool)
            )
        if not valid:
            raise PipelineError(
                E_REPLY,
                "handshake reply field has an invalid type",
                field=f"handshake.{field}",
                details={"cluster_id": cluster.id, "actual_type": type(value).__name__},
            )
    if response["ok"] is not True:
        raise PipelineError(
            E_REPLY,
            "simulator rejected the handshake",
            field="handshake.ok",
            details={"cluster_id": cluster.id, "reply": response},
        )
    if response["service"] != "deterministic-cluster-sim" or response["protocol"] != 1:
        raise PipelineError(
            E_HANDSHAKE,
            "simulator protocol does not match the expected local protocol",
            field="handshake.protocol",
            details={"cluster_id": cluster.id, "reply": response},
        )
    if response["cluster_id"] != cluster.id or response["port"] != cluster.port:
        raise PipelineError(
            E_HANDSHAKE,
            "handshake identified a different cluster endpoint",
            field="handshake.cluster_id",
            details={
                "expected_cluster_id": cluster.id,
                "actual_cluster_id": response["cluster_id"],
                "expected_port": cluster.port,
                "actual_port": response["port"],
            },
        )


class SimulatorProcess:
    def __init__(
        self,
        cluster: Cluster,
        *,
        handshake_timeout: float = 5.0,
        python_executable: str | None = None,
    ) -> None:
        self.cluster = cluster
        self.handshake_timeout = handshake_timeout
        self.python_executable = python_executable or sys.executable
        self.process: subprocess.Popen[bytes] | None = None
        self.hello_reply: dict[str, Any] | None = None

    def _preflight_port(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Reuse only released TIME_WAIT sockets.  On Windows the exclusive
            # option is used instead so an active listener cannot be shared.
            if os.name == "nt":
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((self.cluster.host, self.cluster.port))
        except OSError as exc:
            raise PipelineError(
                E_PORT,
                "declared loopback port is already occupied",
                field="port",
                details={
                    "cluster_id": self.cluster.id,
                    "port": self.cluster.port,
                    "reason": str(exc),
                },
            ) from exc
        finally:
            probe.close()

    def start(self) -> dict[str, Any]:
        self._preflight_port()
        env = os.environ.copy()
        existing_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(PROJECT_ROOT)
            if not existing_path
            else str(PROJECT_ROOT) + os.pathsep + existing_path
        )
        command = [
            self.python_executable,
            "-m",
            "config_pipeline.mock_service",
            "--cluster-id",
            self.cluster.id,
            "--host",
            self.cluster.host,
            "--port",
            str(self.cluster.port),
            "--parent-pid",
            str(os.getpid()),
        ]
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise PipelineError(
                E_HANDSHAKE,
                "failed to spawn the simulator subprocess",
                field="handshake",
                details={"cluster_id": self.cluster.id, "reason": str(exc)},
            ) from exc

        deadline = time.monotonic() + self.handshake_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stderr = self._read_stderr()
                code = E_PORT if E_PORT in stderr or "Address already in use" in stderr else E_HANDSHAKE
                raise PipelineError(
                    code,
                    "simulator exited before completing the socket handshake",
                    field="handshake" if code == E_HANDSHAKE else "port",
                    details={
                        "cluster_id": self.cluster.id,
                        "port": self.cluster.port,
                        "exit_code": self.process.returncode,
                        "stderr": stderr[-2000:],
                    },
                )
            remaining = max(0.05, min(0.5, deadline - time.monotonic()))
            try:
                reply, _ = _request(
                    self.cluster.host,
                    self.cluster.port,
                    {"type": "hello", "protocol": 1},
                    timeout=remaining,
                )
                _validate_handshake(reply, self.cluster)
                self.hello_reply = reply
                return reply
            except (ConnectionRefusedError, OSError, TimeoutError, ValueError) as exc:
                last_error = exc
                time.sleep(0.025)
            except PipelineError:
                self.stop()
                raise

        self.stop()
        raise PipelineError(
            E_HANDSHAKE,
            "simulator did not answer the handshake before the real socket timeout",
            field="handshake",
            details={
                "cluster_id": self.cluster.id,
                "port": self.cluster.port,
                "timeout_seconds": self.handshake_timeout,
                "last_error": str(last_error) if last_error else None,
            },
        )

    def _read_stderr(self) -> str:
        if self.process is None or self.process.stderr is None:
            return ""
        try:
            data = self.process.stderr.read()
            return data.decode("utf-8", errors="replace")
        except (OSError, ValueError):
            return ""

    def dispatch(self, shard: ShardPlan, *, timeout: float = 30.0) -> dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            raise PipelineError(
                E_REPLY,
                "simulator subprocess is not running",
                field="reply",
                shard_id=shard.stable_id,
                details={"cluster_id": self.cluster.id},
            )
        try:
            reply, transport_ms = _request(
                self.cluster.host,
                self.cluster.port,
                shard.dispatch_request,
                timeout=timeout,
            )
        except (OSError, ValueError, ConnectionError) as exc:
            raise PipelineError(
                E_REPLY,
                "failed to obtain a JSONL dispatch reply",
                field="reply",
                shard_id=shard.stable_id,
                details={"cluster_id": self.cluster.id, "reason": str(exc)},
            ) from exc
        self.validate_dispatch_reply(shard, reply)
        reply["_transport_ms"] = int(transport_ms)
        return reply

    @staticmethod
    def validate_dispatch_reply(
        shard: ShardPlan, reply: dict[str, Any]
    ) -> None:
        required: dict[str, type | tuple[type, ...]] = {
            "ok": bool,
            "cluster_id": str,
            "shard_id": str,
            "revision_version": int,
            "aggregate_summary": dict,
            "aggregate_digest": str,
            "elapsed_ms": (int, float),
        }
        for field, expected_type in required.items():
            if field not in reply:
                raise PipelineError(
                    E_REPLY,
                    "dispatch reply is missing a required field",
                    field=f"reply.{field}",
                    shard_id=shard.stable_id,
                )
            value = reply[field]
            if isinstance(expected_type, tuple):
                valid = isinstance(value, expected_type) and not isinstance(value, bool)
            else:
                valid = isinstance(value, expected_type) and not (
                    expected_type is int and isinstance(value, bool)
                )
            if not valid:
                raise PipelineError(
                    E_REPLY,
                    "dispatch reply field has an invalid type",
                    field=f"reply.{field}",
                    shard_id=shard.stable_id,
                    details={"actual_type": type(value).__name__},
                )
        if reply["ok"] is not True:
            raise PipelineError(
                E_REPLY,
                "simulator rejected the dispatch request",
                field="reply.ok",
                shard_id=shard.stable_id,
                details={"reply": reply},
            )
        if reply["cluster_id"] != shard.cluster_id:
            raise PipelineError(
                E_REPLY,
                "reply cluster does not match the planned shard",
                field="reply.cluster_id",
                shard_id=shard.stable_id,
                details={"expected": shard.cluster_id, "actual": reply["cluster_id"]},
            )
        if reply["shard_id"] != shard.stable_id:
            raise PipelineError(
                E_REPLY,
                "reply shard id does not match the planned shard",
                field="reply.shard_id",
                shard_id=shard.stable_id,
                details={"expected": shard.stable_id, "actual": reply["shard_id"]},
            )
        if reply["revision_version"] != shard.revision_version:
            raise PipelineError(
                E_REPLY,
                "reply revision does not match the planned shard",
                field="reply.revision_version",
                shard_id=shard.stable_id,
                details={
                    "expected": shard.revision_version,
                    "actual": reply["revision_version"],
                },
            )
        actual_digest = digest_json(reply["aggregate_summary"])
        if actual_digest != reply["aggregate_digest"]:
            raise PipelineError(
                E_REPLY,
                "aggregate summary digest does not match the returned summary",
                field="reply.aggregate_digest",
                shard_id=shard.stable_id,
                details={"computed": actual_digest, "reported": reply["aggregate_digest"]},
            )

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                _request(
                    self.cluster.host,
                    self.cluster.port,
                    {"type": "shutdown"},
                    timeout=0.5,
                )
            except Exception:
                pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        if process.stderr is not None:
            try:
                process.stderr.close()
            except OSError:
                pass

    def __enter__(self) -> "SimulatorProcess":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


class SimulatorPool:
    def __init__(
        self,
        clusters: list[Cluster] | tuple[Cluster, ...],
        *,
        handshake_timeout: float = 5.0,
    ) -> None:
        self.clusters = {cluster.id: cluster for cluster in clusters}
        self.handshake_timeout = handshake_timeout
        self.processes: dict[str, SimulatorProcess] = {}

    def start(self) -> dict[str, dict[str, Any]]:
        started: dict[str, dict[str, Any]] = {}
        try:
            for cluster in sorted(self.clusters.values(), key=lambda item: item.id):
                process = SimulatorProcess(
                    cluster, handshake_timeout=self.handshake_timeout
                )
                process.start()
                self.processes[cluster.id] = process
                started[cluster.id] = dict(process.hello_reply or {})
        except Exception:
            self.stop()
            raise
        return started

    def dispatch(self, shard: ShardPlan, *, timeout: float = 30.0) -> dict[str, Any]:
        try:
            process = self.processes[shard.cluster_id]
        except KeyError as exc:
            raise PipelineError(
                E_REPLY,
                "no simulator is running for the shard cluster",
                field="cluster_id",
                shard_id=shard.stable_id,
                details={"cluster_id": shard.cluster_id},
            ) from exc
        return process.dispatch(shard, timeout=timeout)

    def stop(self) -> None:
        for cluster_id in sorted(self.processes):
            self.processes[cluster_id].stop()
        self.processes.clear()

    def __enter__(self) -> "SimulatorPool":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
