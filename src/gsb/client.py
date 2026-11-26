"""Parent-side node subprocess lifecycle and TCP client."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable, Sequence

from .errors import GSBTimeoutError, PortError, ProtocolError
from .model import NodeSpec, ShardSpec
from .protocol import JsonLineChannel, PROTOCOL_VERSION, require_fields
from .samples import final_aggregate


class NodeClient:
    def __init__(
        self,
        node: NodeSpec,
        *,
        read_timeout_s: float,
        handshake_timeout_s: float | None = None,
    ) -> None:
        self.node = node
        self.read_timeout_s = read_timeout_s
        self.handshake_timeout_s = read_timeout_s if handshake_timeout_s is None else handshake_timeout_s
        self.channel: JsonLineChannel | None = None

    def _connect(self) -> None:
        try:
            sock = socket.create_connection(
                (self.node.host, self.node.port), timeout=self.handshake_timeout_s
            )
        except ConnectionRefusedError as exc:
            raise PortError(
                "node is not accepting connections on its declared port",
                details={"host": self.node.host, "port": self.node.port},
            ) from exc
        except socket.timeout as exc:
            raise GSBTimeoutError(
                "TCP connection timed out",
                details={"stage": "connect", "host": self.node.host, "port": self.node.port},
            ) from exc
        except OSError as exc:
            raise PortError(
                "cannot connect to node loopback port",
                details={"host": self.node.host, "port": self.node.port, "error": str(exc)},
            ) from exc
        self.channel = JsonLineChannel(sock)

    def handshake(self) -> dict[str, Any]:
        if self.channel is None:
            self._connect()
        assert self.channel is not None
        request_id = f"hello-{uuid.uuid4().hex}"
        self.channel.send(
            {
                "op": "hello",
                "protocol_version": PROTOCOL_VERSION,
                "node_id": self.node.id,
                "request_id": request_id,
            }
        )
        response = self.channel.receive(self.handshake_timeout_s, stage="handshake")
        _raise_remote_error(response, stage="handshake")
        require_fields(
            response,
            ("node_id", "protocol_version", "fixed_scale", "request_id"),
            stage="handshake",
        )
        if response["node_id"] != self.node.id:
            raise ProtocolError(
                "handshake returned the wrong node id",
                details={"expected": self.node.id, "actual": response["node_id"]},
            )
        if response["protocol_version"] != PROTOCOL_VERSION:
            raise ProtocolError(
                "protocol version mismatch",
                details={"expected": PROTOCOL_VERSION, "actual": response["protocol_version"]},
            )
        if response["fixed_scale"] != self.node.fixed_scale:
            raise ProtocolError(
                "node fixed scale differs from its declaration",
                details={"expected": self.node.fixed_scale, "actual": response["fixed_scale"]},
            )
        return response

    def aggregate_batch(
        self,
        *,
        shard: ShardSpec,
        capacity_version: str,
        batch_index: int,
        batch_count: int,
        samples: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.channel is None:
            self._connect()
        assert self.channel is not None
        request_id = f"{shard.id}-{batch_index}-{uuid.uuid4().hex}"
        self.channel.send(
            {
                "op": "aggregate",
                "request_id": request_id,
                "node_id": self.node.id,
                "shard_id": shard.id,
                "capacity_version": capacity_version,
                "fixed_scale": self.node.fixed_scale,
                "batch_index": batch_index,
                "batch_count": batch_count,
                "samples": list(samples),
                "group_count": shard.group_count,
            }
        )
        response = self.channel.receive(self.read_timeout_s, stage="aggregate_reply")
        _raise_remote_error(response, stage="aggregate_reply")
        require_fields(
            response,
            (
                "node_id",
                "shard_id",
                "capacity_version",
                "request_id",
                "batch_index",
                "participating_rows",
                "groups",
                "elapsed_ms",
                "aggregate_digest",
            ),
            stage="aggregate_reply",
        )
        if response["node_id"] != self.node.id:
            raise ProtocolError("reply node_id mismatch", details={"actual": response["node_id"]})
        if response["shard_id"] != shard.id:
            raise ProtocolError("reply shard_id mismatch", details={"actual": response["shard_id"]})
        if response["capacity_version"] != capacity_version:
            raise ProtocolError(
                "reply capacity_version mismatch",
                details={"actual": response["capacity_version"]},
            )
        if response["request_id"] != request_id:
            raise ProtocolError("reply request_id mismatch", details={"actual": response["request_id"]})
        if response["batch_index"] != batch_index:
            raise ProtocolError("reply batch_index mismatch", details={"actual": response["batch_index"]})
        if not isinstance(response["elapsed_ms"], int) or response["elapsed_ms"] < 0:
            raise ProtocolError("reply elapsed_ms must be a non-negative integer")
        if not isinstance(response["groups"], list):
            raise ProtocolError("reply groups must be an array")
        if not isinstance(response["aggregate_digest"], str) or not response["aggregate_digest"]:
            raise ProtocolError("reply aggregate_digest must be a non-empty string")
        return response

    def close(self) -> None:
        if self.channel is not None:
            self.channel.close()
            self.channel = None


def _raise_remote_error(response: dict[str, Any], *, stage: str) -> None:
    if response.get("ok") is not False:
        return
    code = response.get("code", "E_PROTO")
    message = response.get("message", "remote protocol error")
    details = dict(response.get("details") or {})
    details["stage"] = stage
    if code == "E_TIMEOUT":
        raise GSBTimeoutError(message, details=details)
    raise ProtocolError(message, details={**details, "remote_code": code})


def execute_shard(
    node: NodeSpec,
    shard: ShardSpec,
    samples: Sequence[dict[str, Any]],
    capacity_version: str,
    *,
    read_timeout_s: float,
    handshake_timeout_s: float | None = None,
    batch_size: int | None = None,
    before_batch: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute all batches over one real TCP session and return a final aggregate."""

    client = NodeClient(
        node,
        read_timeout_s=read_timeout_s,
        handshake_timeout_s=handshake_timeout_s,
    )
    started_ns = time.monotonic_ns()
    try:
        if before_batch is not None:
            before_batch()
        client.handshake()
        stage_aggregates: list[dict[str, Any]] = []
        node_elapsed_ms = 0
        effective_batch_size = shard.batch_size if batch_size is None else batch_size
        if effective_batch_size < 1:
            raise ValueError("batch_size must be positive")
        batches = [
            samples[offset : offset + effective_batch_size]
            for offset in range(0, len(samples), effective_batch_size)
        ]
        for batch_index, batch in enumerate(batches):
            if before_batch is not None:
                before_batch()
            response = client.aggregate_batch(
                shard=shard,
                capacity_version=capacity_version,
                batch_index=batch_index,
                batch_count=len(batches),
                samples=batch,
            )
            node_elapsed_ms += response["elapsed_ms"]
            stage_aggregates.append(
                {
                    "groups": response["groups"],
                    "participating_rows": response["participating_rows"],
                }
            )
        final = final_aggregate(node, shard, capacity_version, samples, stage_aggregates)
        final["node_elapsed_ms"] = node_elapsed_ms
        final["network_elapsed_ms"] = max(0, (time.monotonic_ns() - started_ns) // 1_000_000)
        final["batch_count"] = len(batches)
        return final
    finally:
        client.close()


class NodeProcess:
    def __init__(self, node: NodeSpec, *, startup_timeout_s: float) -> None:
        self.node = node
        self.startup_timeout_s = startup_timeout_s
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("node process already started")
        env = dict(os.environ)
        source_root = str(Path(__file__).resolve().parents[1])
        old_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = source_root if not old_pythonpath else source_root + os.pathsep + old_pythonpath
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gsb.node_worker",
                "--node-id",
                self.node.id,
                "--host",
                self.node.host,
                "--port",
                str(self.node.port),
                "--fixed-scale",
                str(self.node.fixed_scale),
                "--idle-timeout",
                str(max(30.0, self.startup_timeout_s)),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.process = process
        lines: queue.Queue[str | None] = queue.Queue()

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line.rstrip("\n"))
            lines.put(None)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        deadline = time.monotonic() + self.startup_timeout_s
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GSBTimeoutError(
                        "node startup handshake timed out",
                        details={"stage": "startup", "node_id": self.node.id},
                    )
                try:
                    line = lines.get(timeout=remaining)
                except queue.Empty as exc:
                    raise GSBTimeoutError(
                        "node startup handshake timed out",
                        details={"stage": "startup", "node_id": self.node.id},
                    ) from exc
                if line is None:
                    stderr = ""
                    if process.stderr is not None:
                        try:
                            stderr = process.stderr.read()
                        except OSError:
                            stderr = ""
                    raise ProtocolError(
                        "node process exited before becoming ready",
                        details={"returncode": process.poll(), "stderr": stderr[-2000:]},
                    )
                if line.startswith("GSB_ERROR "):
                    try:
                        payload = json.loads(line[len("GSB_ERROR ") :])
                    except json.JSONDecodeError as exc:
                        raise ProtocolError("invalid node startup error line") from exc
                    code = payload.get("code", "E_PROTO")
                    if code == "E_PORT":
                        raise PortError(
                            payload.get("message", "port bind failed"),
                            details=payload.get("details", {}),
                        )
                    raise ProtocolError(
                        payload.get("message", "node startup failed"),
                        details={**payload.get("details", {}), "remote_code": code},
                    )
                if line.startswith("GSB_READY "):
                    break
            client = NodeClient(self.node, read_timeout_s=self.startup_timeout_s)
            try:
                client.handshake()
            finally:
                client.close()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        self.process = None

    def __enter__(self) -> "NodeProcess":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
