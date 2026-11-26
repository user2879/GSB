from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .errors import PortError, ReplyError, TimeoutError
from .model import PLUGIN_NAMES, Spec
from .util import canonical_json, monotonic_ms


INT64_MAX = int(np.iinfo(np.int64).max)
REPLY_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "pod_id",
        "node_name",
        "parameter_version",
        "scale_exponent",
        "scale_factor",
        "plugin_scores",
        "plugin_weights_myriad",
        "plugin_contributions",
        "total_fixed",
        "total_score",
        "error_bound",
        "elapsed_ms",
    }
)


def compute_score_reply(request: Mapping[str, Any]) -> dict[str, Any]:
    """Compute a deterministic fixed-point reply using NumPy integer arrays."""

    started = monotonic_ms()
    pod_id = str(request["pod_id"])
    node_name = str(request["node_name"])
    parameter_version = int(request["parameter_version"])
    scale_exponent = int(request["scale_exponent"])
    if scale_exponent < 0 or scale_exponent > 3:
        raise ValueError("scale_exponent must be in [0, 3]")
    scale_factor = 10**scale_exponent
    weights_raw = request["weights_myriad"]
    if not isinstance(weights_raw, dict) or set(weights_raw) != set(PLUGIN_NAMES):
        raise ValueError("weights_myriad must contain exactly four plugins")
    weights = {name: int(weights_raw[name]) for name in PLUGIN_NAMES}
    if sum(weights.values()) != 10000 or any(
        value < 0 or value > 10000 for value in weights.values()
    ):
        raise ValueError("weights_myriad must be non-negative and sum to 10000")
    features = request["features"]
    if not isinstance(features, dict):
        raise ValueError("features must be an object")

    has_taints = bool(features["has_taints"])
    untolerated_prefer = max(0, int(features["untolerated_prefer_taints"]))
    has_node_affinity = bool(features["has_node_affinity"])
    node_soft_match = max(0, int(features["node_soft_match_weight"]))
    has_anti_affinity = bool(features["has_pod_anti_affinity"])
    anti_conflicts = max(0, int(features["anti_soft_conflicts"]))
    remaining_cpu = max(0, int(features["remaining_cpu"]))
    remaining_memory = max(0, int(features["remaining_memory"]))
    alloc_cpu = int(features["allocatable_cpu"])
    alloc_memory = int(features["allocatable_memory"])
    if alloc_cpu <= 0 or alloc_memory <= 0:
        raise ValueError("allocatable resources must be positive")

    raw_scores = {
        "taint_toleration": (
            int(np.clip(100 - 25 * untolerated_prefer, 0, 100))
            if has_taints
            else 0
        ),
        "node_affinity": (
            int(np.clip(50 + 10 * node_soft_match, 0, 100))
            if has_node_affinity
            else 0
        ),
        "pod_anti_affinity": (
            int(np.clip(100 - 20 * anti_conflicts, 0, 100))
            if has_anti_affinity
            else 0
        ),
        "resource_fit": int(
            np.clip(
                min(
                    (remaining_cpu * 100) // alloc_cpu,
                    (remaining_memory * 100) // alloc_memory,
                ),
                0,
                100,
            )
        ),
    }

    score_array = np.asarray(
        [raw_scores[name] for name in PLUGIN_NAMES], dtype=np.int64
    )
    weight_array = np.asarray(
        [weights[name] for name in PLUGIN_NAMES], dtype=np.int64
    )
    numerator = score_array * weight_array * np.int64(scale_factor)
    numerator = np.minimum(numerator, np.int64(INT64_MAX))
    contribution_array = numerator // np.int64(10000)
    total_fixed = int(np.minimum(contribution_array.sum(dtype=np.int64), np.int64(INT64_MAX)))
    total_score = total_fixed // scale_factor
    error_bound = (len(PLUGIN_NAMES) + scale_factor - 1) // scale_factor
    elapsed_ms = max(0, monotonic_ms() - started)
    return {
        "schema": "score-reply-v1",
        "pod_id": pod_id,
        "node_name": node_name,
        "parameter_version": parameter_version,
        "scale_exponent": scale_exponent,
        "scale_factor": scale_factor,
        "plugin_scores": raw_scores,
        "plugin_weights_myriad": weights,
        "plugin_contributions": {
            name: int(contribution_array[index])
            for index, name in enumerate(PLUGIN_NAMES)
        },
        "total_fixed": total_fixed,
        "total_score": total_score,
        "error_bound": error_bound,
        "elapsed_ms": elapsed_ms,
    }


def _validate_reply(
    reply: Any,
    *,
    pod_id: str,
    node_name: str,
    parameter_version: int,
    scale_exponent: int,
    scale_factor: int,
) -> dict[str, Any]:
    if not isinstance(reply, dict):
        raise ReplyError("score reply must be a JSON object", pod_id=pod_id, node_name=node_name)
    missing = sorted(REPLY_REQUIRED_FIELDS - set(reply))
    if missing:
        raise ReplyError(
            f"score reply missing field {missing[0]}",
            pod_id=pod_id,
            node_name=node_name,
            details={"missing_field": missing[0]},
        )
    if reply["pod_id"] != pod_id or reply["node_name"] != node_name:
        raise ReplyError(
            "score reply identity mismatch",
            pod_id=pod_id,
            node_name=node_name,
            details={
                "reply_pod_id": reply["pod_id"],
                "reply_node_name": reply["node_name"],
            },
        )
    for field, expected in (
        ("parameter_version", parameter_version),
        ("scale_exponent", scale_exponent),
        ("scale_factor", scale_factor),
    ):
        if reply[field] != expected:
            raise ReplyError(
                f"score reply {field} mismatch",
                pod_id=pod_id,
                node_name=node_name,
                details={
                    "field": field,
                    "expected": expected,
                    "actual": reply[field],
                },
            )
    for field in (
        "plugin_scores",
        "plugin_weights_myriad",
        "plugin_contributions",
    ):
        if not isinstance(reply[field], dict) or set(reply[field]) != set(PLUGIN_NAMES):
            raise ReplyError(
                f"score reply {field} must contain exactly four plugins",
                pod_id=pod_id,
                node_name=node_name,
            )
    for field in ("total_fixed", "total_score", "error_bound", "elapsed_ms"):
        if isinstance(reply[field], bool) or not isinstance(reply[field], int):
            raise ReplyError(
                f"score reply {field} must be an integer",
                pod_id=pod_id,
                node_name=node_name,
            )
    return reply


@dataclass
class WorkerHandle:
    group_name: str
    port: int
    process: subprocess.Popen[str]
    sock: socket.socket
    buffer: bytearray

    def close(self) -> None:
        try:
            if self.sock.fileno() >= 0:
                self.sock.sendall(b'{"command":"shutdown"}\n')
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()


class ScoringManager:
    def __init__(self, spec: Spec) -> None:
        self.spec = spec
        self.workers: dict[str, WorkerHandle] = {}
        self.node_to_group: dict[str, str] = {}
        for group in spec.service.groups:
            for node_name in group.node_names:
                self.node_to_group[node_name] = group.name

    def __enter__(self) -> "ScoringManager":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def start(self) -> None:
        timeout = self.spec.service.timeout_ms / 1000.0
        try:
            for group in self.spec.service.groups:
                if not _port_is_available(self.spec.service.host, group.port):
                    raise PortError(
                        f"scoring port {group.port} is already occupied",
                        details={"group": group.name, "port": group.port},
                    )
                command = [
                    sys.executable,
                    "-m",
                    "affinity_scheduler.worker",
                    "--host",
                    self.spec.service.host,
                    "--port",
                    str(group.port),
                ]
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=os.environ.copy(),
                )
                ready_line = _read_process_line(process, timeout)
                if ready_line is None:
                    stderr = ""
                    if process.poll() is not None and process.stderr is not None:
                        stderr = process.stderr.read()
                    raise PortError(
                        f"scoring service failed to bind port {group.port}",
                        details={
                            "group": group.name,
                            "port": group.port,
                            "stderr": stderr[-1000:],
                        },
                    )
                try:
                    ready = json.loads(ready_line)
                except json.JSONDecodeError as exc:
                    raise PortError(
                        f"scoring service handshake is invalid: {exc.msg}",
                        details={"group": group.name, "port": group.port},
                    ) from exc
                if ready.get("ready") is not True or ready.get("port") != group.port:
                    raise PortError(
                        "scoring service handshake does not confirm the declared port",
                        details={"group": group.name, "port": group.port},
                    )
                try:
                    sock = socket.create_connection(
                        (self.spec.service.host, group.port), timeout=timeout
                    )
                except socket.timeout as exc:
                    process.terminate()
                    raise TimeoutError(
                        f"connecting to scoring port {group.port} timed out",
                        details={"group": group.name, "port": group.port},
                    ) from exc
                except OSError as exc:
                    process.terminate()
                    raise PortError(
                        f"cannot connect to scoring port {group.port}: {exc}",
                        details={"group": group.name, "port": group.port},
                    ) from exc
                sock.settimeout(timeout)
                self.workers[group.name] = WorkerHandle(
                    group_name=group.name,
                    port=group.port,
                    process=process,
                    sock=sock,
                    buffer=bytearray(),
                )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for handle in list(self.workers.values()):
            handle.close()
        self.workers.clear()

    def score(
        self,
        *,
        pod_id: str,
        node_name: str,
        parameter_version: int,
        scale_exponent: int,
        weights_myriad: Mapping[str, int],
        features: Mapping[str, Any],
    ) -> dict[str, Any]:
        group_name = self.node_to_group.get(node_name)
        if group_name is None:
            raise ReplyError(
                "node is not assigned to a scoring group",
                pod_id=pod_id,
                node_name=node_name,
            )
        handle = self.workers[group_name]
        request = {
            "pod_id": pod_id,
            "node_name": node_name,
            "parameter_version": parameter_version,
            "scale_exponent": scale_exponent,
            "weights_myriad": {name: int(weights_myriad[name]) for name in PLUGIN_NAMES},
            "features": dict(features),
        }
        payload = (canonical_json(request) + "\n").encode("utf-8")
        try:
            handle.sock.sendall(payload)
        except socket.timeout as exc:
            raise TimeoutError(
                f"sending score request to port {handle.port} timed out",
                details={"pod_id": pod_id, "node_name": node_name},
            ) from exc
        except OSError as exc:
            raise PortError(
                f"scoring socket send failed: {exc}",
                details={"pod_id": pod_id, "node_name": node_name},
            ) from exc
        reply_line = _read_socket_line(handle.sock, handle.buffer, timeout_ms=self.spec.service.timeout_ms)
        if reply_line is None:
            raise TimeoutError(
                f"score reply from port {handle.port} timed out",
                details={"pod_id": pod_id, "node_name": node_name},
            )
        try:
            reply = json.loads(reply_line)
        except json.JSONDecodeError as exc:
            raise ReplyError(
                f"score reply is not valid JSON: {exc.msg}",
                pod_id=pod_id,
                node_name=node_name,
            ) from exc
        if isinstance(reply, dict) and "error" in reply:
            raise ReplyError(
                f"scoring service rejected request: {reply['error']}",
                pod_id=pod_id,
                node_name=node_name,
                details={"worker_error": reply},
            )
        return _validate_reply(
            reply,
            pod_id=pod_id,
            node_name=node_name,
            parameter_version=parameter_version,
            scale_exponent=scale_exponent,
            scale_factor=10**scale_exponent,
        )


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _read_process_line(
    process: subprocess.Popen[str], timeout: float
) -> str | None:
    if process.stdout is None:
        return None
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        process.terminate()
        return None
    line = process.stdout.readline()
    return line.strip() if line else None


def _read_socket_line(
    sock: socket.socket, buffer: bytearray, *, timeout_ms: int
) -> str | None:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while True:
        newline = buffer.find(b"\n")
        if newline >= 0:
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            return line.decode("utf-8")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        sock.settimeout(remaining)
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            return None
        if not chunk:
            return None
        buffer.extend(chunk)
        if len(buffer) > 1024 * 1024:
            return None
