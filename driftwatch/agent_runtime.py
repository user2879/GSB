from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from .agent_main_support import parent_alive
from .deterministic import generate_observation
from .errors import DriftError
from .models import AGGREGATION_KINDS, METRICS, AgentSpec, AggregationSpec, is_int
from .protocol import MAX_LINE_BYTES, PROTOCOL_VERSION, JsonLineChannel, monotonic_ms


@dataclass
class AgentRuntime:
    spec: AgentSpec
    parent_pid: int | None

    def __post_init__(self) -> None:
        self._stopping = threading.Event()
        self._listener: socket.socket | None = None

    def stop(self) -> None:
        self._stopping.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

    def serve_forever(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener = listener
        try:
            if os.name == "nt":
                exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
                if exclusive is not None:
                    listener.setsockopt(socket.SOL_SOCKET, exclusive, 1)
            else:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.spec.host, self.spec.port))
            listener.listen(64)
            listener.settimeout(0.2)
        except OSError as exc:
            print(
                json.dumps({"ok": False, "code": "E_PORT", "message": str(exc), "port": self.spec.port}),
                file=sys.stderr,
                flush=True,
            )
            listener.close()
            raise

        monitor: threading.Thread | None = None
        if self.parent_pid is not None:
            monitor = threading.Thread(target=self._monitor_parent, daemon=True)
            monitor.start()
        try:
            while not self._stopping.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stopping.is_set():
                        break
                    raise
                threading.Thread(target=self._handle_connection, args=(connection,), daemon=True).start()
        finally:
            try:
                listener.close()
            except OSError:
                pass

    def _monitor_parent(self) -> None:
        assert self.parent_pid is not None
        while not self._stopping.wait(0.2):
            if not parent_alive(self.parent_pid):
                self.stop()
                return

    def _handle_connection(self, connection: socket.socket) -> None:
        channel = JsonLineChannel(connection)
        try:
            connection.settimeout(15.0)
            while not self._stopping.is_set():
                try:
                    request = channel.receive(time.monotonic() + 15.0)
                except DriftError:
                    return
                except TimeoutError:
                    return
                op = request.get("op")
                if op == "hello":
                    if request.get("agent_id") != self.spec.agent_id:
                        channel.send(
                            {
                                "ok": False,
                                "code": "E_RECORD",
                                "message": "agent identifier does not match listener",
                            }
                        )
                        return
                    if request.get("protocol") != PROTOCOL_VERSION:
                        channel.send(
                            {
                                "ok": False,
                                "code": "E_RECORD",
                                "message": "unsupported protocol version",
                            }
                        )
                        return
                    channel.send(
                        {
                            "ok": True,
                            "agent_id": self.spec.agent_id,
                            "protocol": PROTOCOL_VERSION,
                            "pid": os.getpid(),
                        }
                    )
                elif op == "observe":
                    self._handle_observe(channel, request)
                elif op == "shutdown":
                    channel.send({"ok": True, "agent_id": self.spec.agent_id})
                    self.stop()
                    return
                else:
                    channel.send({"ok": False, "code": "E_RECORD", "message": "unknown operation"})
        finally:
            channel.close()

    def _handle_observe(self, channel: JsonLineChannel, request: dict[str, Any]) -> None:
        required = ("snapshot_seq", "host_id", "metric_id", "observation_seq")
        missing = [field for field in required if field not in request]
        if missing:
            channel.send(
                {
                    "ok": False,
                    "code": "E_RECORD",
                    "message": "observation request is missing fields",
                    "fields": missing,
                }
            )
            return
        if not is_int(request["snapshot_seq"]) or not is_int(request["observation_seq"]):
            channel.send(
                {"ok": False, "code": "E_RECORD", "message": "observation sequence fields must be integers"}
            )
            return
        host_id = request["host_id"]
        metric_id = request["metric_id"]
        if not isinstance(host_id, str) or not isinstance(metric_id, str):
            channel.send({"ok": False, "code": "E_RECORD", "message": "host and metric must be strings"})
            return
        if not self.spec.covers(host_id, metric_id):
            channel.send(
                {
                    "ok": False,
                    "code": "E_RECORD",
                    "message": "agent does not cover requested host and metric",
                    "host_id": host_id,
                    "metric_id": metric_id,
                }
            )
            return
        observation = generate_observation(
            self.spec,
            snapshot_seq=int(request["snapshot_seq"]),
            host_id=host_id,
            metric_id=metric_id,
            observation_seq=int(request["observation_seq"]),
            monotonic_ms=monotonic_ms(),
        )
        channel.send({"ok": True, "record": observation.to_dict()})


def spec_from_resolved_dict(raw: dict[str, Any]) -> AgentSpec:
    try:
        aggregation_raw = raw["aggregation"]
        aggregation = AggregationSpec(
            kind=aggregation_raw["kind"],
            window=int(aggregation_raw["window"]),
            version=str(aggregation_raw["version"]),
        )
        spec = AgentSpec(
            agent_id=str(raw["agent_id"]),
            host=str(raw["host"]),
            port=int(raw["port"]),
            seed=int(raw["seed"]),
            aggregation=aggregation,
            hosts=tuple(str(item) for item in raw["hosts"]),
            metrics=tuple(str(item) for item in raw["metrics"]),
            per_agent_concurrency=int(raw.get("per_agent_concurrency", 1)),
            profiles=raw.get("profiles", {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DriftError.agent(
            [{"path": "$", "reason": "resolved agent specification is invalid", "actual": str(exc)}]
        ) from exc
    if spec.host != "127.0.0.1" or not (1024 <= spec.port <= 65535):
        raise DriftError.agent([{"path": "$.host", "reason": "invalid loopback listener declaration"}])
    if spec.aggregation.kind not in AGGREGATION_KINDS:
        raise DriftError.agent([{"path": "$.aggregation.kind", "reason": "unsupported aggregation kind"}])
    if any(metric not in METRICS for metric in spec.metrics):
        raise DriftError.agent([{"path": "$.metrics", "reason": "unsupported metric"}])
    return spec
