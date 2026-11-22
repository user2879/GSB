from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Any

from .errors import DriftError
from .models import AgentSpec, Observation, is_int

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 1024 * 1024


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def _deadline(timeout_ms: int) -> float:
    return time.monotonic() + max(timeout_ms, 1) / 1000.0


def _remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("socket deadline expired")
    return value


class JsonLineChannel:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self._buffer = bytearray()

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def send(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_LINE_BYTES:
            raise DriftError.record("outgoing JSON line exceeds limit", size=len(payload))
        self.sock.sendall(payload + b"\n")

    def receive(self, deadline: float) -> dict[str, Any]:
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                raw = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                if not raw:
                    continue
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise DriftError.record("invalid JSON line", error=str(exc)) from exc
                if not isinstance(value, dict):
                    raise DriftError.record("protocol message must be a JSON object", actual=type(value).__name__)
                return value
            if len(self._buffer) > MAX_LINE_BYTES:
                raise DriftError.record("incoming JSON line exceeds limit", size=len(self._buffer))
            self.sock.settimeout(_remaining(deadline))
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout as exc:
                raise DriftError.timeout("socket read timed out") from exc
            except OSError as exc:
                raise DriftError.port("socket read failed", error=str(exc)) from exc
            if not chunk:
                raise DriftError.record("peer closed the protocol connection before a complete record")
            self._buffer.extend(chunk)


def _connect(
    agent: AgentSpec,
    deadline: float,
    *,
    retry_until_deadline: bool,
    deadline_code: str = "E_PORT",
) -> socket.socket:
    while True:
        timeout = _remaining(deadline)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((agent.host, agent.port))
            return sock
        except socket.timeout as exc:
            sock.close()
            if retry_until_deadline and time.monotonic() < deadline:
                time.sleep(min(0.025, max(0.001, deadline - time.monotonic())))
                continue
            raise DriftError.timeout(
                "agent socket connection timed out",
                agent_id=agent.agent_id,
                host=agent.host,
                port=agent.port,
            ) from exc
        except (ConnectionRefusedError, OSError) as exc:
            sock.close()
            if time.monotonic() >= deadline:
                if deadline_code == "E_TIMEOUT":
                    raise DriftError.timeout(
                        "agent handshake connection deadline expired",
                        agent_id=agent.agent_id,
                        host=agent.host,
                        port=agent.port,
                        error=str(exc),
                    ) from exc
                raise DriftError.port(
                    "cannot connect to agent loopback port",
                    agent_id=agent.agent_id,
                    host=agent.host,
                    port=agent.port,
                    error=str(exc),
                ) from exc
            if not retry_until_deadline:
                raise DriftError.port(
                    "cannot connect to agent loopback port",
                    agent_id=agent.agent_id,
                    host=agent.host,
                    port=agent.port,
                    error=str(exc),
                ) from exc
            time.sleep(min(0.025, max(0.001, deadline - time.monotonic())))


def _validate_handshake(response: dict[str, Any], agent: AgentSpec) -> None:
    if response.get("ok") is not True:
        code = response.get("code")
        message = response.get("message", "agent rejected handshake")
        if code == "E_PORT":
            raise DriftError.port(message, agent_id=agent.agent_id, port=agent.port)
        raise DriftError.record(message, agent_id=agent.agent_id, response=response)
    missing = [
        field
        for field in ("agent_id", "protocol")
        if field not in response
    ]
    if missing:
        raise DriftError.record("handshake record is missing fields", agent_id=agent.agent_id, fields=missing)
    if response["agent_id"] != agent.agent_id:
        raise DriftError.record(
            "handshake agent identifier mismatch",
            expected=agent.agent_id,
            actual=response["agent_id"],
        )
    if response["protocol"] != PROTOCOL_VERSION:
        raise DriftError.record(
            "unsupported agent protocol version",
            expected=PROTOCOL_VERSION,
            actual=response["protocol"],
        )


def handshake(
    agent: AgentSpec,
    timeout_ms: int,
    *,
    retry_until_deadline: bool = True,
    deadline_code: str = "E_PORT",
) -> dict[str, Any]:
    deadline = _deadline(timeout_ms)
    sock = _connect(
        agent,
        deadline,
        retry_until_deadline=retry_until_deadline,
        deadline_code=deadline_code,
    )
    channel = JsonLineChannel(sock)
    try:
        channel.send(
            {
                "op": "hello",
                "protocol": PROTOCOL_VERSION,
                "agent_id": agent.agent_id,
                "client": "driftwatch",
            }
        )
        try:
            response = channel.receive(deadline)
        except DriftError:
            raise
        except TimeoutError as exc:
            raise DriftError.timeout(
                "agent handshake timed out",
                agent_id=agent.agent_id,
                host=agent.host,
                port=agent.port,
            ) from exc
        _validate_handshake(response, agent)
        return response
    finally:
        channel.close()


def _validate_record(response: dict[str, Any], task: Any) -> Observation:
    if response.get("ok") is not True:
        code = response.get("code", "E_RECORD")
        message = response.get("message", "agent rejected observation request")
        if code == "E_TIMEOUT":
            raise DriftError.timeout(message, agent_id=task.agent.agent_id)
        raise DriftError.record(message, agent_id=task.agent.agent_id, response=response)
    record = response.get("record")
    if not isinstance(record, dict):
        raise DriftError.record("observation response has no record object", agent_id=task.agent.agent_id)
    required = {
        "agent_id": str,
        "observation_seq": int,
        "host_id": str,
        "metric_id": str,
        "value": int,
        "aggregation_version": str,
        "monotonic_ms": int,
    }
    problems: list[str] = []
    for field, expected_type in required.items():
        value = record.get(field)
        if field not in record:
            problems.append(f"missing:{field}")
        elif expected_type is int and not is_int(value):
            problems.append(f"type:{field}")
        elif expected_type is str and not isinstance(value, str):
            problems.append(f"type:{field}")
        elif expected_type is int and int(value) < 0:
            problems.append(f"negative:{field}")
    if problems:
        raise DriftError.record(
            "observation record failed validation",
            agent_id=task.agent.agent_id,
            problems=problems,
            record=record,
        )
    if record["agent_id"] != task.agent.agent_id:
        raise DriftError.record("record agent identifier mismatch", expected=task.agent.agent_id, actual=record["agent_id"])
    if record["observation_seq"] != task.observation_seq:
        raise DriftError.record(
            "record observation sequence mismatch",
            expected=task.observation_seq,
            actual=record["observation_seq"],
        )
    if record["host_id"] != task.host_id:
        raise DriftError.record("record host mismatch", expected=task.host_id, actual=record["host_id"])
    if record["metric_id"] != task.metric_id:
        raise DriftError.record("record metric mismatch", expected=task.metric_id, actual=record["metric_id"])
    if not record["aggregation_version"]:
        raise DriftError.record("record aggregation version must not be empty")
    return Observation(
        agent_id=record["agent_id"],
        observation_seq=int(record["observation_seq"]),
        host_id=record["host_id"],
        metric_id=record["metric_id"],
        value=int(record["value"]),
        aggregation_version=record["aggregation_version"],
        monotonic_ms=int(record["monotonic_ms"]),
    )


@dataclass
class AgentClient:
    agent: AgentSpec
    socket_timeout_ms: int
    handshake_timeout_ms: int

    def observe(self, task: Any) -> Observation:
        handshake_deadline = _deadline(self.handshake_timeout_ms)
        sock = _connect(self.agent, handshake_deadline, retry_until_deadline=False)
        channel = JsonLineChannel(sock)
        try:
            channel.send(
                {
                    "op": "hello",
                    "protocol": PROTOCOL_VERSION,
                    "agent_id": self.agent.agent_id,
                    "client": "driftwatch",
                }
            )
            hello = channel.receive(handshake_deadline)
            _validate_handshake(hello, self.agent)
            deadline = _deadline(self.socket_timeout_ms)
            channel.send(
                {
                    "op": "observe",
                    "request_id": task.request_id,
                    "snapshot_seq": task.snapshot_seq,
                    "host_id": task.host_id,
                    "metric_id": task.metric_id,
                    "observation_seq": task.observation_seq,
                }
            )
            response = channel.receive(deadline)
            return _validate_record(response, task)
        except TimeoutError as exc:
            raise DriftError.timeout(
                "observation socket operation timed out",
                agent_id=self.agent.agent_id,
                host_id=task.host_id,
                metric_id=task.metric_id,
            ) from exc
        finally:
            channel.close()


def request_shutdown(agent: AgentSpec, timeout_ms: int) -> bool:
    deadline = _deadline(timeout_ms)
    try:
        sock = _connect(agent, deadline, retry_until_deadline=False)
    except DriftError:
        return False
    channel = JsonLineChannel(sock)
    try:
        channel.send(
            {
                "op": "hello",
                "protocol": PROTOCOL_VERSION,
                "agent_id": agent.agent_id,
                "client": "driftwatch",
            }
        )
        response = channel.receive(deadline)
        _validate_handshake(response, agent)
        channel.send({"op": "shutdown", "agent_id": agent.agent_id})
        channel.receive(deadline)
        return True
    except (DriftError, TimeoutError, OSError):
        return False
    finally:
        channel.close()
