from __future__ import annotations

import json
import socket
from typing import Any

from .errors import LeaseError
from .util import canonical_json_bytes, is_int, mono_ms


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 4 * 1024 * 1024

LEASE_ACK_FIELDS = (
    "ack_version",
    "node_id",
    "request_seq",
    "lease_id",
    "artifact_keys",
    "digests",
    "capacity",
    "expires_mono_ms",
    "elapsed_mono_ms",
    "mono_ms",
)


def validate_lease_ack(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise LeaseError(
            "E_ACK", "lease acknowledgement is not a JSON object", field="ack", expected="object"
        )
    if response.get("ok") is not True:
        code = response.get("code", "E_ACK")
        raise LeaseError(
            str(code),
            str(response.get("message", "node rejected lease request")),
            field=response.get("field", "ack.ok"),
            expected=response.get("expected", "ok=true"),
            actual=response.get("actual", response.get("ok")),
            details={"node_response": response},
        )
    missing = [name for name in LEASE_ACK_FIELDS if name not in response]
    if missing:
        raise LeaseError(
            "E_ACK",
            f"lease acknowledgement missing required field {missing[0]}",
            field=missing[0],
            expected="all required acknowledgement fields",
            actual="<missing>",
            details={"missing": missing},
        )
    if response.get("ack_version") != PROTOCOL_VERSION:
        raise LeaseError(
            "E_ACK",
            "unsupported acknowledgement version",
            field="ack_version",
            expected=PROTOCOL_VERSION,
            actual=response.get("ack_version"),
        )
    for field in ("request_seq", "expires_mono_ms", "elapsed_mono_ms", "mono_ms"):
        if not is_int(response.get(field)) or response[field] < 0:
            raise LeaseError(
                "E_ACK",
                f"acknowledgement field {field} is not a non-negative integer",
                field=field,
                expected="non-negative integer",
                actual=response.get(field),
            )
    digests = response.get("digests")
    artifact_keys = response.get("artifact_keys")
    capacity = response.get("capacity")
    if not isinstance(artifact_keys, list) or not all(
        isinstance(key, str) for key in artifact_keys
    ):
        raise LeaseError(
            "E_ACK",
            "artifact_keys must be a string array",
            field="artifact_keys",
            expected="array of artifact keys",
            actual=artifact_keys,
        )
    if not isinstance(digests, dict) or any(
        key not in digests or not isinstance(digests[key], str) for key in artifact_keys
    ):
        raise LeaseError(
            "E_ACK",
            "digests must contain a digest for every leased artifact key",
            field="digests",
            expected="object keyed by every artifact key",
            actual=digests,
        )
    if not isinstance(capacity, dict):
        raise LeaseError(
            "E_ACK",
            "capacity acknowledgement is missing",
            field="capacity",
            expected="capacity object with used/total/free",
            actual=capacity,
        )
    for field in ("used_slots", "total_slots", "free_slots", "active_leases", "lease_limit"):
        if not is_int(capacity.get(field)) or capacity[field] < 0:
            raise LeaseError(
                "E_ACK",
                "capacity acknowledgement has an invalid field",
                field=f"capacity.{field}",
                expected="non-negative integer",
                actual=capacity.get(field),
            )
    return response


def validate_node_reply(response: Any, required: tuple[str, ...] = ()) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise LeaseError(
            "E_ACK", "node reply is not a JSON object", field="ack", expected="object"
        )
    if response.get("ok") is not True:
        raise LeaseError(
            str(response.get("code", "E_ACK")),
            str(response.get("message", "node rejected request")),
            field=response.get("field", "ack.ok"),
            expected=response.get("expected", "ok=true"),
            actual=response.get("actual", response.get("ok")),
            details={"node_response": response},
        )
    missing = [name for name in required if name not in response]
    if missing:
        raise LeaseError(
            "E_ACK",
            f"node reply missing required field {missing[0]}",
            field=missing[0],
            expected="all required reply fields",
            actual="<missing>",
        )
    return response


class NodeClient:
    def __init__(self, node_id: str, host: str, port: int, timeout_ms: int = 3000) -> None:
        self.node_id = node_id
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms

    def _send(self, request: dict[str, Any], *, phase: str) -> dict[str, Any]:
        started = mono_ms()
        line = canonical_json_bytes(request) + b"\n"
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.timeout_ms / 1000
            ) as connection:
                connection.settimeout(self.timeout_ms / 1000)
                connection.sendall(line)
                received = bytearray()
                while True:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    received.extend(chunk)
                    if len(received) > MAX_LINE_BYTES:
                        raise LeaseError(
                            "E_ACK",
                            "node reply exceeds protocol line limit",
                            field="ack",
                            expected=f"<= {MAX_LINE_BYTES} bytes",
                        )
                    if b"\n" in chunk:
                        break
        except LeaseError:
            raise
        except socket.timeout as exc:
            code = "E_HANDSHAKE" if phase == "handshake" else "E_IO"
            raise LeaseError(
                code,
                f"real socket timeout during {phase}",
                field="socket",
                expected=f"reply within {self.timeout_ms} ms",
                actual=f"timeout after {mono_ms() - started} ms",
                node_id=self.node_id,
            ) from exc
        except OSError as exc:
            code = "E_HANDSHAKE" if phase == "handshake" else "E_IO"
            raise LeaseError(
                code,
                f"socket failure during {phase}: {exc}",
                field="socket",
                expected="live 127.0.0.1 TCP endpoint",
                actual=str(exc),
                node_id=self.node_id,
            ) from exc

        raw = bytes(received).split(b"\n", 1)[0]
        if not raw:
            raise LeaseError(
                "E_ACK",
                "node closed without an acknowledgement line",
                field="ack",
                expected="one JSON acknowledgement line",
                actual="<empty>",
                node_id=self.node_id,
            )
        try:
            reply = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LeaseError(
                "E_ACK",
                "node acknowledgement is not valid UTF-8 JSON",
                field="ack",
                expected="one UTF-8 JSON object line",
                actual=str(exc),
                node_id=self.node_id,
            ) from exc
        return reply

    def handshake(self, timeout_ms: int | None = None) -> dict[str, Any]:
        previous = self.timeout_ms
        if timeout_ms is not None:
            self.timeout_ms = timeout_ms
        try:
            reply = self._send(
                {
                    "op": "ping",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_seq": 0,
                },
                phase="handshake",
            )
            result = validate_node_reply(reply, ("ack_version", "node_id", "mono_ms"))
            if result["node_id"] != self.node_id:
                raise LeaseError(
                    "E_HANDSHAKE",
                    "handshake returned a different node identity",
                    field="node_id",
                    expected=self.node_id,
                    actual=result["node_id"],
                    node_id=self.node_id,
                )
            return result
        finally:
            self.timeout_ms = previous

    def lease(
        self,
        artifact_keys: list[str],
        lease_ms: int,
        request_seq: int,
        *,
        service_delay_ms: int = 0,
        lease_id: str | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "op": "lease",
            "protocol_version": PROTOCOL_VERSION,
            "request_seq": request_seq,
            "artifact_keys": artifact_keys,
            "lease_ms": lease_ms,
            "service_delay_ms": service_delay_ms,
        }
        if lease_id is not None:
            request["lease_id"] = lease_id
        reply = self._send(
            request,
            phase="lease",
        )
        return validate_lease_ack(reply)

    def renew(self, lease_id: str, lease_ms: int) -> dict[str, Any]:
        reply = self._send(
            {
                "op": "renew",
                "protocol_version": PROTOCOL_VERSION,
                "lease_id": lease_id,
                "lease_ms": lease_ms,
            },
            phase="renew",
        )
        return validate_node_reply(reply, ("lease_id", "expires_mono_ms", "elapsed_mono_ms"))

    def release(self, lease_id: str) -> dict[str, Any]:
        reply = self._send(
            {
                "op": "release",
                "protocol_version": PROTOCOL_VERSION,
                "lease_id": lease_id,
            },
            phase="release",
        )
        return validate_node_reply(reply, ("lease_id", "released", "elapsed_mono_ms"))

    def digests(self) -> dict[str, Any]:
        reply = self._send(
            {"op": "digests", "protocol_version": PROTOCOL_VERSION},
            phase="digests",
        )
        return validate_node_reply(reply, ("node_id", "digests", "refs", "mono_ms"))

    def status(self) -> dict[str, Any]:
        reply = self._send(
            {"op": "status", "protocol_version": PROTOCOL_VERSION},
            phase="status",
        )
        return validate_node_reply(
            reply, ("node_id", "capacity", "active_leases", "mono_ms")
        )

    def shutdown(self) -> None:
        self._send(
            {"op": "shutdown", "protocol_version": PROTOCOL_VERSION},
            phase="shutdown",
        )
