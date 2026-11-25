from __future__ import annotations

import json
import socket
from dataclasses import asdict
from typing import Any

from .errors import EdgeLedgerError, LeaseError
from .spec import NodeSpec


PROTOCOL_NAME = "edge-jsonline/1"


class NodeProtocolError(EdgeLedgerError):
    def __init__(self, message: str, code: str = "E_NODE", field: str | None = None):
        super().__init__(code, message, field=field)


class NodeClient:
    def __init__(self, node: NodeSpec, timeout_ms: int = 2000):
        self.node = node
        self.timeout_s = max(0.05, timeout_ms / 1000.0)

    def _rpc(self, message: dict[str, Any]) -> dict[str, Any]:
        try:
            sock = socket.create_connection((self.node.host, self.node.port), timeout=self.timeout_s)
        except OSError as exc:
            raise NodeProtocolError(
                f"cannot connect to {self.node.node_id} at {self.node.host}:{self.node.port}: {exc}",
                code="E_NODE_UNREACHABLE",
            ) from exc
        with sock:
            sock.settimeout(self.timeout_s)
            try:
                stream = sock.makefile("rwb", buffering=0)
                hello = {"type": "hello", "protocol": PROTOCOL_NAME, "node_id": self.node.node_id}
                stream.write((json.dumps(hello, sort_keys=True) + "\n").encode("utf-8"))
                hello_line = stream.readline()
                if not hello_line:
                    raise NodeProtocolError("empty handshake response", code="E_NODE_HANDSHAKE")
                hello_ack = json.loads(hello_line.decode("utf-8"))
                if not hello_ack.get("ok") or hello_ack.get("node_id") != self.node.node_id:
                    raise NodeProtocolError(f"bad handshake ack: {hello_ack}", code="E_NODE_HANDSHAKE")
                stream.write((json.dumps(message, sort_keys=True) + "\n").encode("utf-8"))
                response_line = stream.readline()
                if not response_line:
                    raise NodeProtocolError("node closed connection before response", code="E_NODE_READ")
                response = json.loads(response_line.decode("utf-8"))
            except socket.timeout as exc:
                raise NodeProtocolError(f"socket timeout after {self.timeout_s:.3f}s", code="E_NODE_TIMEOUT") from exc
            except (OSError, json.JSONDecodeError) as exc:
                raise NodeProtocolError(f"JSON-line protocol failure: {exc}", code="E_NODE_PROTOCOL") from exc
        if not response.get("ok"):
            error = response.get("error") or {}
            code = error.get("code", "E_NODE")
            text = error.get("message", "node rejected request")
            if code == "E_LEASE":
                raise LeaseError(text)
            raise NodeProtocolError(text, code=code)
        return response

    def ping(self) -> dict[str, Any]:
        return self._rpc({"type": "ping"})

    def grant_lease(self, lease: Any) -> dict[str, Any]:
        return self._rpc(
            {
                "type": "grant",
                "lease_id": lease.lease_id,
                "lease_version": lease.lease_version,
                "quota_version": lease.quota_version,
                "shard_id": lease.shard_id,
            }
        )

    def renew_lease(self, lease_id: str, expected_version: int, ttl_ms: int) -> dict[str, Any]:
        return self._rpc(
            {
                "type": "renew",
                "lease_id": lease_id,
                "expected_version": expected_version,
                "ttl_ms": ttl_ms,
            }
        )

    def verify_shard(
        self,
        *,
        lease: Any,
        content_key: str,
        shard_seq: int,
        crash_before_reply: bool = False,
    ) -> dict[str, Any]:
        return self._rpc(
            {
                "type": "verify",
                "request_id": f"{lease.lease_id}:{shard_seq}",
                "lease_id": lease.lease_id,
                "lease_version": lease.lease_version,
                "quota_version": lease.quota_version,
                "content_key": content_key,
                "shard_seq": shard_seq,
                "crash_before_reply": crash_before_reply,
            }
        )


def proof_to_dict(proof: Any) -> dict[str, Any]:
    if hasattr(proof, "to_canonical"):
        return proof.to_canonical()
    if hasattr(proof, "__dataclass_fields__"):
        return asdict(proof)
    return dict(proof)
