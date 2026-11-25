"""JSON-line TCP client used by the center."""

from __future__ import annotations

import json
import socket
from typing import Any

from .errors import LeaseError, PortError, ReplyError, TimeoutError


class LeaseNodeClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        node_id: str | None = None,
        timeout: float = 2.0,
    ) -> None:
        self.host = host
        self.port = port
        self.node_id = node_id
        self.timeout = timeout

    def _roundtrip(
        self, payload: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        effective_timeout = self.timeout if timeout is None else timeout
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=effective_timeout
            ) as sock:
                sock.settimeout(effective_timeout)
                wire = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
                sock.sendall(wire)
                file_obj = sock.makefile("rb")
                line = file_obj.readline()
                if not line:
                    raise ReplyError(
                        "node closed connection without a JSON-line reply",
                        field="reply",
                        details={"port": self.port, "op": payload.get("op")},
                    )
                try:
                    reply = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ReplyError(
                        "node reply is not valid JSON",
                        field="reply",
                        details={"line": line[:200].decode("utf-8", "replace")},
                    ) from exc
                if not isinstance(reply, dict):
                    raise ReplyError("node reply must be an object", field="reply")
                self._validate_envelope(reply, payload)
                if reply.get("ok") is False:
                    self._raise_node_error(reply, payload)
                return reply
        except (TimeoutError, ReplyError, LeaseError, PortError):
            raise
        except socket.timeout as exc:
            raise TimeoutError(
                "timed out while waiting for node socket/handshake",
                field="socket.timeout",
                details={"host": self.host, "port": self.port, "op": payload.get("op")},
            ) from exc
        except ConnectionRefusedError as exc:
            raise PortError(
                "node TCP port is not accepting connections",
                field="port",
                details={"host": self.host, "port": self.port},
            ) from exc
        except OSError as exc:
            raise PortError(
                "node TCP connection failed",
                field="socket",
                details={"host": self.host, "port": self.port, "errno": exc.errno},
            ) from exc

    def _validate_envelope(self, reply: dict[str, Any], request: dict[str, Any]) -> None:
        op = request.get("op", "unknown")
        required = ["ok", "node_id"]
        if op != "handshake":
            required.extend(["op", "current_tag"])
        missing = [key for key in required if key not in reply]
        if missing:
            raise ReplyError(
                "node reply is missing required fields",
                field="reply." + missing[0],
                details={"missing": missing, "op": op},
            )
        if self.node_id is not None and reply.get("node_id") != self.node_id:
            raise ReplyError(
                "node reply node_id does not match declaration",
                field="reply.node_id",
                details={"expected": self.node_id, "actual": reply.get("node_id")},
            )
        if op != "handshake" and reply.get("op") != op:
            raise ReplyError(
                "node reply op does not match request",
                field="reply.op",
                details={"expected": op, "actual": reply.get("op")},
            )

    def _raise_node_error(self, reply: dict[str, Any], request: dict[str, Any]) -> None:
        code = reply.get("code", "E_LEASE")
        details = {
            "lease_id": request.get("lease_id"),
            "current_tag": reply.get("current_tag"),
            "requested_tag": request.get("version_tag"),
            "node_id": reply.get("node_id"),
        }
        if code == "E_LEASE":
            raise LeaseError(
                reply.get("message", "node rejected lease generation tag"),
                field="version_tag",
                details=details,
            )
        raise ReplyError(
            reply.get("message", "node returned an error reply"),
            field="reply",
            details={"code": code, **details},
        )

    def handshake(self) -> dict[str, Any]:
        reply = self._roundtrip({"op": "handshake", "protocol_version": 1})
        if reply.get("protocol_version") != 1:
            raise ReplyError(
                "unsupported node protocol_version",
                field="reply.protocol_version",
                details={"actual": reply.get("protocol_version")},
            )
        return reply

    def grant(self, lease_id: str, equity_id: str, ttl_ms: int) -> dict[str, Any]:
        return self._roundtrip(
            {
                "op": "grant",
                "lease_id": lease_id,
                "equity_id": equity_id,
                "ttl_ms": ttl_ms,
            }
        )

    def renew(self, lease_id: str, version_tag: int) -> dict[str, Any]:
        return self._roundtrip(
            {"op": "renew", "lease_id": lease_id, "version_tag": version_tag}
        )

    def release(self, lease_id: str, version_tag: int) -> dict[str, Any]:
        return self._roundtrip(
            {"op": "release", "lease_id": lease_id, "version_tag": version_tag}
        )

    def expire(self, lease_id: str, version_tag: int, reason: str) -> dict[str, Any]:
        return self._roundtrip(
            {
                "op": "expire",
                "lease_id": lease_id,
                "version_tag": version_tag,
                "reason": reason,
            }
        )


def probe_port_available(host: str, port: int, *, timeout: float = 0.2) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.bind((host, port))
    except OSError:
        return False
    finally:
        sock.close()
    return True
