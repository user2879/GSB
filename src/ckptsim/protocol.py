from __future__ import annotations

import json
import socket
from typing import Any

from .canonical import canonical_json
from .errors import CkptError
from .spec import is_int


MAX_LINE_BYTES = 32 * 1024 * 1024


def recv_json_line(sock: socket.socket) -> dict[str, Any]:
    chunks = bytearray()
    while True:
        try:
            chunk = sock.recv(65536)
        except socket.timeout as exc:
            raise CkptError("E_TIMEOUT", "socket read timed out") from exc
        except OSError as exc:
            raise CkptError("E_PORT", f"socket read failed: {exc}") from exc
        if not chunk:
            raise CkptError("E_FIELD", "peer closed before a complete JSON line")
        chunks.extend(chunk)
        if len(chunks) > MAX_LINE_BYTES:
            raise CkptError("E_FIELD", "JSON line exceeds size limit")
        newline = chunks.find(b"\n")
        if newline >= 0:
            raw = bytes(chunks[:newline])
            break
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CkptError("E_FIELD", f"invalid JSON response: {exc}") from exc
    if not isinstance(value, dict):
        raise CkptError("E_FIELD", "JSON response must be an object")
    return value


def send_json_line(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(canonical_json(payload).encode("utf-8") + b"\n")


class TcpShardClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout: float = 5.0,
        connect_timeout: float = 3.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.connect_timeout = connect_timeout

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout
            )
        except socket.timeout as exc:
            raise CkptError(
                "E_TIMEOUT", f"TCP connect to {self.host}:{self.port} timed out"
            ) from exc
        except OSError as exc:
            raise CkptError(
                "E_PORT", f"cannot connect to {self.host}:{self.port}: {exc}"
            ) from exc
        with sock:
            sock.settimeout(self.timeout)
            try:
                send_json_line(sock, payload)
                response = recv_json_line(sock)
            except socket.timeout as exc:
                raise CkptError(
                    "E_TIMEOUT", f"TCP exchange with {self.host}:{self.port} timed out"
                ) from exc
        if response.get("ok") is False:
            error = response.get("error")
            if not isinstance(error, dict) or not isinstance(error.get("code"), str):
                raise CkptError("E_FIELD", "error response is missing error.code")
            raise CkptError(
                error["code"],
                str(error.get("detail", "remote error")),
                item=error.get("item"),
                line=error.get("line"),
                shard_id=error.get("shard_id"),
                batch_id=error.get("batch_id"),
                field=error.get("field"),
            )
        return response

    def health(self) -> dict[str, Any]:
        response = self.request({"op": "health"})
        self._require(response, "shard_id", str)
        return response

    def batch(
        self,
        *,
        shard_id: str,
        batch_id: str,
        sample_ids: list[int],
        parameter_version: int,
        samples: dict[str, Any],
        lease_id: str,
        lease_version: int,
        lease_ttl_ms: int,
    ) -> dict[str, Any]:
        response = self.request(
            {
                "op": "batch",
                "shard_id": shard_id,
                "batch_id": batch_id,
                "sample_ids": list(sample_ids),
                "parameter_version": parameter_version,
                "samples": samples,
                "lease_id": lease_id,
                "lease_version": lease_version,
                "lease_ttl_ms": lease_ttl_ms,
            }
        )
        for field, expected_type in (
            ("shard_id", str),
            ("batch_id", str),
            ("parameter_version", int),
            ("generation", int),
            ("parameter_digest", str),
            ("samples_digest", str),
            ("sample_ids", list),
            ("loss", dict),
            ("loss_digest", str),
            ("duration_ms", int),
        ):
            self._require(response, field, expected_type)
        if response["shard_id"] != shard_id:
            raise CkptError(
                "E_FIELD",
                f"response shard_id {response['shard_id']!r} does not match request",
                shard_id=shard_id,
                batch_id=batch_id,
                field="shard_id",
            )
        if response["batch_id"] != batch_id:
            raise CkptError(
                "E_FIELD",
                f"response batch_id {response['batch_id']!r} does not match request",
                shard_id=shard_id,
                batch_id=batch_id,
                field="batch_id",
            )
        if response["parameter_version"] != parameter_version:
            raise CkptError(
                "E_FIELD",
                "response parameter_version does not match request",
                shard_id=shard_id,
                batch_id=batch_id,
                field="parameter_version",
            )
        if sorted(response["sample_ids"]) != sorted(sample_ids):
            raise CkptError(
                "E_FIELD",
                "response sample_ids do not match request set",
                shard_id=shard_id,
                batch_id=batch_id,
                field="sample_ids",
            )
        if response["duration_ms"] < 0:
            raise CkptError(
                "E_FIELD",
                "response duration_ms must be non-negative",
                shard_id=shard_id,
                batch_id=batch_id,
                field="duration_ms",
            )
        for field in (
            "shard_id",
            "batch_id",
            "sample_ids",
            "parameter_version",
            "parameter_digest",
            "samples_digest",
            "loss_units",
            "per_sample_loss",
        ):
            if field not in response["loss"]:
                raise CkptError(
                    "E_FIELD",
                    f"response loss is missing {field}",
                    shard_id=shard_id,
                    batch_id=batch_id,
                    field=f"loss.{field}",
                )
        return response

    def renew(
        self,
        *,
        shard_id: str,
        lease_id: str,
        expected_version: int,
        lease_ttl_ms: int,
    ) -> dict[str, Any]:
        response = self.request(
            {
                "op": "renew",
                "shard_id": shard_id,
                "lease_id": lease_id,
                "lease_version": expected_version,
                "lease_ttl_ms": lease_ttl_ms,
            }
        )
        for field, expected_type in (
            ("lease_id", str),
            ("lease_version", int),
            ("expires_ms", int),
        ):
            self._require(response, field, expected_type)
        if response["lease_version"] != expected_version + 1:
            raise CkptError(
                "E_LEASE",
                "renewal did not advance the lease version exactly once",
                shard_id=shard_id,
                field="lease_version",
            )
        return response

    @staticmethod
    def _require(response: dict[str, Any], field: str, expected_type: type) -> None:
        if field not in response:
            raise CkptError("E_FIELD", f"response is missing {field}", field=field)
        value = response[field]
        if expected_type is int:
            valid = is_int(value)
        else:
            valid = isinstance(value, expected_type)
        if not valid:
            raise CkptError(
                "E_FIELD", f"response field {field} has the wrong type", field=field
            )
