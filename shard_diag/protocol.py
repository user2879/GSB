from __future__ import annotations

import json
import socket
from contextlib import contextmanager
from typing import Any, Iterator

from .errors import DiagnosticError, ErrorCode
from .spec import DatabaseSpec


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 8 * 1024 * 1024


def _proto(message: str, *, field: str | None = None, **details: Any) -> DiagnosticError:
    return DiagnosticError(ErrorCode.PROTO.value, message, field=field, details=details)


def _timeout(
    message: str,
    *,
    db_id: str | None = None,
    field: str | None = None,
    **details: Any,
) -> DiagnosticError:
    return DiagnosticError(ErrorCode.TIMEOUT.value, message, db_id=db_id, field=field, details=details)


def encode_line(message: dict[str, Any]) -> bytes:
    try:
        return (json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as error:
        raise _proto(f"cannot encode JSON line: {error}") from error


def decode_line(line: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} is not allowed")

    try:
        value = json.loads(line.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _proto(f"invalid JSON line: {error}") from error
    if not isinstance(value, dict):
        raise _proto("JSON line must contain an object")
    return value


def read_json_line(receive: Any, *, timeout_code: str = ErrorCode.TIMEOUT.value) -> dict[str, Any]:
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = receive(1)
        except socket.timeout as error:
            raise DiagnosticError(timeout_code, "socket read timed out") from error
        if not chunk:
            raise _proto("socket closed before a complete JSON line was received")
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_LINE_BYTES:
            raise _proto("JSON line exceeds the protocol size limit")
        if chunk == b"\n":
            break
    return decode_line(b"".join(chunks).strip())


def _require_field(payload: dict[str, Any], field: str, expected_type: type | tuple[type, ...]) -> Any:
    if field not in payload:
        raise _proto(f"response is missing required field {field!r}", field=field, payload=payload)
    value = payload[field]
    expected_types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
    if not isinstance(value, expected_types) or (isinstance(value, bool) and bool not in expected_types):
        raise _proto(f"response field {field!r} has the wrong type", field=field, payload=payload)
    return value


def validate_probe_result(
    result: dict[str, Any],
    *,
    db_id: str,
    event_seq: int,
    expected_route_version: int,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise _proto("probe result must be an object")
    response_db = _require_field(result, "db_id", str)
    if response_db != db_id:
        raise _proto(
            f"probe result db_id {response_db!r} does not match worker {db_id!r}",
            field="db_id",
        )
    response_seq = _require_field(result, "event_seq", int)
    if response_seq != event_seq:
        raise _proto(
            f"probe result event_seq {response_seq} does not match request {event_seq}",
            field="event_seq",
        )
    hit_shards = _require_field(result, "hit_shards", list)
    if not all(isinstance(shard, str) and shard for shard in hit_shards):
        raise _proto("hit_shards must contain non-empty strings", field="hit_shards")
    route_version = _require_field(result, "route_version", int)
    if route_version != expected_route_version:
        raise _proto(
            f"worker used route version {route_version}, expected {expected_route_version}",
            field="route_version",
        )
    elapsed_ms = _require_field(result, "elapsed_ms", (int, float))
    if elapsed_ms < 0:
        raise _proto("elapsed_ms must not be negative", field="elapsed_ms")
    sample_keys = result.get("sample_keys", [])
    if not isinstance(sample_keys, list) or not all(
        isinstance(key, int) and not isinstance(key, bool) for key in sample_keys
    ):
        raise _proto("sample_keys must be a list of integers", field="sample_keys")
    return result


class WorkerClient:
    """Small, timeout-bound JSON line client for one physical database child."""

    def __init__(
        self,
        database: DatabaseSpec,
        *,
        timeout_seconds: float = 2.0,
    ) -> None:
        self.database = database
        self.host = database.host
        self.port = database.port
        self.timeout_seconds = timeout_seconds

    def _connect(self) -> socket.socket:
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout_seconds)
            sock.settimeout(self.timeout_seconds)
            return sock
        except socket.timeout as error:
            raise _timeout(
                f"timed out connecting to {self.database.id} at {self.host}:{self.port}",
                db_id=self.database.id,
            ) from error
        except OSError as error:
            raise _timeout(
                f"cannot connect to {self.database.id} at {self.host}:{self.port}: {error}",
                db_id=self.database.id,
            ) from error

    def handshake(self, sock: socket.socket) -> dict[str, Any]:
        hello = {
            "type": "hello",
            "protocol": PROTOCOL_VERSION,
            "db_id": self.database.id,
            "route_version": self.database.route_version,
        }
        try:
            sock.sendall(encode_line(hello))
            response = read_json_line(sock.recv)
        except socket.timeout as error:
            raise _timeout(
                f"handshake timed out for database {self.database.id}",
                db_id=self.database.id,
            ) from error
        if response.get("type") != "hello_ack":
            raise _proto("handshake response type must be 'hello_ack'", field="type")
        if response.get("db_id") != self.database.id:
            raise _proto("handshake response has an unexpected db_id", field="db_id")
        if response.get("protocol") != PROTOCOL_VERSION:
            raise _proto("handshake response has an unexpected protocol version", field="protocol")
        return response

    @contextmanager
    def connection(self) -> Iterator[socket.socket]:
        sock = self._connect()
        try:
            self.handshake(sock)
            yield sock
        finally:
            sock.close()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with self.connection() as sock:
                sock.sendall(encode_line(payload))
                response = read_json_line(sock.recv)
        except socket.timeout as error:
            raise _timeout(
                f"response timed out for database {self.database.id}",
                db_id=self.database.id,
            ) from error
        if not response.get("ok", False):
            code = response.get("code", ErrorCode.PROTO.value)
            raise DiagnosticError(
                code,
                str(response.get("message", "worker returned an error")),
                db_id=response.get("db_id", self.database.id),
                field=response.get("field"),
                details=response,
            )
        return response

    def ping(self) -> dict[str, Any]:
        response = self.request({"type": "ping"})
        _require_field(response, "database", dict)
        return response

    def probe_batch(
        self,
        probes: list[dict[str, Any]],
        *,
        batch_id: str | None = None,
    ) -> list[dict[str, Any]]:
        payload = {
            "type": "probe_batch",
            "batch_id": batch_id,
            "probes": probes,
        }
        response = self.request(payload)
        results = _require_field(response, "results", list)
        if len(results) != len(probes):
            raise _proto(
                f"worker returned {len(results)} results for {len(probes)} probes",
                field="results",
            )
        validated: list[dict[str, Any]] = []
        for probe, result in zip(probes, results, strict=True):
            validated.append(
                validate_probe_result(
                    result,
                    db_id=self.database.id,
                    event_seq=probe["event_seq"],
                    expected_route_version=self.database.route_version,
                )
            )
        return validated

    def grant_lease(
        self,
        *,
        lease_id: str,
        shard_id: str,
        event_seq: int,
        lease_version: int,
    ) -> dict[str, Any]:
        return self.request(
            {
                "type": "lease_grant",
                "lease_id": lease_id,
                "shard_id": shard_id,
                "event_seq": event_seq,
                "lease_version": lease_version,
            }
        )

    def renew_lease(
        self,
        *,
        lease_id: str,
        expected_lease_version: int,
        event_seq: int,
        shard_id: str,
    ) -> dict[str, Any]:
        response = self.request(
            {
                "type": "lease_renew",
                "lease_id": lease_id,
                "expected_lease_version": expected_lease_version,
                "event_seq": event_seq,
                "shard_id": shard_id,
            }
        )
        new_version = _require_field(response, "lease_version", int)
        return {"lease_version": new_version, "response": response}

    def revoke_lease(self, *, lease_id: str, shard_id: str, event_seq: int) -> dict[str, Any]:
        return self.request(
            {
                "type": "lease_revoke",
                "lease_id": lease_id,
                "shard_id": shard_id,
                "event_seq": event_seq,
            }
        )
