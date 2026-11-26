"""JSON-lines loopback client with real socket timeout semantics."""

from __future__ import annotations

import json
import socket
import threading
from typing import Any, Iterable

from .canonical import canonical_bytes
from .errors import EdgeError, LeaseError, PortError, TimeoutErrorEdge
from .spec import NodeSpec


REQUIRED_RECEIPT_FIELDS = (
    "request_id",
    "node_id",
    "fragment_id",
    "sampled_blocks",
    "sampling_counts",
    "covered_blocks",
    "uncovered_blocks",
    "uncovered_ratio_ppm",
    "quantization_error_bound",
    "error_count",
    "error_signed_fixed",
    "error_abs_fixed",
    "status",
    "quota_version",
    "duration_monotonic_ms",
    "applied_fragment_rate_ppm",
    "applied_block_rate_ppm",
)


class NodeClient:
    def __init__(self, node: NodeSpec, timeout_ms: int):
        self.node = node
        self.timeout_ms = int(timeout_ms)
        self._sequence = 0
        self._lock = threading.Lock()

    def _request_id(self) -> str:
        with self._lock:
            self._sequence += 1
            return f"{self.node.node_id}-req-{self._sequence}"

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout_s = self.timeout_ms / 1000.0
        request = dict(payload)
        request.setdefault("request_id", self._request_id())
        try:
            with socket.create_connection((self.node.host, self.node.port), timeout=timeout_s) as connection:
                connection.settimeout(timeout_s)
                connection.sendall(canonical_bytes(request) + b"\n")
                file_object = connection.makefile("rb")
                line = file_object.readline()
        except socket.timeout as exc:
            raise TimeoutErrorEdge(
                self.node.node_id,
                self.timeout_ms,
                request_id=request.get("request_id"),
            ) from exc
        except (ConnectionRefusedError, OSError) as exc:
            raise PortError(
                self.node.node_id,
                self.node.host,
                self.node.port,
                os_error=str(exc),
                request_id=request.get("request_id"),
            ) from exc
        if not line:
            raise EdgeError(
                "E_PROTOCOL",
                "empty response from node",
                details={"node_id": self.node.node_id},
                exit_code=9,
            )
        try:
            response = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise EdgeError(
                "E_PROTOCOL",
                "node response is not JSON",
                details={"node_id": self.node.node_id},
                exit_code=9,
            ) from exc
        if not isinstance(response, dict):
            raise EdgeError("E_PROTOCOL", "node response must be an object", exit_code=9)
        if not response.get("ok", False):
            code = str(response.get("code", "E_PROTOCOL"))
            if code == "E_LEASE":
                details = response.get("details", {})
                raise LeaseError(
                    str(details.get("lease_id", request.get("lease_id"))),
                    int(details.get("expected_version", request.get("lease_version", 0))),
                    details.get("actual_version"),
                )
            raise EdgeError(
                code,
                str(response.get("message", "node rejected request")),
                details=dict(response.get("details", {})),
                exit_code=9,
            )
        return response

    def ping(self) -> dict[str, Any]:
        return self.request({"op": "ping"})

    def register_lease(self, lease_id: str, version: int, expires_monotonic_ms: int) -> dict[str, Any]:
        return self.request(
            {
                "op": "register_lease",
                "lease_id": lease_id,
                "version": version,
                "expires_monotonic_ms": expires_monotonic_ms,
            }
        )

    def renew_lease(self, lease_id: str, expected_version: int, ttl_ms: int) -> dict[str, Any]:
        return self.request(
            {
                "op": "renew_lease",
                "lease_id": lease_id,
                "expected_version": expected_version,
                "ttl_ms": ttl_ms,
            }
        )

    def validate_batch(
        self,
        *,
        fragment_id: str,
        quota_version: int,
        fragment_rate_ppm: int,
        block_rate_ppm: int,
        lease_id: str,
        lease_version: int,
    ) -> dict[str, Any]:
        response = self.request(
            {
                "op": "validate_batch",
                "fragment_id": fragment_id,
                "quota_version": quota_version,
                "fragment_rate_ppm": fragment_rate_ppm,
                "block_rate_ppm": block_rate_ppm,
                "lease_id": lease_id,
                "lease_version": lease_version,
            }
        )
        missing = [field for field in REQUIRED_RECEIPT_FIELDS if field not in response]
        if missing:
            raise EdgeError(
                "E_RECEIPT",
                "node receipt is missing required fields",
                details={"missing_fields": missing, "node_id": self.node.node_id},
                exit_code=9,
            )
        if response.get("node_id") != self.node.node_id:
            raise EdgeError("E_RECEIPT", "receipt node_id does not match", exit_code=9)
        if response.get("fragment_id") != fragment_id:
            raise EdgeError("E_RECEIPT", "receipt fragment_id does not match", exit_code=9)
        if int(response.get("quota_version", -1)) != int(quota_version):
            raise EdgeError("E_RECEIPT", "receipt quota_version does not match", exit_code=9)
        if int(response.get("applied_fragment_rate_ppm", -1)) != int(fragment_rate_ppm):
            raise EdgeError("E_RATE", "node distorted the fragment sampling rate", exit_code=9)
        if int(response.get("applied_block_rate_ppm", -1)) != int(block_rate_ppm):
            raise EdgeError("E_RATE", "node distorted the block sampling rate", exit_code=9)
        duration = response.get("duration_monotonic_ms")
        if not isinstance(duration, int) or duration < 0:
            raise EdgeError("E_RECEIPT", "receipt monotonic duration is invalid", exit_code=9)
        return response

