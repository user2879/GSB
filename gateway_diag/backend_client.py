from __future__ import annotations

import json
import socket
from typing import Any

from .errors import E_BACKEND, GatewayError
from .models import Backend, Request
from .util import canonical_json, digest_value


class BackendClient:
    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def _round_trip(self, backend: Backend, payload: dict[str, Any]) -> dict[str, Any]:
        material = (canonical_json(payload) + "\n").encode("utf-8")
        try:
            with socket.create_connection((backend.host, backend.port), timeout=self.timeout) as sock:
                sock.settimeout(self.timeout)
                sock.sendall(material)
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > 8 * 1024 * 1024:
                        raise GatewayError(
                            E_BACKEND,
                            "backend response exceeded the protocol limit",
                            field="response",
                            details={"backend_id": backend.id, "port": backend.port},
                        )
                    if b"\n" in chunk:
                        break
        except GatewayError:
            raise
        except OSError as exc:
            raise GatewayError(
                E_BACKEND,
                "cannot reach local mock inference backend",
                field="port",
                details={
                    "backend_id": backend.id,
                    "host": backend.host,
                    "port": backend.port,
                    "reason": str(exc),
                },
            )
        data = b"".join(chunks).split(b"\n", 1)[0]
        try:
            response = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GatewayError(
                E_BACKEND,
                "backend returned an invalid JSON line",
                field="response",
                details={"backend_id": backend.id, "reason": str(exc)},
            )
        if not isinstance(response, dict):
            raise GatewayError(E_BACKEND, "backend response must be an object", field="response")
        if not response.get("ok"):
            raise GatewayError(
                E_BACKEND,
                str(response.get("message", "backend rejected the request")),
                field=str(response.get("field", "response")),
                details={"backend_id": backend.id, "response": response},
            )
        return response

    def ping(self, backend: Backend) -> dict[str, Any]:
        return self._round_trip(backend, {"op": "ping"})

    def infer_batch(self, backend: Backend, requests: list[Request]) -> dict[str, Any]:
        if not requests:
            raise ValueError("an inference batch must not be empty")
        payload = {
            "op": "infer",
            "protocol": "infer.v1",
            "backend_id": backend.id,
            "params": {
                "model_dim": backend.model_dim,
                "hidden_dim": backend.hidden_dim,
                "seed": backend.seed,
                "quantization": backend.quantization,
            },
            "requests": [
                {"request_id": request.request_id, "vector": list(request.vector)}
                for request in requests
            ],
        }
        response = self._round_trip(backend, payload)
        if response.get("backend_id") != backend.id:
            raise GatewayError(
                E_BACKEND,
                "backend response identifies a different backend",
                field="backend_id",
                details={"expected": backend.id, "actual": response.get("backend_id")},
            )
        if response.get("request_count") != len(requests):
            raise GatewayError(
                E_BACKEND,
                "backend response request count does not match the batch",
                field="request_count",
                details={"expected": len(requests), "actual": response.get("request_count")},
            )
        output_digest = response.get("output_digest")
        if not isinstance(output_digest, str) or not output_digest.startswith("sha256:"):
            raise GatewayError(
                E_BACKEND,
                "backend response has no valid output_digest",
                field="output_digest",
                details={"backend_id": backend.id},
            )
        expected_params = {
            "id": backend.id,
            "model_dim": backend.model_dim,
            "hidden_dim": backend.hidden_dim,
            "seed": backend.seed,
            "quantization": backend.quantization,
        }
        if response.get("param_digest") != digest_value(expected_params):
            raise GatewayError(
                E_BACKEND,
                "backend response parameter digest does not match the declaration",
                field="param_digest",
                details={
                    "backend_id": backend.id,
                    "expected": digest_value(expected_params),
                    "actual": response.get("param_digest"),
                },
            )
        if sorted(response.get("request_ids", [])) != sorted(
            request.request_id for request in requests
        ):
            raise GatewayError(
                E_BACKEND,
                "backend response request ids do not match the batch",
                field="request_ids",
                details={"backend_id": backend.id},
            )
        return response
