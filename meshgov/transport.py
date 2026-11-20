from __future__ import annotations

import http.client
import socket
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode

from .rules import InstanceRule, RequestSpec


@dataclass(frozen=True)
class AttemptResponse:
    success: bool
    code: str | None
    detail: str
    http_status: int | None
    body: str = ""


class Transport(Protocol):
    def request(
        self, instance: InstanceRule, request: RequestSpec, attempt_no: int
    ) -> AttemptResponse: ...


class HttpTransport:
    def __init__(self, timeout_ms: int | None = None) -> None:
        self.fallback_timeout_ms = timeout_ms

    def request(
        self, instance: InstanceRule, request: RequestSpec, attempt_no: int
    ) -> AttemptResponse:
        query = urlencode(
            {"mesh_request_id": request.request_id, "mesh_attempt": attempt_no}
        )
        path = request.path + ("&" if "?" in request.path else "?") + query
        timeout_ms = request.route.timeout_ms
        if self.fallback_timeout_ms is not None:
            timeout_ms = self.fallback_timeout_ms
        timeout_s = max(timeout_ms / 1000.0, 0.001)
        connection = http.client.HTTPConnection(
            instance.host, instance.port, timeout=timeout_s
        )
        try:
            connection.request(
                request.method,
                path,
                headers={"Connection": "close", "Accept": "application/json"},
            )
            response = connection.getresponse()
            body_bytes = response.read(64 * 1024)
            body = body_bytes.decode("utf-8", errors="replace")
            if 200 <= response.status < 300:
                return AttemptResponse(True, None, "ok", response.status, body)
            return AttemptResponse(
                False,
                "E_UPSTREAM",
                f"upstream returned HTTP {response.status}",
                response.status,
                body,
            )
        except (socket.timeout, TimeoutError):
            return AttemptResponse(
                False,
                "E_UPSTREAM",
                f"attempt timeout after {timeout_ms} ms",
                None,
            )
        except OSError as exc:
            return AttemptResponse(
                False,
                "E_UPSTREAM",
                f"upstream unreachable: {exc}",
                None,
            )
        except http.client.HTTPException as exc:
            return AttemptResponse(
                False,
                "E_UPSTREAM",
                f"upstream protocol error: {exc}",
                None,
            )
        finally:
            connection.close()
