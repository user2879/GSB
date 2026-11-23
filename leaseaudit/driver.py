from __future__ import annotations

import http.client
import json
import socket
from dataclasses import dataclass, field

from .engine import Lease
from .scenario import SourceSpec


@dataclass
class RequestResult:
    source_id: str
    request_seq: int
    lease_id: str
    outcome: str
    status_code: int | None = None
    detail: str | None = None

    def as_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "request_seq": self.request_seq,
            "lease_id": self.lease_id,
            "outcome": self.outcome,
            "status_code": self.status_code,
            "detail": self.detail,
        }


@dataclass
class MockSourceClient:
    spec: SourceSpec
    request_counter: int = 0
    results: list[RequestResult] = field(default_factory=list)
    next_request_at_ms: int = 0
    next_renew_at_ms: int = 0

    def lease_is_valid_at(self, lease: Lease, now_ms: int) -> bool:
        return lease.source_id == self.spec.source_id and now_ms < lease.expires_at_ms

    def request(self, lease: Lease, now_ms: int) -> RequestResult:
        if not self.lease_is_valid_at(lease, now_ms):
            result = RequestResult(
                source_id=self.spec.source_id,
                request_seq=self.request_counter,
                lease_id=lease.lease_id,
                outcome="client_rejected_expired_lease",
                detail="monotonic now_ms must be strictly before expires_at_ms",
            )
            self.results.append(result)
            return result

        self.request_counter += 1
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            self.spec.port,
            timeout=self.spec.request_timeout_ms / 1000.0,
        )
        try:
            connection.request(
                "POST",
                "/work",
                body=b"deterministic-load",
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Lease-Id": lease.lease_id,
                    "X-Lease-Expires-Ms": str(lease.expires_at_ms),
                    "X-Request-Seq": str(self.request_counter),
                },
            )
            response = connection.getresponse()
            body = response.read()
            if response.status != 200:
                result = RequestResult(
                    source_id=self.spec.source_id,
                    request_seq=self.request_counter,
                    lease_id=lease.lease_id,
                    outcome="http_error",
                    status_code=response.status,
                    detail=body.decode("utf-8", errors="replace"),
                )
            else:
                try:
                    document = json.loads(body)
                    detail = str(document.get("status", "success"))
                except ValueError:
                    detail = "invalid_json"
                result = RequestResult(
                    source_id=self.spec.source_id,
                    request_seq=self.request_counter,
                    lease_id=lease.lease_id,
                    outcome="success",
                    status_code=response.status,
                    detail=detail,
                )
        except ConnectionRefusedError:
            result = RequestResult(
                source_id=self.spec.source_id,
                request_seq=self.request_counter,
                lease_id=lease.lease_id,
                outcome="connection_refused",
                detail="loopback listener refused the connection",
            )
        except (ConnectionResetError, BrokenPipeError):
            result = RequestResult(
                source_id=self.spec.source_id,
                request_seq=self.request_counter,
                lease_id=lease.lease_id,
                outcome="connection_reset",
            )
        except (socket.timeout, TimeoutError):
            result = RequestResult(
                source_id=self.spec.source_id,
                request_seq=self.request_counter,
                lease_id=lease.lease_id,
                outcome="timeout",
            )
        except OSError as exc:
            result = RequestResult(
                source_id=self.spec.source_id,
                request_seq=self.request_counter,
                lease_id=lease.lease_id,
                outcome="os_error",
                detail=str(exc),
            )
        finally:
            connection.close()
        self.results.append(result)
        return result

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.outcome] = counts.get(result.outcome, 0) + 1
        return {
            "source_id": self.spec.source_id,
            "requests": self.request_counter,
            "outcomes": {key: counts[key] for key in sorted(counts)},
        }

    def next_request_delay_ms(self) -> int:
        from .numeric import numpy_request_delay_ms

        return numpy_request_delay_ms(
            self.spec.seed,
            self.request_counter,
            self.spec.request_interval_ms,
        )
