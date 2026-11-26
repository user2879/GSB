"""Loopback JSON-lines cache node child process."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes
from .errors import EdgeError, LeaseError, PortError
from .jsonlog import StructuredLogger
from .sampling import compute_sampling
from .spec import FragmentEvent, NodeSpec, RevisionSpec, load_spec


class NodeState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.leases: dict[str, dict[str, int]] = {}

    def register(self, lease_id: str, version: int, expires_ms: int) -> None:
        with self.lock:
            self.leases[lease_id] = {"version": int(version), "expires_ms": int(expires_ms)}

    def assert_lease(self, lease_id: str, version: int) -> None:
        with self.lock:
            lease = self.leases.get(lease_id)
            if lease is None:
                raise EdgeError(
                    "E_LEASE",
                    "lease is not registered on node",
                    details={"lease_id": lease_id},
                    exit_code=5,
                )
            now_ms = int(time.monotonic() * 1000)
            if now_ms >= lease["expires_ms"]:
                raise EdgeError(
                    "E_LEASE",
                    "lease has really expired",
                    details={"lease_id": lease_id, "now_ms": now_ms, "expires_ms": lease["expires_ms"]},
                    exit_code=5,
                )
            if lease["version"] != int(version):
                raise LeaseError(lease_id, int(version), int(lease["version"]))

    def renew(self, lease_id: str, expected_version: int, ttl_ms: int) -> dict[str, int]:
        with self.lock:
            lease = self.leases.get(lease_id)
            if lease is None:
                raise LeaseError(lease_id, int(expected_version), None)
            if lease["version"] != int(expected_version):
                raise LeaseError(lease_id, int(expected_version), int(lease["version"]))
            new_version = int(expected_version) + 1
            new_expiry = int(time.monotonic() * 1000) + int(ttl_ms)
            lease["version"] = new_version
            lease["expires_ms"] = new_expiry
            return {"version": new_version, "expires_ms": new_expiry}


class CacheNodeServer:
    def __init__(
        self,
        node: NodeSpec,
        fragment_by_id: dict[str, FragmentEvent],
        revision_by_id: dict[int, RevisionSpec],
        *,
        delay_ms: int = 0,
        missing_receipt_field: str | None = None,
        rate_skew_ppm: int = 0,
        logger: StructuredLogger | None = None,
    ) -> None:
        self.node = node
        self.fragment_by_id = fragment_by_id
        self.revision_by_id = revision_by_id
        self.delay_ms = int(delay_ms)
        self.missing_receipt_field = missing_receipt_field
        self.rate_skew_ppm = int(rate_skew_ppm)
        self.state = NodeState()
        self.logger = logger or StructuredLogger()
        self._stop = threading.Event()

    def _error_response(self, error: EdgeError, request_id: Any = None) -> dict[str, Any]:
        payload = error.to_dict()
        payload["request_id"] = request_id
        return payload

    def _validate_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic_ns()
        request_id = request.get("request_id")
        fragment_id = request.get("fragment_id")
        if not isinstance(fragment_id, str) or fragment_id not in self.fragment_by_id:
            raise EdgeError(
                "E_PROTOCOL",
                "unknown or missing fragment_id",
                details={"fragment_id": fragment_id},
                exit_code=9,
            )
        fragment = self.fragment_by_id[fragment_id]
        revision_id = request.get("quota_version")
        if not isinstance(revision_id, int) or revision_id not in self.revision_by_id:
            raise EdgeError("E_PROTOCOL", "unknown quota_version", exit_code=9)
        revision = self.revision_by_id[revision_id]
        if revision.node_id != self.node.node_id or fragment.node_id != self.node.node_id:
            raise EdgeError("E_PROTOCOL", "request does not belong to this node", exit_code=9)
        expected_fragment_rate = int(request.get("fragment_rate_ppm", revision.fragment_rate_ppm))
        expected_block_rate = int(request.get("block_rate_ppm", revision.block_rate_ppm))
        if expected_fragment_rate != revision.fragment_rate_ppm or expected_block_rate != revision.block_rate_ppm:
            raise EdgeError(
                "E_RATE",
                "request rates do not match the declared revision",
                details={
                    "expected_fragment_rate_ppm": revision.fragment_rate_ppm,
                    "requested_fragment_rate_ppm": expected_fragment_rate,
                    "expected_block_rate_ppm": revision.block_rate_ppm,
                    "requested_block_rate_ppm": expected_block_rate,
                },
                exit_code=9,
            )
        lease_id = request.get("lease_id")
        lease_version = request.get("lease_version")
        if isinstance(lease_id, str) and isinstance(lease_version, int):
            self.state.assert_lease(lease_id, lease_version)
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        applied_fragment_rate = revision.fragment_rate_ppm
        applied_block_rate = min(1_000_000, max(0, revision.block_rate_ppm + self.rate_skew_ppm))
        if self.rate_skew_ppm:
            skewed_revision = RevisionSpec(
                revision_id=revision.revision_id,
                node_id=revision.node_id,
                parent_revision=revision.parent_revision,
                quota=revision.quota,
                fragment_rate_ppm=applied_fragment_rate,
                block_rate_ppm=applied_block_rate,
            )
            outcome = compute_sampling(self.node, fragment, skewed_revision)
        else:
            outcome = compute_sampling(self.node, fragment, revision)
        elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
        payload = dict(outcome.payload)
        payload.update(
            {
                "ok": True,
                "request_id": request_id,
                "node_id": self.node.node_id,
                "elapsed_monotonic_ms": int(elapsed_ms),
                "duration_monotonic_ms": int(elapsed_ms),
                "applied_fragment_rate_ppm": applied_fragment_rate,
                "applied_block_rate_ppm": applied_block_rate,
                "lease_id": lease_id,
                "lease_version": lease_version,
            }
        )
        if self.missing_receipt_field:
            payload.pop(self.missing_receipt_field, None)
        return payload

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        request_id = request.get("request_id")
        if op == "ping":
            return {"ok": True, "code": "OK", "request_id": request_id, "node_id": self.node.node_id}
        if op == "register_lease":
            self.state.register(
                str(request["lease_id"]),
                int(request["version"]),
                int(request["expires_monotonic_ms"]),
            )
            return {
                "ok": True,
                "code": "OK",
                "request_id": request_id,
                "lease_id": request["lease_id"],
                "version": int(request["version"]),
            }
        if op == "renew_lease":
            renewed = self.state.renew(
                str(request["lease_id"]),
                int(request["expected_version"]),
                int(request["ttl_ms"]),
            )
            return {
                "ok": True,
                "code": "OK",
                "request_id": request_id,
                "lease_id": request["lease_id"],
                **renewed,
            }
        if op == "validate_batch":
            return self._validate_batch(request)
        if op == "shutdown":
            self._stop.set()
            return {"ok": True, "code": "OK", "request_id": request_id}
        raise EdgeError("E_PROTOCOL", "unknown operation", details={"op": op}, exit_code=9)

    def serve_forever(self, ready_event: threading.Event | None = None) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self.node.host, self.node.port))
            listener.listen(64)
        except OSError as exc:
            raise PortError(
                self.node.node_id,
                self.node.host,
                self.node.port,
                os_error=str(exc),
            ) from exc
        listener.settimeout(0.2)
        if ready_event is not None:
            ready_event.set()
        self.logger.emit(
            "NODE_READY",
            node_id=self.node.node_id,
            host=self.node.host,
            port=self.node.port,
        )
        try:
            while not self._stop.is_set():
                try:
                    connection, address = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                thread = threading.Thread(
                    target=self._serve_connection,
                    args=(connection, address),
                    daemon=True,
                )
                thread.start()
        finally:
            listener.close()

    def _serve_connection(self, connection: socket.socket, address: tuple[str, int]) -> None:
        with connection:
            connection.settimeout(2.0)
            buffer = b""
            while not self._stop.is_set():
                try:
                    chunk = connection.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line:
                        continue
                    request_id: Any = None
                    try:
                        request = json.loads(line.decode("utf-8"))
                        if not isinstance(request, dict):
                            raise ValueError("request is not a JSON object")
                        request_id = request.get("request_id")
                        response = self.handle(request)
                    except EdgeError as error:
                        response = self._error_response(error, request_id)
                        self.logger.emit(
                            error.code,
                            level="ERROR",
                            node_id=self.node.node_id,
                            **error.details,
                        )
                    except Exception as error:
                        response = self._error_response(
                            EdgeError(
                                "E_PROTOCOL",
                                f"request handling failed: {type(error).__name__}",
                                details={"error": str(error)},
                                exit_code=9,
                            ),
                            request_id,
                        )
                    connection.sendall(canonical_bytes(response) + b"\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="edge-cache loopback node")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--log")
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--missing-receipt-field")
    parser.add_argument("--rate-skew-ppm", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.spec)
        node = spec.node_by_id().get(args.node_id)
        if node is None:
            raise EdgeError("E_SPEC", "unknown node id", details={"node_id": args.node_id}, exit_code=3)
        logger = StructuredLogger(args.log)
        server = CacheNodeServer(
            node,
            spec.fragment_by_id(),
            spec.revision_by_id(),
            delay_ms=args.delay_ms,
            missing_receipt_field=args.missing_receipt_field,
            rate_skew_ppm=args.rate_skew_ppm,
            logger=logger,
        )
        ready = threading.Event()
        server.serve_forever(ready)
        return 0
    except PortError as error:
        structured = StructuredLogger(args.log)
        structured.emit(error.code, level="ERROR", **error.details)
        print(json.dumps(error.to_dict(), ensure_ascii=False), file=sys.stderr)
        return error.exit_code
    except EdgeError as error:
        structured = StructuredLogger(args.log)
        structured.emit(error.code, level="ERROR", **error.details)
        print(json.dumps(error.to_dict(), ensure_ascii=False), file=sys.stderr)
        return error.exit_code
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

