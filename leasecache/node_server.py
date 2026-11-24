from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .digest import ArtifactStore
from .errors import LeaseError
from .protocol import MAX_LINE_BYTES, PROTOCOL_VERSION
from .spec import NodeSpec, PipelineSpec, load_spec
from .util import StructuredLogger, is_int, mono_ms


class LeaseDenied(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str,
        expected: str,
        actual: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.expected = expected
        self.actual = actual

    def response(self, node_id: str) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "node_id": node_id,
            "mono_ms": mono_ms(),
        }


class NodeRuntime:
    def __init__(
        self,
        spec: PipelineSpec,
        node: NodeSpec,
        workspace: Path,
        logger: StructuredLogger,
    ) -> None:
        self.spec = spec
        self.node = node
        self.logger = logger
        self.store = ArtifactStore(workspace, node)
        self.digests: dict[str, str] = {}
        self.refs: dict[str, list[str]] = {}
        self.manifest: list[dict[str, object]] = []
        self.digests, self.refs, self.manifest = self.store.materialize(spec.artifacts)
        self.lock = threading.RLock()
        self.used_slots = 0
        self.leases: dict[str, dict[str, Any]] = {}
        self.request_cache: dict[int, dict[str, Any]] = {}
        self.stopping = threading.Event()
        self.logger.info(
            "node artifacts materialized",
            node_id=node.node_id,
            artifact_count=len(self.digests),
            capacity_slots=node.capacity_slots,
            lease_limit=node.lease_limit,
            random_seed=node.random_seed,
        )

    def capacity(self) -> dict[str, int]:
        return {
            "used_slots": self.used_slots,
            "total_slots": self.node.capacity_slots,
            "free_slots": self.node.capacity_slots - self.used_slots,
            "active_leases": len(self.leases),
            "lease_limit": self.node.lease_limit,
        }

    def reap(self) -> list[str]:
        now = mono_ms()
        expired: list[str] = []
        with self.lock:
            for lease_id, lease in sorted(self.leases.items()):
                if lease["expires_mono_ms"] <= now:
                    self.used_slots -= len(lease["artifact_keys"])
                    del self.leases[lease_id]
                    expired.append(lease_id)
                    self.logger.info(
                        "node lease expired",
                        node_id=self.node.node_id,
                        lease_id=lease_id,
                        capacity=self.capacity(),
                    )
        return expired

    def handle(self, request: Any) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        self.reap()
        if not isinstance(request, dict):
            raise LeaseDenied(
                "E_PROTOCOL",
                "request must be a JSON object",
                field="request",
                expected="JSON object",
            )
        if request.get("protocol_version") != PROTOCOL_VERSION:
            raise LeaseDenied(
                "E_PROTOCOL",
                "unsupported protocol version",
                field="protocol_version",
                expected=str(PROTOCOL_VERSION),
                actual=request.get("protocol_version"),
            )
        operation = request.get("op")
        if operation == "ping":
            response = {"ok": True, "ack_version": PROTOCOL_VERSION}
        elif operation == "lease":
            response = self._lease(request)
        elif operation == "renew":
            response = self._renew(request)
        elif operation == "release":
            response = self._release(request)
        elif operation == "digests":
            response = self._digests()
        elif operation == "status":
            response = self._status()
        elif operation == "shutdown":
            response = {"ok": True, "shutdown": True}
            threading.Thread(target=self._delayed_stop, daemon=True).start()
        else:
            raise LeaseDenied(
                "E_PROTOCOL",
                "unknown operation",
                field="op",
                expected="ping|lease|renew|release|digests|status|shutdown",
                actual=operation,
            )
        response.update(
            {
                "node_id": self.node.node_id,
                "mono_ms": mono_ms(),
                "elapsed_mono_ms": (time.monotonic_ns() - started_ns) // 1_000_000,
            }
        )
        return response

    def _delayed_stop(self) -> None:
        time.sleep(0.05)
        self.stopping.set()

    def _lease(self, request: dict[str, Any]) -> dict[str, Any]:
        request_seq = request.get("request_seq")
        if not is_int(request_seq) or request_seq < 1:
            raise LeaseDenied(
                "E_PROTOCOL",
                "request_seq must be a positive integer",
                field="request_seq",
                expected="positive integer",
                actual=request_seq,
            )
        raw_keys = request.get("artifact_keys")
        if not isinstance(raw_keys, list) or not raw_keys:
            raise LeaseDenied(
                "E_PROTOCOL",
                "artifact_keys must be a non-empty array",
                field="artifact_keys",
                expected="non-empty string array",
                actual=raw_keys,
            )
        if not all(isinstance(key, str) for key in raw_keys):
            raise LeaseDenied(
                "E_PROTOCOL",
                "artifact_keys contains a non-string value",
                field="artifact_keys",
                expected="string array",
                actual=raw_keys,
            )
        keys = sorted(set(raw_keys))
        if len(keys) != len(raw_keys):
            raise LeaseDenied(
                "E_PROTOCOL",
                "artifact_keys contains duplicates",
                field="artifact_keys",
                expected="unique artifact keys",
                actual=raw_keys,
            )
        lease_ms = request.get("lease_ms")
        if not is_int(lease_ms) or lease_ms <= 0:
            raise LeaseDenied(
                "E_PROTOCOL",
                "lease_ms must be a positive integer",
                field="lease_ms",
                expected="positive integer",
                actual=lease_ms,
            )
        service_delay_ms = request.get("service_delay_ms", 0)
        if not is_int(service_delay_ms) or not 0 <= service_delay_ms <= 5000:
            raise LeaseDenied(
                "E_PROTOCOL",
                "service_delay_ms must be in [0, 5000]",
                field="service_delay_ms",
                expected="integer in [0, 5000]",
                actual=service_delay_ms,
            )
        missing = [key for key in keys if key not in self.digests]
        if missing:
            raise LeaseDenied(
                "E_ARTIFACT",
                "unknown artifact key",
                field="artifact_keys",
                expected="keys present in the pipeline specification",
                actual=missing,
            )
        if service_delay_ms:
            time.sleep(service_delay_ms / 1000)
            self.reap()

        with self.lock:
            cached = self.request_cache.get(request_seq)
            if cached is not None and cached.get("lease_id") in self.leases:
                reply = dict(cached)
                reply["capacity"] = self.capacity()
                reply.update(self.capacity())
                return reply
            if cached is not None:
                self.request_cache.pop(request_seq, None)
            if len(self.leases) >= self.node.lease_limit:
                raise LeaseDenied(
                    "E_LEASE_LIMIT",
                    "node lease limit reached; request must queue",
                    field="lease_limit",
                    expected=f"active leases < {self.node.lease_limit}",
                    actual=len(self.leases),
                )
            needed = len(keys)
            if self.used_slots + needed > self.node.capacity_slots:
                raise LeaseDenied(
                    "E_CAPACITY",
                    "node capacity exhausted; request must queue",
                    field="capacity_slots",
                    expected=f"free slots >= {needed}",
                    actual=self.node.capacity_slots - self.used_slots,
                )
            requested_lease_id = request.get("lease_id")
            if requested_lease_id is not None and (
                not isinstance(requested_lease_id, str) or not requested_lease_id
            ):
                raise LeaseDenied(
                    "E_PROTOCOL",
                    "lease_id must be a non-empty string when supplied",
                    field="lease_id",
                    expected="non-empty string",
                    actual=requested_lease_id,
                )
            lease_id = requested_lease_id or f"{self.node.node_id}:lease:{request_seq}"
            if lease_id in self.leases:
                raise LeaseDenied(
                    "E_PROTOCOL",
                    "lease identifier collision",
                    field="request_seq",
                    expected="globally unique request sequence",
                    actual=request_seq,
                )
            expires = mono_ms() + lease_ms
            self.used_slots += needed
            self.leases[lease_id] = {
                "artifact_keys": keys,
                "expires_mono_ms": expires,
            }
            reply = {
                "ok": True,
                "ack_version": PROTOCOL_VERSION,
                "request_seq": request_seq,
                "lease_id": lease_id,
                "artifact_keys": keys,
                "digests": {key: self.digests[key] for key in keys},
                "refs": {key: self.refs[key] for key in keys},
                "capacity": self.capacity(),
                "expires_mono_ms": expires,
            }
            self.request_cache[request_seq] = dict(reply)
            self.logger.info(
                "lease granted",
                node_id=self.node.node_id,
                lease_id=lease_id,
                request_seq=request_seq,
                artifact_keys=keys,
                expires_mono_ms=expires,
                capacity=self.capacity(),
            )
            return reply

    def _renew(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        lease_ms = request.get("lease_ms")
        if not isinstance(lease_id, str) or not lease_id:
            raise LeaseDenied(
                "E_PROTOCOL",
                "lease_id is required",
                field="lease_id",
                expected="non-empty string",
                actual=lease_id,
            )
        if not is_int(lease_ms) or lease_ms <= 0:
            raise LeaseDenied(
                "E_PROTOCOL",
                "lease_ms must be a positive integer",
                field="lease_ms",
                expected="positive integer",
                actual=lease_ms,
            )
        with self.lock:
            lease = self.leases.get(lease_id)
            if lease is None:
                raise LeaseDenied(
                    "E_LEASE_UNKNOWN",
                    "active lease not found",
                    field="lease_id",
                    expected="active lease identifier",
                    actual=lease_id,
                )
            lease["expires_mono_ms"] = mono_ms() + lease_ms
            return {
                "ok": True,
                "lease_id": lease_id,
                "expires_mono_ms": lease["expires_mono_ms"],
            }

    def _release(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        if not isinstance(lease_id, str) or not lease_id:
            raise LeaseDenied(
                "E_PROTOCOL",
                "lease_id is required",
                field="lease_id",
                expected="non-empty string",
                actual=lease_id,
            )
        with self.lock:
            lease = self.leases.pop(lease_id, None)
            if lease is None:
                return {"ok": True, "lease_id": lease_id, "released": False}
            self.used_slots -= len(lease["artifact_keys"])
            return {
                "ok": True,
                "lease_id": lease_id,
                "released": True,
                "capacity": self.capacity(),
            }

    def _digests(self) -> dict[str, Any]:
        return {
            "ok": True,
            "node_id": self.node.node_id,
            "random_seed": self.node.random_seed,
            "digests": dict(sorted(self.digests.items())),
            "refs": {key: list(self.refs[key]) for key in sorted(self.refs)},
            "manifest": self.manifest,
        }

    def _status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "node_id": self.node.node_id,
            "capacity": self.capacity(),
            "active_leases": sorted(self.leases),
        }


class RequestHandler(socketserver.StreamRequestHandler):
    server: "LeaseTCPServer"

    def handle(self) -> None:
        raw = self.rfile.readline(MAX_LINE_BYTES + 1)
        if len(raw) > MAX_LINE_BYTES:
            response = {
                "ok": False,
                "code": "E_PROTOCOL",
                "message": "request line too large",
                "field": "request",
                "expected": f"<= {MAX_LINE_BYTES} bytes",
            }
        else:
            try:
                request = json.loads(raw.decode("utf-8"))
                response = self.server.runtime.handle(request)
            except LeaseDenied as exc:
                self.server.runtime.logger.warning(
                    "request denied",
                    code=exc.code,
                    field=exc.field,
                    node_id=self.server.runtime.node.node_id,
                    expected=exc.expected,
                    actual=exc.actual,
                )
                response = exc.response(self.server.runtime.node.node_id)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                response = {
                    "ok": False,
                    "code": "E_PROTOCOL",
                    "message": f"request is not UTF-8 JSON: {exc}",
                    "field": "request",
                    "expected": "one UTF-8 JSON object line",
                    "node_id": self.server.runtime.node.node_id,
                }
            except Exception as exc:  # pragma: no cover - defensive process boundary
                self.server.runtime.logger.error(
                    "E_NODE",
                    "unhandled node request failure",
                    node_id=self.server.runtime.node.node_id,
                    error=repr(exc),
                )
                response = {
                    "ok": False,
                    "code": "E_NODE",
                    "message": repr(exc),
                    "field": "request",
                    "expected": "successful node operation",
                }
        encoded = json.dumps(
            response, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.wfile.write(encoded + b"\n")
        self.wfile.flush()


class LeaseTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = os.name != "nt"
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: NodeRuntime) -> None:
        self.runtime = runtime
        super().__init__(address, RequestHandler)


def _parent_alive_windows(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    synchronize = 0x00100000
    process_query_limited_information = 0x1000
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(
        synchronize | process_query_limited_information, False, int(pid)
    )
    if not handle:
        return False
    try:
        status = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(status)):
            return False
        return status.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def parent_is_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    if os.name == "nt":
        try:
            return _parent_alive_windows(pid)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def watchdog_parent(runtime: NodeRuntime, parent_pid: int) -> None:
    while not runtime.stopping.wait(0.1):
        if not parent_is_alive(parent_pid):
            runtime.logger.warning(
                "parent process disappeared; node exiting", node_id=runtime.node.node_id
            )
            runtime.stopping.set()
            return


def reaper_loop(runtime: NodeRuntime) -> None:
    while not runtime.stopping.wait(0.02):
        runtime.reap()


def run_node(
    spec_path: str,
    node_id: str,
    workspace: str,
    log_path: str,
    parent_pid: int = 0,
) -> int:
    logger = StructuredLogger(log_path, f"node:{node_id}")
    try:
        spec = load_spec(spec_path)
        node = spec.node_by_id.get(node_id)
        if node is None:
            raise LeaseError(
                "E_SPEC",
                "node identifier is not present in the specification",
                field="nodes[].id",
                expected=f"one node with id {node_id}",
                actual=node_id,
            )
        runtime = NodeRuntime(spec, node, Path(workspace), logger)
    except LeaseError as exc:
        logger.error(exc.code, exc.message, **exc.log_fields())
        return 2

    try:
        server = LeaseTCPServer((node.host, node.port), runtime)
    except OSError as exc:
        logger.error(
            "E_PORT",
            "cannot bind loopback lease port",
            node_id=node.node_id,
            host=node.host,
            port=node.port,
            error=str(exc),
        )
        return 40

    threads = [threading.Thread(target=reaper_loop, args=(runtime,), daemon=True)]
    if parent_pid:
        threads.append(
            threading.Thread(
                target=watchdog_parent, args=(runtime, parent_pid), daemon=True
            )
        )
    for thread in threads:
        thread.start()

    def stop(_signum: int, _frame: Any) -> None:
        runtime.stopping.set()

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, stop)

    serving = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    serving.start()
    logger.info(
        "node listening",
        node_id=node.node_id,
        host=node.host,
        port=node.port,
        parent_pid=parent_pid,
    )
    try:
        while not runtime.stopping.is_set():
            time.sleep(0.02)
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)
        logger.info("node stopped", node_id=node.node_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one loopback artifact cache node")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args(argv)
    return run_node(
        spec_path=args.spec,
        node_id=args.node_id,
        workspace=args.workspace,
        log_path=args.log,
        parent_pid=args.parent_pid,
    )


if __name__ == "__main__":
    raise SystemExit(main())
