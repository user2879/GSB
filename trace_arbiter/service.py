from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import socket
import socketserver
import sys
import threading
import time
from typing import Any

from .adjudicate import authoritative_summary, batch_decision_summary
from .canonical import canonical_bytes, canonical_text, digest_canonical
from .errors import ArbiterError


MAX_REQUEST_BYTES = 16 * 1024 * 1024
PROTOCOL_VERSION = 1


def _response_bytes(response: dict[str, Any]) -> bytes:
    return canonical_bytes(response) + b"\n"


def _error_response(exc: ArbiterError, *, elapsed_us: int) -> dict[str, Any]:
    response = exc.as_dict()
    response.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "processing_us": max(0, elapsed_us),
        }
    )
    return response


def _parent_is_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        process = kernel32.OpenProcess(0x00100000, False, pid)
        if not process:
            return False
        try:
            return kernel32.WaitForSingleObject(process, 0) != 0
        finally:
            kernel32.CloseHandle(process)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _watch_parent(parent_pid: int, stop_event: threading.Event) -> None:
    while not stop_event.wait(0.2):
        if not _parent_is_alive(parent_pid):
            os._exit(0)


class _ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = os.name != "nt"
    daemon_threads = True

    def server_bind(self) -> None:
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            line = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        except (OSError, ValueError):
            return
        if not line:
            return
        if len(line) > MAX_REQUEST_BYTES:
            response = _error_response(
                ArbiterError("E_PROTOCOL", "request exceeds maximum line size"),
                elapsed_us=0,
            )
            self.wfile.write(_response_bytes(response))
            return

        started_ns = time.perf_counter_ns()
        try:
            try:
                request = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArbiterError("E_PROTOCOL", f"invalid request JSON: {exc}") from exc
            response = self._dispatch(request, started_ns)
        except ArbiterError as exc:
            response = _error_response(exc, elapsed_us=(time.perf_counter_ns() - started_ns) // 1000)
        except Exception as exc:  # defensive worker boundary
            response = _error_response(
                ArbiterError("E_PROTOCOL", f"worker failure: {type(exc).__name__}: {exc}"),
                elapsed_us=(time.perf_counter_ns() - started_ns) // 1000,
            )
        try:
            self.wfile.write(_response_bytes(response))
            self.wfile.flush()
        except OSError:
            pass

    @staticmethod
    def _dispatch(request: dict[str, Any], started_ns: int) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ArbiterError("E_PROTOCOL", "request must be a JSON object")
        if request.get("protocol_version") != PROTOCOL_VERSION:
            raise ArbiterError("E_PROTOCOL", "unsupported protocol_version")
        delay_ms = request.get("simulate_delay_ms", 0)
        if not isinstance(delay_ms, int) or isinstance(delay_ms, bool) or not (0 <= delay_ms <= 60_000):
            raise ArbiterError("E_PROTOCOL", "simulate_delay_ms must be an integer in [0,60000]")
        if delay_ms:
            time.sleep(delay_ms / 1000)

        mode = request.get("mode")
        nodes = request.get("nodes")
        spans = request.get("spans")
        ruleset_version = request.get("ruleset_version")
        if not isinstance(nodes, list) or not isinstance(spans, list):
            raise ArbiterError("E_PROTOCOL", "nodes and spans must be arrays")
        if not isinstance(ruleset_version, int) or isinstance(ruleset_version, bool):
            raise ArbiterError("E_PROTOCOL", "ruleset_version must be an integer")

        batch_id = digest_canonical(request)
        if mode == "finalize":
            summary, digest, payload = authoritative_summary(
                nodes,
                spans,
                ruleset_version=ruleset_version,
            )
            processing_us = (time.perf_counter_ns() - started_ns) // 1000
            return {
                "batch_id": batch_id,
                "decision_digest": digest,
                "decision_summary": summary,
                "mode": "finalize",
                "node_id": request.get("worker_node_id"),
                "ok": True,
                "processing_us": max(0, processing_us),
                "protocol_version": PROTOCOL_VERSION,
                "trace_count": payload["trace_count"],
                "trace_results": payload["trace_decisions"],
            }
        if mode == "batch":
            target_node_id = request.get("target_node_id")
            target_span_ids = request.get("target_span_ids")
            target_spans = request.get("target_spans")
            if not isinstance(target_node_id, str) or not target_node_id:
                raise ArbiterError("E_PROTOCOL", "target_node_id must be a non-empty string")
            if target_spans is not None and (
                not isinstance(target_spans, list)
                or not all(isinstance(item, dict) for item in target_spans)
            ):
                raise ArbiterError("E_PROTOCOL", "target_spans must be an object array")
            if target_span_ids is not None and (
                not isinstance(target_span_ids, list)
                or not all(isinstance(item, str) for item in target_span_ids)
            ):
                raise ArbiterError("E_PROTOCOL", "target_span_ids must be a string array")
            if target_spans is None and target_span_ids is None:
                raise ArbiterError("E_PROTOCOL", "batch request needs target_spans")
            summary, digest, payload = batch_decision_summary(
                nodes,
                spans,
                ruleset_version=ruleset_version,
                target_node_id=target_node_id,
                target_span_ids=target_span_ids,
                target_spans=target_spans,
            )
            processing_us = (time.perf_counter_ns() - started_ns) // 1000
            return {
                "batch_id": batch_id,
                "decision_digest": digest,
                "decision_summary": summary,
                "mode": "batch",
                "node_id": target_node_id,
                "ok": True,
                "processing_us": max(0, processing_us),
                "protocol_version": PROTOCOL_VERSION,
                "trace_context": payload["trace_context"],
            }
        raise ArbiterError("E_PROTOCOL", "mode must be 'batch' or 'finalize'")


def _emit_startup_error(code: str, message: str, **details: Any) -> None:
    payload = {"type": "error", "code": code, "message": message, "ok": False}
    if details:
        payload["details"] = details
    sys.stdout.write(canonical_text(payload) + "\n")
    sys.stdout.flush()


def serve(node: dict[str, Any], *, startup_delay_ms: int, parent_pid: int) -> int:
    host = node.get("host")
    port = node.get("port")
    if host != "127.0.0.1" or not isinstance(port, int):
        _emit_startup_error("E_SPEC", "worker node must bind literal 127.0.0.1 with an integer port")
        return 2

    server: _ThreadingServer
    try:
        server = _ThreadingServer((host, port), _Handler)
    except OSError as exc:
        _emit_startup_error("E_PORT", f"cannot bind {host}:{port}: {exc}", errno=exc.errno)
        return 3

    stop_event = threading.Event()
    if parent_pid > 0:
        threading.Thread(
            target=_watch_parent,
            args=(parent_pid, stop_event),
            name="parent-watchdog",
            daemon=True,
        ).start()

    if startup_delay_ms:
        time.sleep(startup_delay_ms / 1000)

    ready = {
        "type": "ready",
        "ok": True,
        "node_id": node.get("node_id"),
        "host": host,
        "port": port,
        "protocol_version": PROTOCOL_VERSION,
        "pid": os.getpid(),
    }
    sys.stdout.write(canonical_text(ready) + "\n")
    sys.stdout.flush()

    def stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
    except (AttributeError, ValueError):
        pass

    server.timeout = 0.2
    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic trace aggregation worker")
    parser.add_argument("--node-json", required=True)
    parser.add_argument("--startup-delay-ms", type=int, default=0)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        node = json.loads(args.node_json)
    except json.JSONDecodeError:
        _emit_startup_error("E_SPEC", "--node-json is not valid JSON")
        return 2
    return serve(
        node,
        startup_delay_ms=max(0, args.startup_delay_ms),
        parent_pid=max(0, args.parent_pid),
    )


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
