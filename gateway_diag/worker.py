"""Loopback JSON-lines inference worker launched as a subprocess."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from typing import Any

from .model import canonical_json
from .parameters import (
    batch_output_digest,
    generate_parameters,
    infer_request,
)


def _json_line(value: dict[str, Any]) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _error_payload(code: str, message: str, backend_id: str, port: int) -> dict[str, Any]:
    return {
        "type": "error",
        "code": code,
        "message": message,
        "backend_id": backend_id,
        "port": port,
    }


def _load_items(message: dict[str, Any], backend_id: str, port: int) -> list[tuple[int, list[float]]]:
    items = message.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("items must be a non-empty array")
    parsed: list[tuple[int, list[float]]] = []
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each item must be an object")
        request_id = item.get("request_id")
        if isinstance(request_id, bool) or not isinstance(request_id, int):
            raise ValueError("request_id must be an integer")
        if request_id in seen:
            raise ValueError("request_id values must be unique inside a batch")
        seen.add(request_id)
        vector = item.get("input")
        if not isinstance(vector, list):
            raise ValueError("input must be an array")
        try:
            converted = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError("input values must be numeric") from exc
        parsed.append((request_id, converted))
    return parsed


def _make_receipt(
    message: dict[str, Any],
    params: Any,
    backend_id: str,
    port: int,
) -> dict[str, Any]:
    batch_id = message.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("batch_id must be a non-empty string")
    items = _load_items(message, backend_id, port)
    delay_ms = int(os.environ.get("GATEWAY_DIAG_TEST_BATCH_DELAY_MS", "0") or "0")
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
    started = time.monotonic_ns()
    digests: dict[int, str] = {}
    for request_id, vector in items:
        _, digest = infer_request(params, request_id, vector)
        digests[request_id] = digest
    elapsed_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
    return {
        "type": "receipt",
        "protocol_version": 1,
        "backend_id": backend_id,
        "port": port,
        "parameter_digest": params.parameter_digest,
        "batch_id": batch_id,
        "request_ids": [request_id for request_id, _ in items],
        "output_digests": {
            str(request_id): digests[request_id] for request_id in sorted(digests)
        },
        "output_digest": batch_output_digest(backend_id, batch_id, digests),
        "elapsed_ms": elapsed_ms,
        "worker_pid": os.getpid(),
    }


def _serve_connection(
    connection: socket.socket,
    params: Any,
    backend_id: str,
    port: int,
) -> None:
    with connection:
        connection.settimeout(10.0)
        stream = connection.makefile("rwb")
        try:
            raw = stream.readline(10_000_001)
            if not raw or len(raw) > 10_000_000:
                raise ValueError("request line is missing or too large")
            message = json.loads(raw.decode("utf-8"))
            if not isinstance(message, dict):
                raise ValueError("request must be a JSON object")
            kind = message.get("type")
            if kind == "hello":
                response = {
                    "type": "hello_ack",
                    "protocol_version": 1,
                    "backend_id": backend_id,
                    "port": port,
                    "parameter_digest": params.parameter_digest,
                }
            elif kind == "infer":
                if message.get("backend_id") != backend_id:
                    raise ValueError("receipt backend_id mismatch")
                response = _make_receipt(message, params, backend_id, port)
            else:
                raise ValueError("unsupported message type")
            stream.write(_json_line(response))
            stream.flush()
            try:
                stream.read(1)
            except socket.timeout:
                pass
        except Exception as exc:
            error = _error_payload("E_RECEIPT", str(exc), backend_id, port)
            try:
                stream.write(_json_line(error))
                stream.flush()
            except OSError:
                pass


def _stdin_watchdog(stop: threading.Event, server: socket.socket) -> None:
    try:
        while not stop.is_set():
            line = sys.stdin.buffer.readline()
            if not line or line.strip().lower() == b"shutdown":
                break
    finally:
        stop.set()
        try:
            server.close()
        except OSError:
            pass


def _port_accepts_connections(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.25)
    try:
        probe.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _create_listener(host: str, port: int) -> socket.socket:
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        exclusive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        exclusive.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            exclusive.bind((host, port))
            exclusive.listen(32)
            return exclusive
        except OSError as first_error:
            exclusive.close()
            if _port_accepts_connections(host, port):
                raise first_error
            reusable = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reusable.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                reusable.bind((host, port))
                reusable.listen(32)
                return reusable
            except OSError:
                reusable.close()
                raise first_error
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((host, port))
        listener.listen(32)
        return listener
    except OSError:
        listener.close()
        raise


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
    parser = argparse.ArgumentParser(prog="gateway_diag.worker")
    parser.add_argument("--backend-id", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--model-dim", required=True, type=int)
    parser.add_argument("--hidden-dim", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--quantization", required=True)
    args = parser.parse_args(argv)

    if args.host != "127.0.0.1":
        print(
            canonical_json(
                _error_payload(
                    "E_REJECTED_BIND_HOST",
                    "worker may only bind to 127.0.0.1",
                    args.backend_id,
                    args.port,
                )
            ),
            flush=True,
        )
        return 2

    backend = type(
        "WorkerBackend",
        (),
        {
            "id": args.backend_id,
            "model_dim": args.model_dim,
            "hidden_dim": args.hidden_dim,
            "seed": args.seed,
            "quantization": args.quantization,
        },
    )()
    params = generate_parameters(backend)  # type: ignore[arg-type]

    try:
        server = _create_listener(args.host, args.port)
    except OSError as exc:
        print(
            canonical_json(
                _error_payload("E_PORT", str(exc), args.backend_id, args.port)
            ),
            flush=True,
        )
        return 3

    stop = threading.Event()
    watchdog = threading.Thread(
        target=_stdin_watchdog, args=(stop, server), name="worker-stdin", daemon=True
    )
    watchdog.start()

    def handle_signal(_signum: int, _frame: Any) -> None:
        stop.set()
        try:
            server.close()
        except OSError:
            pass

    for signal_name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, signal_name):
            try:
                signal.signal(getattr(signal, signal_name), handle_signal)
            except (OSError, ValueError):
                pass

    print(
        canonical_json(
            {
                "type": "ready",
                "protocol_version": 1,
                "backend_id": args.backend_id,
                "port": args.port,
                "parameter_digest": params.parameter_digest,
                "pid": os.getpid(),
            }
        ),
        flush=True,
    )
    server.settimeout(0.25)
    try:
        while not stop.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            _serve_connection(connection, params, args.backend_id, args.port)
    finally:
        stop.set()
        try:
            server.close()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
