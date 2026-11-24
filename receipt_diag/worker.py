"""Loopback TCP worker subprocess.

The worker only performs deterministic NumPy inference. It never sleeps to
simulate work and it never accepts a non-loopback bind address.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import socketserver
import sys
import threading
import time
from typing import Any

import numpy as np

from .canonical import canonical_json
from .parameters import (
    batch_output_digest,
    forward,
    generate_parameters,
    parameter_digest,
    vector_digest,
)


MAX_MESSAGE_BYTES = 8 * 1024 * 1024


def _parent_alive(pid: int) -> bool:
    if os.name == "nt":
        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(process, 0) == 0x102
        finally:
            ctypes.windll.kernel32.CloseHandle(process)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64


class _Handler(socketserver.BaseRequestHandler):
    def _read_message(self) -> dict[str, Any]:
        chunks = bytearray()
        while True:
            chunk = self.request.recv(65536)
            if not chunk:
                break
            chunks.extend(chunk)
            if len(chunks) > MAX_MESSAGE_BYTES:
                raise ValueError("request exceeds maximum message size")
            if b"\n" in chunk:
                break
        if not chunks:
            raise ValueError("empty request")
        line = bytes(chunks).split(b"\n", 1)[0]
        document = json.loads(line.decode("utf-8"))
        if not isinstance(document, dict):
            raise ValueError("request must be a JSON object")
        return document

    def handle(self) -> None:
        server: "_WorkerTCPServer" = self.server  # type: ignore[assignment]
        local_host, local_port = self.request.getsockname()[:2]
        peer_host, peer_port = self.request.getpeername()[:2]
        try:
            self.request.settimeout(server.io_timeout_seconds)
            request = self._read_message()
            if request.get("op") == "ping":
                response = {
                    "ok": True,
                    "worker_id": server.worker_id,
                    "pid": os.getpid(),
                    "host": local_host,
                    "port": local_port,
                    "peer": {"host": peer_host, "port": peer_port},
                }
            elif request.get("op") == "forward":
                response = server.forward(request)
                response.update(
                    {
                        "ok": True,
                        "worker_id": server.worker_id,
                        "pid": os.getpid(),
                        "host": local_host,
                        "port": local_port,
                        "peer": {"host": peer_host, "port": peer_port},
                    }
                )
            else:
                response = {"ok": False, "code": "E_FIELD", "detail": "unknown operation"}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            response = {"ok": False, "code": "E_FIELD", "detail": str(exc)}
        except (socket.timeout, TimeoutError) as exc:
            response = {
                "ok": False,
                "code": "E_WORKER",
                "detail": f"server socket timeout: {exc}",
            }
        except Exception as exc:  # pragma: no cover - defensive protocol boundary
            response = {"ok": False, "code": "E_WORKER", "detail": repr(exc)}
        try:
            encoded = canonical_json(response).encode("utf-8") + b"\n"
            self.request.sendall(encoded)
        except OSError:
            pass


class _WorkerTCPServer(_ThreadingServer):
    def __init__(
        self,
        address: tuple[str, int],
        *,
        worker_id: str,
        seed: int,
        param_version: int,
        model_dim: int,
        hidden_dim: int,
        io_timeout_seconds: float,
    ) -> None:
        self.worker_id = worker_id
        self.seed = seed
        self.param_version = param_version
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.io_timeout_seconds = io_timeout_seconds
        self.parameters = generate_parameters(
            seed=seed,
            param_version=param_version,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
        )
        from .specs import WorkerSpec

        self.identity = WorkerSpec(
            worker_id=worker_id,
            host=address[0],
            port=address[1],
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            seed=seed,
            param_version=param_version,
            batch_size=1,
            concurrency=1,
        )
        self.param_digest = parameter_digest(self.identity, self.parameters)
        super().__init__(address, _Handler)

    def forward(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("worker_id") != self.worker_id:
            raise ValueError("worker_id does not match this process")
        if request.get("param_version") != self.param_version:
            raise ValueError("param_version does not match this process")
        if request.get("seed") != self.seed:
            raise ValueError("seed does not match this process")
        if request.get("model_dim") != self.model_dim:
            raise ValueError("model_dim does not match this process")
        if request.get("hidden_dim") != self.hidden_dim:
            raise ValueError("hidden_dim does not match this process")
        raw_requests = request.get("requests")
        if not isinstance(raw_requests, list) or not raw_requests:
            raise ValueError("requests must be a non-empty array")
        ordered = sorted(
            raw_requests,
            key=lambda item: (item["arrival_seq"], item["request_id"]),
        )
        request_ids = [item["request_id"] for item in ordered]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("duplicate request_id in forward batch")
        matrix = np.asarray([item["input"] for item in ordered], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.model_dim:
            raise ValueError(
                f"input matrix dimension {matrix.shape} does not match model_dim {self.model_dim}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("input contains non-finite values")
        started = time.monotonic_ns()
        output = forward(self.parameters, matrix)
        duration_ms = (time.monotonic_ns() - started) // 1_000_000
        outputs = []
        for request_id, vector in zip(request_ids, output):
            outputs.append(
                {
                    "request_id": request_id,
                    "vector_digest": vector_digest(request_id, vector),
                }
            )
        digest = batch_output_digest(
            batch_id=str(request["batch_id"]),
            request_ids=request_ids,
            param_version=self.param_version,
            param_digest=self.param_digest,
            vector_digests=[
                (item["request_id"], item["vector_digest"]) for item in outputs
            ],
        )
        return {
            "param_digest": self.param_digest,
            "output_digest": digest,
            "duration_ms": int(duration_ms),
            "outputs": outputs,
            "transport": "tcp-loopback",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Loopback deterministic worker")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--param-version", required=True, type=int)
    parser.add_argument("--model-dim", required=True, type=int)
    parser.add_argument("--hidden-dim", required=True, type=int)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--io-timeout-ms", type=int, default=30000)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        parser.error("host must be the literal 127.0.0.1")
    if not (0 < args.port <= 65535):
        parser.error("port must be in [1, 65535]")

    server = _WorkerTCPServer(
        (args.host, args.port),
        worker_id=args.worker_id,
        seed=args.seed,
        param_version=args.param_version,
        model_dim=args.model_dim,
        hidden_dim=args.hidden_dim,
        io_timeout_seconds=max(args.io_timeout_ms, 1) / 1000.0,
    )
    stop = threading.Event()

    def monitor_parent() -> None:
        while not stop.wait(0.25):
            if not _parent_alive(args.parent_pid):
                server.shutdown()
                return

    monitor = threading.Thread(target=monitor_parent, name="parent-monitor", daemon=True)
    monitor.start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
