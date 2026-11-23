from __future__ import annotations

import argparse
import json
import os
import signal
import socketserver
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .loaders import load_backends_file
from .models import SUPPORTED_QUANTIZATION, Backend
from .util import canonical_json, digest_value, sha256_hex

MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_BATCH = 4096
MAX_DIMENSION = 4096


@dataclass(frozen=True)
class ComputeParams:
    model_dim: int
    hidden_dim: int
    seed: int
    quantization: str

    @classmethod
    def from_payload(cls, payload: Any) -> "ComputeParams":
        if not isinstance(payload, dict):
            raise ValueError("params must be an object")
        model_dim = payload.get("model_dim")
        hidden_dim = payload.get("hidden_dim")
        seed = payload.get("seed")
        quantization = payload.get("quantization")
        for name, value in (("model_dim", model_dim), ("hidden_dim", hidden_dim), ("seed", seed)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"params.{name} must be an integer")
        if not (1 <= model_dim <= MAX_DIMENSION):
            raise ValueError(f"params.model_dim must be in 1..{MAX_DIMENSION}")
        if not (1 <= hidden_dim <= MAX_DIMENSION):
            raise ValueError(f"params.hidden_dim must be in 1..{MAX_DIMENSION}")
        if not (0 <= seed <= 2**63 - 1):
            raise ValueError("params.seed must be a non-negative integer")
        if not isinstance(quantization, str) or quantization.lower() not in SUPPORTED_QUANTIZATION:
            raise ValueError("params.quantization must be fp32, fp16 or int8")
        return cls(model_dim, hidden_dim, seed, quantization.lower())


def _quantize(array: np.ndarray, quantization: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if quantization == "fp32":
        return value.astype(np.float32).astype(np.float64)
    if quantization == "fp16":
        return value.astype(np.float16).astype(np.float64)
    if quantization == "int8":
        maximum = float(np.max(np.abs(value))) if value.size else 0.0
        if maximum == 0.0:
            return np.zeros_like(value, dtype=np.float64)
        scale = maximum / 127.0
        return np.clip(np.rint(value / scale), -127, 127).astype(np.float64) * scale
    raise ValueError(f"unsupported quantization {quantization!r}")


def deterministic_forward(params: ComputeParams, vectors: list[list[Any]]) -> str:
    """Return a platform-stable digest of deterministic NumPy forward outputs."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != params.model_dim:
        raise ValueError("vector dimension does not match model_dim")
    if matrix.shape[0] > MAX_BATCH:
        raise ValueError(f"batch exceeds {MAX_BATCH} requests")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must contain finite numbers")

    rng = np.random.default_rng(params.seed)
    w1 = _quantize(rng.standard_normal((params.model_dim, params.hidden_dim)), params.quantization)
    b1 = _quantize(rng.standard_normal((params.hidden_dim,)), params.quantization)
    w2 = _quantize(rng.standard_normal((params.hidden_dim,)), params.quantization)
    b2 = _quantize(rng.standard_normal((1,)), params.quantization)[0]
    hidden = np.tanh(matrix @ w1 + b1)
    if params.quantization == "fp32":
        hidden = hidden.astype(np.float32).astype(np.float64)
    elif params.quantization == "fp16":
        hidden = hidden.astype(np.float16).astype(np.float64)
    elif params.quantization == "int8":
        hidden = _quantize(hidden, "int8")
    output = hidden @ w2 + b2
    rendered = ["%.12e" % float(value) for value in output.reshape(-1)]
    material = canonical_json(
        {
            "engine": "numpy-forward.v1",
            "model_dim": params.model_dim,
            "hidden_dim": params.hidden_dim,
            "seed": params.seed,
            "quantization": params.quantization,
            "outputs": rendered,
        }
    )
    return "sha256:" + sha256_hex(material)


def _error(message: str, field: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "code": "E_BACKEND", "message": message}
    if field:
        payload["field"] = field
    return payload


def handle_request(payload: Any, *, declared: dict[str, Backend], allow_dynamic: bool) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _error("request must be a JSON object", "request")
    op = payload.get("op", "infer")
    if op == "ping":
        return {
            "ok": True,
            "service": "deterministic-mock-inference",
            "pid": os.getpid(),
            "mono_ns": time.monotonic_ns(),
        }
    if op != "infer":
        return _error("unsupported operation", "op")
    backend_id = payload.get("backend_id")
    if not isinstance(backend_id, str) or not backend_id:
        return _error("backend_id must be a non-empty string", "backend_id")
    if backend_id not in declared and not allow_dynamic:
        return _error("backend_id is not declared on this mock server", "backend_id")
    try:
        params = ComputeParams.from_payload(payload.get("params"))
    except ValueError as exc:
        return _error(str(exc), "params")
    if not allow_dynamic and backend_id in declared:
        backend = declared[backend_id]
        expected = (backend.model_dim, backend.hidden_dim, backend.seed, backend.quantization)
        actual = (params.model_dim, params.hidden_dim, params.seed, params.quantization)
        if expected != actual:
            return _error("request parameters differ from the declared backend", "params")
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        return _error("requests must be a non-empty array", "requests")
    if len(requests) > MAX_BATCH:
        return _error(f"requests cannot exceed {MAX_BATCH}", "requests")
    vectors: list[list[Any]] = []
    request_ids: list[str] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            return _error("request entry must be an object", f"requests[{index}]")
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return _error("request_id must be a non-empty string", f"requests[{index}].request_id")
        vector = request.get("vector")
        if not isinstance(vector, list) or len(vector) != params.model_dim:
            return _error("vector length must equal params.model_dim", f"requests[{index}].vector")
        vectors.append(vector)
        request_ids.append(request_id)
    try:
        output_digest = deterministic_forward(params, vectors)
    except (ValueError, FloatingPointError) as exc:
        return _error(str(exc), "requests")
    return {
        "ok": True,
        "backend_id": backend_id,
        "request_count": len(request_ids),
        "request_ids": request_ids,
        "output_digest": output_digest,
        "engine": "numpy-forward.v1",
        "param_digest": digest_value({"id": backend_id, **params.__dict__}),
    }


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _make_handler(declared: dict[str, Backend], allow_dynamic: bool):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            while True:
                line = self.rfile.readline(MAX_LINE_BYTES + 1)
                if not line:
                    return
                if len(line) > MAX_LINE_BYTES:
                    response = _error("request line is too large", "request")
                else:
                    try:
                        payload = json.loads(line.decode("utf-8"))
                        response = handle_request(
                            payload, declared=declared, allow_dynamic=allow_dynamic
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        response = _error(f"invalid JSON line: {exc}", "request")
                    except Exception as exc:  # keep protocol errors machine readable
                        response = _error(f"internal inference failure: {type(exc).__name__}", "inference")
                self.wfile.write((canonical_json(response) + "\n").encode("utf-8"))
                self.wfile.flush()

    return Handler


class MockBackendServer:
    """Starts one loopback JSONL listener per declared backend port."""

    def __init__(
        self,
        backends: list[Backend],
        *,
        allow_dynamic: bool = False,
        log_path: str | Path | None = None,
    ) -> None:
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends = {backend.id: backend for backend in backends}
        self.allow_dynamic = allow_dynamic
        self.log_path = Path(log_path) if log_path else None
        self._servers: list[_ThreadingTCPServer] = []
        self._threads: list[threading.Thread] = []

    @classmethod
    def from_config(cls, path: str | Path, *, allow_dynamic: bool = False) -> "MockBackendServer":
        return cls(load_backends_file(path), allow_dynamic=allow_dynamic)

    def start(self) -> "MockBackendServer":
        seen_ports: set[int] = set()
        handler = _make_handler(self.backends, self.allow_dynamic)
        for backend in self.backends.values():
            if backend.host != "127.0.0.1":
                raise ValueError("mock backend refuses non-loopback host")
            if backend.port in seen_ports:
                raise ValueError(f"duplicate listener port {backend.port}")
            seen_ports.add(backend.port)
            server = _ThreadingTCPServer((backend.host, backend.port), handler)
            thread = threading.Thread(
                target=server.serve_forever,
                name=f"mock-backend-{backend.id}-{backend.port}",
                daemon=True,
            )
            thread.start()
            self._servers.append(server)
            self._threads.append(thread)
        self._log("mock_backends_started", ports=sorted(seen_ports))
        return self

    def stop(self) -> None:
        for server in self._servers:
            server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join(timeout=2)
        self._servers.clear()
        self._threads.clear()
        self._log("mock_backends_stopped")

    def __enter__(self) -> "MockBackendServer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _log(self, event: str, **details: Any) -> None:
        if not self.log_path:
            return
        record = {
            "mono_ns": time.monotonic_ns(),
            "level": "INFO",
            "component": "mock-backend",
            "event": event,
            **details,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        synchronize = 0x00100000
        wait_timeout = 0x00000102
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic loopback inference mock")
    parser.add_argument("--config", required=True, help="JSON file containing backends")
    parser.add_argument("--allow-dynamic", action="store_true", help="use params supplied by each request")
    parser.add_argument("--log", help="optional JSONL server log")
    parser.add_argument("--parent-pid", type=int, help="exit if this parent process disappears")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        server = MockBackendServer.from_config(args.config, allow_dynamic=args.allow_dynamic)
        server.start()
    except Exception as exc:
        print(json.dumps({"ok": False, "code": "E_BACKEND", "message": str(exc)}), flush=True)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "event": "ready",
                "ports": sorted({backend.port for backend in server.backends.values()}),
            }
        ),
        flush=True,
    )
    stop_event = threading.Event()

    def request_stop(signum=None, frame=None):
        stop_event.set()

    if args.parent_pid:
        parent_pid = int(args.parent_pid)

        def watch_parent() -> None:
            while not stop_event.wait(0.5):
                if not _process_alive(parent_pid):
                    stop_event.set()

        threading.Thread(target=watch_parent, name="parent-watchdog", daemon=True).start()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stop_event.wait(0.5):
            pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
