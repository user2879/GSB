from __future__ import annotations

import argparse
import json
import socket
import socketserver
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .fixedpoint import deterministic_forward

DEFAULT_SEED = 20261125
PORT_MIN = 20_000
PORT_MAX = 39_999


def find_free_loopback_port(
    port_min: int = PORT_MIN,
    port_max: int = PORT_MAX,
    *,
    preferred: int | None = None,
    max_scan: int = 500,
) -> int:
    candidates: list[int] = []
    if preferred and port_min <= preferred <= port_max:
        candidates.append(preferred)
    for offset in range(max_scan):
        port = port_min + offset
        if port not in candidates:
            candidates.append(port)
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free loopback port found in [{port_min},{port_max}]")


class DeterministicWorkerHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                request = json.loads(line.decode("utf-8"))
                response = self.server.backend.infer(request)  # type: ignore[attr-defined]
            except Exception as exc:  # keep subprocess alive for protocol errors
                response = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
            payload = (json.dumps(response, sort_keys=True) + "\n").encode("utf-8")
            self.wfile.write(payload)
            self.wfile.flush()


class WorkerBackend:
    def __init__(self, worker_id: str, seed: int, model_dim: int = 16, hidden_dim: int = 32):
        self.worker_id = worker_id
        self.seed = seed
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter_ns()
        requests = request.get("requests")
        if requests is None:
            requests = [
                {
                    "request_id": request["request_id"],
                    "request_number": request.get("request_number", 1),
                }
            ]
        digests: dict[str, str] = {}
        vectors: dict[str, list[int]] = {}
        mac_total = 0
        param_count = 0
        for item in requests:
            number = int(item["request_number"])
            result = deterministic_forward(
                seed=int(request.get("seed", self.seed)),
                arrival_seq=number,
                model_dim=int(request.get("model_dim", self.model_dim)),
                hidden_dim=int(request.get("hidden_dim", self.hidden_dim)),
            )
            digests[item["request_id"]] = result.digest
            vectors[item["request_id"]] = result.vector
            mac_total += result.mac_count
            param_count = result.param_count
        elapsed_ms = max(1, int((time.perf_counter_ns() - started) // 1_000_000))
        return {
            "ok": True,
            "worker_id": self.worker_id,
            "batch_id": request.get("batch_id"),
            "output_digests": digests,
            "vectors": vectors,
            "mac_count": mac_total,
            "param_count": param_count,
            "measured_elapsed_ms": elapsed_ms,
            "compute_backend": "cpu-int-fixed-point-numpy",
        }


class LoopbackWorkerServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port: int, backend: WorkerBackend):
        self.backend = backend
        super().__init__(("127.0.0.1", port), DeterministicWorkerHandler)


def run_worker_process(args: argparse.Namespace) -> int:
    if not (args.port == 0 or args.port_min <= args.port <= args.port_max):
        event = {
            "event": "bind_failed",
            "worker_id": args.worker_id,
            "host": "127.0.0.1",
            "port": args.port,
            "reason": f"port outside [{args.port_min},{args.port_max}]",
        }
        print(json.dumps(event), file=sys.stderr, flush=True)
        return 2
    backend = WorkerBackend(args.worker_id, args.seed, args.model_dim, args.hidden_dim)
    try:
        server = LoopbackWorkerServer(args.port, backend)
    except OSError as exc:
        event = {
            "event": "bind_failed",
            "worker_id": args.worker_id,
            "host": "127.0.0.1",
            "port": args.port,
            "error": type(exc).__name__,
            "message": str(exc),
            "troubleshooting": "check configured port range and startup logs",
        }
        print(json.dumps(event), file=sys.stderr, flush=True)
        return 2
    bound_port = server.server_address[1]
    if not (args.port_min <= bound_port <= args.port_max):
        server.server_close()
        print(
            json.dumps({"event": "bind_failed", "reason": "assigned port outside allowed range"}),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "event": "listening",
                "worker_id": args.worker_id,
                "host": "127.0.0.1",
                "port": bound_port,
                "seed": args.seed,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def launch_worker_subprocess(
    worker_id: str,
    *,
    preferred: int | None = None,
    seed: int = DEFAULT_SEED,
    port_min: int = PORT_MIN,
    port_max: int = PORT_MAX,
    model_dim: int = 16,
    hidden_dim: int = 32,
    project_root: str | Path | None = None,
    startup_timeout_s: float = 8.0,
) -> tuple[subprocess.Popen[str], int]:
    resolved_port = find_free_loopback_port(port_min, port_max, preferred=preferred)
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "batch_ledger.worker",
        "--worker-id",
        worker_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(resolved_port),
        "--port-min",
        str(port_min),
        "--port-max",
        str(port_max),
        "--seed",
        str(seed),
        "--model-dim",
        str(model_dim),
        "--hidden-dim",
        str(hidden_dim),
    ]
    process = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        line = process.stdout.readline()
        if line:
            event = json.loads(line)
            if event.get("event") == "listening":
                return process, int(event["port"])
            if event.get("event") == "bind_failed":
                raise RuntimeError(f"worker bind failed: {event}")
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"worker exited before listening: {stderr}")
        time.sleep(0.02)
    process.terminate()
    raise TimeoutError("worker startup timed out")


def infer_over_loopback(port: int, payload: dict[str, Any], timeout_s: float = 5.0) -> dict[str, Any]:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        file = sock.makefile("r", encoding="utf-8")
        line = file.readline()
        if not line:
            raise RuntimeError("worker closed connection without a response")
        return json.loads(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic local receipt inference worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-min", type=int, default=PORT_MIN)
    parser.add_argument("--port-max", type=int, default=PORT_MAX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args(argv)
    return run_worker_process(args)


if __name__ == "__main__":
    raise SystemExit(main())
