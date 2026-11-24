"""127.0.0.1 loopback inference worker process."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any

from .compute import WorkerRuntime, build_receipt_rows
from .config import load_config, load_requests
from .util import canonical_json


def _handle_request(
    request: dict[str, Any],
    *,
    runtime: WorkerRuntime,
    request_by_id: dict[str, Any],
) -> dict[str, Any]:
    batch_id = request.get("batch_id")
    request_ids = request.get("request_ids")
    version = request.get("receipt_version", "r3")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("batch_id must be a non-empty string")
    if not isinstance(request_ids, list) or not request_ids:
        raise ValueError("request_ids must be a non-empty list")
    if version not in {"r1", "r2", "r3"}:
        raise ValueError("receipt_version must be r1, r2 or r3")
    missing = [value for value in request_ids if value not in request_by_id]
    if missing:
        raise ValueError(f"unknown request ids: {missing}")
    requests = [request_by_id[value] for value in request_ids]
    covered_refs = request.get("covered_refs", {})
    if not isinstance(covered_refs, dict):
        raise ValueError("covered_refs must be an object")
    rows = build_receipt_rows(
        runtime,
        batch_id=batch_id,
        requests=requests,
        version=version,
        duration_ms=request.get("duration_ms"),
        covered_refs={
            str(key): [str(item) for item in value]
            for key, value in covered_refs.items()
            if isinstance(value, list)
        },
    )
    return {
        "ok": True,
        "worker_id": runtime.worker.worker_id,
        "param_version": runtime.param_version,
        "batch_id": batch_id,
        "duration_ms": runtime.duration_ms(len(rows)),
        "receipts": rows,
    }


def serve(
    *,
    config_path: Path,
    requests_path: Path,
    worker_id: str,
    host: str = "127.0.0.1",
    port_override: int | None = None,
) -> None:
    config = load_config(config_path)
    requests = load_requests(requests_path, batch_size=config.batch_size)
    worker = config.worker_by_id.get(worker_id)
    if worker is None:
        raise SystemExit(f"unknown worker {worker_id!r}")
    if host != "127.0.0.1":
        raise SystemExit("worker may only bind to 127.0.0.1")
    port = worker.port if port_override is None else port_override
    if not (config.port_min <= port <= config.port_max):
        raise SystemExit(f"port {port} is outside configured range")
    runtime = WorkerRuntime(worker, worker.param_version, config.throughput_ops_per_ms)
    request_by_id = {request.request_id: request for request in requests}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(16)
        print(
            canonical_json(
                {
                    "event": "READY",
                    "worker_id": worker_id,
                    "host": host,
                    "port": port,
                    "pid": __import__("os").getpid(),
                }
            ),
            flush=True,
        )
        while True:
            try:
                connection, address = server.accept()
            except KeyboardInterrupt:
                break
            with connection:
                connection.settimeout(30)
                buffer = bytearray()
                while b"\n" not in buffer:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                if not buffer:
                    continue
                line = bytes(buffer).split(b"\n", 1)[0]
                try:
                    request = json.loads(line.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("request root must be an object")
                    response = _handle_request(
                        request, runtime=runtime, request_by_id=request_by_id
                    )
                except Exception as exc:  # networked protocol must isolate bad clients
                    response = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
                connection.sendall((canonical_json(response) + "\n").encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one deterministic loopback inference worker")
    parser.add_argument("--project", default=".")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--requests", default="requests.ndjson")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    project = Path(args.project).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project / config_path
    requests_path = Path(args.requests)
    if not requests_path.is_absolute():
        requests_path = project / requests_path
    try:
        serve(
            config_path=config_path,
            requests_path=requests_path,
            worker_id=args.worker_id,
            host=args.host,
            port_override=args.port,
        )
    except OSError as exc:
        print(f"worker failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

