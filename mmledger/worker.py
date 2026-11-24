"""Deterministic NumPy loopback inference worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import load_config
from .jsonutil import canonical_dumps
from .models import IMAGE_ONLY, IMAGE_TEXT, ModelSpec, WorkerSpec


def _mix_seed(seed: int, request_id: str, sequence: int) -> int:
    digest = hashlib.sha256(f"{seed}|{request_id}|{sequence}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**63 - 1)


def forward_summary(
    *,
    model: ModelSpec,
    seed: int,
    request_id: str,
    sequence: int,
    kind: str,
    worker_id: str,
    parameter_version: int,
    parameter_digest: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(np.uint64(_mix_seed(seed, request_id, sequence)))
    qmax = (1 << (model.quantization_bits - 1)) - 1
    x_int = rng.integers(-qmax, qmax + 1, size=model.input_dim, dtype=np.int64)
    w1_int = rng.integers(-qmax, qmax + 1, size=(model.input_dim, model.hidden_dim), dtype=np.int64)
    b1_int = rng.integers(-qmax, qmax + 1, size=model.hidden_dim, dtype=np.int64)
    w2_int = rng.integers(-qmax, qmax + 1, size=(model.hidden_dim, 2), dtype=np.int64)

    hidden_int = x_int @ w1_int + (b1_int << (model.quantization_bits - 1))
    hidden_int = np.maximum(hidden_int, 0)
    logits_int = hidden_int @ w2_int
    fixed_score = int(logits_int.sum())

    scale = float(max(1, qmax))
    x_ref = x_int.astype(np.float64) / scale
    w1_ref = w1_int.astype(np.float64) / scale
    b1_ref = b1_int.astype(np.float64) / scale
    w2_ref = w2_int.astype(np.float64) / scale
    hidden_ref = np.maximum(x_ref @ w1_ref + b1_ref, 0.0)
    logits_ref = np.tanh(hidden_ref @ w2_ref)
    reference_score = float(logits_ref.sum())
    if not math.isfinite(reference_score):
        reference_score = 0.0
    reference_ppm = int(round(reference_score * 1_000_000.0))
    fixed_ppm = int(round(fixed_score * 1_000_000.0 / (scale * scale)))
    output_digest = hashlib.sha256(
        canonical_dumps(
            {
                "worker_id": worker_id,
                "parameter_version": parameter_version,
                "parameter_digest": parameter_digest,
                "request_id": request_id,
                "kind": kind,
                "fixed_score": fixed_score,
                "fixed_score_ppm": fixed_ppm,
                "reference_score_ppm": reference_ppm,
                "quantization_bits": model.quantization_bits,
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "request_id": request_id,
        "kind": kind,
        "sequence": sequence,
        "output_digest": output_digest,
        "fixed_score": fixed_score,
        "fixed_score_ppm": fixed_ppm,
        "reference_score_ppm": reference_ppm,
        "absolute_error_ppm": abs(fixed_ppm - reference_ppm),
    }


def infer_response(
    body: Mapping[str, Any],
    *,
    model: ModelSpec,
    seed: int,
    worker: WorkerSpec,
) -> dict[str, Any]:
    batch_id = body.get("batch_id")
    requests = body.get("requests")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("batch_id must be a non-empty string")
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")
    normalized = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"request {index} must be an object")
        request_id = request.get("request_id")
        kind = request.get("kind")
        sequence = request.get("sequence", index)
        request_seed = request.get("input_seed", seed + index)
        overwritten_ref = request.get("overwritten_ref")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"request {index} request_id must be non-empty")
        if kind not in (IMAGE_TEXT, IMAGE_ONLY):
            raise ValueError(f"request {request_id} has invalid kind")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise ValueError(f"request {request_id} sequence must be an integer")
        if not isinstance(request_seed, int) or isinstance(request_seed, bool):
            raise ValueError(f"request {request_id} input_seed must be an integer")
        if overwritten_ref is not None and (not isinstance(overwritten_ref, str) or not overwritten_ref):
            raise ValueError(f"request {request_id} overwritten_ref must be null or non-empty")
        normalized.append(
            {
                "request_id": request_id,
                "kind": kind,
                "sequence": sequence,
                "input_seed": request_seed,
                "overwritten_ref": overwritten_ref,
            }
        )

    ranked = sorted(normalized, key=lambda item: item["request_id"])
    rank_by_id = {item["request_id"]: index + 1 for index, item in enumerate(ranked)}
    summaries = []
    for request in normalized:
        summary = forward_summary(
            model=model,
            seed=seed,
            request_id=request["request_id"],
            sequence=request["sequence"],
            kind=request["kind"],
            worker_id=worker.worker_id,
            parameter_version=worker.parameter_version,
            parameter_digest=worker.parameter_digest,
        )
        summary["stable_rank"] = rank_by_id[request["request_id"]]
        summary["overwritten_ref"] = request["overwritten_ref"]
        summaries.append(summary)
    per_request_ms = max(
        1,
        (model.macs_per_request * model.quantization_bits // 8) // 64,
    )
    duration_ms = max(1, len(summaries) * per_request_ms)
    return {
        "batch_id": batch_id,
        "worker_id": worker.worker_id,
        "parameter_version": worker.parameter_version,
        "parameter_digest": worker.parameter_digest,
        "duration_ms": duration_ms,
        "requests": sorted(summaries, key=lambda item: item["request_id"]),
        "macs_per_request": model.macs_per_request,
        "quantized_parameter_units": model.quantized_parameter_units,
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "MMLedgerWorker/1.0"
    protocol_version = "HTTP/1.1"
    worker: WorkerSpec
    model: ModelSpec
    seed: int

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return

    def _send(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = canonical_dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "worker_id": self.worker.worker_id,
                    "parameter_version": self.worker.parameter_version,
                },
            )
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4 * 1024 * 1024:
                raise ValueError("invalid content length")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            response = infer_response(
                body,
                model=self.model,
                seed=self.seed,
                worker=self.worker,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send(HTTPStatus.OK, response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one deterministic loopback inference worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = load_config(args.config)
    if result.bundle is None:
        for failure in result.failures:
            print(f"{failure.code}: {failure.message}", file=sys.stderr)
        return 2
    bundle = result.bundle
    worker = bundle.workers_by_id.get(args.worker_id)
    if worker is None:
        print(f"unknown worker_id {args.worker_id}", file=sys.stderr)
        return 2
    handler_class = type("WorkerHandler", (_Handler,), {})
    handler_class.worker = worker
    handler_class.model = bundle.model
    handler_class.seed = bundle.seed
    server = ThreadingHTTPServer((worker.host, worker.port), handler_class)
    server.daemon_threads = True

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        canonical_dumps(
            {
                "ready": True,
                "worker_id": worker.worker_id,
                "host": worker.host,
                "port": worker.port,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
