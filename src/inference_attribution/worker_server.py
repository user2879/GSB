"""Loopback socket worker process used by the integration harness."""

from __future__ import annotations

import argparse
import base64
import json
import os
import socketserver
import sys
import threading
import time
from typing import Any

from .errors import DiagnosticError, is_int
from .jsonio import canonical_json, parse_json_text
from .model import batch_output_digest, output_digest, parameter_digest
from .planner import batch_id_for
from .spec import WorkerSpec, parse_workers


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 16 * 1024 * 1024


def _decode_worker(encoded: str) -> WorkerSpec:
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        value = parse_json_text(
            raw.decode("utf-8"), object_name="worker-spec", code="E_SPEC"
        )
    except (ValueError, UnicodeDecodeError) as exc:
        raise DiagnosticError(
            code="E_SPEC", message="invalid encoded worker declaration", field="spec"
        ) from exc
    document = {
        "schema_version": 1,
        "worker_set_version": 1,
        "workers": [value],
    }
    return parse_workers(document, object_name="worker-spec").workers[0]


def _protocol_error(field: str, message: str) -> DiagnosticError:
    return DiagnosticError(
        code="E_PROTO", message=message, object="worker-request", field=field
    )


def _validate_request(value: Any, worker: WorkerSpec) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _protocol_error("<object>", "request must be a JSON object")
    for field in ("protocol_version", "batch_id", "worker_id", "batch_seq", "requests"):
        if field not in value:
            raise _protocol_error(field, "missing request field")
    if value["protocol_version"] != PROTOCOL_VERSION:
        raise _protocol_error("protocol_version", "unsupported protocol version")
    if value["worker_id"] != worker.worker_id:
        raise _protocol_error("worker_id", "worker declaration mismatch")
    if not is_int(value["batch_seq"]) or value["batch_seq"] < 0:
        raise _protocol_error("batch_seq", "expected a non-negative integer")
    requests = value["requests"]
    if not isinstance(requests, list) or not requests:
        raise _protocol_error("requests", "expected a non-empty array")
    seen_ids: set[str] = set()
    seen_arrivals: set[int] = set()
    for index, request in enumerate(requests):
        prefix = f"requests[{index}]"
        if not isinstance(request, dict):
            raise _protocol_error(prefix, "request entry must be an object")
        for field in ("request_id", "arrival_seq", "input"):
            if field not in request:
                raise _protocol_error(f"{prefix}.{field}", "missing request field")
        request_id = request["request_id"]
        if not isinstance(request_id, str) or not request_id:
            raise _protocol_error(f"{prefix}.request_id", "invalid request id")
        if request_id in seen_ids:
            raise _protocol_error(f"{prefix}.request_id", "duplicate request id")
        seen_ids.add(request_id)
        arrival_seq = request["arrival_seq"]
        if not is_int(arrival_seq) or arrival_seq < 0:
            raise _protocol_error(
                f"{prefix}.arrival_seq", "expected a non-negative integer"
            )
        if arrival_seq in seen_arrivals:
            raise _protocol_error(
                f"{prefix}.arrival_seq", "duplicate arrival sequence"
            )
        seen_arrivals.add(arrival_seq)
        vector = request["input"]
        if not isinstance(vector, list) or len(vector) != worker.input_dim:
            raise _protocol_error(
                f"{prefix}.input", "input dimension does not match worker"
            )
        for value_index, item in enumerate(vector):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise _protocol_error(
                    f"{prefix}.input[{value_index}]", "input value must be numeric"
                )
    arrival_seqs = tuple(
        sorted(int(request["arrival_seq"]) for request in requests)
    )
    expected_batch_id = batch_id_for(
        worker.worker_id, int(value["batch_seq"]), arrival_seqs
    )
    if value["batch_id"] != expected_batch_id:
        raise _protocol_error("batch_id", "batch id does not match stable derivation")
    return value


def _process_batch(value: Any, worker: WorkerSpec) -> dict[str, Any]:
    request = _validate_request(value, worker)
    started = time.perf_counter_ns()
    ordered = sorted(
        request["requests"],
        key=lambda item: (int(item["arrival_seq"]), str(item["request_id"])),
    )
    param_digest = parameter_digest(worker)
    receipts: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    for rank, item in enumerate(ordered):
        request_id = str(item["request_id"])
        digest = output_digest(worker, tuple(float(number) for number in item["input"]))
        outputs[request_id] = digest
        receipts.append(
            {
                "batch_id": request["batch_id"],
                "request_id": request_id,
                "output_digest": digest,
                "param_version_seq": worker.param_version_seq,
                "latency_ms": 0,
                "worker_id": worker.worker_id,
                "rank_in_batch": rank,
                "superseded_request_ids": [],
                "param_digest": param_digest,
            }
        )
    elapsed_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)
    for receipt in receipts:
        receipt["latency_ms"] = elapsed_ms
    return {
        "protocol_version": PROTOCOL_VERSION,
        "ok": True,
        "batch_id": request["batch_id"],
        "worker_id": worker.worker_id,
        "param_version_seq": worker.param_version_seq,
        "param_digest": param_digest,
        "latency_ms": elapsed_ms,
        "batch_output_digest": batch_output_digest(
            outputs,
            param_version_seq=worker.param_version_seq,
            param_digest=param_digest,
        ),
        "receipts": receipts,
    }


class _WorkerHandler(socketserver.StreamRequestHandler):
    worker: WorkerSpec

    def handle(self) -> None:
        line = self.rfile.readline(MAX_LINE_BYTES + 1)
        if not line:
            return
        if len(line) > MAX_LINE_BYTES:
            response = {
                "ok": False,
                "error": {
                    "code": "E_PROTO",
                    "message": "request line exceeds size limit",
                    "field": "<request>",
                },
            }
        else:
            try:
                value = parse_json_text(
                    line.decode("utf-8"),
                    object_name="worker-request",
                    code="E_PROTO",
                )
                response = _process_batch(value, self.worker)
            except DiagnosticError as exc:
                response = {"ok": False, "error": exc.to_dict()}
            except Exception as exc:
                response = {
                    "ok": False,
                    "error": {
                        "code": "E_PROTO",
                        "message": f"worker failed to process batch: {exc}",
                        "field": "<batch>",
                    },
                }
        self.wfile.write((canonical_json(response) + "\n").encode("utf-8"))
        self.wfile.flush()


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _watch_parent(parent_pid: int | None) -> None:
    if parent_pid is None:
        return
    while True:
        if os.getppid() != parent_pid:
            os._exit(0)
        time.sleep(0.2)


def serve(worker: WorkerSpec, parent_pid: int | None) -> int:
    _WorkerHandler.worker = worker
    try:
        server = _ThreadingServer((worker.host, worker.port), _WorkerHandler)
    except OSError as exc:
        error = DiagnosticError(
            code="E_PORT",
            message=f"cannot listen on {worker.host}:{worker.port}: {exc}",
            object=worker.worker_id,
            field="port",
        )
        print(canonical_json(error.to_dict()), file=sys.stderr, flush=True)
        return 2
    watcher = threading.Thread(
        target=_watch_parent,
        args=(parent_pid,),
        name=f"worker-parent-watch-{worker.worker_id}",
        daemon=True,
    )
    watcher.start()
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inference-attribution-worker")
    parser.add_argument("--spec-b64", required=True)
    parser.add_argument("--parent-pid", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        worker = _decode_worker(args.spec_b64)
    except DiagnosticError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr, flush=True)
        return 2
    return serve(worker, args.parent_pid)


if __name__ == "__main__":
    raise SystemExit(main())
