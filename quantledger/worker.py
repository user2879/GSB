"""Subprocess worker: loopback JSON-lines server with leased async jobs."""

from __future__ import annotations

import signal
import socketserver
import threading
import time
from typing import Any

import numpy as np

from .config import ALLOWED_FRAC_BITS, ALLOWED_QUANT_LEVELS, LayerSpec, WorkerSpec
from .errors import (
    E_BAD_DIMENSION,
    E_BAD_FRAC_BITS,
    E_BAD_QUANT_LEVEL,
    E_LEASE_CAPACITY,
    E_LEASE_EXPIRED,
    E_NONFINITE_WEIGHT,
    E_PROTOCOL,
    E_QUANT_LEVEL,
    QLabError,
)
from .leases import LeaseBook, LeaseReaper
from .protocol import MAX_LINE_BYTES, decode_message, encode_message
from .quant import evaluate_chain, output_summary, quant_summary


def _code_for_error(exc: QLabError) -> str:
    mapping = {
        E_BAD_DIMENSION: "E_DIMENSION",
        E_BAD_QUANT_LEVEL: E_QUANT_LEVEL,
    }
    return mapping.get(exc.code, exc.code)


def _finite_array(value: Any, *, label: str, code: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QLabError(code, f"{label} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise QLabError(E_NONFINITE_WEIGHT if label == "weight" else code, f"{label} contains non-finite values")
    return array


def validate_submit(payload: dict[str, Any]) -> tuple[list[LayerSpec], np.ndarray, int, int, int, int, int]:
    batch_id = payload.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise QLabError(E_PROTOCOL, "batch_id must be a non-empty string")
    revision = payload.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise QLabError(E_PROTOCOL, "revision must be a positive integer")
    frac_bits = payload.get("frac_bits")
    if frac_bits not in ALLOWED_FRAC_BITS:
        raise QLabError(E_BAD_FRAC_BITS, f"frac_bits must be one of {list(ALLOWED_FRAC_BITS)}")
    quant_level = payload.get("quant_level")
    if quant_level not in ALLOWED_QUANT_LEVELS:
        raise QLabError(E_BAD_QUANT_LEVEL, f"quant_level must be one of {list(ALLOWED_QUANT_LEVELS)}")
    ttl_ms = payload.get("ttl_ms", 5000)
    if not isinstance(ttl_ms, int) or isinstance(ttl_ms, bool) or ttl_ms < 1:
        raise QLabError(E_LEASE_CAPACITY, "ttl_ms must be a positive integer")
    delay = payload.get("compute_delay_ms", 0)
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0 or delay > 60_000:
        raise QLabError(E_PROTOCOL, "compute_delay_ms must be an integer in 0..60000")
    samples = _finite_array(payload.get("samples"), label="samples", code="E_DIMENSION")
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise QLabError(E_PROTOCOL, "layers must be a non-empty list")
    layers: list[LayerSpec] = []
    previous_rows: int | None = None
    names: set[str] = set()
    for index, raw in enumerate(raw_layers):
        if not isinstance(raw, dict):
            raise QLabError(E_PROTOCOL, f"layer {index} must be an object")
        name = raw.get("name")
        rows, cols, group_size = raw.get("rows"), raw.get("cols"), raw.get("group_size")
        if not isinstance(name, str) or not name or name in names:
            raise QLabError("E_UNKNOWN_LAYER", f"invalid or duplicate layer name at index {index}")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (rows, cols, group_size)):
            raise QLabError(E_BAD_DIMENSION, f"layer {name} dimensions and group_size must be positive integers")
        if (rows * cols) % group_size != 0:
            raise QLabError("E_GROUP_NOT_DIVISIBLE", f"layer {name} group_size does not divide element count")
        if previous_rows is not None and cols != previous_rows:
            raise QLabError("E_DIMENSION", f"layer {name} input width differs from previous output width")
        weight = _finite_array(raw.get("weight"), label="weight", code=E_NONFINITE_WEIGHT)
        if weight.shape != (rows, cols):
            raise QLabError("E_DIMENSION", f"layer {name} weight shape {list(weight.shape)} does not match {(rows, cols)}")
        bias = None
        if raw.get("bias") is not None:
            bias = _finite_array(raw.get("bias"), label="bias", code=E_NONFINITE_WEIGHT)
            if bias.shape != (rows,):
                raise QLabError("E_DIMENSION", f"layer {name} bias shape must be [{rows}]")
        layers.append(LayerSpec(name, rows, cols, group_size, weight, raw.get("weight_seed"), bias))
        names.add(name)
        previous_rows = rows
    if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] != layers[0].cols:
        raise QLabError("E_DIMENSION", f"sample shape {list(samples.shape)} does not match input width {layers[0].cols}")
    return layers, samples, int(revision), int(frac_bits), int(quant_level), int(ttl_ms), int(delay)


class WorkerRuntime:
    def __init__(self, spec: WorkerSpec) -> None:
        self.spec = spec
        self.book = LeaseBook(
            worker_id=spec.worker_id,
            concurrency_quota=spec.concurrency_quota,
            lease_limit=spec.lease_limit,
        )
        self.reaper = LeaseReaper(self.book)
        self.reaper.start()
        self.compute_slots = threading.BoundedSemaphore(spec.concurrency_quota)
        self._shutdown = threading.Event()

    def dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:
        layers, samples, revision, frac_bits, quant_level, ttl_ms, delay = validate_submit(payload)
        batch_id = str(payload["batch_id"])
        lease = self.book.grant(batch_id=batch_id, revision=revision, ttl_ms=ttl_ms)
        thread = threading.Thread(
            target=self._run_job,
            name=f"job-{lease.lease_id}",
            args=(lease.lease_id, lease.generation, batch_id, layers, samples, revision, frac_bits, quant_level, delay),
            daemon=True,
        )
        thread.start()
        return {
            "ok": True,
            "op": "submit",
            "status": "leased",
            "lease": lease.to_dict(now_ms=self.book._now()),
            "capacity": self.book.status(),
        }

    def _run_job(
        self,
        lease_id: str,
        generation: int,
        batch_id: str,
        layers: list[LayerSpec],
        samples: np.ndarray,
        revision: int,
        frac_bits: int,
        quant_level: int,
        delay_ms: int,
    ) -> None:
        with self.compute_slots:
            if not self.book.begin_run(lease_id, generation):
                return
            lease = self.book.get(lease_id)
            if lease is not None and delay_ms:
                lease.cancelled.wait(delay_ms / 1000.0)
            if lease is not None and lease.cancelled.is_set():
                self.book.finish_run(
                    lease_id,
                    generation,
                    {
                        "ok": False,
                        "code": E_LEASE_EXPIRED,
                        "message": "job cancelled after lease loss",
                        "batch_id": batch_id,
                    },
                )
                return
            started_ns = time.monotonic_ns()
            try:
                output, records, _, result_meta = evaluate_chain(
                    layers,
                    samples,
                    revision=revision,
                    frac_bits=frac_bits,
                    quant_level=quant_level,
                    batch_id=batch_id,
                )
                duration_ms = max(0, (time.monotonic_ns() - started_ns) // 1_000_000)
                receipt = {
                    "q": 3,
                    "batch_id": batch_id,
                    "layer_name": layers[-1].name,
                    "output_summary": output_summary(output),
                    "revision": revision,
                    "duration_ms": int(duration_ms),
                    "worker_id": self.spec.worker_id,
                    "stable_rank": 1,
                    "superseded_refs": [],
                    "quant_params_summary": quant_summary(records),
                }
                result = {
                    "ok": True,
                    "batch_id": batch_id,
                    "revision": revision,
                    "worker_id": self.spec.worker_id,
                    "duration_ms": int(duration_ms),
                    "receipt": receipt,
                    "layers": records,
                    "result_meta": result_meta,
                    "output": output.astype(float).tolist(),
                }
                self.book.finish_run(lease_id, generation, result)
            except QLabError as exc:
                self.book.finish_run(
                    lease_id,
                    generation,
                    {
                        "ok": False,
                        "code": _code_for_error(exc),
                        "message": exc.message,
                        "batch_id": batch_id,
                    },
                )
            except Exception as exc:  # pragma: no cover - defensive process isolation
                self.book.finish_run(
                    lease_id,
                    generation,
                    {
                        "ok": False,
                        "code": "E_WORKER_INTERNAL",
                        "message": f"{type(exc).__name__}: {exc}",
                        "batch_id": batch_id,
                    },
                )

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        op = message.get("op", message.get("type"))
        request_id = message.get("request_id")
        try:
            if op == "ping":
                response = {
                    "ok": True,
                    "op": "ping",
                    "worker_id": self.spec.worker_id,
                    "pid": __import__("os").getpid(),
                    "protocol": "jsonl-v1",
                    "status": self.book.status(),
                }
            elif op == "submit":
                response = self.dispatch(message)
            elif op == "renew":
                lease = self.book.renew(
                    lease_id=str(message.get("lease_id")),
                    generation=int(message.get("generation")),
                    ttl_ms=int(message.get("ttl_ms")),
                    revision=message.get("revision"),
                )
                response = {"ok": True, "op": "renew", "lease": lease.to_dict(now_ms=self.book._now())}
            elif op == "release":
                response = {
                    "ok": True,
                    "op": "release",
                    "released": self.book.release(
                        lease_id=str(message.get("lease_id")),
                        generation=int(message.get("generation")),
                        revision=message.get("revision"),
                    ),
                }
            elif op == "fetch":
                response = {"ok": True, "op": "fetch", **self.book.fetch(str(message.get("lease_id")))}
            elif op == "status":
                response = {"ok": True, "op": "status", "status": self.book.status()}
            elif op == "events":
                events = self.book.drain_events()
                response = {"ok": True, "op": "events", "events": events, "count": len(events)}
            elif op == "set_lease_limit":
                status = self.book.set_lease_limit(message.get("value"))
                response = {"ok": True, "op": "set_lease_limit", "status": status}
            elif op == "shutdown":
                response = {"ok": True, "op": "shutdown", "status": "stopping"}
                self._shutdown.set()
            else:
                raise QLabError(E_PROTOCOL, f"unknown operation {op!r}")
        except (TypeError, ValueError) as exc:
            response = {"ok": False, "code": E_PROTOCOL, "message": f"invalid request field: {exc}"}
        except QLabError as exc:
            response = exc.as_dict()
            response["code"] = _code_for_error(exc)
        except Exception as exc:  # pragma: no cover - protocol isolation boundary
            response = {"ok": False, "code": "E_WORKER_INTERNAL", "message": f"{type(exc).__name__}: {exc}"}
        if request_id is not None:
            response["request_id"] = request_id
        return response

    def close(self) -> None:
        self.reaper.stop()
        self.book.stop()


class _ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        runtime: WorkerRuntime = self.server.runtime  # type: ignore[attr-defined]
        while not runtime._shutdown.is_set():
            line = self.rfile.readline(MAX_LINE_BYTES + 1)
            if not line:
                return
            if len(line) > MAX_LINE_BYTES:
                self.wfile.write(encode_message({"ok": False, "code": E_PROTOCOL, "message": "request line too large"}))
                self.wfile.flush()
                return
            try:
                message = decode_message(line)
            except QLabError as exc:
                self.wfile.write(encode_message(exc.as_dict()))
                self.wfile.flush()
                continue
            response = runtime.handle(message)
            self.wfile.write(encode_message(response))
            self.wfile.flush()
            if message.get("op") == "shutdown":
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return


def serve_worker(spec: WorkerSpec) -> None:
    runtime = WorkerRuntime(spec)
    server = _ThreadingServer((spec.host, spec.port), _Handler)
    server.runtime = runtime  # type: ignore[attr-defined]
    stop_event = threading.Event()

    def _stop(signum: int, frame: Any) -> None:  # noqa: ARG001
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        server.serve_forever(poll_interval=0.02)
    finally:
        server.server_close()
        runtime.close()
