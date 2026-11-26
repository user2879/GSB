"""Lease-bounded parallel decoding with one TCP worker per block group."""

from __future__ import annotations

import base64
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .errors import SocketTimeoutError, TransportError


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DecodeJob:
    block_group_id: str
    columns: tuple[str, ...] = ()
    block_bytes: bytes | None = None
    row_count: int = 0
    rows: list[Any] | None = None
    response_delay_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def coerce(value: Any) -> "DecodeJob":
        if isinstance(value, DecodeJob):
            return value
        if isinstance(value, str):
            return DecodeJob(block_group_id=value)
        if isinstance(value, dict):
            block_bytes = value.get("block_bytes")
            if isinstance(block_bytes, str):
                block_bytes = base64.b64decode(block_bytes)
            columns = value.get("columns") or ()
            if isinstance(columns, str):
                columns = (columns,)
            return DecodeJob(
                block_group_id=value["block_group_id"],
                columns=tuple(columns),
                block_bytes=block_bytes,
                row_count=int(value.get("row_count", 0) or 0),
                rows=value.get("rows"),
                response_delay_ms=int(value.get("response_delay_ms", value.get("delay_ms", 0)) or 0),
                metadata=dict(value.get("metadata", {})),
            )
        raise TypeError(f"unsupported decode job: {type(value)!r}")

    def request(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "decode",
            "block_group_id": self.block_group_id,
            "row_count": self.row_count,
            "response_delay_ms": self.response_delay_ms,
            "metadata": self.metadata,
        }
        if self.block_bytes is not None:
            payload["block_b64"] = base64.b64encode(self.block_bytes).decode("ascii")
        if self.rows is not None:
            payload["rows"] = self.rows
        return payload


def tcp_transact(
    host: str,
    port: int,
    request: dict[str, Any],
    timeout_ms: int = 2000,
) -> dict[str, Any]:
    timeout_seconds = max(float(timeout_ms), 1.0) / 1000.0
    started = time.perf_counter()
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_seconds) as sock:
            sock.settimeout(timeout_seconds)
            sock.sendall((json.dumps(request, sort_keys=True) + "\n").encode("utf-8"))
            chunks = bytearray()
            while b"\n" not in chunks:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.extend(chunk)
            if b"\n" not in chunks:
                raise TransportError("worker closed without a complete response", port=port)
            response = json.loads(chunks.split(b"\n", 1)[0].decode("utf-8"))
    except (TimeoutError, socket.timeout) as exc:
        elapsed = (time.perf_counter() - started) * 1000
        raise SocketTimeoutError(
            f"TCP call timed out after {timeout_ms}ms",
            host=host,
            port=port,
            timeout_ms=timeout_ms,
            elapsed_ms=round(elapsed, 3),
        ) from exc
    except OSError as exc:
        raise TransportError(f"TCP call failed: {exc}", host=host, port=port) from exc

    if not response.get("ok"):
        raise TransportError("worker returned an error", **response)
    return response["receipt"]


# Backwards-friendly alias.
send_decode = tcp_transact


class Receipt(dict):
    """Dictionary receipt with attribute access for tolerant integrations."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class WorkerProcess:
    """A child-process executor bound to a unique 127.0.0.1 port."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        timeout_ms: int = 5000,
        python_executable: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.host = host
        self.timeout_ms = timeout_ms
        self.python_executable = python_executable or sys.executable
        self.env = env
        self.process: subprocess.Popen[str] | None = None
        self.port: int | None = None
        self.ready: dict[str, Any] | None = None

    def start(self, start_timeout_ms: int | None = None) -> "WorkerProcess":
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        self.process = subprocess.Popen(
            [self.python_executable, "-m", "gsb.worker_server", "--host", self.host, "--port", "0"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        timeout_seconds = max(float(start_timeout_ms or self.timeout_ms), 1.0) / 1000.0
        ready_line = ""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise TransportError("worker process exited before becoming ready", stderr=stderr)
            assert self.process.stdout is not None
            readable, _, _ = select.select([self.process.stdout], [], [], 0.02)
            if readable:
                ready_line = self.process.stdout.readline().strip()
                if ready_line:
                    break
        if not ready_line:
            self.close()
            raise TransportError("worker start timed out", timeout_ms=start_timeout_ms or self.timeout_ms)
        self.ready = json.loads(ready_line)
        self.port = int(self.ready["port"])
        return self

    def decode(self, request: dict[str, Any], timeout_ms: int | None = None) -> dict[str, Any]:
        if self.port is None:
            self.start()
        assert self.port is not None
        receipt = Receipt(tcp_transact(self.host, self.port, request, timeout_ms or self.timeout_ms))
        receipt.setdefault("worker_port", self.port)
        return receipt

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)
        for stream in (self.process.stdout, self.process.stderr, self.process.stdin):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        self.process = None

    def __enter__(self) -> "WorkerProcess":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class Scheduler:
    def __init__(
        self,
        global_budget: int | None = None,
        per_column_lease: int | dict[str, int] | None = None,
        *,
        global_decode_budget: int | None = None,
        max_global_in_flight: int | None = None,
        max_global: int | None = None,
        per_column_limit: int | dict[str, int] | None = None,
        max_per_column: int | dict[str, int] | None = None,
        socket_timeout_ms: int | None = None,
        timeout_ms: int | None = None,
        max_concurrent_groups: int | None = None,
        max_concurrent_block_groups: int | None = None,
        max_in_flight: int | None = None,
        max_per_column_concurrency: int | dict[str, int] | None = None,
        worker_start_timeout_ms: int = 6000,
        worker_response_delay_ms: int = 0,
        worker_factory: Callable[[], WorkerProcess] | None = None,
    ) -> None:
        self.global_budget = int(
            global_budget
            or global_decode_budget
            or max_global_in_flight
            or max_global
            or max_concurrent_groups
            or max_concurrent_block_groups
            or max_in_flight
            or 2
        )
        lease = per_column_lease if per_column_lease is not None else per_column_limit
        lease = lease if lease is not None else max_per_column
        lease = lease if lease is not None else max_per_column_concurrency
        self.per_column_lease = lease if lease is not None else 1
        self.socket_timeout_ms = int(socket_timeout_ms or timeout_ms or 4000)
        self.worker_start_timeout_ms = int(worker_start_timeout_ms)
        self.worker_response_delay_ms = int(worker_response_delay_ms)
        self.worker_factory = worker_factory or (lambda: WorkerProcess(timeout_ms=self.socket_timeout_ms))

        self._global_slots = threading.BoundedSemaphore(self.global_budget)
        self._column_slots: dict[str, threading.BoundedSemaphore] = {}
        self._lock = threading.RLock()
        self.global_in_flight = 0
        self.peak_global_in_flight = 0
        self.column_in_flight: dict[str, int] = {}
        self.peak_per_column: dict[str, int] = {}
        self.lease_events: list[dict[str, Any]] = []
        self.worker_ports: dict[str, int] = {}
        self._executor: ThreadPoolExecutor | None = None

    def _column_limit(self, column: str) -> int:
        if isinstance(self.per_column_lease, dict):
            return int(self.per_column_lease.get(column, self.per_column_lease.get("*", 1)))
        return int(self.per_column_lease)

    def _column_slot(self, column: str) -> threading.BoundedSemaphore:
        with self._lock:
            if column not in self._column_slots:
                self._column_slots[column] = threading.BoundedSemaphore(self._column_limit(column))
                self.column_in_flight.setdefault(column, 0)
                self.peak_per_column.setdefault(column, 0)
            return self._column_slots[column]

    def _record_event(self, event: str, job: DecodeJob, **extra: Any) -> None:
        payload = {
            "event": event,
            "block_group_id": job.block_group_id,
            "monotonic_ms": round(time.monotonic() * 1000, 3),
            "global_in_flight": self.global_in_flight,
            "columns": list(job.columns),
        }
        payload.update(extra)
        self.lease_events.append(payload)

    def _run_one(self, job_value: Any) -> dict[str, Any]:
        job = DecodeJob.coerce(job_value)
        columns = tuple(sorted(set(job.columns)))
        acquired_columns: list[str] = []
        worker: WorkerProcess | None = None
        self._global_slots.acquire()
        try:
            with self._lock:
                self.global_in_flight += 1
                self.peak_global_in_flight = max(self.peak_global_in_flight, self.global_in_flight)
                self._record_event("global_acquired", job)

            for column in columns:
                self._column_slot(column).acquire()
                acquired_columns.append(column)
                with self._lock:
                    self.column_in_flight[column] = self.column_in_flight.get(column, 0) + 1
                    self.peak_per_column[column] = max(
                        self.peak_per_column.get(column, 0),
                        self.column_in_flight[column],
                    )
                    self._record_event("column_acquired", job, column=column)

            request = job.request()
            request.setdefault("response_delay_ms", self.worker_response_delay_ms)
            worker = self.worker_factory()
            worker.start(self.worker_start_timeout_ms)
            with self._lock:
                self.worker_ports[job.block_group_id] = int(worker.port or 0)
            receipt = Receipt(worker.decode(request, self.socket_timeout_ms))
            receipt["worker_port"] = worker.port
            receipt.setdefault("block_group_id", job.block_group_id)
            receipt.setdefault("decoded_rows", job.row_count)
            receipt.setdefault("elapsed_ms", 0)
            return receipt
        finally:
            if worker is not None:
                worker.close()
            for column in reversed(acquired_columns):
                self._column_slot(column).release()
                with self._lock:
                    self.column_in_flight[column] -= 1
            with self._lock:
                self.global_in_flight -= 1
                self._record_event("released", job)
            self._global_slots.release()

    def run(self, jobs: Iterable[Any]) -> list[dict[str, Any]]:
        coerced = [DecodeJob.coerce(job) for job in jobs]
        if not coerced:
            return []
        max_workers = min(32, max(self.global_budget, len(coerced)))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gsb-decode") as executor:
            futures = [executor.submit(self._run_one, job) for job in coerced]
            receipts = [future.result() for future in futures]
        return receipts

    schedule = run
    schedule_batch = run
    run_batch = run
    decode = run
    map = run

    def lease_peaks(self) -> dict[str, Any]:
        return {
            "global_budget": self.global_budget,
            "peak_global_in_flight": self.peak_global_in_flight,
            "per_column_limit": self.per_column_lease,
            "peak_per_column": dict(sorted(self.peak_per_column.items())),
            "worker_ports": dict(sorted(self.worker_ports.items())),
        }

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "Scheduler":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


DecodeScheduler = Scheduler
LeaseScheduler = Scheduler
ParallelDecodeScheduler = Scheduler
