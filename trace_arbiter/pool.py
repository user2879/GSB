from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .canonical import canonical_bytes, canonical_text, digest_bytes
from .errors import ArbiterError, EXIT_CODES
from .spec import Node


MAX_RESPONSE_BYTES = 64 * 1024 * 1024


@dataclass
class WorkerProcess:
    node: Node
    process: subprocess.Popen[bytes]
    pid: int


@dataclass
class ClientResult:
    response: dict[str, Any]
    attempts: int
    elapsed_us: int
    request_digest: str


@dataclass
class ScheduledBatch:
    batch_index: int
    node_id: str
    request: dict[str, Any]
    request_text: str
    request_digest: str
    submitted_ns: int = field(default_factory=time.perf_counter_ns)


@dataclass
class BatchOutcome:
    batch: ScheduledBatch
    response: dict[str, Any]
    attempts: int
    elapsed_us: int
    queue_enter_ns: int
    queue_exit_ns: int
    error: ArbiterError | None = None
    finished_ns: int = field(default_factory=time.perf_counter_ns)

    @property
    def queue_wait_us(self) -> int:
        return max(0, (self.queue_exit_ns - self.queue_enter_ns) // 1000)

    @property
    def queue_wait_ms(self) -> int:
        return self.queue_wait_us // 1000


@dataclass
class ConcurrencySnapshot:
    max_global_running: int
    max_node_running: dict[str, int]
    wait_intervals_us: list[tuple[int, str, int]]
    completed_batches: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "completed_batches": self.completed_batches,
            "max_global_running": self.max_global_running,
            "max_node_running": dict(sorted(self.max_node_running.items())),
            "wait_intervals_us": [
                {
                    "batch_index": index,
                    "node_id": node_id,
                    "wait_us": wait,
                    "wait_ms": wait // 1000,
                }
                for index, node_id, wait in self.wait_intervals_us
            ],
        }


class _Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.global_running = 0
        self.node_running: dict[str, int] = {}
        self.max_global_running = 0
        self.max_node_running: dict[str, int] = {}
        self.wait_intervals_us: list[tuple[int, str, int]] = []
        self.completed_batches = 0

    def enter(self, node_id: str, batch_index: int, wait_us: int) -> None:
        with self.lock:
            self.global_running += 1
            self.node_running[node_id] = self.node_running.get(node_id, 0) + 1
            self.max_global_running = max(self.max_global_running, self.global_running)
            self.max_node_running[node_id] = max(
                self.max_node_running.get(node_id, 0), self.node_running[node_id]
            )
            self.wait_intervals_us.append((batch_index, node_id, wait_us))

    def leave(self, node_id: str) -> None:
        with self.lock:
            self.global_running -= 1
            self.node_running[node_id] -= 1
            self.completed_batches += 1

    def snapshot(self) -> ConcurrencySnapshot:
        with self.lock:
            return ConcurrencySnapshot(
                max_global_running=self.max_global_running,
                max_node_running=dict(self.max_node_running),
                wait_intervals_us=sorted(self.wait_intervals_us),
                completed_batches=self.completed_batches,
            )


def _read_line_with_deadline(stream: Any, timeout_s: float) -> bytes | None:
    result: queue.Queue[bytes | None] = queue.Queue(maxsize=1)

    def reader() -> None:
        try:
            result.put(stream.readline())
        except Exception:
            result.put(None)

    threading.Thread(target=reader, name="worker-handshake-reader", daemon=True).start()
    try:
        return result.get(timeout=timeout_s)
    except queue.Empty:
        return None


class WorkerPool:
    """Owns one local subprocess per declared node."""

    def __init__(
        self,
        nodes: Iterable[Node],
        *,
        handshake_timeout_ms: int = 5_000,
        python_executable: str | None = None,
        command_factory: Callable[[Node], list[str]] | None = None,
    ) -> None:
        self.nodes = tuple(sorted(nodes, key=lambda node: node.node_id))
        self.handshake_timeout_ms = handshake_timeout_ms
        self.python_executable = python_executable or sys.executable
        self.command_factory = command_factory
        self.workers: dict[str, WorkerProcess] = {}
        self._closed = False

    def _command(self, node: Node) -> list[str]:
        if self.command_factory is not None:
            return self.command_factory(node)
        return [
            self.python_executable,
            "-u",
            "-m",
            "trace_arbiter.service",
            "--node-json",
            canonical_text(node.as_dict()),
            "--parent-pid",
            str(os.getpid()),
        ]

    def start(self) -> None:
        if self.workers:
            return
        try:
            for node in self.nodes:
                command = self._command(node)
                env = os.environ.copy()
                project_root = str(Path(__file__).resolve().parent.parent)
                env["PYTHONPATH"] = project_root + (
                    os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
                )
                creationflags = 0
                if os.name == "nt":
                    creationflags = (
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    )
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        cwd=project_root,
                        env=env,
                        bufsize=0,
                        creationflags=creationflags,
                    )
                except (OSError, ValueError) as exc:
                    raise ArbiterError(
                        "E_SPAWN",
                        f"cannot spawn aggregation service for {node.node_id}: {exc}",
                        details={"node_id": node.node_id},
                    ) from exc

                worker = WorkerProcess(node=node, process=process, pid=process.pid)
                self.workers[node.node_id] = worker
                timeout_s = max(0.001, self.handshake_timeout_ms / 1000)
                line = _read_line_with_deadline(process.stdout, timeout_s)
                if line is None:
                    raise ArbiterError(
                        "E_HANDSHAKE",
                        f"handshake timeout for node {node.node_id}",
                        details={"node_id": node.node_id, "timeout_ms": self.handshake_timeout_ms},
                    )
                if not line:
                    code = process.poll()
                    raise ArbiterError(
                        "E_SPAWN",
                        f"aggregation service for {node.node_id} exited before handshake",
                        details={"node_id": node.node_id, "returncode": code},
                    )
                try:
                    hello = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ArbiterError(
                        "E_HANDSHAKE",
                        f"invalid handshake from node {node.node_id}: {exc}",
                        details={"node_id": node.node_id},
                    ) from exc
                if hello.get("type") == "error" or hello.get("ok") is False:
                    code = hello.get("code", "E_SPAWN")
                    if code not in EXIT_CODES:
                        code = "E_SPAWN"
                    raise ArbiterError(
                        code,
                        hello.get("message", "worker startup error"),
                        details={"node_id": node.node_id, **hello.get("details", {})},
                    )
                if hello.get("type") != "ready" or hello.get("node_id") != node.node_id:
                    raise ArbiterError(
                        "E_HANDSHAKE",
                        f"unexpected handshake for node {node.node_id}",
                        details={"node_id": node.node_id, "handshake": hello},
                    )
        except Exception:
            self.terminate()
            raise

    def terminate(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self.workers.values():
            process = worker.process
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        deadline = time.monotonic() + 2.0
        for worker in self.workers.values():
            process = worker.process
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass
        self.workers.clear()

    def __enter__(self) -> "WorkerPool":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.terminate()


class JsonLineClient:
    def __init__(
        self,
        node: Node,
        *,
        timeout_s: float,
        retries: int,
    ) -> None:
        self.node = node
        self.timeout_s = timeout_s
        self.retries = max(0, retries)

    def request(self, request: dict[str, Any]) -> ClientResult:
        body = canonical_bytes(request) + b"\n"
        request_digest = digest_bytes(body)
        started_ns = time.perf_counter_ns()
        last_error: Exception | None = None
        attempts = self.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._once(body)
                elapsed_us = (time.perf_counter_ns() - started_ns) // 1000
                if not isinstance(response, dict):
                    raise ArbiterError("E_PROTOCOL", "worker response must be a JSON object")
                if response.get("ok") is False:
                    code = response.get("code", "E_PROTOCOL")
                    if code not in EXIT_CODES:
                        code = "E_PROTOCOL"
                    raise ArbiterError(
                        code,
                        response.get("message", "worker rejected request"),
                        trace_id=response.get("trace_id"),
                        field=response.get("field"),
                        details={
                            "node_id": self.node.node_id,
                            "attempts": attempt,
                            **response.get("details", {}),
                        },
                    )
                return ClientResult(
                    response=response,
                    attempts=attempt,
                    elapsed_us=elapsed_us,
                    request_digest=request_digest,
                )
            except ArbiterError:
                raise
            except (OSError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(min(0.02, 0.002 * attempt))
        elapsed_us = (time.perf_counter_ns() - started_ns) // 1000
        raise ArbiterError(
            "E_TIMEOUT",
            f"socket request to node {self.node.node_id} timed out or failed after {attempts} attempts",
            details={
                "node_id": self.node.node_id,
                "attempts": attempts,
                "timeout_ms": int(self.timeout_s * 1000),
                "elapsed_us": elapsed_us,
                "last_error": str(last_error) if last_error else None,
            },
        )

    def _once(self, body: bytes) -> Any:
        with socket.create_connection(
            (self.node.host, self.node.port),
            timeout=self.timeout_s,
        ) as sock:
            sock.settimeout(self.timeout_s)
            sock.sendall(body)
            chunks = bytearray()
            while True:
                chunk = sock.recv(65_536)
                if not chunk:
                    break
                chunks.extend(chunk)
                if len(chunks) > MAX_RESPONSE_BYTES:
                    raise ArbiterError("E_PROTOCOL", "worker response exceeds maximum size")
                if b"\n" in chunks:
                    break
            if not chunks:
                raise ConnectionError("worker closed the connection without a response")
            line = bytes(chunks).split(b"\n", 1)[0]
            try:
                return json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArbiterError(
                    "E_PROTOCOL",
                    f"worker returned invalid JSON: {exc}",
                ) from exc


class BatchScheduler:
    """Schedule batches with both a global cap and a cap per node."""

    def __init__(
        self,
        pool: WorkerPool,
        *,
        global_concurrency: int,
        per_node_concurrency: int,
        timeout_ms: int,
        retries: int,
    ) -> None:
        if global_concurrency < 1 or per_node_concurrency < 1:
            raise ArbiterError("E_SPEC", "concurrency limits must be >= 1")
        self.pool = pool
        self.global_concurrency = global_concurrency
        self.per_node_concurrency = per_node_concurrency
        self.timeout_s = max(0.001, timeout_ms / 1000)
        self.retries = max(0, retries)
        self.stats = _Stats()

    def run(
        self,
        batches: Iterable[ScheduledBatch],
        *,
        on_result: Callable[[BatchOutcome], None] | None = None,
        abort_on_error: bool = True,
    ) -> tuple[list[BatchOutcome], list[ArbiterError]]:
        batch_list = sorted(batches, key=lambda batch: batch.batch_index)
        if not batch_list:
            return [], []
        by_node: dict[str, list[ScheduledBatch]] = {}
        for batch in batch_list:
            by_node.setdefault(batch.node_id, []).append(batch)

        global_sem = threading.Semaphore(self.global_concurrency)
        abort_event = threading.Event()
        outcomes: list[BatchOutcome] = []
        errors: list[ArbiterError] = []
        outcomes_lock = threading.Lock()
        executors: dict[str, ThreadPoolExecutor] = {}

        def execute(batch: ScheduledBatch) -> BatchOutcome:
            if abort_event.is_set():
                return BatchOutcome(
                    batch=batch,
                    response={},
                    attempts=0,
                    elapsed_us=0,
                    queue_enter_ns=batch.submitted_ns,
                    queue_exit_ns=time.perf_counter_ns(),
                    error=ArbiterError("E_RECOVERY", "batch cancelled before execution"),
                )
            global_sem.acquire()
            queue_exit_ns = time.perf_counter_ns()
            wait_us = max(0, (queue_exit_ns - batch.submitted_ns) // 1000)
            self.stats.enter(batch.node_id, batch.batch_index, wait_us)
            try:
                worker = self.pool.workers.get(batch.node_id)
                if worker is None:
                    raise ArbiterError(
                        "E_SPAWN",
                        f"no running worker for node {batch.node_id}",
                        details={"node_id": batch.node_id},
                    )
                client = JsonLineClient(
                    worker.node,
                    timeout_s=self.timeout_s,
                    retries=self.retries,
                )
                client_result = client.request(batch.request)
                return BatchOutcome(
                    batch=batch,
                    response=client_result.response,
                    attempts=client_result.attempts,
                    elapsed_us=client_result.elapsed_us,
                    queue_enter_ns=batch.submitted_ns,
                    queue_exit_ns=queue_exit_ns,
                )
            except ArbiterError as exc:
                return BatchOutcome(
                    batch=batch,
                    response={},
                    attempts=int(exc.details.get("attempts", self.retries + 1)),
                    elapsed_us=int(exc.details.get("elapsed_us", 0)),
                    queue_enter_ns=batch.submitted_ns,
                    queue_exit_ns=queue_exit_ns,
                    error=exc,
                )
            finally:
                self.stats.leave(batch.node_id)
                global_sem.release()

        futures: list[Future[BatchOutcome]] = []
        try:
            for node_id, node_batches in sorted(by_node.items()):
                executor = ThreadPoolExecutor(
                    max_workers=min(self.per_node_concurrency, len(node_batches)),
                    thread_name_prefix=f"batch-{node_id}",
                )
                executors[node_id] = executor
                for batch in node_batches:
                    futures.append(executor.submit(execute, batch))

            for future in as_completed(futures):
                try:
                    outcome = future.result()
                except Exception as exc:
                    error = ArbiterError(
                        "E_PROTOCOL",
                        f"scheduler worker failed: {type(exc).__name__}: {exc}",
                    )
                    errors.append(error)
                    if abort_on_error:
                        abort_event.set()
                    continue
                with outcomes_lock:
                    outcomes.append(outcome)
                    if outcome.error is not None:
                        errors.append(outcome.error)
                        if abort_on_error:
                            abort_event.set()
                if on_result is not None:
                    try:
                        on_result(outcome)
                    except ArbiterError as exc:
                        errors.append(exc)
                        if abort_on_error:
                            abort_event.set()
                    except Exception as exc:
                        errors.append(
                            ArbiterError(
                                "E_PROTOCOL",
                                f"result callback failed: {type(exc).__name__}: {exc}",
                                details={"batch_index": outcome.batch.batch_index},
                            )
                        )
                        if abort_on_error:
                            abort_event.set()
        finally:
            for executor in executors.values():
                executor.shutdown(wait=True, cancel_futures=True)
        outcomes.sort(key=lambda outcome: outcome.batch.batch_index)
        errors.sort(key=lambda err: (err.details.get("batch_index", 10**18), err.code))
        return outcomes, errors

    def snapshot(self) -> ConcurrencySnapshot:
        return self.stats.snapshot()
