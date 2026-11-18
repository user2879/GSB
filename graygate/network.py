from __future__ import annotations

import json
import socket
import socketserver
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

from .backend import BackendResult, ForwardEngine
from .errors import GatewayError, RequestError
from .spec import LOOPBACK_HOST, GatewaySpec
from .util import canonical_json


def send_json_line(
    port: int,
    payload: dict[str, Any],
    *,
    host: str = LOOPBACK_HOST,
    timeout: float = 10.0,
) -> dict[str, Any]:
    wire = (canonical_json(payload) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.sendall(wire)
        stream = connection.makefile("rb")
        line = stream.readline()
    if not line:
        raise RequestError(
            "E_TRANSPORT", "$.network", f"connection to {host}:{port} closed without reply"
        )
    try:
        response = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RequestError(
            "E_TRANSPORT", "$.network", f"invalid JSON reply from {host}:{port}"
        ) from exc
    if not isinstance(response, dict):
        raise RequestError(
            "E_TRANSPORT", "$.network", f"non-object reply from {host}:{port}"
        )
    return response


class _ThreadingJsonServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _BackendHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                request = json.loads(line.decode("utf-8"))
                if not isinstance(request, dict):
                    raise RequestError(
                        "E_REQUEST", "$.backend", "request must be an object"
                    )
                if request.get("op") != "infer":
                    raise RequestError(
                        "E_REQUEST", "$.backend.op", "expected 'infer'"
                    )
                version = request.get("model_version")
                samples = request.get("samples")
                if not isinstance(version, str):
                    raise RequestError(
                        "E_REQUEST",
                        "$.backend.model_version",
                        "must be a string",
                    )
                if not isinstance(samples, list):
                    raise RequestError(
                        "E_REQUEST", "$.backend.samples", "must be an array"
                    )
                engine: ForwardEngine = self.server.engine  # type: ignore[attr-defined]
                results = engine.infer_batch(version, samples)
                response = {
                    "ok": True,
                    "results": [result.canonical() for result in results],
                }
            except GatewayError as exc:
                response = exc.to_dict()
            except Exception as exc:  # pragma: no cover - protocol safety net
                response = {
                    "ok": False,
                    "code": "E_BACKEND",
                    "field_path": "$.backend",
                    "message": str(exc),
                }
            self.wfile.write((canonical_json(response) + "\n").encode("utf-8"))
            self.wfile.flush()


class BackendTCPServer(_ThreadingJsonServer):
    def __init__(self, address: tuple[str, int], engine: ForwardEngine):
        self.engine = engine
        super().__init__(address, _BackendHandler)


@dataclass
class _WaitResult:
    result: BackendResult | None = None
    batch_seq: int | None = None
    error: BaseException | None = None


@dataclass
class _BatchJob:
    sample: Sequence[float]
    event: threading.Event
    outcome: _WaitResult | None = None


class BatchDispatcher:
    """Groups samples by rule/backend while keeping each inference deterministic."""

    def __init__(
        self,
        client: Any,
        *,
        wait_window_ms: int = 2,
    ):
        self.client = client
        self.wait_window_ms = wait_window_ms
        self._condition = threading.Condition()
        self._pending: dict[tuple[str, str, int], deque[_BatchJob]] = {}
        self._batch_counters: dict[tuple[str, str, int], int] = {}
        self._stop = False
        self._thread = threading.Thread(
            target=self._run, name="batch-dispatcher", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        *,
        rule_id: str,
        model_version: str,
        batch_size: int,
        sample: Sequence[float],
    ) -> tuple[BackendResult, int]:
        key = (rule_id, model_version, batch_size)
        job = _BatchJob(sample=sample, event=threading.Event())
        with self._condition:
            if self._stop:
                raise RequestError(
                    "E_TRANSPORT", "$.gateway", "batch dispatcher is stopped"
                )
            self._pending.setdefault(key, deque()).append(job)
            self._condition.notify_all()
        job.event.wait(timeout=30.0)
        if not job.event.is_set() or job.outcome is None:
            raise RequestError(
                "E_TRANSPORT", "$.gateway", "batch inference timed out"
            )
        if job.outcome.error is not None:
            if isinstance(job.outcome.error, GatewayError):
                raise job.outcome.error
            raise RequestError(
                "E_BACKEND", "$.backend", str(job.outcome.error)
            ) from job.outcome.error
        assert job.outcome.result is not None
        assert job.outcome.batch_seq is not None
        return job.outcome.result, job.outcome.batch_seq

    def _next_batch(self) -> tuple[tuple[str, str, int], list[_BatchJob]] | None:
        with self._condition:
            while not self._pending and not self._stop:
                self._condition.wait()
            if self._stop and not self._pending:
                return None
            key = next(iter(self._pending))
            queue = self._pending[key]
            batch_size = key[2]
            batch: list[_BatchJob] = []
            while queue and len(batch) < batch_size:
                batch.append(queue.popleft())
            if not queue:
                del self._pending[key]
            return key, batch

    def _collect_remaining(
        self,
        key: tuple[str, str, int],
        batch: list[_BatchJob],
        deadline: float,
    ) -> None:
        batch_size = key[2]
        while len(batch) < batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            with self._condition:
                queue = self._pending.get(key)
                if queue:
                    batch.append(queue.popleft())
                    if not queue:
                        del self._pending[key]
                    continue
                self._condition.wait(timeout=remaining)

    def _run(self) -> None:
        while True:
            selected = self._next_batch()
            if selected is None:
                return
            key, batch = selected
            deadline = time.monotonic() + (self.wait_window_ms / 1000.0)
            self._collect_remaining(key, batch, deadline)
            with self._condition:
                self._batch_counters[key] = self._batch_counters.get(key, 0) + 1
                batch_seq = self._batch_counters[key]
            results: list[BackendResult] | None = None
            error: BaseException | None = None
            try:
                results = self.client.infer(
                    key[1], [job.sample for job in batch]
                )
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"backend returned {len(results)} results for {len(batch)} samples"
                    )
            except BaseException as exc:  # propagate to every waiter
                error = exc
            for index, job in enumerate(batch):
                job.outcome = _WaitResult(
                    result=None if results is None else results[index],
                    batch_seq=batch_seq,
                    error=error,
                )
                job.event.set()

    def close(self) -> None:
        with self._condition:
            self._stop = True
            pending = [
                job for queue in self._pending.values() for job in queue
            ]
            self._pending.clear()
            self._condition.notify_all()
        error = RequestError(
            "E_TRANSPORT", "$.gateway", "batch dispatcher stopped"
        )
        for job in pending:
            job.outcome = _WaitResult(error=error)
            job.event.set()
        self._thread.join(timeout=5.0)


class RemoteBackendClient:
    def __init__(self, spec: GatewaySpec, *, timeout: float = 10.0):
        self.spec = spec
        self.timeout = timeout

    def infer(
        self, model_version: str, samples: Sequence[Sequence[float]]
    ) -> list[BackendResult]:
        model = self.spec.models_by_version[model_version]
        response = send_json_line(
            model.port,
            {
                "op": "infer",
                "model_version": model_version,
                "samples": [list(sample) for sample in samples],
            },
            timeout=self.timeout,
        )
        if not response.get("ok"):
            raise RequestError(
                str(response.get("code", "E_BACKEND")),
                str(response.get("field_path", "$.backend")),
                str(response.get("message", "backend request failed")),
            )
        results = response.get("results")
        if not isinstance(results, list):
            raise RequestError(
                "E_BACKEND", "$.backend.results", "must be an array"
            )
        parsed: list[BackendResult] = []
        for index, item in enumerate(results):
            if not isinstance(item, dict):
                raise RequestError(
                    "E_BACKEND", f"$.backend.results[{index}]", "must be an object"
                )
            digest = item.get("output_digest")
            elapsed = item.get("elapsed_us")
            value = item.get("output_value")
            if not isinstance(digest, str):
                raise RequestError(
                    "E_BACKEND",
                    f"$.backend.results[{index}].output_digest",
                    "must be a string",
                )
            if not isinstance(elapsed, int) or isinstance(elapsed, bool) or elapsed < 0:
                raise RequestError(
                    "E_BACKEND",
                    f"$.backend.results[{index}].elapsed_us",
                    "must be a non-negative integer",
                )
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RequestError(
                    "E_BACKEND",
                    f"$.backend.results[{index}].output_value",
                    "must be a number",
                )
            parsed.append(BackendResult(digest, elapsed, float(value)))
        return parsed


class BackendCluster:
    def __init__(self, spec: GatewaySpec, engine: ForwardEngine):
        self.spec = spec
        self.engine = engine
        self.servers: list[BackendTCPServer] = []
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        try:
            for model in self.spec.models:
                server = BackendTCPServer((model.host, model.port), self.engine)
                thread = threading.Thread(
                    target=server.serve_forever,
                    name=f"backend-{model.version}",
                    daemon=True,
                )
                thread.start()
                self.servers.append(server)
                self.threads.append(thread)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        for server in self.servers:
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=2.0)
        self.servers.clear()
        self.threads.clear()

