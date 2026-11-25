"""Worker subprocess lifecycle and protocol client."""

from __future__ import annotations

import base64
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import DiagnosticError, is_int
from .jsonio import canonical_bytes, canonical_json, parse_json_text
from .model import batch_output_digest, output_digest
from .planner import BatchPlan
from .receipts import HEX_64
from .spec import RequestSpec, WorkerSpec


PROTOCOL_VERSION = 1
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
R3_FIELDS = {
    "batch_id",
    "request_id",
    "output_digest",
    "param_version_seq",
    "latency_ms",
    "worker_id",
    "rank_in_batch",
    "superseded_request_ids",
    "param_digest",
}


@dataclass(frozen=True)
class WorkerResponse:
    batch_id: str
    worker_id: str
    param_version_seq: int
    param_digest: str
    latency_ms: int
    batch_output_digest: str
    receipts: tuple[dict[str, Any], ...]


def _is_port_available(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


class WorkerProcess:
    def __init__(self, worker: WorkerSpec, *, startup_timeout: float = 8.0) -> None:
        self.worker = worker
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> "WorkerProcess":
        if not _is_port_available(self.worker.host, self.worker.port):
            raise DiagnosticError(
                code="E_PORT",
                message=(
                    f"port {self.worker.port} on {self.worker.host} is already in use"
                ),
                object=self.worker.worker_id,
                field="port",
            )
        encoded = base64.b64encode(
            canonical_json(self.worker.to_dict()).encode("utf-8")
        ).decode("ascii")
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1])
        previous = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            source_root if not previous else source_root + os.pathsep + previous
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "inference_attribution.worker_server",
                "--spec-b64",
                encoded,
                "--parent-pid",
                str(os.getpid()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                message = self._stderr_text()
                raise DiagnosticError(
                    code="E_PORT",
                    message=f"worker exited before listening: {message}",
                    object=self.worker.worker_id,
                    field="port",
                )
            try:
                with socket.create_connection(
                    (self.worker.host, self.worker.port), timeout=0.25
                ):
                    return self
            except OSError:
                time.sleep(0.02)
        self.stop()
        raise DiagnosticError(
            code="E_PORT",
            message=f"worker did not listen within {self.startup_timeout}s",
            object=self.worker.worker_id,
            field="port",
        )

    def _stderr_text(self) -> str:
        if self.process is None or self.process.stderr is None:
            return "no stderr"
        try:
            return self.process.stderr.read().strip() or "no stderr"
        except Exception:
            return "stderr unavailable"

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.process.stderr is not None:
            self.process.stderr.close()
        self.process = None

    def __enter__(self) -> "WorkerProcess":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


class WorkerClient:
    def __init__(self, worker: WorkerSpec, *, timeout: float = 30.0) -> None:
        self.worker = worker
        self.timeout = timeout

    def dispatch_batch(
        self,
        batch: BatchPlan,
        requests: dict[str, RequestSpec],
    ) -> WorkerResponse:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "batch_id": batch.batch_id,
            "worker_id": batch.worker_id,
            "batch_seq": batch.batch_seq,
            "requests": [
                {
                    "request_id": request_id,
                    "arrival_seq": requests[request_id].arrival_seq,
                    "input": list(requests[request_id].input),
                }
                for request_id in batch.request_ids
            ],
        }
        try:
            with socket.create_connection(
                (self.worker.host, self.worker.port), timeout=self.timeout
            ) as connection:
                connection.settimeout(self.timeout)
                connection.sendall(canonical_bytes(payload) + b"\n")
                stream = connection.makefile("rb")
                line = stream.readline(MAX_RESPONSE_BYTES + 1)
        except OSError as exc:
            raise DiagnosticError(
                code="E_PROTO",
                message=f"worker protocol connection failed: {exc}",
                object=batch.batch_id,
                field="<socket>",
            ) from exc
        if not line:
            raise DiagnosticError(
                code="E_PROTO",
                message="worker returned an empty response",
                object=batch.batch_id,
                field="<response>",
            )
        if len(line) > MAX_RESPONSE_BYTES:
            raise DiagnosticError(
                code="E_PROTO",
                message="worker response exceeds size limit",
                object=batch.batch_id,
                field="<response>",
            )
        response = parse_json_text(
            line.decode("utf-8"), object_name=batch.batch_id, code="E_PROTO"
        )
        return self._validate_response(response, batch, requests)

    def _validate_response(
        self,
        response: Any,
        batch: BatchPlan,
        requests: dict[str, RequestSpec],
    ) -> WorkerResponse:
        if not isinstance(response, dict):
            raise self._proto(batch, "<response>", "response must be a JSON object")
        if response.get("ok") is not True:
            error = response.get("error")
            message = canonical_json(error) if isinstance(error, dict) else "worker rejected batch"
            raise self._proto(batch, "ok", message)
        required_top = {
            "protocol_version",
            "batch_id",
            "worker_id",
            "param_version_seq",
            "param_digest",
            "latency_ms",
            "batch_output_digest",
            "receipts",
        }
        for field in sorted(required_top):
            if field not in response:
                raise self._proto(batch, field, "missing response field")
        if response["protocol_version"] != PROTOCOL_VERSION:
            raise self._proto(batch, "protocol_version", "protocol version mismatch")
        if response["batch_id"] != batch.batch_id:
            raise self._proto(batch, "batch_id", "response batch id mismatch")
        if response["worker_id"] != batch.worker_id:
            raise self._proto(batch, "worker_id", "response worker id mismatch")
        if response["param_version_seq"] != batch.param_version_seq:
            raise self._proto(
                batch, "param_version_seq", "response parameter version mismatch"
            )
        if response["param_digest"] != batch.param_digest:
            raise self._proto(batch, "param_digest", "response parameter digest mismatch")
        if not is_int(response["latency_ms"]) or response["latency_ms"] < 0:
            raise self._proto(batch, "latency_ms", "invalid latency")
        if (
            not isinstance(response["batch_output_digest"], str)
            or HEX_64.fullmatch(response["batch_output_digest"]) is None
        ):
            raise self._proto(batch, "batch_output_digest", "invalid batch digest")
        raw_receipts = response["receipts"]
        if not isinstance(raw_receipts, list) or not raw_receipts:
            raise self._proto(batch, "receipts", "expected a non-empty receipt array")

        receipts: list[dict[str, Any]] = []
        outputs: dict[str, str] = {}
        ranks: set[int] = set()
        request_ids: set[str] = set()
        for index, receipt in enumerate(raw_receipts):
            prefix = f"receipts[{index}]"
            if not isinstance(receipt, dict):
                raise self._proto(batch, prefix, "receipt must be an object")
            for field in sorted(R3_FIELDS):
                if field not in receipt:
                    raise self._proto(batch, f"{prefix}.{field}", "missing receipt field")
            if receipt["batch_id"] != batch.batch_id:
                raise self._proto(batch, f"{prefix}.batch_id", "receipt batch mismatch")
            if receipt["worker_id"] != batch.worker_id:
                raise self._proto(batch, f"{prefix}.worker_id", "receipt worker mismatch")
            if receipt["param_version_seq"] != batch.param_version_seq:
                raise self._proto(
                    batch,
                    f"{prefix}.param_version_seq",
                    "receipt parameter version mismatch",
                )
            if receipt["param_digest"] != batch.param_digest:
                raise self._proto(
                    batch, f"{prefix}.param_digest", "receipt parameter digest mismatch"
                )
            if not is_int(receipt["latency_ms"]) or receipt["latency_ms"] < 0:
                raise self._proto(batch, f"{prefix}.latency_ms", "invalid latency")
            if not is_int(receipt["rank_in_batch"]) or receipt["rank_in_batch"] < 0:
                raise self._proto(batch, f"{prefix}.rank_in_batch", "invalid rank")
            if not isinstance(receipt["superseded_request_ids"], list):
                raise self._proto(
                    batch,
                    f"{prefix}.superseded_request_ids",
                    "invalid duplicate-delivery references",
                )
            request_id = receipt["request_id"]
            if not isinstance(request_id, str) or request_id not in requests:
                raise self._proto(batch, f"{prefix}.request_id", "unknown request")
            if request_id in request_ids:
                raise self._proto(batch, f"{prefix}.request_id", "duplicate request receipt")
            request_ids.add(request_id)
            rank = int(receipt["rank_in_batch"])
            if rank >= len(batch.request_ids) or rank in ranks:
                raise self._proto(batch, f"{prefix}.rank_in_batch", "rank out of bounds or duplicate")
            ranks.add(rank)
            digest = receipt["output_digest"]
            if not isinstance(digest, str) or HEX_64.fullmatch(digest) is None:
                raise self._proto(batch, f"{prefix}.output_digest", "invalid output digest")
            expected_digest = output_digest(
                self.worker, requests[request_id].input
            )
            if digest != expected_digest:
                raise self._proto(
                    batch,
                    f"{prefix}.output_digest",
                    "worker output digest does not match local deterministic forward pass",
                )
            outputs[request_id] = digest
            receipts.append(dict(receipt))
        if request_ids != set(batch.request_ids):
            raise self._proto(batch, "receipts", "receipt request set does not match batch")
        expected_batch_digest = batch_output_digest(
            outputs,
            param_version_seq=batch.param_version_seq,
            param_digest=batch.param_digest,
        )
        if response["batch_output_digest"] != expected_batch_digest:
            raise self._proto(
                batch,
                "batch_output_digest",
                "batch output digest does not bind the receipt set",
            )
        ordered_receipts = tuple(
            sorted(receipts, key=lambda item: (int(item["rank_in_batch"]), item["request_id"]))
        )
        return WorkerResponse(
            batch_id=batch.batch_id,
            worker_id=batch.worker_id,
            param_version_seq=batch.param_version_seq,
            param_digest=batch.param_digest,
            latency_ms=int(response["latency_ms"]),
            batch_output_digest=expected_batch_digest,
            receipts=ordered_receipts,
        )

    @staticmethod
    def _proto(batch: BatchPlan, field: str, message: str) -> DiagnosticError:
        return DiagnosticError(
            code="E_PROTO",
            message=message,
            object=batch.batch_id,
            field=field,
        )


class WorkerPool:
    def __init__(self, workers: Iterable[WorkerSpec]) -> None:
        self.worker_specs = {worker.worker_id: worker for worker in workers}
        self.processes: dict[str, WorkerProcess] = {}

    def start(self) -> "WorkerPool":
        try:
            for worker_id in sorted(self.worker_specs):
                process = WorkerProcess(self.worker_specs[worker_id])
                process.start()
                self.processes[worker_id] = process
        except Exception:
            self.stop()
            raise
        return self

    def client(self, worker_id: str) -> WorkerClient:
        return WorkerClient(self.worker_specs[worker_id])

    def stop(self) -> None:
        for worker_id in sorted(self.processes, reverse=True):
            self.processes[worker_id].stop()
        self.processes.clear()

    def __enter__(self) -> "WorkerPool":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
