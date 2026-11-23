"""Subprocess lifecycle and loopback JSON-lines client for inference workers."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .errors import HandshakeError, PortError, ReceiptError
from .model import Backend, Spec, canonical_json
from .parameters import BackendParameters, batch_output_digest, generate_parameters

PROTOCOL_VERSION = 1


def _monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class InferenceReceipt:
    backend_id: str
    port: int
    parameter_digest: str
    batch_id: str
    request_ids: tuple[int, ...]
    output_digests: Mapping[int, str]
    output_digest: str
    elapsed_ms: int
    worker_pid: int
    receipt_digest: str
    raw: Mapping[str, Any]

    @property
    def ordered_output_digests(self) -> dict[int, str]:
        return {
            request_id: self.output_digests[request_id]
            for request_id in sorted(self.output_digests)
        }


def _receipt_error(
    message: str,
    backend: Backend,
    *,
    batch_id: str | None = None,
    field: str | None = None,
) -> ReceiptError:
    return ReceiptError(
        message,
        backend_id=backend.id,
        port=backend.port,
        batch_id=batch_id,
        field=field,
    )


def validate_receipt(
    raw: Any,
    backend: Backend,
    expected_batch_id: str,
    expected_request_ids: Iterable[int],
    expected_parameter_digest: str,
) -> InferenceReceipt:
    if not isinstance(raw, dict):
        raise _receipt_error("receipt must be a JSON object", backend, batch_id=expected_batch_id)
    if raw.get("type") == "error":
        raise _receipt_error(
            f"worker rejected batch: {raw.get('message', 'unknown error')}",
            backend,
            batch_id=expected_batch_id,
            field="type",
        )
    required = (
        "type",
        "backend_id",
        "parameter_digest",
        "batch_id",
        "request_ids",
        "output_digests",
        "output_digest",
        "elapsed_ms",
        "worker_pid",
    )
    for field in required:
        if field not in raw:
            raise _receipt_error(
                f"receipt field {field!r} is missing",
                backend,
                batch_id=expected_batch_id,
                field=field,
            )
    if raw.get("type") != "receipt":
        raise _receipt_error(
            "receipt type must be 'receipt'", backend, batch_id=expected_batch_id, field="type"
        )
    if raw.get("backend_id") != backend.id:
        raise _receipt_error(
            "receipt backend_id does not match the requested backend",
            backend,
            batch_id=expected_batch_id,
            field="backend_id",
        )
    if raw.get("parameter_digest") != expected_parameter_digest:
        raise _receipt_error(
            "receipt parameter_digest does not match the parent construction",
            backend,
            batch_id=expected_batch_id,
            field="parameter_digest",
        )
    if raw.get("batch_id") != expected_batch_id:
        raise _receipt_error(
            "receipt batch_id does not match the submitted batch",
            backend,
            batch_id=expected_batch_id,
            field="batch_id",
        )
    expected_ids = set(expected_request_ids)
    raw_ids = raw.get("request_ids")
    if (
        not isinstance(raw_ids, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_ids)
        or len(raw_ids) != len(set(raw_ids))
        or set(raw_ids) != expected_ids
    ):
        raise _receipt_error(
            "receipt request_ids do not exactly match the batch membership",
            backend,
            batch_id=expected_batch_id,
            field="request_ids",
        )
    output_digests = raw.get("output_digests")
    if not isinstance(output_digests, dict):
        raise _receipt_error(
            "receipt output_digests must be an object",
            backend,
            batch_id=expected_batch_id,
            field="output_digests",
        )
    normalized_digests: dict[int, str] = {}
    for key, value in output_digests.items():
        try:
            request_id = int(key)
        except (TypeError, ValueError):
            raise _receipt_error(
                "receipt output_digests has a non-integer key",
                backend,
                batch_id=expected_batch_id,
                field="output_digests",
            )
        if not _is_digest(value):
            raise _receipt_error(
                "receipt output digest is malformed",
                backend,
                batch_id=expected_batch_id,
                field=f"output_digests.{key}",
            )
        normalized_digests[request_id] = value
    if set(normalized_digests) != expected_ids:
        raise _receipt_error(
            "receipt output_digests does not cover the exact batch membership",
            backend,
            batch_id=expected_batch_id,
            field="output_digests",
        )
    output_digest = raw.get("output_digest")
    if not _is_digest(output_digest):
        raise _receipt_error(
            "receipt output_digest is malformed",
            backend,
            batch_id=expected_batch_id,
            field="output_digest",
        )
    if output_digest != batch_output_digest(
        backend.id, expected_batch_id, normalized_digests
    ):
        raise _receipt_error(
            "receipt batch output_digest is inconsistent with request output summaries",
            backend,
            batch_id=expected_batch_id,
            field="output_digest",
        )
    elapsed_ms = raw.get("elapsed_ms")
    if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, int) or elapsed_ms < 0:
        raise _receipt_error(
            "receipt elapsed_ms must be a non-negative integer",
            backend,
            batch_id=expected_batch_id,
            field="elapsed_ms",
        )
    worker_pid = raw.get("worker_pid")
    if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 0:
        raise _receipt_error(
            "receipt worker_pid must be a positive integer",
            backend,
            batch_id=expected_batch_id,
            field="worker_pid",
        )
    receipt_digest = hashlib.sha256(
        b"gateway-receipt-v1\0" + canonical_json(raw).encode("utf-8")
    ).hexdigest()
    return InferenceReceipt(
        backend_id=backend.id,
        port=backend.port,
        parameter_digest=expected_parameter_digest,
        batch_id=expected_batch_id,
        request_ids=tuple(sorted(expected_ids)),
        output_digests=normalized_digests,
        output_digest=output_digest,
        elapsed_ms=elapsed_ms,
        worker_pid=worker_pid,
        receipt_digest=receipt_digest,
        raw=raw,
    )


class BackendProcess:
    def __init__(
        self,
        backend: Backend,
        parameters: BackendParameters,
        *,
        startup_timeout: float = 8.0,
        receipt_timeout: float = 20.0,
        worker_command: list[str] | None = None,
    ) -> None:
        self.backend = backend
        self.parameters = parameters
        self.startup_timeout = startup_timeout
        self.receipt_timeout = receipt_timeout
        self.worker_command = worker_command
        self.process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stderr_lines: deque[str] = deque(maxlen=40)
        self._reader_threads: list[threading.Thread] = []

    @property
    def stderr_tail(self) -> str:
        return "".join(self._stderr_lines).strip()

    def _command(self) -> list[str]:
        if self.worker_command is not None:
            return list(self.worker_command)
        return [
            sys.executable,
            "-u",
            "-m",
            "gateway_diag.worker",
            "--backend-id",
            self.backend.id,
            "--host",
            self.backend.host,
            "--port",
            str(self.backend.port),
            "--model-dim",
            str(self.backend.model_dim),
            "--hidden-dim",
            str(self.backend.hidden_dim),
            "--seed",
            str(self.backend.seed),
            "--quantization",
            self.backend.quantization,
        ]

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                line = self.process.stdout.readline()
                self._stdout_queue.put(line if line else None)
                if not line:
                    break
        finally:
            self._stdout_queue.put(None)

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while True:
            line = self.process.stderr.readline()
            if not line:
                break
            self._stderr_lines.append(line)

    def start(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        project_root = str(Path(__file__).resolve().parent.parent)
        existing_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            project_root + os.pathsep + existing_path if existing_path else project_root
        )
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                **kwargs,
            )
        except OSError as exc:
            raise PortError(
                f"failed to launch backend process: {exc}",
                backend_id=self.backend.id,
                port=self.backend.port,
            ) from exc
        stdout_thread = threading.Thread(
            target=self._read_stdout, name=f"stdout-{self.backend.id}", daemon=True
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr, name=f"stderr-{self.backend.id}", daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        self._reader_threads.extend([stdout_thread, stderr_thread])

        deadline = time.monotonic() + self.startup_timeout
        line: str | None = None
        while time.monotonic() < deadline:
            timeout = max(0.01, deadline - time.monotonic())
            try:
                line = self._stdout_queue.get(timeout=timeout)
            except queue.Empty:
                break
            if line is not None:
                break
        if line is None:
            self.close()
            detail = f"; stderr: {self.stderr_tail}" if self.stderr_tail else ""
            raise HandshakeError(
                f"backend handshake timed out after {self.startup_timeout:.1f}s{detail}",
                backend_id=self.backend.id,
                port=self.backend.port,
                field="ready",
            )
        try:
            ready = json.loads(line)
        except json.JSONDecodeError as exc:
            self.close()
            raise HandshakeError(
                f"backend emitted an invalid ready line: {exc.msg}",
                backend_id=self.backend.id,
                port=self.backend.port,
                field="ready",
            ) from exc
        if not isinstance(ready, dict):
            self.close()
            raise HandshakeError(
                "backend ready line must be an object",
                backend_id=self.backend.id,
                port=self.backend.port,
                field="ready",
            )
        if ready.get("type") == "error":
            code = ready.get("code")
            message = str(ready.get("message", "worker startup failed"))
            self.close()
            if code == "E_PORT":
                raise PortError(
                    message, backend_id=self.backend.id, port=self.backend.port
                )
            raise HandshakeError(
                message,
                backend_id=self.backend.id,
                port=self.backend.port,
                field="ready",
            )
        required = ("type", "backend_id", "port", "parameter_digest")
        missing = [field for field in required if field not in ready]
        if missing:
            self.close()
            raise HandshakeError(
                f"backend ready line is missing {missing[0]!r}",
                backend_id=self.backend.id,
                port=self.backend.port,
                field=missing[0],
            )
        if (
            ready.get("type") != "ready"
            or ready.get("backend_id") != self.backend.id
            or ready.get("port") != self.backend.port
            or ready.get("parameter_digest") != self.parameters.parameter_digest
        ):
            self.close()
            raise HandshakeError(
                "backend ready line does not match the declared backend and parameter digest",
                backend_id=self.backend.id,
                port=self.backend.port,
                field="ready",
            )
        try:
            ack = self._roundtrip({"type": "hello"}, timeout=self.startup_timeout)
        except (OSError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            self.close()
            raise HandshakeError(
                f"loopback handshake failed: {exc}",
                backend_id=self.backend.id,
                port=self.backend.port,
                field="hello_ack",
            ) from exc
        if (
            not isinstance(ack, dict)
            or ack.get("type") != "hello_ack"
            or ack.get("backend_id") != self.backend.id
            or ack.get("port") != self.backend.port
            or ack.get("parameter_digest") != self.parameters.parameter_digest
        ):
            self.close()
            raise HandshakeError(
                "backend hello acknowledgement is invalid",
                backend_id=self.backend.id,
                port=self.backend.port,
                field="hello_ack",
            )

    def _roundtrip(self, message: Mapping[str, Any], *, timeout: float) -> Any:
        payload = (canonical_json(dict(message)) + "\n").encode("utf-8")
        with socket.create_connection(
            (self.backend.host, self.backend.port), timeout=timeout
        ) as connection:
            connection.settimeout(timeout)
            connection.sendall(payload)
            with connection.makefile("rb") as stream:
                raw = stream.readline(10_000_001)
        if not raw:
            raise TimeoutError("backend closed the connection without a response")
        if len(raw) > 10_000_000:
            raise ValueError("backend response is too large")
        value = json.loads(raw.decode("utf-8"))
        return value

    def infer(
        self,
        batch_id: str,
        items: Iterable[tuple[int, Iterable[float]]],
    ) -> InferenceReceipt:
        materialized = [
            {"request_id": request_id, "input": [float(value) for value in vector]}
            for request_id, vector in items
        ]
        message = {
            "type": "infer",
            "backend_id": self.backend.id,
            "batch_id": batch_id,
            "items": materialized,
        }
        try:
            raw = self._roundtrip(message, timeout=self.receipt_timeout)
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            raise _receipt_error(
                f"loopback inference failed: {exc}",
                self.backend,
                batch_id=batch_id,
                field="receipt",
            ) from exc
        return validate_receipt(
            raw,
            self.backend,
            batch_id,
            [int(item["request_id"]) for item in materialized],
            self.parameters.parameter_digest,
        )

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write("shutdown\n")
                    process.stdin.flush()
                    process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        for stream in (process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        self.process = None

    def __enter__(self) -> "BackendProcess":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class BackendManager:
    def __init__(
        self,
        spec: Spec,
        parameters: Mapping[str, BackendParameters] | None = None,
        *,
        startup_timeout: float = 8.0,
        receipt_timeout: float = 20.0,
        worker_command_factory: Callable[[Backend], list[str]] | None = None,
    ) -> None:
        self.spec = spec
        self.parameters = dict(parameters or {}) or {
            backend.id: generate_parameters(backend) for backend in spec.backends
        }
        self.startup_timeout = startup_timeout
        self.receipt_timeout = receipt_timeout
        self.worker_command_factory = worker_command_factory
        self.processes: dict[str, BackendProcess] = {}

    def start(self) -> "BackendManager":
        try:
            for backend in sorted(self.spec.backends, key=lambda item: item.id):
                command = (
                    self.worker_command_factory(backend)
                    if self.worker_command_factory is not None
                    else None
                )
                process = BackendProcess(
                    backend,
                    self.parameters[backend.id],
                    startup_timeout=self.startup_timeout,
                    receipt_timeout=self.receipt_timeout,
                    worker_command=command,
                )
                process.start()
                self.processes[backend.id] = process
        except Exception:
            self.close()
            raise
        return self

    def infer(
        self,
        backend_id: str,
        batch_id: str,
        items: Iterable[tuple[int, Iterable[float]]],
    ) -> InferenceReceipt:
        process = self.processes.get(backend_id)
        if process is None:
            raise ReceiptError(
                f"backend {backend_id!r} is not running",
                backend_id=backend_id,
                field="backend_id",
            )
        return process.infer(batch_id, items)

    def close(self) -> None:
        for backend_id in sorted(self.processes, reverse=True):
            self.processes[backend_id].close()
        self.processes.clear()

    def __enter__(self) -> "BackendManager":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()


def millisecond_now() -> int:
    return _monotonic_ms()
