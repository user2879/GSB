"""Parent-side subprocess orchestration over genuine loopback sockets."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .engine import BatchPayload
from .errors import DiagnosticError
from .model import ParameterCache
from .protocol import LineReader, perform_handshake, require_fields, send_json_line
from .receipts import ReceiptContext, normalize_receipt
from .spec import WorkerSpec, WorkloadSpec, dump_workload


def ensure_port_available(
    host: str,
    port: int,
    *,
    field: str = "port",
    line: int | None = None,
    worker_id: str | None = None,
) -> None:
    if host != "127.0.0.1":
        raise DiagnosticError("E_SPEC", "only 127.0.0.1 is allowed", field=field, line=line)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        raise DiagnosticError(
            "E_PORT",
            f"loopback port {port} is unavailable: {exc}",
            field=field,
            line=line,
            details={"host": host, "port": port, "worker_id": worker_id, "errno": getattr(exc, "errno", None)},
        ) from exc
    finally:
        sock.close()


def probe_workload_ports(spec: WorkloadSpec) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for worker in sorted(spec.workers, key=lambda item: item.source_index):
        ensure_port_available(
            worker.host,
            worker.port,
            field=f"workers[{worker.source_index}].port",
            line=worker.source_line,
            worker_id=worker.worker_id,
        )
        results.append({"worker_id": worker.worker_id, "host": worker.host, "port": worker.port, "available": True})
    return {"host_scope": "loopback", "ports": results, "all_available": True}


def find_free_loopback_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _connect_loopback(host: str, port: int, timeout_s: float) -> socket.socket:
    # Linux can choose the destination port itself as the ephemeral source
    # port on loopback.  That creates a TCP self-connect and makes the client
    # read its own hello.  Bind a different source port explicitly and verify
    # the resulting 4-tuple before returning the socket.
    last_error: OSError | None = None
    for _attempt in range(16):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, 0))
            if int(sock.getsockname()[1]) == port:
                sock.close()
                continue
            sock.settimeout(max(0.001, timeout_s))
            sock.connect((host, port))
            if int(sock.getsockname()[1]) == port:
                sock.close()
                continue
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            raise
        except BaseException:
            sock.close()
            raise
    raise last_error or OSError("unable to allocate a non-self loopback source port")


class WorkerProcess:
    def __init__(
        self,
        spec: WorkloadSpec,
        worker: WorkerSpec,
        spec_path: str | Path,
        *,
        request_timeout_ms: int | None = None,
    ) -> None:
        self.spec = spec
        self.worker = worker
        self.spec_path = str(Path(spec_path).resolve())
        self.request_timeout_ms = request_timeout_ms or spec.limits.io_timeout_ms
        self.parameters = ParameterCache().get(worker, spec.audio.speaker_count)
        self.process: subprocess.Popen[str] | None = None
        self.ready: dict[str, Any] | None = None
        self.startup_socket: socket.socket | None = None

    def _wait_for_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.spec.limits.handshake_timeout_ms / 1000.0
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stderr = ""
                if self.process.stderr is not None:
                    try:
                        stderr = self.process.stderr.read()
                    except OSError:
                        stderr = ""
                raise DiagnosticError(
                    "E_PORT",
                    "worker subprocess exited before accepting a loopback connection",
                    field=f"workers[{self.worker.source_index}].port",
                    line=self.worker.source_line,
                    details={
                        "worker_id": self.worker.worker_id,
                        "port": self.worker.port,
                        "exit_code": self.process.returncode,
                        "stderr": stderr.strip()[:2000],
                    },
                )
            remaining = max(0.001, min(0.1, deadline - time.monotonic()))
            try:
                sock = _connect_loopback(self.worker.host, self.worker.port, remaining)
            except OSError as exc:
                last_error = exc
                continue
            reader = LineReader(sock)
            try:
                self.ready = perform_handshake(
                    sock,
                    reader,
                    expected_worker_id=self.worker.worker_id,
                    expected_param_version=self.worker.parameter_version,
                    expected_param_digest=self.parameters.digest,
                    timeout_ms=self.spec.limits.handshake_timeout_ms,
                    phase="startup_handshake",
                )
                self.startup_socket = sock
                return
            except BaseException:
                sock.close()
                raise
        raise DiagnosticError(
            "E_HANDSHAKE",
            "worker did not accept a loopback handshake before the deadline",
            field="socket.connect",
            details={
                "worker_id": self.worker.worker_id,
                "port": self.worker.port,
                "timeout_ms": self.spec.limits.handshake_timeout_ms,
                "timeout_source": "socket.connect",
                "last_error": str(last_error) if last_error else None,
            },
        )

    def start(self) -> "WorkerProcess":
        ensure_port_available(
            self.worker.host,
            self.worker.port,
            field=f"workers[{self.worker.source_index}].port",
            line=self.worker.source_line,
            worker_id=self.worker.worker_id,
        )
        package_root = Path(__file__).resolve().parent.parent
        environment = os.environ.copy()
        old_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(package_root) + (os.pathsep + old_pythonpath if old_pythonpath else "")
        command = [
            sys.executable,
            "-m",
            "speaker_diag.worker",
            "--spec",
            self.spec_path,
            "--worker-id",
            self.worker.worker_id,
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(package_root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            self._wait_for_ready()
        except BaseException:
            self.stop(force=True)
            raise
        return self

    def request_batch(self, payload: BatchPayload) -> dict[str, Any]:
        if self.process is None or self.ready is None:
            raise DiagnosticError("E_HANDSHAKE", "worker has not completed startup handshake", field="socket")
        if self.process.poll() is not None:
            raise DiagnosticError(
                "E_HANDSHAKE",
                "worker subprocess exited before the batch request",
                field="socket",
                details={"worker_id": self.worker.worker_id, "exit_code": self.process.returncode},
            )
        try:
            sock = _connect_loopback(
                self.worker.host,
                self.worker.port,
                self.spec.limits.handshake_timeout_ms / 1000.0,
            )
        except OSError as exc:
            raise DiagnosticError(
                "E_HANDSHAKE",
                f"cannot connect to worker loopback socket: {exc}",
                field="socket.connect",
                details={"worker_id": self.worker.worker_id, "timeout_source": "socket.connect"},
            ) from exc
        reader = LineReader(sock)
        try:
            perform_handshake(
                sock,
                reader,
                expected_worker_id=self.worker.worker_id,
                expected_param_version=self.worker.parameter_version,
                expected_param_digest=self.parameters.digest,
                timeout_ms=self.spec.limits.handshake_timeout_ms,
                phase="batch_handshake",
            )
            send_json_line(sock, payload.to_message())
            receipt = reader.receive(self.request_timeout_ms, phase="batch_receipt")
            require_fields(
                receipt,
                (
                    "batch_id",
                    "worker_id",
                    "param_version",
                    "elapsed_ms",
                    "parameter_digest",
                    "participant_window_ids",
                    "windows",
                ),
                phase="batch_receipt",
            )
            if receipt["batch_id"] != payload.batch_id:
                raise DiagnosticError("E_FIELD", "receipt batch_id does not match request", field="batch_id")
            if receipt["worker_id"] != payload.worker_id:
                raise DiagnosticError("E_FIELD", "receipt worker_id does not match request", field="worker_id")
            if receipt["param_version"] != payload.param_version:
                raise DiagnosticError("E_FIELD", "receipt parameter version mismatch", field="param_version")
            if receipt["parameter_digest"] != payload.param_digest:
                raise DiagnosticError("E_FIELD", "receipt parameter digest mismatch", field="parameter_digest")
            if list(receipt["participant_window_ids"]) != list(payload.participant_window_ids):
                raise DiagnosticError(
                    "E_FIELD",
                    "receipt participant window set/order mismatch",
                    field="participant_window_ids",
                )
            return receipt
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def stop(self, *, force: bool = False) -> None:
        if self.startup_socket is not None:
            try:
                send_json_line(self.startup_socket, {"type": "shutdown"})
            except OSError:
                pass
            try:
                self.startup_socket.close()
            except OSError:
                pass
            self.startup_socket = None
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=1.0)
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        self.process.kill()
                        self.process.wait(timeout=1.0)
                    except (subprocess.TimeoutExpired, OSError):
                        pass
            for stream in (self.process.stdout, self.process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self.process = None
            self.ready = None

    def __enter__(self) -> "WorkerProcess":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop(force=exc is not None)


def write_runtime_spec(spec: WorkloadSpec, path: str | Path) -> Path:
    target = Path(path)
    target.write_text(dump_workload(spec), encoding="utf-8")
    return target


def start_workers(
    spec: WorkloadSpec,
    spec_path: str | Path,
    *,
    worker_ids: list[str] | None = None,
) -> dict[str, WorkerProcess]:
    selected = set(worker_ids) if worker_ids is not None else None
    workers: list[WorkerProcess] = []
    try:
        for worker in spec.workers:
            if selected is not None and worker.worker_id not in selected:
                continue
            workers.append(WorkerProcess(spec, worker, spec_path).start())
    except BaseException:
        for worker in workers:
            worker.stop(force=True)
        raise
    return {worker.worker.worker_id: worker for worker in workers}


def stop_workers(workers: dict[str, WorkerProcess]) -> None:
    for worker_id in sorted(workers):
        workers[worker_id].stop()


def validate_receipt_against_context(
    receipt: dict[str, Any],
    context: ReceiptContext,
) -> None:
    normalized = normalize_receipt(receipt, context=context)
    if normalized.batch_id != context.batch_id:
        raise DiagnosticError("E_FIELD", "normalized batch does not match context", field="batch_id")
