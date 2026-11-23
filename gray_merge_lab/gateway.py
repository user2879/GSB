from __future__ import annotations

import base64
import json
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json
from .engine import GatewayEngine, InferenceResult, ModelBank
from .errors import LabError
from .specs import BackendSpec, LoadedSpec, validate_samples


LOOPBACK = "127.0.0.1"


def _decode_response(line: bytes) -> dict[str, Any]:
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabError("E_RUNTIME", f"backend returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LabError("E_RUNTIME", "backend response must be a JSON object")
    if not value.get("ok"):
        raise LabError(
            value.get("error", "E_RUNTIME"),
            value.get("message", "backend request failed"),
            field=value.get("field"),
            batch_id=value.get("batch_id"),
            details=value.get("details", {}),
        )
    return value


class BackendClient:
    def __init__(self, spec: BackendSpec, timeout: float = 10.0) -> None:
        self.spec = spec
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.file: Any | None = None
        self.lock = threading.Lock()

    def connect(self, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                sock = socket.create_connection((self.spec.host, self.spec.port), timeout=0.5)
                sock.settimeout(self.timeout)
                self.sock = sock
                self.file = sock.makefile("rwb")
                response = self.request({"op": "ping"})
                if response.get("backend_id") != self.spec.backend_id:
                    raise LabError("E_RUNTIME", "backend ping identity mismatch")
                return
            except (OSError, LabError) as exc:
                last_error = exc
                if self.file is not None:
                    try:
                        self.file.close()
                    finally:
                        self.file = None
                if self.sock is not None:
                    self.sock.close()
                    self.sock = None
                time.sleep(0.05)
        raise LabError(
            "E_RUNTIME",
            f"cannot connect to backend {self.spec.backend_id} at {LOOPBACK}:{self.spec.port}: {last_error}",
            details={"backend_id": self.spec.backend_id, "port": self.spec.port},
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.sock is None or self.file is None:
                raise LabError("E_RUNTIME", "backend socket is not connected")
            try:
                self.file.write((canonical_json(payload) + "\n").encode("utf-8"))
                self.file.flush()
                line = self.file.readline()
            except OSError as exc:
                raise LabError("E_RUNTIME", f"backend communication failed: {exc}") from exc
            if not line:
                raise LabError("E_RUNTIME", "backend closed the connection")
            return _decode_response(line)

    def close(self) -> None:
        if self.file is not None:
            try:
                self.file.close()
            finally:
                self.file = None
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None


class WorkerManager:
    def __init__(self, backends: Iterable[BackendSpec]) -> None:
        self.backends = tuple(backends)
        self.processes: dict[str, subprocess.Popen[bytes]] = {}

    def start(self) -> "WorkerManager":
        project_root = Path(__file__).resolve().parent.parent
        for spec in self.backends:
            encoded = base64.b64encode(canonical_json(spec.declaration()).encode("utf-8")).decode("ascii")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gray_merge_lab.backend_service",
                    "--backend-b64",
                    encoded,
                    "--version",
                    "1",
                ],
                cwd=str(project_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self.processes[spec.backend_id] = process
        for spec in self.backends:
            client = BackendClient(spec)
            try:
                client.connect(timeout=12.0)
                client.close()
            except LabError:
                process = self.processes[spec.backend_id]
                stderr = b""
                if process.poll() is not None and process.stderr is not None:
                    stderr = process.stderr.read()
                self.stop()
                raise LabError(
                    "E_RUNTIME",
                    f"backend process failed to become ready: {spec.backend_id}",
                    details={
                        "backend_id": spec.backend_id,
                        "port": spec.port,
                        "stderr": stderr.decode("utf-8", "replace")[-2000:],
                    },
                )
        return self

    def stop(self) -> None:
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
        for process in self.processes.values():
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
            if process.stderr is not None:
                process.stderr.close()
        self.processes.clear()

    def __enter__(self) -> "WorkerManager":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


class TcpBackendTransport:
    def __init__(self, bank: ModelBank) -> None:
        self.bank = bank
        self.specs = {item.backend_id: item for item in bank.spec.backends}
        self.clients: dict[str, BackendClient] = {}
        self.lock = threading.Lock()

    def _client(self, backend_id: str) -> BackendClient:
        with self.lock:
            client = self.clients.get(backend_id)
            if client is None:
                spec = self.specs.get(backend_id)
                if spec is None:
                    raise LabError("E_RUNTIME", f"unknown backend {backend_id!r}")
                client = BackendClient(spec)
                client.connect()
                self.clients[backend_id] = client
            return client

    def infer(self, backend_id: str, samples: list[Any], batch_id: str) -> InferenceResult:
        params = self.bank.get(backend_id)
        payload = {
            "op": "infer",
            "batch_id": batch_id,
            "param_version": params.version,
            "samples": [
                {"sample_id": item.sample_id, "input_seed": item.input_seed} for item in samples
            ],
        }
        response = self._client(backend_id).request(payload)
        if response.get("backend_param_digest") != params.param_digest:
            raise LabError(
                "E_RUNTIME",
                "backend parameter digest differs from gateway model bank",
                batch_id=batch_id,
                details={
                    "backend": response.get("backend_param_digest"),
                    "gateway": params.param_digest,
                },
            )
        if response.get("backend_param_version") != params.version:
            raise LabError("E_RUNTIME", "backend parameter version mismatch", batch_id=batch_id)
        return InferenceResult(
            output_digest=response["output_digest"],
            sample_outputs=response["sample_outputs"],
            duration_ms=int(response["duration_ms"]),
            backend_param_digest=response["backend_param_digest"],
            backend_param_version=int(response["backend_param_version"]),
            input_digests=list(response["input_digests"]),
        )

    def update_backend(self, backend_id: str, spec: BackendSpec, version: int) -> None:
        self.specs[backend_id] = spec
        response = self._client(backend_id).request(
            {"op": "update", "spec": spec.declaration(), "version": version}
        )
        if response.get("summary", {}).get("param_digest") != self.bank.get(backend_id).param_digest:
            raise LabError("E_RUNTIME", "hot update digest differs between gateway and backend")

    def close(self) -> None:
        for client in self.clients.values():
            client.close()
        self.clients.clear()


class GatewayRequestHandler(socketserver.StreamRequestHandler):
    def _send(self, value: dict[str, Any]) -> None:
        self.wfile.write((canonical_json(value) + "\n").encode("utf-8"))
        self.wfile.flush()

    def handle(self) -> None:
        server: GatewayTCPServer = self.server  # type: ignore[assignment]
        for raw in self.rfile:
            try:
                value = json.loads(raw.decode("utf-8"))
                if not isinstance(value, dict):
                    raise LabError("E_SPEC", "request line must be a JSON object")
                operation = value.get("op")
                if operation == "flush":
                    self._send(server.flush())
                elif operation in {"sample", None}:
                    sample = value.get("sample") if operation == "sample" else value
                    if not isinstance(sample, dict):
                        raise LabError("E_SPEC", "sample must be a JSON object")
                    count = server.submit(sample)
                    self._send(
                        {
                            "ok": True,
                            "op": "accepted",
                            "sample_id": sample.get("sample_id"),
                            "pending_samples": count,
                        }
                    )
                else:
                    raise LabError("E_SPEC", "unsupported gateway operation", field="op")
            except LabError as exc:
                self._send({"ok": False, **exc.to_dict()})
            except Exception as exc:
                self._send(
                    {
                        "ok": False,
                        "error": "E_RUNTIME",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )


class GatewayTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    address_family = socket.AF_INET

    def __init__(self, engine: GatewayEngine, host: str = LOOPBACK, port: int = 0) -> None:
        if host != LOOPBACK:
            raise LabError("E_SPEC", "gateway host must be literal 127.0.0.1", field="host")
        self.engine = engine
        self.pending: list[dict[str, Any]] = []
        self.pending_lock = threading.Lock()
        self.flush_lock = threading.Lock()
        super().__init__((host, port), GatewayRequestHandler)

    @property
    def host(self) -> str:
        return self.server_address[0]

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def submit(self, sample: dict[str, Any]) -> int:
        with self.pending_lock:
            self.pending.append(dict(sample))
            return len(self.pending)

    def flush(self) -> dict[str, Any]:
        with self.flush_lock:
            with self.pending_lock:
                raw_samples = list(self.pending)
                self.pending.clear()
            if not raw_samples:
                return {"ok": True, "op": "flush", "samples": 0, "batches": 0}
            samples = validate_samples(raw_samples, self.engine.spec.rules)
            executions = self.engine.process_samples(samples)
            sample_to_batch: dict[str, dict[str, Any]] = {}
            batches = []
            for execution in executions:
                plan = execution.plan
                row = {
                    "batch_id": plan.batch_id,
                    "rule_id": plan.rule_id,
                    "destination": plan.destination,
                    "client_bucket": plan.buckets[0],
                    "client_key": plan.samples[0].client_key,
                    "sample_ids": plan.sample_ids(),
                    "output_digest": execution.conclusion["output_digest"],
                    "backend_param_digest": execution.conclusion["backend_param_digest"],
                    "backend_param_version": execution.conclusion["backend_param_version"],
                    "merge_seq": execution.conclusion["merge_seq"],
                    "duration_ms": execution.record["raw_duration_ms"],
                }
                batches.append(row)
                for sample_id in plan.sample_ids():
                    sample_to_batch[sample_id] = {
                        "batch_id": plan.batch_id,
                        "destination": plan.destination,
                    }
            return {
                "ok": True,
                "op": "flush",
                "samples": len(samples),
                "batches": len(executions),
                "batch_rows": batches,
                "sample_to_batch": sample_to_batch,
            }

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, name="gateway-jsonl", daemon=True)
        thread.start()
        return thread


class GatewayClient:
    def __init__(self, host: str = LOOPBACK, port: int = 0, timeout: float = 20.0) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.file = self.sock.makefile("rwb")

    def _round_trip(self, value: dict[str, Any]) -> dict[str, Any]:
        self.file.write((canonical_json(value) + "\n").encode("utf-8"))
        self.file.flush()
        line = self.file.readline()
        if not line:
            raise LabError("E_RUNTIME", "gateway closed the connection")
        response = json.loads(line.decode("utf-8"))
        if not isinstance(response, dict):
            raise LabError("E_RUNTIME", "gateway response must be an object")
        return response

    def accept(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self._round_trip({"op": "sample", "sample": sample})

    def flush(self) -> dict[str, Any]:
        response = self._round_trip({"op": "flush"})
        if not response.get("ok"):
            raise LabError(
                response.get("error", "E_RUNTIME"),
                response.get("message", "gateway flush failed"),
                field=response.get("field"),
                batch_id=response.get("batch_id"),
                details=response.get("details", {}),
            )
        return response

    def close(self) -> None:
        try:
            self.file.close()
        finally:
            self.sock.close()

    def __enter__(self) -> "GatewayClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
