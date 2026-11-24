from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .errors import ProtocolError
from .spec import EnvironmentSpec, SessionSpec, SliceSpec


@dataclass(frozen=True)
class Receipt:
    request_id: str
    service_id: str
    slice_version: int
    aggregate_summary: dict[str, int]
    sample_digest: str
    digest: str
    elapsed_ms_monotonic: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "service_id": self.service_id,
            "slice_version": self.slice_version,
            "aggregate_summary": dict(sorted(self.aggregate_summary.items())),
            "sample_digest": self.sample_digest,
            "digest": self.digest,
            "elapsed_ms_monotonic": self.elapsed_ms_monotonic,
        }


class ServiceProcess:
    def __init__(self, environment: EnvironmentSpec, seed: int) -> None:
        self.environment = environment
        self.seed = seed
        self.process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self._stderr: list[str] = []
        self._pump_threads: list[threading.Thread] = []

    def start(self, timeout_ms: int) -> None:
        if self.process is not None:
            return
        package_root = str(Path(__file__).resolve().parent.parent)
        env = os.environ.copy()
        previous_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = package_root if not previous_path else package_root + os.pathsep + previous_path
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            "-u",
            "-m",
            "config_arbiter.service",
            "--service-id",
            self.environment.id,
            "--port",
            str(self.environment.port),
            "--seed",
            str(self.seed),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            raise ProtocolError(
                "E_PORT",
                f"$.environments.{self.environment.id}.port",
                f"cannot start service subprocess: {exc}",
                environment=self.environment.id,
                port=self.environment.port,
            ) from exc
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._start_pump(self.process.stdout, "stdout")
        self._start_pump(self.process.stderr, "stderr")
        self._await_ready(timeout_ms)

    def _start_pump(self, stream: Any, kind: str) -> None:
        def pump() -> None:
            try:
                for line in iter(stream.readline, ""):
                    self._lines.put((kind, line.rstrip("\r\n")))
            finally:
                self._lines.put((kind, None))

        thread = threading.Thread(target=pump, name=f"service-{kind}-{self.environment.id}", daemon=True)
        thread.start()
        self._pump_threads.append(thread)

    def _await_ready(self, timeout_ms: int) -> None:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                raise ProtocolError(
                    "E_TIMEOUT",
                    "service.ready",
                    f"service {self.environment.id!r} did not become ready in {timeout_ms} ms",
                    environment=self.environment.id,
                    port=self.environment.port,
                )
            try:
                kind, line = self._lines.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                if self.process is not None and self.process.poll() is not None:
                    details = "\n".join(self._stderr[-5:])
                    self.stop()
                    raise ProtocolError(
                        "E_PORT",
                        f"$.environments.{self.environment.id}.port",
                        f"service exited before ready: {details or 'no stderr'}",
                        environment=self.environment.id,
                        port=self.environment.port,
                    )
                continue
            if line is None:
                continue
            if kind == "stderr":
                self._stderr.append(line)
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self._stderr.append(line)
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("event") == "READY":
                return
            if payload.get("event") == "ERROR":
                self.stop()
                code = payload.get("code") or "E_PORT"
                raise ProtocolError(
                    code if code in {"E_PORT", "E_TIMEOUT", "E_FIELD"} else "E_PORT",
                    f"$.environments.{self.environment.id}.port",
                    str(payload.get("message", "service failed to start")),
                    environment=self.environment.id,
                    port=self.environment.port,
                )

    def send_slice(self, slice_spec: SliceSpec, session: SessionSpec) -> Receipt:
        if self.process is None or self.process.poll() is not None:
            raise ProtocolError(
                "E_PORT",
                "socket.connect",
                f"service {self.environment.id!r} is not running",
                environment=self.environment.id,
                port=self.environment.port,
            )
        timeout_s = session.handshake_timeout_ms / 1000.0
        request_id = f"{self.environment.id}:{slice_spec.version}:{time.monotonic_ns()}"
        request = {
            "request_id": request_id,
            "service_id": self.environment.id,
            "scale_exponent": session.scale_exponent,
            "seed": session.seed,
            "sample_count": session.sample_count,
            "group_size": session.group_size,
            "noise_bound": session.noise_bound,
            "slice": {"version": slice_spec.version, "values": dict(slice_spec.values)},
        }
        try:
            with socket.create_connection(
                ("127.0.0.1", self.environment.port), timeout=timeout_s
            ) as connection:
                connection.settimeout(timeout_s)
                connection.sendall(canonical_json(request).encode("utf-8") + b"\n")
                reader = connection.makefile("rb")
                raw = reader.readline(1_048_577)
        except (socket.timeout, TimeoutError) as exc:
            raise ProtocolError(
                "E_TIMEOUT",
                "socket.read",
                f"handshake with {self.environment.id!r} timed out after {session.handshake_timeout_ms} ms",
                environment=self.environment.id,
                port=self.environment.port,
            ) from exc
        except ConnectionRefusedError as exc:
            raise ProtocolError(
                "E_PORT",
                "socket.connect",
                f"connection refused by {self.environment.id!r} on port {self.environment.port}",
                environment=self.environment.id,
                port=self.environment.port,
            ) from exc
        except OSError as exc:
            raise ProtocolError(
                "E_PORT",
                "socket",
                f"socket failure for {self.environment.id!r}: {exc}",
                environment=self.environment.id,
                port=self.environment.port,
            ) from exc
        if not raw:
            raise ProtocolError(
                "E_FIELD",
                "response",
                f"service {self.environment.id!r} closed the socket without a receipt",
                environment=self.environment.id,
            )
        if len(raw) > 1_048_576:
            raise ProtocolError(
                "E_FIELD",
                "response",
                f"receipt from {self.environment.id!r} exceeds 1 MiB",
                environment=self.environment.id,
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(
                "E_FIELD",
                "response",
                f"receipt from {self.environment.id!r} is not valid UTF-8 JSON",
                environment=self.environment.id,
            ) from exc
        if not isinstance(payload, dict):
            raise ProtocolError(
                "E_FIELD",
                "response",
                "receipt must be a JSON object",
                environment=self.environment.id,
            )
        if "error" in payload:
            error = payload["error"]
            if isinstance(error, dict):
                code = str(error.get("code", "E_FIELD"))
                raise ProtocolError(
                    code if code in {"E_PORT", "E_TIMEOUT", "E_FIELD"} else "E_FIELD",
                    str(error.get("field", "response")),
                    str(error.get("message", "service rejected request")),
                    environment=self.environment.id,
                )
            raise ProtocolError(
                "E_FIELD", "response.error", "malformed service error", environment=self.environment.id
            )
        required = (
            "service_id",
            "slice_version",
            "aggregate_summary",
            "sample_digest",
            "digest",
            "elapsed_ms_monotonic",
        )
        for field in required:
            if field not in payload:
                raise ProtocolError(
                    "E_FIELD",
                    f"response.{field}",
                    f"receipt from {self.environment.id!r} is missing {field!r}",
                    environment=self.environment.id,
                )
        if payload["service_id"] != self.environment.id:
            raise ProtocolError(
                "E_FIELD",
                "response.service_id",
                f"expected service_id {self.environment.id!r}, got {payload['service_id']!r}",
                environment=self.environment.id,
            )
        if payload["slice_version"] != slice_spec.version:
            raise ProtocolError(
                "E_FIELD",
                "response.slice_version",
                f"expected slice_version {slice_spec.version}, got {payload['slice_version']!r}",
                environment=self.environment.id,
            )
        aggregates = payload["aggregate_summary"]
        if not isinstance(aggregates, dict):
            raise ProtocolError(
                "E_FIELD",
                "response.aggregate_summary",
                "aggregate_summary must be an object",
                environment=self.environment.id,
            )
        normalized_aggregates: dict[str, int] = {}
        for key in sorted(aggregates):
            value = aggregates[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ProtocolError(
                    "E_FIELD",
                    f"response.aggregate_summary.{key}",
                    "aggregate values must be integers",
                    environment=self.environment.id,
                )
            normalized_aggregates[key] = value
        for key in sorted(slice_spec.values):
            if key not in normalized_aggregates:
                raise ProtocolError(
                    "E_FIELD",
                    f"response.aggregate_summary.{key}",
                    f"receipt is missing aggregate for logical key {key!r}",
                    environment=self.environment.id,
                )
        elapsed = payload["elapsed_ms_monotonic"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
            raise ProtocolError(
                "E_FIELD",
                "response.elapsed_ms_monotonic",
                "elapsed_ms_monotonic must be a non-negative integer",
                environment=self.environment.id,
            )
        for field in ("sample_digest", "digest"):
            if not isinstance(payload[field], str) or len(payload[field]) != 64:
                raise ProtocolError(
                    "E_FIELD",
                    f"response.{field}",
                    f"{field} must be a 64-character hex digest",
                    environment=self.environment.id,
                )
        return Receipt(
            request_id=request_id,
            service_id=payload["service_id"],
            slice_version=payload["slice_version"],
            aggregate_summary=normalized_aggregates,
            sample_digest=payload["sample_digest"],
            digest=payload["digest"],
            elapsed_ms_monotonic=elapsed,
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.5)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


class ServiceManager:
    def __init__(self, environments: dict[str, EnvironmentSpec], seed: int) -> None:
        self._services = {
            env_id: ServiceProcess(environment, seed)
            for env_id, environment in sorted(environments.items())
        }
        self.start_errors: dict[str, dict[str, Any]] = {}

    def start_all(self, timeout_ms: int) -> dict[str, dict[str, Any]]:
        errors: dict[str, dict[str, Any]] = {}
        for env_id, service in self._services.items():
            try:
                service.start(timeout_ms)
            except ProtocolError as exc:
                errors[env_id] = exc.to_dict()
        self.start_errors = errors
        if len(errors) == len(self._services):
            first_env = sorted(errors)[0]
            first = errors[first_env]
            raise ProtocolError(
                str(first["code"]),
                str(first["field"]),
                f"all environment services failed to start; first failure: {first['message']}",
                environment=first_env,
            )
        return errors

    def get(self, env_id: str) -> ServiceProcess:
        return self._services[env_id]

    def stop_all(self) -> None:
        for service in self._services.values():
            service.stop()

    def __enter__(self) -> "ServiceManager":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_all()
