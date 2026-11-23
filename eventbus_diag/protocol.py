"""Subprocess orchestration and the loopback line protocol client."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .canonical import digest_value, stable_serialize
from .errors import AckError, BusDiagnosticError, PortError, ProtocolTimeoutError
from .logging_utils import StructuredLogger
from .models import Ack, PlanBatch, Subscriber


def _read_line(sock: socket.socket, *, limit: int = 16 * 1024 * 1024) -> bytes:
    chunks = bytearray()
    while b"\n" not in chunks:
        part = sock.recv(65536)
        if not part:
            raise ConnectionError("peer closed the connection before a line was received")
        chunks.extend(part)
        if len(chunks) > limit:
            raise ValueError("protocol line exceeds size limit")
    line, _, _rest = bytes(chunks).partition(b"\n")
    return line


def _configure_listener_socket(sock: socket.socket) -> None:
    """Allow TIME_WAIT reuse without allowing a second live listener.

    Linux/macOS need SO_REUSEADDR for a restarted listener.  Windows instead
    uses SO_EXCLUSIVEADDRUSE so an actually occupied port is never stolen.
    """

    if os.name == "nt":
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def _port_is_available(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _configure_listener_socket(probe)
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


class SubscriberProcess:
    def __init__(
        self,
        subscriber: Subscriber,
        *,
        handshake_timeout_ms: int,
        logger: StructuredLogger,
        worker_log_path: str | Path | None = None,
        inject_processing_delay_ms: int = 0,
        omit_ack_field: str | None = None,
        port_release_grace_ms: int = 0,
    ) -> None:
        self.subscriber = subscriber
        self.handshake_timeout_ms = int(handshake_timeout_ms)
        self.logger = logger
        self.worker_log_path = str(worker_log_path) if worker_log_path else None
        self.inject_processing_delay_ms = int(inject_processing_delay_ms)
        self.omit_ack_field = omit_ack_field
        self.port_release_grace_ms = max(0, int(port_release_grace_ms))
        self.process: subprocess.Popen[str] | None = None
        self.ready = False
        self.ready_greeting: dict[str, Any] | None = None
        self._package_root = str(Path(__file__).resolve().parent.parent)

    def start(self) -> None:
        if self.ready:
            return
        port_deadline = time.monotonic() + self.port_release_grace_ms / 1000.0
        waiting_logged = False
        while not _port_is_available(self.subscriber.host, self.subscriber.port):
            if time.monotonic() >= port_deadline:
                error = PortError(
                    f"port {self.subscriber.port} is already occupied",
                    subscriber_id=self.subscriber.subscriber_id,
                    port=self.subscriber.port,
                )
                self.logger.diagnostic(error)
                raise error
            if self.port_release_grace_ms and not waiting_logged:
                self.logger.warning(
                    "RECOVERY_PORT_WAIT",
                    "loopback port is still held by a terminating process; retrying",
                    subscriber_id=self.subscriber.subscriber_id,
                    port=self.subscriber.port,
                    grace_ms=self.port_release_grace_ms,
                )
                waiting_logged = True
            time.sleep(0.02)
        if waiting_logged:
            self.logger.info(
                "RECOVERY_PORT_READY",
                "loopback port became available for recovery",
                subscriber_id=self.subscriber.subscriber_id,
                port=self.subscriber.port,
            )

        command = [
            sys.executable,
            "-m",
            "eventbus_diag.worker",
            "--subscriber-id",
            self.subscriber.subscriber_id,
            "--host",
            self.subscriber.host,
            "--port",
            str(self.subscriber.port),
            "--budget-ms",
            str(self.subscriber.processing_budget_ms),
            "--seed",
            str(self.subscriber.seed),
        ]
        if self.worker_log_path:
            command.extend(["--log-file", self.worker_log_path])
        if self.inject_processing_delay_ms:
            command.extend(["--processing-delay-ms", str(self.inject_processing_delay_ms)])
        if self.omit_ack_field:
            command.extend(["--omit-ack-field", self.omit_ack_field])

        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            self._package_root
            if not existing_pythonpath
            else self._package_root + os.pathsep + existing_pythonpath
        )
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=environment,
                creationflags=creationflags,
            )
        except OSError as exc:
            error = PortError(
                f"could not start subscriber process: {exc}",
                subscriber_id=self.subscriber.subscriber_id,
                port=self.subscriber.port,
            )
            self.logger.diagnostic(error)
            raise error from exc

        deadline = time.monotonic() + self.handshake_timeout_ms / 1000.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate(timeout=1.0)
                error_code = self._startup_error_code(stdout)
                if error_code == "E_PORT":
                    error = PortError(
                        f"subscriber process failed to bind loopback port {self.subscriber.port}",
                        subscriber_id=self.subscriber.subscriber_id,
                        port=self.subscriber.port,
                        details={"stderr": stderr[-2000:]},
                    )
                else:
                    error = BusDiagnosticError(
                        "E_PROCESS",
                        "subscriber process exited during startup",
                        details={
                            "subscriber_id": self.subscriber.subscriber_id,
                            "port": self.subscriber.port,
                            "returncode": self.process.returncode,
                            "stdout": stdout[-1000:],
                            "stderr": stderr[-2000:],
                        },
                    )
                self.logger.diagnostic(error)
                raise error
            try:
                greeting = self._connect_and_read_greeting(timeout_seconds=0.3)
                if greeting.get("event") != "ready":
                    raise ValueError("first line was not a ready greeting")
                if greeting.get("subscriber_id") != self.subscriber.subscriber_id:
                    raise ValueError("ready greeting has wrong subscriber_id")
                self.ready_greeting = greeting
                self.ready = True
                self.logger.info(
                    "SERVICE_HANDSHAKE",
                    "subscriber handshake succeeded",
                    subscriber_id=self.subscriber.subscriber_id,
                    port=self.subscriber.port,
                    pid=greeting.get("pid"),
                )
                return
            except (OSError, ValueError, ConnectionError) as exc:
                last_error = exc
                time.sleep(0.02)

        self.stop()
        error = ProtocolTimeoutError(
            "subscriber handshake timed out",
            subscriber_id=self.subscriber.subscriber_id,
            port=self.subscriber.port,
            details={
                "handshake_timeout_ms": self.handshake_timeout_ms,
                "last_error": str(last_error) if last_error else None,
            },
        )
        self.logger.diagnostic(error)
        raise error

    def dispatch(
        self,
        batch: PlanBatch,
        *,
        request_timeout_ms: int,
    ) -> Ack:
        if not self.ready:
            self.start()
        request = {
            "type": "batch",
            "protocol_version": 1,
            "protocol": "eventbus.line.v1",
            "subscriber_id": self.subscriber.subscriber_id,
            "channel_id": batch.channel_id,
            "topic": batch.topic,
            "batch_id": batch.batch_id,
            "batch_index": batch.batch_index,
            "event_ids": list(batch.event_ids),
            "events": batch.content["events"],
            "processing_budget_ms": self.subscriber.processing_budget_ms,
            "seed": self.subscriber.seed,
        }
        timeout_seconds = max(0.05, request_timeout_ms / 1000.0)
        try:
            sock = socket.create_connection(
                (self.subscriber.host, self.subscriber.port), timeout=timeout_seconds
            )
        except OSError as exc:
            error = ProtocolTimeoutError(
                f"could not connect to subscriber for batch: {exc}",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
                port=self.subscriber.port,
            )
            self.logger.diagnostic(error)
            raise error from exc

        try:
            with sock:
                sock.settimeout(timeout_seconds)
                greeting = self._parse_greeting(_read_line(sock))
                if greeting.get("subscriber_id") != self.subscriber.subscriber_id:
                    error = AckError(
                        "connection greeting contains the wrong subscriber_id",
                        subscriber_id=self.subscriber.subscriber_id,
                        batch_id=batch.batch_id,
                    )
                    self.logger.diagnostic(error)
                    raise error
                sock.sendall((stable_serialize(request) + "\n").encode("utf-8"))
                response_line = _read_line(sock)
        except socket.timeout as exc:
            error = ProtocolTimeoutError(
                "timed out waiting for batch confirmation",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
                port=self.subscriber.port,
            )
            self.logger.diagnostic(error)
            raise error from exc
        except (OSError, ConnectionError, ValueError) as exc:
            error = ProtocolTimeoutError(
                f"subscriber socket failed while dispatching batch: {exc}",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
                port=self.subscriber.port,
            )
            self.logger.diagnostic(error)
            raise error from exc

        return self._parse_ack(response_line, batch)

    def _parse_ack(self, raw_line: bytes, batch: PlanBatch) -> Ack:
        try:
            response = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            error = AckError(
                f"confirmation line is not valid JSON: {exc}",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
            )
            self.logger.diagnostic(error)
            raise error from exc
        if not isinstance(response, dict):
            error = AckError(
                "confirmation line must be a JSON object",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
            )
            self.logger.diagnostic(error)
            raise error
        if "code" in response and response.get("event") == "protocol_error":
            error = AckError(
                f"subscriber rejected batch: {response.get('message', 'protocol error')}",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
                details={"remote_code": response.get("code")},
            )
            self.logger.diagnostic(error)
            raise error

        required = ("subscriber_id", "batch_id", "event_ids", "processing_ms")
        missing = [field for field in required if field not in response]
        if missing:
            error = AckError(
                "confirmation is missing required fields",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
                missing_fields=missing,
            )
            self.logger.diagnostic(error)
            raise error
        if response["subscriber_id"] != self.subscriber.subscriber_id:
            error = AckError(
                "confirmation subscriber_id does not match request",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
            )
            self.logger.diagnostic(error)
            raise error
        if response["batch_id"] != batch.batch_id:
            error = AckError(
                "confirmation batch_id does not match request",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
            )
            self.logger.diagnostic(error)
            raise error
        processing_ms = response["processing_ms"]
        if isinstance(processing_ms, bool) or not isinstance(processing_ms, int):
            error = AckError(
                "confirmation processing_ms must be an integer",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
            )
            self.logger.diagnostic(error)
            raise error
        if processing_ms < 0:
            error = AckError(
                "confirmation processing_ms must be non-negative",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
            )
            self.logger.diagnostic(error)
            raise error
        event_ids_raw = response["event_ids"]
        if not isinstance(event_ids_raw, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in event_ids_raw
        ):
            error = AckError(
                "confirmation event_ids must be an array of integers",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
            )
            self.logger.diagnostic(error)
            raise error
        event_ids = tuple(sorted(int(value) for value in event_ids_raw))
        if len(event_ids) != len(set(event_ids)):
            error = AckError(
                "confirmation event_ids contains duplicates",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
            )
            self.logger.diagnostic(error)
            raise error
        if set(event_ids) != set(batch.event_ids):
            error = AckError(
                "confirmation event_ids do not match the dispatched batch",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
                details={
                    "expected_event_ids": list(batch.event_ids),
                    "actual_event_ids": list(event_ids),
                },
            )
            self.logger.diagnostic(error)
            raise error
        ack_digest = response.get("ack_digest")
        if ack_digest is not None and not isinstance(ack_digest, str):
            error = AckError(
                "confirmation ack_digest must be a string when present",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
            )
            self.logger.diagnostic(error)
            raise error
        processed_order_raw = response.get("processed_order", sorted(event_ids))
        if not isinstance(processed_order_raw, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in processed_order_raw
        ):
            error = AckError(
                "confirmation processed_order must be an array of integers",
                subscriber_id=self.subscriber.subscriber_id,
                batch_id=batch.batch_id,
            )
            self.logger.diagnostic(error)
            raise error

        processed_order = tuple(int(value) for value in processed_order_raw)
        if not ack_digest:
            ack_digest = digest_value(
                {
                    "subscriber_id": response["subscriber_id"],
                    "batch_id": response["batch_id"],
                    "event_ids": list(event_ids),
                    "processing_ms": processing_ms,
                    "processed_order": list(processed_order),
                    "protocol_version": int(response.get("protocol_version", 1)),
                }
            )

        return Ack(
            subscriber_id=response["subscriber_id"],
            batch_id=response["batch_id"],
            event_ids=event_ids,
            processing_ms=processing_ms,
            ack_digest=str(ack_digest),
            processed_order=processed_order,
            raw=response,
        )

    def _connect_and_read_greeting(self, *, timeout_seconds: float) -> dict[str, Any]:
        sock = socket.create_connection(
            (self.subscriber.host, self.subscriber.port), timeout=timeout_seconds
        )
        try:
            sock.settimeout(timeout_seconds)
            return self._parse_greeting(_read_line(sock))
        finally:
            sock.close()

    @staticmethod
    def _parse_greeting(raw_line: bytes) -> dict[str, Any]:
        obj = json.loads(raw_line.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("greeting must be a JSON object")
        return obj

    @staticmethod
    def _startup_error_code(stdout: str) -> str | None:
        for line in reversed(stdout.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("code"):
                return str(value["code"])
        return None

    def stop(self) -> None:
        process = self.process
        self.ready = False
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin:
                    process.stdin.write("STOP\n")
                    process.stdin.flush()
                    process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self.process = None


class SubscriberPool:
    def __init__(self, processes: dict[str, SubscriberProcess], logger: StructuredLogger) -> None:
        self.processes = processes
        self.logger = logger

    @classmethod
    def from_declaration(
        cls,
        declaration,
        *,
        logger: StructuredLogger,
        worker_log_path: str | Path | None = None,
    ) -> "SubscriberPool":
        processes = {
            subscriber.subscriber_id: SubscriberProcess(
                subscriber,
                handshake_timeout_ms=declaration.handshake_timeout_ms,
                logger=logger,
                worker_log_path=worker_log_path,
            )
            for subscriber in declaration.subscribers
        }
        return cls(processes, logger)

    def start_all(self) -> dict[str, BusDiagnosticError]:
        errors: dict[str, BusDiagnosticError] = {}
        for subscriber_id in sorted(self.processes):
            try:
                self.processes[subscriber_id].start()
            except BusDiagnosticError as exc:
                errors[subscriber_id] = exc
        return errors

    def stop_all(self) -> None:
        for subscriber_id in sorted(self.processes, reverse=True):
            try:
                self.processes[subscriber_id].stop()
            except Exception as exc:  # cleanup must not hide the primary failure
                self.logger.error(
                    "E_PROCESS_STOP",
                    f"failed to stop subscriber process: {exc}",
                    subscriber_id=subscriber_id,
                )
