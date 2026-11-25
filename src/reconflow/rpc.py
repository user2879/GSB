from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from .canonical import digest_value
from .errors import PortBusyError, ReceiptFieldError, SocketTimeoutFailure

REQUIRED_RECEIPT_FIELDS = ("type", "task_id", "ok", "result", "digest")


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def shutdown_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    for pipe in (process.stdout, process.stderr, process.stdin):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass


def assert_port_available(port: int, host: str = "127.0.0.1") -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        raise PortBusyError(f"TCP port {port} is occupied or cannot bind: {exc}", host=host, port=port) from exc
    finally:
        sock.close()


@dataclass
class WorkerHandle:
    process: subprocess.Popen[str]
    host: str
    port: int
    timeout_ms: int

    def close(self) -> None:
        shutdown_process(self.process)

    def __enter__(self) -> "WorkerHandle":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()


def start_worker_subprocess(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    timeout_ms: int = 2_000,
    delay_ms: int = 0,
    inject_missing_field: str = "",
    ready_timeout_ms: int = 5_000,
) -> WorkerHandle:
    src_dir = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-u",
        "-m",
        "reconflow.worker_server",
        "--host",
        host,
        "--port",
        str(port),
        "--timeout-ms",
        str(timeout_ms),
        "--delay-ms",
        str(delay_ms),
        "--inject-missing-field",
        inject_missing_field,
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    deadline = time.monotonic() + ready_timeout_ms / 1000.0
    ready_line: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            try:
                stderr = process.stderr.read() if process.stderr else ""
            finally:
                shutdown_process(process)
            raise PortBusyError(
                f"shard worker exited before handshake-ready (rc={process.returncode})",
                host=host,
                port=port,
                stderr=stderr[-500:],
            )
        readable, _, _ = select.select([process.stdout], [], [], 0.05)
        if readable:
            ready_line = process.stdout.readline().strip()
            if ready_line:
                break
    if not ready_line:
        shutdown_process(process)
        raise SocketTimeoutFailure("timed out waiting for worker ready line", host=host, port=port, phase="subprocess_ready")
    try:
        ready = json.loads(ready_line)
    except json.JSONDecodeError as exc:
        shutdown_process(process)
        raise ReceiptFieldError("worker ready line is not JSON", line=ready_line) from exc
    if ready.get("code") == "E_PORT" or ready.get("event") == "error":
        shutdown_process(process)
        raise PortBusyError(str(ready.get("message", "port error")), host=host, port=port or ready.get("port"))
    if ready.get("event") != "ready":
        shutdown_process(process)
        raise ReceiptFieldError("worker ready event missing event=ready", ready=ready)
    actual_port = int(ready["port"])
    return WorkerHandle(process=process, host=str(ready.get("host", host)), port=actual_port, timeout_ms=timeout_ms)


class ComputeClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 2_000) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.sock: socket.socket | None = None
        self.file: Any = None

    def __enter__(self) -> "ComputeClient":
        self.connect()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()

    def connect(self) -> None:
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_ms / 1000.0)
        except OSError as exc:
            if isinstance(exc, socket.timeout):
                raise SocketTimeoutFailure("TCP connect timed out", host=self.host, port=self.port, phase="connect") from exc
            raise PortBusyError(f"cannot connect to shard worker: {exc}", host=self.host, port=self.port) from exc
        try:
            self.sock.settimeout(self.timeout_ms / 1000.0)
            self.file = self.sock.makefile("rwb")
            hello = self._read_json_line(phase="hello")
            if hello.get("type") != "hello" or hello.get("protocol") != 1:
                raise ReceiptFieldError("invalid worker hello", field="hello.protocol", receipt=hello)
            self._write_json_line({"type": "handshake", "protocol": 1, "client": "reconflow-scheduler"})
            ack = self._read_json_line(phase="handshake_ack")
            if ack.get("type") != "handshake_ack" or ack.get("ok") is not True:
                raise ReceiptFieldError("invalid handshake acknowledgement", field="handshake_ack.ok", receipt=ack)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        try:
            if self.file is not None:
                self.file.close()
        finally:
            if self.sock is not None:
                self.sock.close()

    def _write_json_line(self, payload: dict[str, Any]) -> None:
        assert self.file is not None
        self.file.write((json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode())
        self.file.flush()

    def _read_json_line(self, *, phase: str) -> dict[str, Any]:
        assert self.file is not None
        try:
            line = self.file.readline()
        except (socket.timeout, TimeoutError) as exc:
            raise SocketTimeoutFailure(f"socket timeout during {phase}", host=self.host, port=self.port, phase=phase) from exc
        except OSError as exc:
            raise SocketTimeoutFailure(f"socket read failed during {phase}: {exc}", host=self.host, port=self.port, phase=phase) from exc
        if not line:
            raise SocketTimeoutFailure(f"socket closed during {phase}", host=self.host, port=self.port, phase=phase)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReceiptFieldError(f"non-JSON response during {phase}", field="$", line=line[:200].decode("utf-8", "replace")) from exc
        if not isinstance(payload, dict):
            raise ReceiptFieldError("response must be an object", field="$")
        return payload

    def submit(self, task: dict[str, Any]) -> dict[str, Any]:
        self._write_json_line(task)
        receipt = self._read_json_line(phase="receipt")
        for field in REQUIRED_RECEIPT_FIELDS:
            if field not in receipt:
                raise ReceiptFieldError(
                    f"receipt is missing field {field!r}",
                    field=field,
                    task_id=task.get("task_id"),
                    receipt_keys=sorted(receipt),
                )
        if receipt["task_id"] != task["task_id"]:
            raise ReceiptFieldError("receipt task_id does not match task", field="task_id", expected=task["task_id"], actual=receipt["task_id"])
        if receipt.get("ok") is not True:
            raise ReceiptFieldError("worker returned ok=false", field="ok", receipt=receipt)
        result = receipt["result"]
        if not isinstance(result, dict):
            raise ReceiptFieldError("receipt result must be an object", field="result")
        expected_digest = digest_value(result)
        if receipt["digest"] != expected_digest:
            raise ReceiptFieldError("receipt digest does not match result", field="digest", expected=expected_digest, actual=receipt["digest"])
        return result
