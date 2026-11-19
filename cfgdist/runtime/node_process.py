"""Platform-side lifecycle management for one node child process."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.manifest import NodeSpec
from ..errors import E_HANDSHAKE, E_NODE, E_PORT, ErrorDetail, PlatformError
from ..util.logging_util import StructuredLogger


def _module_target() -> str:
    return "cfgdist.runtime.node_worker"


@dataclass
class NodeProcess:
    spec: NodeSpec
    proc: subprocess.Popen[bytes]
    port: int
    address: tuple[str, int]
    logger: StructuredLogger
    handshake_timeout_ms: int = 5000
    operation_timeout_ms: int = 10000
    reaper_timeout_ms: int = 5000

    def _connect(self) -> socket.socket:
        sock = socket.create_connection(self.address, timeout=self.handshake_timeout_ms / 1000.0)
        sock.settimeout(self.handshake_timeout_ms / 1000.0)
        return sock

    def handshake(self) -> None:
        deadline = time.monotonic() + self.handshake_timeout_ms / 1000.0
        try:
            sock = self._connect()
        except (OSError, socket.timeout) as exc:
            raise PlatformError(
                E_HANDSHAKE,
                "node unreachable for handshake",
                [ErrorDetail(reason=str(exc), key=self.spec.node_id, source="handshake")],
            ) from exc
        try:
            with sock:
                sock.sendall(
                    (json.dumps({"kind": "hello", "node_id": self.spec.node_id}, sort_keys=True) + "\n").encode()
                )
                sock.settimeout(max(0.1, deadline - time.monotonic()))
                line = sock.makefile("rb").readline()
                if not line:
                    raise PlatformError(
                        E_HANDSHAKE,
                        "handshake returned no data",
                        [ErrorDetail(key=self.spec.node_id, source="handshake")],
                    )
                ack = json.loads(line.decode("utf-8"))
                if ack.get("kind") != "hello_ack" or ack.get("node_id") != self.spec.node_id:
                    raise PlatformError(
                        E_HANDSHAKE,
                        "malformed handshake acknowledgement",
                        [ErrorDetail(reason=json.dumps(ack, sort_keys=True), key=self.spec.node_id, source="handshake")],
                    )
        except (socket.timeout, TimeoutError) as exc:
            raise PlatformError(
                E_HANDSHAKE,
                "handshake timed out",
                [ErrorDetail(reason=str(exc), key=self.spec.node_id, source="handshake")],
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise PlatformError(
                E_HANDSHAKE,
                "handshake transport error",
                [ErrorDetail(reason=str(exc), key=self.spec.node_id, source="handshake")],
            ) from exc
        self.logger.log("node_handshake_ok", key=self.spec.node_id, port=self.port)

    def request(
        self,
        request_id: str,
        attempt: int,
        slice_payload: dict[str, Any],
    ) -> dict[str, Any]:
        sock = socket.create_connection(self.address, timeout=self.operation_timeout_ms / 1000.0)
        try:
            with sock:
                sock.settimeout(self.operation_timeout_ms / 1000.0)
                message = {"kind": "config", "request_id": request_id, "attempt": attempt, "slice": slice_payload}
                sock.sendall((json.dumps(message, sort_keys=True) + "\n").encode("utf-8"))
                line = sock.makefile("rb").readline()
                if not line:
                    raise ConnectionError("node closed connection without receipt")
                receipt = json.loads(line.decode("utf-8"))
                if receipt.get("kind") != "receipt":
                    raise ConnectionError(f"unexpected node reply: {receipt.get('kind')}")
                return receipt
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def shutdown(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            sock = socket.create_connection(self.address, timeout=1.0)
            with sock:
                sock.sendall((json.dumps({"kind": "shutdown"}) + "\n").encode())
                sock.makefile("rb").readline()
        except OSError:
            pass
        self._reap()

    def force_reap(self) -> None:
        """Forcefully terminate and reclaim the child within the reaper budget."""
        if self.proc.poll() is not None:
            self._close_pipes()
            return
        self.logger.log("node_force_terminate", key=self.spec.node_id)
        self.proc.terminate()
        try:
            self.proc.wait(timeout=self.reaper_timeout_ms / 1000.0)
            self.logger.log("node_reaped", key=self.spec.node_id, returncode=self.proc.returncode)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=self.reaper_timeout_ms / 1000.0)
            self.logger.log("node_reaped_after_kill", key=self.spec.node_id, returncode=self.proc.returncode)
        finally:
            self._close_pipes()

    def _reap(self) -> None:
        try:
            self.proc.wait(timeout=self.reaper_timeout_ms / 1000.0)
        except subprocess.TimeoutExpired:
            self.force_reap()
        finally:
            self._close_pipes()

    def _close_pipes(self) -> None:
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass


def spawn_node(
    spec: NodeSpec,
    logger: StructuredLogger,
    *,
    log_path: str | None = None,
    inject: list[str] | None = None,
    startup_timeout_ms: int = 8000,
    handshake_timeout_ms: int = 5000,
    operation_timeout_ms: int = 10000,
) -> NodeProcess:
    command = [
        sys.executable,
        "-u",
        "-m",
        _module_target(),
        "--node-id",
        spec.node_id,
        "--aggregate",
        spec.aggregate,
        "--sample",
        json.dumps(list(spec.sample)),
        "--log-path",
        log_path or "",
    ]
    if spec.port is not None:
        command += ["--port", str(spec.port)]
    for flag in inject or []:
        command += ["--inject", flag]

    creation_flags = 0
    if sys.platform == "win32":  # pragma: no cover - Windows-only branch
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags,
        )
    except OSError as exc:
        raise PlatformError(
            E_NODE,
            "failed to spawn node subprocess",
            [ErrorDetail(reason=str(exc), key=spec.node_id, source="spawn")],
        ) from exc

    # Read the single ready line within the startup budget.
    assert proc.stdout is not None
    deadline = time.monotonic() + startup_timeout_ms / 1000.0
    line: bytes | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            logger.log(
                "node_startup_exit",
                code=E_NODE,
                key=spec.node_id,
                returncode=proc.returncode,
                stderr=stderr[:500],
            )
            raise PlatformError(
                E_NODE,
                "node process exited during startup",
                [
                    ErrorDetail(
                        reason=f"exit={proc.returncode}; {stderr[:200]}",
                        key=spec.node_id,
                        source="spawn",
                    )
                ],
            )
        proc.stdout._checkReadable  # noqa: B018 - keep attribute access explicit
        ready = _read_with_deadline(proc.stdout, 0.2)
        if ready:
            line = ready
            break
    if line is None:
        proc.force_kill = proc.kill  # type: ignore[attr-defined]
        proc.kill()
        proc.wait(timeout=2.0)
        raise PlatformError(
            E_NODE,
            "node startup timed out before announcing readiness",
            [ErrorDetail(key=spec.node_id, source="spawn")],
        )

    try:
        announcement = json.loads(line.decode("utf-8"))
    except json.JSONDecodeError as exc:
        proc.kill()
        raise PlatformError(
            E_NODE,
            "node emitted an unreadable readiness line",
            [ErrorDetail(reason=str(exc), key=spec.node_id, source="spawn")],
        ) from exc

    if announcement.get("event") == "bind_error":
        proc.wait(timeout=2.0)
        logger.log("node_port_busy", code=E_PORT, key=spec.node_id, reason=announcement.get("reason"))
        raise PlatformError(
            E_PORT,
            "loopback port already occupied",
            [ErrorDetail(reason=announcement.get("reason", "bind failed"), key=spec.node_id, source="bind")],
        )
    if announcement.get("event") != "ready":
        proc.kill()
        raise PlatformError(
            E_NODE,
            "node announced an unexpected startup event",
            [ErrorDetail(reason=json.dumps(announcement, sort_keys=True), key=spec.node_id, source="spawn")],
        )

    process = NodeProcess(
        spec=spec,
        proc=proc,
        port=int(announcement["port"]),
        address=("127.0.0.1", int(announcement["port"])),
        logger=logger,
        handshake_timeout_ms=handshake_timeout_ms,
        operation_timeout_ms=operation_timeout_ms,
    )
    logger.log("node_ready", key=spec.node_id, port=process.port, pid=proc.pid)
    return process


def _read_with_deadline(stream: Any, timeout_s: float) -> bytes | None:
    import select

    if sys.platform == "win32":  # pragma: no cover - Windows uses blocking read
        # On Windows select() cannot wait on pipes; rely on the short poll loop
        # in the caller and a blocking readline with a thread would add cost, so
        # we attempt a non-blocking peek via the buffered stream instead.
        stream.flush if hasattr(stream, "flush") else None
    readable, _, _ = select.select([stream], [], [], timeout_s)
    if not readable:
        return None
    line = stream.readline()
    return line or None
