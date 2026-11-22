"""Subprocess orchestration and loopback TCP JSON-lines client protocol.

Every node is an independent :mod:`edge_lease.node` child process.  This module
owns process lifecycle (start, ready detection, dead-process classification and
clean shutdown) and the per-dispatch socket conversation, mapping every failure
mode to its mandated code: E_PORT for bind failure, E_HANDSHAKE for connect /
handshake timeout, E_REPLY for malformed or incomplete receipts.
"""
from __future__ import annotations
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import HandshakeError, PlatformError, PortError, ReplyError


class DispatchTimeout(PlatformError):
    """Per-dispatch socket timeout with a stable code."""

    code = "E_HANDSHAKE"
from .fixedpoint import quant_version
from .logging_setup import JsonlLogger
from .models import NodeDef, PlanItem, Receipt, VersionManifest

REQUIRED_RECEIPT_FIELDS = (
    "node_id",
    "version",
    "path",
    "derived_digest",
    "shard_hit_digest",
    "cost_ms",
    "quant_version",
)


def _read_ready_line(
    process: subprocess.Popen, deadline: float, log_path: Path
) -> Dict[str, Any]:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = ""
            if log_path.exists():
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-500:]
            raise PortError(
                f"node process exited before ready (code {process.returncode})",
                "node.startup",
                {"returncode": process.returncode, "stderr_tail": tail},
            )
        line = process.stdout.readline()
        if line:
            doc = json.loads(line.decode("utf-8"))
            if doc.get("type") == "ready":
                return doc
        else:
            time.sleep(0.01)
    raise PortError("node did not emit ready line in time", "node.startup")


class NodeProcess:
    """A live edge node child process with readiness and liveness tracking."""

    def __init__(
        self,
        node: NodeDef,
        scale: int,
        workdir: Path,
        logger: JsonlLogger,
        startup_timeout_s: float = 15.0,
    ) -> None:
        self.node = node
        self.scale = scale
        self.workdir = Path(workdir)
        self.logger = logger
        self.startup_timeout_s = startup_timeout_s
        self.process: Optional[subprocess.Popen] = None
        self.log_path = self.workdir / f"node-{node.node_id}.log"
        self.available = False
        self.death_code: Optional[int] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "edge_lease.node",
            "--node-id",
            self.node.node_id,
            "--host",
            self.node.host,
            "--port",
            str(self.node.port),
            "--scale",
            str(self.scale),
            "--fail-mode",
            self.node.fail_mode,
        ]
        log_handle = self.log_path.open("wb")
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=log_handle,
            )
        except OSError as exc:  # pragma: no cover - defensive
            log_handle.close()
            raise PortError(str(exc), "node.spawn") from exc
        self._log_handle = log_handle
        try:
            _read_ready_line(
                self.process,
                time.monotonic() + self.startup_timeout_s,
                self.log_path,
            )
        except PortError:
            # Bind failures surface either as early exit or as a logged
            # bind_error record; normalize both to E_PORT.
            self.available = False
            self.logger.error(
                "node_port_failure",
                node_id=self.node.node_id,
                code="E_PORT",
                log_path=str(self.log_path),
            )
            self.stop()
            raise
        self.available = True
        self.logger.info(
            "node_started",
            node_id=self.node.node_id,
            port=self.node.port,
            pid=self.process.pid,
        )

    def check_liveness(self) -> bool:
        with self._lock:
            if self.process is None:
                self.available = False
                return False
            code = self.process.poll()
            if code is not None:
                self.available = False
                self.death_code = code
                self.logger.error(
                    "node_exited",
                    node_id=self.node.node_id,
                    returncode=code,
                    code="E_HANDSHAKE",
                )
            return self.available

    def stop(self) -> None:
        with self._lock:
            if self.process is None:
                return
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.available = False
            handle = getattr(self, "_log_handle", None)
            if handle is not None:
                handle.close()
            self.process = None


class NodeClient:
    """One connect/handshake/dispatch conversation per call."""

    def __init__(
        self,
        node_process: NodeProcess,
        logger: JsonlLogger,
        connect_timeout_s: float = 3.0,
    ) -> None:
        self.np = node_process
        self.node = node_process.node
        self.logger = logger
        self.connect_timeout_s = connect_timeout_s

    def dispatch(
        self,
        item: PlanItem,
        manifest: VersionManifest,
        content_seed: int,
        attempt: int,
    ) -> Receipt:
        timeout_s = self.node.dispatch_timeout_ms / 1000.0
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout_s)
        try:
            try:
                sock.connect((self.node.host, self.node.port))
            except (OSError, socket.timeout) as exc:
                self._mark_dead_or_timeout(exc)
                raise HandshakeError(
                    f"connect failed: {exc}",
                    f"item {item.item_id}",
                    {"node_id": self.node.node_id, "attempt": attempt},
                ) from exc
            sock.settimeout(timeout_s)
            self._handshake(sock, item, attempt)
            receipt_doc = self._send_dispatch(sock, item, manifest, content_seed, attempt)
        finally:
            sock.close()
        return self._validate_receipt(receipt_doc, item, attempt)

    def _mark_dead_or_timeout(self, exc: Exception) -> None:
        was_alive = self.np.available
        alive = self.np.check_liveness()
        if not alive and not was_alive:
            self.logger.error(
                "node_unavailable",
                node_id=self.node.node_id,
                reason="process_dead",
                code="E_HANDSHAKE",
            )
        else:
            self.logger.warning(
                "handshake_timeout",
                node_id=self.node.node_id,
                error=str(exc),
                code="E_HANDSHAKE",
            )

    def _send_line(self, sock: socket.socket, doc: Dict[str, Any]) -> None:
        sock.sendall((json.dumps(doc, sort_keys=True) + "\n").encode("utf-8"))

    def _recv_line(self, sock: socket.socket) -> Dict[str, Any]:
        buffer = bytearray()
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("connection closed before reply")
            buffer.extend(chunk)
            if b"\n" in buffer:
                line, _ = buffer.split(b"\n", 1)
                return json.loads(line.decode("utf-8"))

    def _handshake(
        self, sock: socket.socket, item: PlanItem, attempt: int
    ) -> None:
        try:
            self._send_line(
                sock,
                {
                    "type": "handshake",
                    "node_id": self.node.node_id,
                    "quant_version": item.quant_version,
                },
            )
            hello = self._recv_line(sock)
        except (socket.timeout, OSError, json.JSONDecodeError, ConnectionError) as exc:
            self.np.check_liveness()
            self.logger.error(
                "handshake_failure",
                node_id=self.node.node_id,
                item_id=item.item_id,
                attempt=attempt,
                code="E_HANDSHAKE",
            )
            raise HandshakeError(
                f"handshake failed: {exc}",
                f"item {item.item_id}",
                {"node_id": self.node.node_id, "attempt": attempt},
            ) from exc
        if hello.get("type") != "hello" or hello.get("node_id") != self.node.node_id:
            raise HandshakeError(
                "invalid hello frame",
                f"item {item.item_id}",
                {"hello": hello},
            )

    def _send_dispatch(
        self,
        sock: socket.socket,
        item: PlanItem,
        manifest: VersionManifest,
        content_seed: int,
        attempt: int,
    ) -> Dict[str, Any]:
        path_table = [entry.to_dict() for entry in manifest.entries]
        payload = {
            "type": "dispatch",
            "item_id": item.item_id,
            "node_id": self.node.node_id,
            "version": item.version,
            "path": item.path,
            "content_seed": content_seed,
            "path_table": path_table,
        }
        try:
            self._send_line(sock, payload)
            return self._recv_line(sock)
        except socket.timeout as exc:
            self.logger.warning(
                "dispatch_timeout",
                node_id=self.node.node_id,
                item_id=item.item_id,
                attempt=attempt,
                timeout_ms=self.node.dispatch_timeout_ms,
            )
            raise DispatchTimeout(
                "dispatch socket timed out",
                f"item {item.item_id}",
                {
                    "node_id": self.node.node_id,
                    "attempt": attempt,
                    "timeout_ms": self.node.dispatch_timeout_ms,
                },
            ) from exc
        except (OSError, ConnectionError, json.JSONDecodeError) as exc:
            self.np.check_liveness()
            raise ReplyError(
                f"reply transport/parse failure: {exc}",
                f"item {item.item_id}",
                {"node_id": self.node.node_id, "attempt": attempt},
            ) from exc

    def _validate_receipt(
        self, doc: Dict[str, Any], item: PlanItem, attempt: int
    ) -> Receipt:
        missing = [field for field in REQUIRED_RECEIPT_FIELDS if field not in doc]
        if missing:
            self.logger.error(
                "reply_missing_fields",
                node_id=self.node.node_id,
                item_id=item.item_id,
                missing=missing,
                attempt=attempt,
                code="E_REPLY",
            )
            raise ReplyError(
                f"receipt missing fields {missing}",
                f"item {item.item_id}.receipt",
                {"missing": missing, "attempt": attempt},
            )
        return Receipt(
            item_id=item.item_id,
            node_id=doc["node_id"],
            version=int(doc["version"]),
            path=doc["path"],
            derived_digest=int(doc["derived_digest"]),
            shard_hit_digest=int(doc["shard_hit_digest"]),
            cost_ms=int(doc["cost_ms"]),
            quant_version=str(doc["quant_version"]),
            attempt=attempt,
            status="ok",
        )
