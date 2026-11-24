"""Real loopback UDP delivery and acknowledgement handling."""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import E_FIELD, E_TIMEOUT, fail
from .fragmenter import context_for_plan
from .samples import MessagePlan
from .spec import Endpoint
from .util import monotonic_ms, sha256_hex
from .wire import NormalizedFragment, decode_fragment, encode_fragment, with_retry_count


@dataclass(frozen=True)
class SendReceipt:
    message_key: str
    fragment_index: int
    tunnel_id: str
    attempt: int
    duplicate_copy: int
    wire: bytes
    wire_sha256: str
    ack: Mapping[str, Any]
    sent_monotonic_ms: float
    ack_received_monotonic_ms: float

    def to_dict(self, *, include_wire: bool = False) -> dict[str, Any]:
        result = {
            "message_key": self.message_key,
            "fragment_index": self.fragment_index,
            "tunnel_id": self.tunnel_id,
            "attempt": self.attempt,
            "duplicate_copy": self.duplicate_copy,
            "wire_length": len(self.wire),
            "wire_sha256": self.wire_sha256,
            "ack": dict(self.ack),
            "sent_monotonic_ms": self.sent_monotonic_ms,
            "ack_received_monotonic_ms": self.ack_received_monotonic_ms,
            "rtt_ms": self.ack_received_monotonic_ms - self.sent_monotonic_ms,
        }
        if include_wire:
            import base64

            result["wire_b64"] = base64.b64encode(self.wire).decode("ascii")
        return result


class ReceiverProcess:
    def __init__(
        self,
        endpoint: Endpoint,
        *,
        ack_mode: str = "normal",
        drop_indices: set[int] | None = None,
        drop_all: bool = False,
        idle_timeout_ms: int = 10_000,
        log_file: str | None = None,
        start_timeout_ms: int = 5_000,
    ) -> None:
        self.endpoint = endpoint
        self.ack_mode = ack_mode
        self.drop_indices = drop_indices or set()
        self.drop_all = drop_all
        self.idle_timeout_ms = idle_timeout_ms
        self.log_file = log_file
        self.start_timeout_ms = start_timeout_ms
        self.process: subprocess.Popen[str] | None = None
        self._ready_dir = tempfile.TemporaryDirectory(prefix="tunnel-interop-ready-")
        self.ready_file = Path(self._ready_dir.name) / "ready.json"

    def start(self) -> "ReceiverProcess":
        package_root = str(Path(__file__).resolve().parent.parent)
        env = os.environ.copy()
        env["PYTHONPATH"] = package_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        command = [
            sys.executable,
            "-m",
            "tunnel_interop.udp_receiver",
            "--endpoint-id",
            self.endpoint.endpoint_id,
            "--host",
            self.endpoint.host,
            "--port",
            str(self.endpoint.port),
            "--ack-mode",
            self.ack_mode,
            "--idle-timeout-ms",
            str(self.idle_timeout_ms),
            "--ready-file",
            str(self.ready_file),
        ]
        if self.drop_all:
            command.append("--drop-all")
        if self.drop_indices:
            command.extend(["--drop-indices", ",".join(str(item) for item in sorted(self.drop_indices))])
        if self.log_file:
            command.extend(["--log-file", self.log_file])
        self.process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=package_root,
        )
        deadline = time.monotonic() + self.start_timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if self.ready_file.exists():
                try:
                    ready = json.loads(self.ready_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    time.sleep(0.01)
                    continue
                if not ready.get("ok"):
                    self.stop()
                    fail(
                        E_TIMEOUT,
                        "receiver.ready",
                        "loopback UDP receiver failed to bind",
                        endpoint_id=self.endpoint.endpoint_id,
                        port=self.endpoint.port,
                        receiver_error=ready.get("error"),
                    )
                return self
            if self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr is not None else ""
                self.stop()
                fail(
                    E_TIMEOUT,
                    "receiver.ready",
                    "loopback UDP receiver exited before becoming ready",
                    endpoint_id=self.endpoint.endpoint_id,
                    port=self.endpoint.port,
                    stderr=stderr[-2000:],
                )
            time.sleep(0.01)
        self.stop()
        fail(
            E_TIMEOUT,
            "receiver.ready",
            "loopback UDP receiver readiness timed out",
            endpoint_id=self.endpoint.endpoint_id,
            port=self.endpoint.port,
            timeout_ms=self.start_timeout_ms,
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        try:
            self._ready_dir.cleanup()
        except OSError:
            pass

    def __enter__(self) -> "ReceiverProcess":
        return self.start()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.stop()


class ReceiverPool:
    def __init__(
        self,
        endpoints: list[Endpoint] | tuple[Endpoint, ...],
        *,
        ack_mode: str = "normal",
        drop_indices: set[int] | None = None,
        drop_all: bool = False,
        idle_timeout_ms: int = 10_000,
        log_file: str | None = None,
    ) -> None:
        self.receivers = [
            ReceiverProcess(
                endpoint,
                ack_mode=ack_mode,
                drop_indices=drop_indices,
                drop_all=drop_all,
                idle_timeout_ms=idle_timeout_ms,
                log_file=log_file,
            )
            for endpoint in endpoints
        ]

    def start(self) -> "ReceiverPool":
        started: list[ReceiverProcess] = []
        try:
            for receiver in self.receivers:
                receiver.start()
                started.append(receiver)
        except Exception:
            for receiver in reversed(started):
                receiver.stop()
            raise
        return self

    def stop(self) -> None:
        for receiver in reversed(self.receivers):
            receiver.stop()

    def __enter__(self) -> "ReceiverPool":
        return self.start()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.stop()


REQUIRED_ACK_FIELDS = (
    "ack",
    "endpoint_id",
    "fragment_index",
    "tunnel_id",
    "datagram_sha256",
    "received_at_ms",
)


def _exchange_once(
    *,
    endpoint: Endpoint,
    wire: bytes,
    expected_index: int,
    expected_tunnel_id: str,
    timeout_ms: int,
) -> tuple[Mapping[str, Any], float, float]:
    sent = monotonic_ms()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(timeout_ms / 1000.0)
        sock.sendto(wire, (endpoint.host, endpoint.port))
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise socket.timeout()
            sock.settimeout(remaining)
            try:
                payload, _peer = sock.recvfrom(65535)
            except socket.timeout:
                raise
            try:
                ack = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(ack, dict):
                continue
            for field in REQUIRED_ACK_FIELDS:
                if field not in ack:
                    fail(
                        E_FIELD,
                        f"ack.{field}",
                        "loopback receiver acknowledgement is missing a required field",
                        field=field,
                        fragment_index=ack.get("fragment_index", expected_index),
                        ack=ack,
                    )
            if ack.get("fragment_index") != expected_index or ack.get("tunnel_id") != expected_tunnel_id:
                continue
            if not ack.get("ack"):
                fail(
                    E_FIELD,
                    "ack.ack",
                    "loopback receiver rejected the datagram",
                    fragment_index=expected_index,
                    ack=ack,
                )
            if ack.get("datagram_sha256") != sha256_hex(wire):
                fail(
                    E_FIELD,
                    "ack.datagram_sha256",
                    "acknowledgement digest does not match the datagram that was sent",
                    expected=sha256_hex(wire),
                    actual=ack.get("datagram_sha256"),
                    fragment_index=expected_index,
                )
            if ack.get("endpoint_id") != endpoint.endpoint_id:
                fail(
                    E_FIELD,
                    "ack.endpoint_id",
                    "acknowledgement endpoint does not match the route next hop",
                    expected=endpoint.endpoint_id,
                    actual=ack.get("endpoint_id"),
                    fragment_index=expected_index,
                )
            if not isinstance(ack.get("received_at_ms"), int) or isinstance(ack.get("received_at_ms"), bool):
                fail(
                    E_FIELD,
                    "ack.received_at_ms",
                    "acknowledgement received time must be an integer millisecond value",
                    fragment_index=expected_index,
                )
            return ack, sent, monotonic_ms()


def send_fragment(
    *,
    plan: MessagePlan,
    fragment: NormalizedFragment,
    target_format: str,
    timeout_ms: int,
    max_retries: int,
    duplicate_copies: int = 1,
    reassembly_deadline_monotonic_ms: float | None = None,
) -> tuple[SendReceipt, ...]:
    receipts: list[SendReceipt] = []
    for copy_index in range(duplicate_copies):
        last_reason: str | None = None
        for attempt in range(max_retries + 1):
            attempt_timeout_ms = timeout_ms
            if reassembly_deadline_monotonic_ms is not None:
                remaining_ms = reassembly_deadline_monotonic_ms - monotonic_ms()
                if remaining_ms <= 0:
                    fail(
                        E_TIMEOUT,
                        "reassembly.deadline",
                        "declared reassembly timeout elapsed before a fragment could be sent",
                        endpoint_id=plan.endpoint.endpoint_id,
                        fragment_index=fragment.fragment_index,
                        attempt=attempt,
                    )
                attempt_timeout_ms = max(1, min(timeout_ms, math.ceil(remaining_ms)))
            attempted = with_retry_count(fragment, attempt) if target_format == "f3" else fragment
            wire = encode_fragment(attempted, target_format)
            try:
                ack, sent, received = _exchange_once(
                    endpoint=plan.endpoint,
                    wire=wire,
                    expected_index=fragment.fragment_index,
                    expected_tunnel_id=fragment.tunnel_id,
                    timeout_ms=attempt_timeout_ms,
                )
            except socket.timeout:
                last_reason = "socket recv timeout"
                continue
            receipts.append(
                SendReceipt(
                    message_key=plan.message_key,
                    fragment_index=fragment.fragment_index,
                    tunnel_id=fragment.tunnel_id,
                    attempt=attempt,
                    duplicate_copy=copy_index,
                    wire=wire,
                    wire_sha256=sha256_hex(wire),
                    ack=ack,
                    sent_monotonic_ms=sent,
                    ack_received_monotonic_ms=received,
                )
            )
            break
        else:
            fail(
                E_TIMEOUT,
                f"fragments[{fragment.fragment_index}].ack",
                "real loopback UDP acknowledgement timed out",
                endpoint_id=plan.endpoint.endpoint_id,
                port=plan.endpoint.port,
                fragment_index=fragment.fragment_index,
                attempts=max_retries + 1,
                timeout_ms=timeout_ms,
                reassembly_deadline_monotonic_ms=reassembly_deadline_monotonic_ms,
                last_reason=last_reason,
            )
    return tuple(receipts)


def decode_receipts(
    plan: MessagePlan,
    receipts: list[SendReceipt],
    *,
    total_fragments: int,
) -> list[NormalizedFragment]:
    context = context_for_plan(plan, total_fragments)
    decoded = []
    for receipt in receipts:
        decoded.append(decode_fragment(receipt.wire, context))
    return decoded
