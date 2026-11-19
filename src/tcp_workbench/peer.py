"""Deterministic scripted peer over a real 127.0.0.1 TCP socket."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from .clock import monotonic_ms
from .errors import E_CONNECT, E_PORT, E_SCRIPT, BusinessError
from .models import PeerScript

MAX_FRAME_BYTES = 8 * 1024 * 1024


def _send_frame(conn: socket.socket, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)) + payload)


def _recv_exact(conn: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while reading frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(conn: socket.socket) -> Any:
    header = _recv_exact(conn, 4)
    length = struct.unpack("!I", header)[0]
    if length > MAX_FRAME_BYTES:
        raise ValueError("frame exceeds maximum size")
    return json.loads(_recv_exact(conn, length).decode("utf-8"))


@dataclass
class PeerBatchResult:
    responses: list[dict[str, Any]]
    dropped_seqs: list[int]
    server_mono_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "responses": self.responses,
            "dropped_seqs": self.dropped_seqs,
            "server_mono_ms": self.server_mono_ms,
        }


class DeterministicPeer:
    def __init__(
        self, script: PeerScript, host: str = "127.0.0.1", port: int = 43177
    ) -> None:
        self.script = script
        self.host = host
        self.requested_port = port
        self.port = port
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: BaseException | None = None

    def start(self) -> DeterministicPeer:
        if self.requested_port != 0 and self.requested_port < 1024:
            raise BusinessError(
                E_PORT, "a high TCP port (>=1024) is required", ["port"]
            )
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.bind((self.host, self.requested_port))
            server.listen(8)
            server.settimeout(0.25)
        except OSError as exc:
            server.close()
            raise BusinessError(
                E_PORT,
                f"cannot listen on {self.host}:{self.requested_port}: {exc}",
                ["port"],
            ) from exc
        self._socket = server
        self.port = int(server.getsockname()[1])
        self._thread = threading.Thread(
            target=self._serve, name="deterministic-peer", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._socket = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def _serve(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    self._error = exc
                return
            with conn:
                try:
                    request = _recv_frame(conn)
                    if not isinstance(request, dict) or request.get("kind") != "batch":
                        raise ValueError("expected a batch request")
                    result = self.respond_to_batch(request.get("packets", []))
                    _send_frame(conn, {"kind": "batch_result", **result.as_dict()})
                except Exception as exc:  # noqa: BLE001
                    try:
                        _send_frame(
                            conn,
                            {"kind": "error", "code": E_SCRIPT, "message": str(exc)},
                        )
                    except OSError:
                        pass

    def respond_to_batch(self, packets: list[dict[str, Any]]) -> PeerBatchResult:
        if not isinstance(packets, list) or not packets:
            raise ValueError("packet batch must be a non-empty list")
        normal: list[dict[str, Any]] = []
        reordered: list[dict[str, Any]] = []
        dropped: list[int] = []
        for index, packet in enumerate(packets):
            for_seq = packet.get("seq")
            payload_bytes = packet.get("payload_bytes", 0)
            if not isinstance(for_seq, int) or isinstance(for_seq, bool):
                raise TypeError("packet seq must be an integer")
            if (
                not isinstance(payload_bytes, int)
                or isinstance(payload_bytes, bool)
                or payload_bytes < 0
            ):
                raise ValueError("packet payload_bytes must be a non-negative integer")
            response = self.script.response_map.get(for_seq)
            if response is None:
                if self.script.default_policy == "drop":
                    dropped.append(for_seq)
                    continue
                action = "ack"
                ack_seq = for_seq + payload_bytes
                delay_ms = self.script.default_delay_ms
                reorder = False
                is_default = True
            else:
                action = response.action
                ack_seq = response.ack_seq
                delay_ms = response.delay_ms
                reorder = response.reorder
                is_default = False
            if action == "drop":
                if delay_ms:
                    time.sleep(delay_ms / 1000.0)
                dropped.append(for_seq)
                continue
            if delay_ms:
                time.sleep(delay_ms / 1000.0)
            item = {
                "source_index": index,
                "for_seq": for_seq,
                "action": action,
                "ack_seq": ack_seq,
                "delay_ms": delay_ms,
                "reordered": bool(reorder),
                "default_policy": is_default,
                "server_mono_ms": monotonic_ms(),
            }
            if reorder:
                reordered.append(item)
            else:
                normal.append(item)
        responses = normal + reordered
        for arrival_index, response in enumerate(responses):
            response["arrival_index"] = arrival_index
        return PeerBatchResult(
            responses=responses, dropped_seqs=dropped, server_mono_ms=monotonic_ms()
        )


def send_batch(
    host: str,
    port: int,
    packets: list[dict[str, Any]],
    *,
    timeout_s: float = 5.0,
) -> PeerBatchResult:
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as conn:
            conn.settimeout(timeout_s)
            _send_frame(conn, {"kind": "batch", "packets": packets})
            response = _recv_frame(conn)
    except (OSError, ConnectionError, ValueError, json.JSONDecodeError) as exc:
        raise BusinessError(
            E_CONNECT, f"cannot connect or exchange with peer: {exc}", ["host", "port"]
        ) from exc
    if not isinstance(response, dict):
        raise BusinessError(
            E_CONNECT, "peer returned a malformed response", ["peer.response"]
        )
    if response.get("kind") == "error":
        raise BusinessError(
            str(response.get("code", E_SCRIPT)),
            str(response.get("message", "peer protocol error")),
            ["peer.response"],
        )
    if response.get("kind") != "batch_result":
        raise BusinessError(
            E_CONNECT, "peer returned an unexpected response kind", ["peer.response"]
        )
    return PeerBatchResult(
        responses=list(response.get("responses", [])),
        dropped_seqs=[int(value) for value in response.get("dropped_seqs", [])],
        server_mono_ms=int(response.get("server_mono_ms", 0)),
    )
