from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Iterable

from .errors import SimError
from .protocol import FRAME_HEADER_SIZE, FRAME_HEADER_STRUCT, Frame, build_frame, parse_frame
from .quantization import parse_packet


MAX_FRAME_BYTES = 512 * 1024 * 1024


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass
class LoopbackReceiver:
    worker_id: int
    host: str = "127.0.0.1"

    def __post_init__(self) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, 0))
        self._server.listen(128)
        self._server.settimeout(0.2)
        self.port = int(self._server.getsockname()[1])
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._packets: dict[tuple[int, int, str], bytes] = {}
        self._frames: list[Frame] = []
        self.receive_errors: list[dict[str, object]] = []
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name=f"loopback-receiver-{self.worker_id}",
            daemon=True,
        )
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._read_connection,
                args=(connection,),
                name=f"loopback-reader-{self.worker_id}",
                daemon=True,
            ).start()

    def _read_connection(self, connection: socket.socket) -> None:
        try:
            with connection:
                connection.settimeout(5.0)
                header = _recv_exact(connection, FRAME_HEADER_SIZE)
                if header is None:
                    return
                if len(header) < FRAME_HEADER_SIZE:
                    raise SimError("E_COMM_FRAME_INVALID", "truncated transport header")
                try:
                    _magic, _version, _origin, _src, _dst, _round, _window, packet_len = (
                        FRAME_HEADER_STRUCT.unpack_from(header, 0)
                    )
                except struct.error as exc:
                    raise SimError("E_COMM_FRAME_INVALID", "invalid transport header") from exc
                if packet_len > MAX_FRAME_BYTES:
                    raise SimError("E_COMM_FRAME_INVALID", "transport frame exceeds size limit")
                payload = _recv_exact(connection, packet_len)
                if payload is None:
                    raise SimError("E_COMM_FRAME_INVALID", "truncated transport payload")
                frame = parse_frame(header + payload)
                if frame.dst_worker != self.worker_id:
                    raise SimError(
                        "E_COMM_FRAME_INVALID",
                        f"frame for worker {frame.dst_worker} arrived at worker {self.worker_id}",
                    )
                packet_view = parse_packet(frame.packet)
                tensor_name = packet_view.header.tensor_name
                key = (frame.window, frame.origin_worker, tensor_name)
                with self._condition:
                    self._packets[key] = frame.packet
                    self._frames.append(frame)
                    self._condition.notify_all()
        except BaseException as exc:  # receiver threads must never terminate the process
            with self._condition:
                self.receive_errors.append(
                    {
                        "worker_id": self.worker_id,
                        "code": getattr(exc, "code", "E_COMM_RECEIVE_FAILED"),
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                self._condition.notify_all()

    def wait_for_keys(self, keys: Iterable[tuple[int, int, str]], timeout: float = 5.0) -> list[tuple[int, int, str]]:
        expected = set(keys)
        deadline = time.monotonic() + timeout
        with self._condition:
            while not expected.issubset(self._packets):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if not self._condition.wait(remaining):
                    break
            return sorted(expected.difference(self._packets))

    def get_packet(self, window: int, origin_worker: int, tensor_name: str) -> bytes | None:
        with self._condition:
            return self._packets.get((window, origin_worker, tensor_name))

    def keys(self) -> list[tuple[int, int, str]]:
        with self._condition:
            return sorted(self._packets)

    def stop(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        self._accept_thread.join(timeout=1.0)


class LoopbackHub:
    """One TCP receiver per logical worker; all gradient bytes cross 127.0.0.1 sockets."""

    def __init__(self, workers: int, host: str = "127.0.0.1") -> None:
        self.host = host
        self.receivers = {worker_id: LoopbackReceiver(worker_id, host=host) for worker_id in range(workers)}

    def __enter__(self) -> "LoopbackHub":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def stop(self) -> None:
        for receiver in self.receivers.values():
            receiver.stop()

    def send(
        self,
        *,
        origin_worker: int,
        src_holder: int,
        dst_worker: int,
        route_round: int,
        window: int,
        packet: bytes,
        timeout: float = 5.0,
    ) -> int:
        frame = build_frame(
            origin_worker=origin_worker,
            src_holder=src_holder,
            dst_worker=dst_worker,
            route_round=route_round,
            window=window,
            packet=packet,
        )
        receiver = self.receivers[dst_worker]
        with socket.create_connection((self.host, receiver.port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall(frame)
        return len(frame)


class BudgetLedger:
    def __init__(self, link_budgets: dict[str, int], window_budget: int) -> None:
        self._link_budgets = dict(link_budgets)
        self._window_budget = int(window_budget)
        self._link_sent = {link: 0 for link in self._link_budgets}
        self._total_sent = 0
        self._lock = threading.Lock()

    def reserve(self, link: str, wire_bytes: int) -> tuple[bool, int, int]:
        with self._lock:
            link_sent = self._link_sent.get(link, 0)
            link_budget = self._link_budgets.get(link, 0)
            if link_sent + wire_bytes > link_budget:
                return False, link_sent, link_budget
            if self._total_sent + wire_bytes > self._window_budget:
                return False, self._total_sent, self._window_budget
            self._link_sent[link] = link_sent + wire_bytes
            self._total_sent += wire_bytes
            return True, link_sent, link_budget

    def release(self, link: str, wire_bytes: int) -> None:
        with self._lock:
            self._link_sent[link] = max(0, self._link_sent.get(link, 0) - wire_bytes)
            self._total_sent = max(0, self._total_sent - wire_bytes)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "total_sent_bytes": self._total_sent,
                "window_budget_bytes": self._window_budget,
                "link_sent_bytes": dict(sorted(self._link_sent.items())),
                "link_budgets": dict(sorted(self._link_budgets.items())),
            }
