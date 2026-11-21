"""Real loopback UDP and deterministic in-memory transport backends."""

from __future__ import annotations

import socket
from collections import defaultdict, deque
from typing import Deque, Iterable, Protocol

from .errors import LedgerError


LoopbackHost = "127.0.0.1"


class Transport(Protocol):
    @property
    def ports(self) -> dict[str, int]: ...

    @property
    def datagrams_sent(self) -> int: ...

    @property
    def datagrams_received(self) -> int: ...

    @property
    def sink_datagrams(self) -> int: ...

    def receive_stage(self, stage_id: str, *, timeout: float = 3.0) -> bytes: ...

    def deliver_stage(self, stage_id: str, datagram: bytes) -> bytes: ...

    def deliver_sink(self, datagram: bytes) -> None: ...

    def close(self) -> None: ...


class InMemoryTransport:
    """No-network backend used by pure replay/unit tests; same byte boundaries."""

    def __init__(self, stage_ids: Iterable[str], *, base_port: int = 49152) -> None:
        ordered = sorted(stage_ids)
        self._ports = {stage_id: base_port + index for index, stage_id in enumerate(ordered)}
        self._queues: dict[str, Deque[bytes]] = defaultdict(deque)
        self._sent = 0
        self._received = 0
        self._sink_datagrams = 0
        self._sink_bytes = 0
        self._closed = False

    @property
    def ports(self) -> dict[str, int]:
        return dict(self._ports)

    @property
    def datagrams_sent(self) -> int:
        return self._sent

    @property
    def datagrams_received(self) -> int:
        return self._received

    @property
    def sink_datagrams(self) -> int:
        return self._sink_datagrams

    @property
    def sink_bytes(self) -> int:
        return self._sink_bytes

    def send_stage(self, stage_id: str, datagram: bytes) -> None:
        if self._closed:
            raise LedgerError("transport is closed")
        if stage_id not in self._ports:
            raise LedgerError(f"unknown stage: {stage_id}")
        self._queues[stage_id].append(bytes(datagram))
        self._sent += 1

    def receive_stage(self, stage_id: str, *, timeout: float = 3.0) -> bytes:
        del timeout
        try:
            datagram = self._queues[stage_id].popleft()
        except (KeyError, IndexError) as exc:
            raise LedgerError(f"no datagram queued for stage {stage_id}") from exc
        self._received += 1
        return datagram

    def deliver_stage(self, stage_id: str, datagram: bytes) -> bytes:
        self.send_stage(stage_id, datagram)
        return self.receive_stage(stage_id)

    def deliver_sink(self, datagram: bytes) -> None:
        if self._closed:
            raise LedgerError("transport is closed")
        self._sink_datagrams += 1
        self._sink_bytes += len(datagram)
        self._sent += 1

    def close(self) -> None:
        self._closed = True


class LoopbackTransport:
    """One real UDP ingress socket per shaper plus one final receiver socket."""

    def __init__(
        self,
        stage_ids: Iterable[str],
        *,
        base_port: int = 49152,
        timeout: float = 3.0,
    ) -> None:
        if not 49152 <= base_port <= 65535:
            raise ValueError("base_port must be in [49152, 65535]")
        self._stage_ids = tuple(sorted(stage_ids))
        self._timeout = timeout
        self._sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sockets: dict[str, socket.socket] = {}
        self._ports: dict[str, int] = {}
        self._sent = 0
        self._received = 0
        self._sink_datagrams = 0
        self._sink_bytes = 0
        self._closed = False
        next_port = base_port
        try:
            for stage_id in self._stage_ids:
                sock, next_port = self._bind_next(next_port)
                self._sockets[stage_id] = sock
                self._ports[stage_id] = sock.getsockname()[1]
            self._sink, next_port = self._bind_next(next_port)
            self._sink_port = self._sink.getsockname()[1]
        except Exception:
            self.close()
            raise

    def _bind_next(self, start_port: int) -> tuple[socket.socket, int]:
        last_error: OSError | None = None
        for port in range(start_port, 65536):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind((LoopbackHost, port))
                sock.settimeout(self._timeout)
                return sock, port + 1
            except OSError as exc:
                last_error = exc
                sock.close()
        raise LedgerError(
            f"cannot allocate a free loopback UDP port at or above {start_port}: {last_error}"
        )

    @property
    def ports(self) -> dict[str, int]:
        return dict(self._ports)

    @property
    def sink_port(self) -> int:
        return self._sink_port

    @property
    def datagrams_sent(self) -> int:
        return self._sent

    @property
    def datagrams_received(self) -> int:
        return self._received

    @property
    def sink_datagrams(self) -> int:
        return self._sink_datagrams

    @property
    def sink_bytes(self) -> int:
        return self._sink_bytes

    def send_stage(self, stage_id: str, datagram: bytes) -> None:
        if self._closed:
            raise LedgerError("transport is closed")
        try:
            port = self._ports[stage_id]
        except KeyError as exc:
            raise LedgerError(f"unknown stage: {stage_id}") from exc
        self._sender.sendto(datagram, (LoopbackHost, port))
        self._sent += 1

    def receive_stage(self, stage_id: str, *, timeout: float | None = None) -> bytes:
        if self._closed:
            raise LedgerError("transport is closed")
        try:
            sock = self._sockets[stage_id]
        except KeyError as exc:
            raise LedgerError(f"unknown stage: {stage_id}") from exc
        old_timeout = sock.gettimeout()
        if timeout is not None:
            sock.settimeout(timeout)
        try:
            datagram, _ = sock.recvfrom(65535)
        except socket.timeout as exc:
            raise LedgerError(f"timed out receiving UDP at stage {stage_id}") from exc
        finally:
            sock.settimeout(old_timeout)
        self._received += 1
        return datagram

    def deliver_stage(self, stage_id: str, datagram: bytes) -> bytes:
        self.send_stage(stage_id, datagram)
        return self.receive_stage(stage_id)

    def deliver_sink(self, datagram: bytes) -> None:
        if self._closed:
            raise LedgerError("transport is closed")
        self._sender.sendto(datagram, (LoopbackHost, self._sink_port))
        self._sent += 1
        try:
            received, _ = self._sink.recvfrom(65535)
        except socket.timeout as exc:
            raise LedgerError("timed out receiving final sink datagram") from exc
        if received != datagram:
            raise LedgerError("loopback final sink datagram corruption")
        self._received += 1
        self._sink_datagrams += 1
        self._sink_bytes += len(received)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sock in self._sockets.values():
            try:
                sock.close()
            except OSError:
                pass
        sink = getattr(self, "_sink", None)
        if sink is not None:
            try:
                sink.close()
            except OSError:
                pass
        try:
            self._sender.close()
        except OSError:
            pass

    def __enter__(self) -> "LoopbackTransport":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
