from __future__ import annotations

import socket
from dataclasses import dataclass, field


LOOPBACK_HOST = "127.0.0.1"


class LoopbackSignalError(RuntimeError):
    pass


def check_loopback_port(port: int = 0, host: str = LOOPBACK_HOST) -> int:
    """Bind-check a loopback port and return the selected port.

    Port 0 asks the operating system for an available ephemeral port. No
    non-loopback address is ever contacted.
    """

    if host != LOOPBACK_HOST:
        raise ValueError("only the 127.0.0.1 loopback host is allowed")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((LOOPBACK_HOST, port))
        return int(probe.getsockname()[1])


@dataclass
class LoopbackSignalBroker:
    """A local TCP handshake used as a deterministic connection signal."""

    requested_port: int = 0
    timeout_seconds: float = 2.0
    _server: socket.socket | None = field(default=None, init=False, repr=False)
    _peers: dict[str, tuple[socket.socket, socket.socket]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.requested_port < 0 or self.requested_port > 65535:
            raise ValueError("requested_port must be between 0 and 65535")

    def start(self) -> "LoopbackSignalBroker":
        if self._server is not None:
            return self
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((LOOPBACK_HOST, self.requested_port))
            server.listen(16)
            server.settimeout(self.timeout_seconds)
        except OSError:
            server.close()
            raise
        self._server = server
        return self

    @property
    def port(self) -> int:
        if self._server is None:
            raise LoopbackSignalError("loopback broker is not started")
        return int(self._server.getsockname()[1])

    def establish(self, connection_id: str) -> dict[str, object]:
        if self._server is None:
            raise LoopbackSignalError("loopback broker is not started")
        if connection_id in self._peers:
            raise LoopbackSignalError(f"connection already established: {connection_id}")
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(self.timeout_seconds)
        peer: socket.socket | None = None
        try:
            client.connect((LOOPBACK_HOST, self.port))
            peer, _ = self._server.accept()
            peer.settimeout(self.timeout_seconds)
            peer.sendall(b"ESTABLISH")
            request_signal = client.recv(16)
            if request_signal != b"ESTABLISH":
                raise LoopbackSignalError(f"unexpected establish signal: {request_signal!r}")
            client.sendall(b"ESTABLISHED")
            acknowledgement = peer.recv(16)
            if acknowledgement != b"ESTABLISHED":
                raise LoopbackSignalError(f"unexpected handshake acknowledgement: {acknowledgement!r}")
            peer.sendall(b"ACK")
            if client.recv(16) != b"ACK":
                raise LoopbackSignalError("missing establish ACK")
        except OSError as exc:
            client.close()
            if peer is not None:
                peer.close()
            raise LoopbackSignalError(f"loopback establish failed for {connection_id}: {exc}") from exc
        self._peers[connection_id] = (client, peer)
        return {
            "transport": "loopback_tcp",
            "host": LOOPBACK_HOST,
            "port": self.port,
            "signal": "ESTABLISHED",
        }

    def close(self, connection_id: str) -> dict[str, object]:
        pair = self._peers.pop(connection_id, None)
        if pair is None:
            raise LoopbackSignalError(f"connection is not held by broker: {connection_id}")
        client, peer = pair
        try:
            client.sendall(b"CLOSE")
            close_signal = peer.recv(16)
            if close_signal != b"CLOSE":
                raise LoopbackSignalError(f"unexpected close signal: {close_signal!r}")
            peer.sendall(b"CLOSED")
            if client.recv(16) != b"CLOSED":
                raise LoopbackSignalError("missing close acknowledgement")
        except OSError as exc:
            raise LoopbackSignalError(f"loopback close failed for {connection_id}: {exc}") from exc
        finally:
            for socket_object in (client, peer):
                try:
                    socket_object.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                socket_object.close()
        return {
            "transport": "loopback_tcp",
            "host": LOOPBACK_HOST,
            "port": self.port,
            "signal": "CLOSED",
        }

    def resync(self, connection_ids: list[str]) -> list[str]:
        """Reopen loopback handshakes after process recovery.

        A forced process termination necessarily destroys in-memory sockets.
        The committed connection state remains authoritative; this method only
        restores the local signal transport used for later close acknowledgements.
        """

        reopened: list[str] = []
        for connection_id in sorted(set(connection_ids)):
            if connection_id in self._peers:
                continue
            self.establish(connection_id)
            reopened.append(connection_id)
        return reopened

    def stop(self) -> None:
        for connection_id in list(self._peers):
            try:
                self.close(connection_id)
            except LoopbackSignalError:
                pass
        if self._server is not None:
            self._server.close()
            self._server = None

    def __enter__(self) -> "LoopbackSignalBroker":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


class NullSignalBroker:
    """Replay-only signal broker; it deliberately opens no sockets."""

    def establish(self, connection_id: str) -> dict[str, object]:
        return {"transport": "replay", "signal": "ESTABLISHED", "socket_opened": False}

    def close(self, connection_id: str) -> dict[str, object]:
        return {"transport": "replay", "signal": "CLOSED", "socket_opened": False}

    def resync(self, connection_ids: list[str]) -> list[str]:
        return []
