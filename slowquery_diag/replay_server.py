"""Deterministic loopback replay service and its socket client.

The service is intentionally launched as a separate subprocess (Windows:
``python -m slowquery_diag.replay_server``) and listens on 127.0.0.1 with
a real TCP socket.  Frames are length-prefixed JSON::

    u32 big-endian frame length | UTF-8 JSON payload

No artificial sleeps are used to emulate quotas: concurrency is enforced
by the scheduler, while a real ``SO_RCVTIMEO``/connection timeout governs
each socket.  A ``stall`` catalog entry exists solely to exercise the
real timeout path.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
from typing import Any

from .errors import AckError, PortError, ReplayTimeoutError

HOST = "127.0.0.1"
FRAME_HEADER = 4
ACK_REQUIRED_FIELDS = (
    "type", "candidate_id", "source_id", "est_latency_us", "est_rows_scanned",
)


def _send_frame(sock: socket.socket, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sock.sendall(len(body).to_bytes(FRAME_HEADER, "big") + body)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while reading frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, FRAME_HEADER)
    length = int.from_bytes(header, "big")
    body = _recv_exact(sock, length)
    return json.loads(body.decode("utf-8"))


class DeterministicReplayService:
    """Threaded TCP service returning catalog-declared estimates."""

    def __init__(self, catalog: dict[str, dict[str, Any]], host: str = HOST) -> None:
        self._catalog = dict(catalog)
        self._host = host
        self._lock = threading.Lock()
        self._inflight = 0
        self.peak_inflight = 0
        self.total_requests = 0
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.port: int | None = None

    # ------------------------------------------------------------------
    def start(self, port: int = 0) -> int:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self._host, port))
        except OSError as exc:
            server.close()
            raise PortError(
                f"cannot bind {self._host}:{port}: {exc}",
                ref=f"{self._host}:{port}", field="bind",
            )
        server.listen(32)
        server.settimeout(0.2)
        self._server = server
        self.port = server.getsockname()[1]
        self._thread = threading.Thread(
            target=self._accept_loop, name="replay-service", daemon=True
        )
        self._thread.start()
        return self.port

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle, args=(conn,), daemon=True
            ).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            request = _recv_frame(conn)
            kind = request.get("type")
            if kind == "hello":
                _send_frame(conn, {
                    "type": "hello_ack",
                    "status": "ready",
                    "catalog_size": len(self._catalog),
                })
            elif kind == "stats":
                with self._lock:
                    payload = {
                        "type": "stats_ack",
                        "peak_inflight": self.peak_inflight,
                        "total_requests": self.total_requests,
                        "inflight": self._inflight,
                    }
                _send_frame(conn, payload)
            elif kind == "replay":
                self._handle_replay(conn, request)
            else:
                _send_frame(conn, {"type": "error", "error": "unknown request type"})
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _handle_replay(self, conn: socket.socket, request: dict[str, Any]) -> None:
        with self._lock:
            self._inflight += 1
            self.total_requests += 1
            self.peak_inflight = max(self.peak_inflight, self._inflight)
        try:
            candidate_id = request.get("candidate_id", "")
            source_id = request.get("source_id", "")
            entry = self._catalog.get(f"{candidate_id}@{source_id}")
            if entry is None:
                _send_frame(conn, {
                    "type": "ack",
                    "candidate_id": candidate_id,
                    "source_id": source_id,
                    "status": "unknown_candidate",
                })
                return
            # A real blocking sleep is used only for the dedicated timeout
            # test candidate; quotas themselves never rely on sleeps.
            stall_ms = int(entry.get("stall_ms", 0))
            if stall_ms > 0:
                self._stop.wait(stall_ms / 1000.0)
            _send_frame(conn, {
                "type": "ack",
                "candidate_id": candidate_id,
                "source_id": entry["source_id"],
                "est_latency_us": int(entry["est_latency_us"]),
                "est_rows_scanned": int(entry["est_rows_scanned"]),
                "status": "ok",
            })
        finally:
            with self._lock:
                self._inflight -= 1


class ReplayClient:
    """Synchronous client enforcing a real per-socket timeout."""

    def __init__(self, host: str = HOST, port: int = 0,
                 timeout_ms: int = 2000) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        timeout = self.timeout_ms / 1000.0
        sock.settimeout(timeout)
        try:
            sock.connect((self.host, self.port))
        except socket.timeout:
            sock.close()
            raise ReplayTimeoutError(
                f"handshake connect timed out after {self.timeout_ms}ms",
                ref=f"{self.host}:{self.port}", field="connect",
            )
        except ConnectionRefusedError as exc:
            sock.close()
            raise PortError(
                f"no replay service on {self.host}:{self.port}: {exc}",
                ref=f"{self.host}:{self.port}", field="connect",
            )
        except OSError as exc:
            sock.close()
            raise PortError(str(exc), ref=f"{self.host}:{self.port}", field="connect")
        return sock

    def hello(self) -> dict[str, Any]:
        sock = self._connect()
        try:
            _send_frame(sock, {"type": "hello"})
            return _recv_frame(sock)
        except socket.timeout:
            raise ReplayTimeoutError(
                "hello handshake timed out", field="hello"
            )
        finally:
            sock.close()

    def replay(self, candidate_id: str, source_id: str,
               rewritten_sql: str) -> dict[str, Any]:
        sock = self._connect()
        try:
            _send_frame(sock, {
                "type": "replay",
                "candidate_id": candidate_id,
                "source_id": source_id,
                "rewritten_sql": rewritten_sql,
            })
            try:
                reply = _recv_frame(sock)
            except socket.timeout:
                raise ReplayTimeoutError(
                    f"replay receipt timed out after {self.timeout_ms}ms",
                    ref=candidate_id, field="ack",
                )
            for required in ACK_REQUIRED_FIELDS:
                if required not in reply:
                    raise AckError(
                        f"receipt missing required field {required!r}",
                        ref=candidate_id, field=required, detail=reply,
                    )
            if reply["source_id"] != source_id:
                raise AckError(
                    "receipt source_id disagrees with the requested source",
                    ref=candidate_id, field="source_id", detail=reply,
                )
            return reply
        finally:
            sock.close()

    def stats(self) -> dict[str, Any]:
        sock = self._connect()
        try:
            _send_frame(sock, {"type": "stats"})
            return _recv_frame(sock)
        finally:
            sock.close()


def _load_catalog(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        f"{entry['candidate_id']}@{entry['source_id']}": entry
        for entry in payload["replay_catalog"]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback replay service")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--host", default=HOST)
    args = parser.parse_args(argv)

    catalog = _load_catalog(args.catalog)
    service = DeterministicReplayService(catalog, host=args.host)
    port = service.start(args.port)
    print(f"REPLAY_SERVICE_READY host={args.host} port={port}", flush=True)
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
