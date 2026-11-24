"""Optional local TCP status endpoint (localhost only)."""
from __future__ import annotations

import json
import socket
import threading
from dataclasses import asdict
from typing import TYPE_CHECKING

from .viewmodel import build_window_rows

if TYPE_CHECKING:  # pragma: no cover
    from ..engine import ObsFlowEngine


class StatusServer:
    def __init__(self, engine: "ObsFlowEngine", host: str = "127.0.0.1",
                 port: int = 0):
        self._engine = engine
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((host, port))
        self._socket.listen(4)
        self.host, self.port = self._socket.getsockname()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._running = True

    def start(self) -> tuple[str, int]:
        self._thread.start()
        return self.host, self.port

    def _serve(self) -> None:
        self._socket.settimeout(0.2)
        while self._running:
            try:
                conn, _ = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                rows = build_window_rows(self._engine.query_windows(),
                                         self._engine.lease_manager)
                payload = json.dumps(
                    [asdict(row) for row in rows], sort_keys=True)
                conn.sendall(payload.encode("utf-8"))

    def stop(self) -> None:
        self._running = False
        self._socket.close()
        self._thread.join(timeout=2)
