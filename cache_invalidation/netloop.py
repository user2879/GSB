"""回环端口上的确定性 JSON Lines 服务。

仅绑定 127.0.0.1，不访问任何外部地址。每条请求/响应都是单行 JSON，
便于用标准库 socket 做可重复的端到端验证。
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass

from .engine import Engine
from .errors import BusinessError
from .model import EntryID


def find_free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@dataclass
class LoopbackServer:
    engine: Engine
    host: str = "127.0.0.1"
    port: int = 0

    def __post_init__(self) -> None:
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(8)
        sock.settimeout(0.2)
        self._socket = sock
        self.port = int(sock.getsockname()[1])
        self._thread = threading.Thread(
            target=self._serve, name="cache-loopback", daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _serve(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._socket.accept()
            except (socket.timeout, OSError):
                continue
            with conn:
                self._handle(conn)

    def _handle(self, conn: socket.socket) -> None:
        file = conn.makefile("rwb", buffering=0)
        while not self._stop.is_set():
            line = file.readline()
            if not line:
                return
            try:
                request = json.loads(line.decode("utf-8"))
                response = self.dispatch(request)
            except BusinessError as exc:
                response = {"ok": False, "error": exc.to_dict()}
            except Exception as exc:  # 防御性：服务线程不能因单请求退出
                response = {
                    "ok": False,
                    "error": {"code": "E_INTERNAL", "message": str(exc)},
                }
            file.write((json.dumps(response, ensure_ascii=False)
                        + "\n").encode("utf-8"))

    def dispatch(self, request: dict) -> dict:
        command = request.get("command")
        if command == "ping":
            return {"ok": True, "pong": True}
        if command == "invalidate":
            result = self.engine.explicit_invalidate(
                EntryID(request["namespace"], int(request["key"])),
                int(request["now_ms"]))
            return {"ok": True,
                    "affected": list(result.affected)}
        if command == "reap":
            result = self.engine.reap_expired(int(request["now_ms"]))
            return {"ok": True,
                    "affected": list(result.affected)}
        if command == "refresh":
            result = self.engine.refresh_namespace(
                request["namespace"], int(request["now_ms"]))
            return {"ok": True,
                    "affected": list(result.affected)}
        if command == "state":
            return {
                "ok": True,
                "format_version": self.engine.format_version,
                "versions": {
                    entry_id.to_str(): version
                    for entry_id, version in sorted(
                        self.engine.versions.items())
                },
                "alive": sorted(entry_id.to_str()
                                for entry_id in self.engine.alive),
            }
        return {"ok": False,
                "error": {"code": "E_FORMAT",
                          "message": f"未知命令: {command!r}"}}


class LoopbackClient:
    def __init__(self, port: int, host: str = "127.0.0.1",
                 timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def call(self, request: dict) -> dict:
        with socket.create_connection((self.host, self.port),
                                     timeout=self.timeout) as conn:
            file = conn.makefile("rwb", buffering=0)
            file.write((json.dumps(request, ensure_ascii=False)
                        + "\n").encode("utf-8"))
            line = file.readline()
        return json.loads(line.decode("utf-8"))
