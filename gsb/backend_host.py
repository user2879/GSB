"""父进程侧后端托管：启动子进程、读取动态端口、发起真实套接字请求。"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .spec import BackendDecl

REPO_ROOT = Path(__file__).resolve().parent.parent


class BackendTimeoutError(Exception):
    """套接字超时（可重试）。"""


class BackendStaleError(Exception):
    """后端当前参数版本高于派发绑定版本（热更新交错）。"""

    def __init__(self, bound_version: int, current_version: int):
        self.bound_version = bound_version
        self.current_version = current_version
        super().__init__(
            f"绑定版本 {bound_version} 已过期，后端当前版本 {current_version}"
        )


class BackendRpcError(Exception):
    """后端返回的其它错误。"""


def _recv_exact(conn: socket.socket, length: int) -> bytes:
    payload = b""
    while len(payload) < length:
        chunk = conn.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("后端连接提前关闭")
        payload += chunk
    return payload


class BackendHost:
    def __init__(self, decl: BackendDecl):
        self.decl = decl
        self._proc: subprocess.Popen | None = None
        self.actual_port: int | None = None

    @property
    def backend_id(self) -> str:
        return self.decl.backend_id

    def start(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "gsb.backend_server",
            json.dumps(asdict(self.decl), sort_keys=True),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(REPO_ROOT),
        )
        line = self._proc.stdout.readline().strip()
        if not line.startswith("PORT "):
            raise RuntimeError(f"后端 {self.backend_id} 未报告端口: {line!r}")
        self.actual_port = int(line.split(" ", 1)[1])

    def request(self, payload: dict, timeout_ms: int) -> dict:
        if self.actual_port is None:
            raise RuntimeError("后端尚未启动")
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        frame = f"GSB-REQ {len(body)}\n".encode("utf-8") + body
        try:
            with socket.create_connection(
                ("127.0.0.1", self.actual_port), timeout=timeout_ms / 1000.0
            ) as conn:
                conn.settimeout(timeout_ms / 1000.0)
                conn.sendall(frame)
                header = b""
                while not header.endswith(b"\n"):
                    chunk = conn.recv(1)
                    if not chunk:
                        raise ConnectionError("后端连接提前关闭")
                    header += chunk
                parts = header.decode("utf-8").strip().split(" ")
                if len(parts) != 2 or parts[0] != "GSB-RESP":
                    raise BackendRpcError(f"非法响应帧头: {header!r}")
                payload_bytes = _recv_exact(conn, int(parts[1]))
        except (socket.timeout, TimeoutError) as exc:
            raise BackendTimeoutError(str(exc)) from exc
        response = json.loads(payload_bytes.decode("utf-8"))
        if not response.get("ok"):
            if response.get("error") == "STALE_VERSION":
                raise BackendStaleError(
                    response["bound_version"], response["current_version"]
                )
            raise BackendRpcError(response.get("error", "未知后端错误"))
        return response

    def hot_update(self, new_seed: int, timeout_ms: int = 5000) -> dict:
        return self.request({"cmd": "update", "seed": new_seed}, timeout_ms)

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self.request({"cmd": "shutdown"}, timeout_ms=1000)
        except Exception:
            pass
        try:
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()
        self._proc = None

    def kill(self) -> None:
        """模拟强制终止（用于崩溃恢复测试）。"""
        if self._proc is not None:
            self._proc.kill()
            self._proc = None
