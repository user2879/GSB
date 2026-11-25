"""评估器子进程生命周期管理与同步租约客户端。"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

from .errors import LeaseError
from .evaluator import (
    GRANT,
    RENEW,
    REVOKE,
    REAP,
    SHUTDOWN,
    SNAPSHOT,
    _recv_frame,
    _send_frame,
)


class EvaluatorProcess:
    """以本地子进程方式启动评估器，并通过 AF_UNIX 访问。"""

    def __init__(
        self,
        socket_path: str,
        global_limit: int,
        per_service_limit: int,
        log_path: str | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.global_limit = global_limit
        self.per_service_limit = per_service_limit
        self.log_path = log_path
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        from . import evaluator

        log_handle = None
        if self.log_path:
            log_handle = open(self.log_path, "a", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "slo_burn.evaluator",
                self.socket_path,
                str(self.global_limit),
                str(self.per_service_limit),
            ],
            stdout=log_handle,
            stderr=log_handle,
        )
        for _ in range(100):
            if Path(self.socket_path).exists():
                try:
                    with LeaseClient(self.socket_path) as client:
                        client.snapshot()
                    return
                except (ConnectionError, FileNotFoundError, OSError):
                    pass
            if self.process.poll() is not None:
                raise LeaseError("评估器子进程启动后立即退出")
            import time

            time.sleep(0.02)
        raise LeaseError("评估器子进程启动超时")

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            with LeaseClient(self.socket_path) as client:
                client.request({"kind": SHUTDOWN})
        except (ConnectionError, FileNotFoundError, OSError, LeaseError):
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        finally:
            self.process = None
            Path(self.socket_path).unlink(missing_ok=True)

    def __enter__(self) -> "EvaluatorProcess":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class LeaseClient:
    """每次请求建立短连接：服务端单帧处理后关闭，避免异常帧破坏同步。"""

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path

    def connect(self) -> None:
        # 兼容历史接口；真正连接在 request 内按需建立
        self._probe()

    def _probe(self) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.connect(self.socket_path)

    def close(self) -> None:
        return None

    def __enter__(self) -> "LeaseClient":
        self._probe()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def request(self, payload: dict) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(self.socket_path)
            _send_frame(connection, payload)
            response = _recv_frame(connection)
        if not response.get("ok"):
            error = response.get("error", {})
            raise LeaseError(
                error.get("message", "租约请求失败"),
                **error.get("context", {}),
            )
        return response

    def grant(self, service_id: str, ttl_ms: int) -> dict:
        return self.request({"kind": GRANT, "service_id": service_id, "ttl_ms": ttl_ms})

    def renew(self, lease_id: str, version: int, ttl_ms: int) -> dict:
        return self.request(
            {
                "kind": RENEW,
                "lease_id": lease_id,
                "version": version,
                "ttl_ms": ttl_ms,
            }
        )

    def revoke(self, lease_id: str, version: int) -> dict:
        return self.request(
            {"kind": REVOKE, "lease_id": lease_id, "version": version}
        )

    def reap(self) -> dict:
        return self.request({"kind": REAP})

    def snapshot(self) -> dict:
        return self.request({"kind": SNAPSHOT})
