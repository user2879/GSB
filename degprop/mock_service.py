"""绑定 127.0.0.1 的确定性模拟服务（真实回环 TCP）。

每个服务持有两个套接字：
    * live 套接字：正常 listen/accept，按共享 LCG 规则返回成功或慢响应；
    * refused 套接字：仅 bind 不 listen，connect 必然得到真实
      ConnectionRefusedError（Windows 上对应 WSAECONNREFUSED）。

慢响应分支在服务端线程内真实睡眠，再写回响应；客户端超时完全由
socket.settimeout 驱动。
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .models import LOOPBACK, Outcome
from .outcome import plan_outcome

logger = logging.getLogger("degprop.mock")


@dataclass
class ServiceBinding:
    service_id: str
    port: int
    refused_port: int


class MockService(threading.Thread):
    def __init__(self, service_id: str, port: int, seed: int,
                 *, stop_event: threading.Event, slow_ms: int = 400,
                 handle_ms: int = 12) -> None:
        super().__init__(name=f"mock-{service_id}", daemon=True)
        self.service_id = service_id
        self.port = port
        self.seed = seed
        self.stop_event = stop_event
        self.slow_ms = slow_ms
        self.handle_ms = handle_ms
        self._live = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._live.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._live.bind((LOOPBACK, port))
        self._live.listen(32)
        self._live.settimeout(0.2)
        # 仅 bind 不 listen：保留端口并使后续 connect 被真实拒绝。
        self._refused = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._refused.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._refused.bind((LOOPBACK, 0))
        self.binding = ServiceBinding(service_id, port,
                                      self._refused.getsockname()[1])

    def run(self) -> None:
        logger.info("mock_service_started", extra={"service_id": self.service_id,
                                                    "port": self.port,
                                                    "refused_port": self.binding.refused_port})
        while not self.stop_event.is_set():
            try:
                conn, _ = self._live.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                payload = b""
                conn.settimeout(5.0)
                while not payload.endswith(b"\n"):
                    chunk = conn.recv(256)
                    if not chunk:
                        return
                    payload += chunk
                request = json.loads(payload.decode("utf-8"))
                plan = plan_outcome(self.seed, request["seq"],
                                    slow_ms=self.slow_ms,
                                    handle_ms=self.handle_ms)
                # 真实处理耗时：慢响应分支睡眠足够久，让客户端 socket
                # 超时先于响应到达；普通分支睡眠固定处理毫秒。
                delay_ms = (self.slow_ms
                            if plan.outcome in (Outcome.SLOW_SUCCESS.value,
                                                Outcome.TIMEOUT.value)
                            else plan.handle_ms)
                time.sleep(delay_ms / 1000.0)
                body = json.dumps({
                    "service_id": self.service_id,
                    "seq": request["seq"],
                    "outcome": plan.outcome,
                    "bucket": plan.bucket,
                    "handle_ms": plan.handle_ms,
                }).encode("utf-8")
                conn.sendall(body)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.debug("mock_handle_error", extra={"service_id": self.service_id,
                                                          "error": str(exc)})

    def shutdown(self) -> None:
        self._live.close()
        self._refused.close()


class MockCluster:
    """一组模拟服务的进程内生命周期管理，并写出拒绝端口映射文件。"""

    def __init__(self, topology, *, slow_ms: int = 400) -> None:
        self.topology = topology
        self.slow_ms = slow_ms
        self.stop_event = threading.Event()
        self.services: list[MockService] = []
        self.bindings: dict[str, ServiceBinding] = {}

    def start(self) -> dict[str, ServiceBinding]:
        for service in self.topology.services.values():
            mock = MockService(service.service_id, service.port, service.seed,
                               stop_event=self.stop_event, slow_ms=self.slow_ms)
            mock.start()
            self.services.append(mock)
            self.bindings[service.service_id] = mock.binding
        self._wait_ready()
        return self.bindings

    def _wait_ready(self) -> None:
        for binding in self.bindings.values():
            probe = socket.socket()
            probe.settimeout(2.0)
            for _ in range(50):
                try:
                    probe.connect((LOOPBACK, binding.port))
                    probe.close()
                    break
                except OSError:
                    self.stop_event.wait(0.02)
            else:
                raise RuntimeError(f"模拟服务 {binding.service_id} 未就绪")

    def write_port_map(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            sid: {"port": b.port, "refused_port": b.refused_port}
            for sid, b in self.bindings.items()
        }, indent=2), encoding="utf-8")

    def stop(self) -> None:
        self.stop_event.set()
        for mock in self.services:
            mock.shutdown()
            mock.join(timeout=2.0)
        logger.info("mock_cluster_stopped", extra={"count": len(self.services)})
