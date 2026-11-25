"""执行器子进程编排与双约束并行调度。

约束：
- 全局在途分片上限 ``max_in_flight``；
- 单执行器并发配额 ``per_executor_slots``；
- 越界即排队，等待起止用单调时钟毫秒记录，不使用任何人造延迟制造排队；
- 握手与读取均走真实套接字超时。

回收模型：等待配额期间收到的回执由 ``_acquire_handle`` 显式返回，
不依赖共享可变缓冲，避免丢帧。
"""

import os
import select
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .errors import PortError, ProtocolError
from .protocol import READY, find_free_port, is_port_in_use, recv_frame, send_frame


def now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class ExecutorHandle:
    executor_id: str
    port: int
    process: subprocess.Popen
    socket: Any
    in_flight: int = 0
    dispatched: int = 0


@dataclass
class WaitEvent:
    shard_id: str
    reason: str
    wait_start_ms: int
    wait_end_ms: int

    @property
    def waited_ms(self) -> int:
        return max(0, self.wait_end_ms - self.wait_start_ms)


@dataclass
class ScheduleReport:
    receipts: List[Dict[str, Any]] = field(default_factory=list)
    waits: List[WaitEvent] = field(default_factory=list)
    max_global_in_flight: int = 0
    max_executor_in_flight: Dict[str, int] = field(default_factory=dict)


class Scheduler:
    def __init__(
        self,
        data_dir: str,
        executor_count: int,
        max_in_flight: int,
        per_executor_slots: int,
        socket_timeout: float = 10.0,
        declared_ports: Optional[List[int]] = None,
    ):
        if executor_count <= 0 or max_in_flight <= 0 or per_executor_slots <= 0:
            raise ValueError("并发参数必须为正整数")
        self.data_dir = data_dir
        self.executor_count = executor_count
        self.max_in_flight = max_in_flight
        self.per_executor_slots = per_executor_slots
        self.socket_timeout = socket_timeout
        self.declared_ports = declared_ports
        self.handles: List[ExecutorHandle] = []

    def _allocate_port(self, index: int) -> int:
        if self.declared_ports and index < len(self.declared_ports):
            port = self.declared_ports[index]
            if is_port_in_use(port):
                raise PortError(
                    "声明端口已被占用", object_id=f"executor-{index}", field="port"
                )
            return port
        return find_free_port()

    def start(self) -> None:
        for index in range(self.executor_count):
            port = self._allocate_port(index)
            executor_id = f"executor-{index}"
            log_dir = os.path.join(self.data_dir, "..", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_handle = open(
                os.path.join(log_dir, f"{executor_id}.log"), "a", encoding="utf-8"
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "coljoin.executor",
                    "--data-dir",
                    self.data_dir,
                    "--port",
                    str(port),
                    "--id",
                    executor_id,
                    "--timeout",
                    str(self.socket_timeout),
                ],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self._await_listening(process, port, executor_id, log_handle.name)
            sock = self._connect(port, executor_id)
            self.handles.append(ExecutorHandle(executor_id, port, process, sock))

    def _await_listening(self, process, port, executor_id, log_path):
        deadline = time.monotonic() + self.socket_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                with open(log_path, "r", encoding="utf-8") as handle:
                    detail = handle.read()
                raise ProtocolError(
                    f"执行器 {executor_id} 提前退出: {detail}",
                    object_id=executor_id,
                    field="process",
                )
            if is_port_in_use(port):
                return
            time.sleep(0.01)
        raise ProtocolError(
            f"执行器 {executor_id} 监听超时", object_id=executor_id, field="port"
        )

    def _connect(self, port: int, executor_id: str):
        deadline = time.monotonic() + self.socket_timeout
        last_error = None
        while time.monotonic() < deadline:
            sock = socket.socket()
            sock.settimeout(self.socket_timeout)
            try:
                sock.connect(("127.0.0.1", port))
                hello = recv_frame(sock)
                if hello.get("protocol") != "coljoin-tcp-v1":
                    raise ProtocolError("握手协议不匹配", object_id=executor_id)
                send_frame(sock, READY)
                return sock
            except OSError as exc:
                last_error = exc
                sock.close()
                time.sleep(0.01)
        raise ProtocolError(
            f"执行器 {executor_id} 连接失败: {last_error}",
            object_id=executor_id,
            field="connect",
        )

    def _global_busy(self) -> int:
        return sum(handle.in_flight for handle in self.handles)

    def _drain_ready(self) -> List[Dict[str, Any]]:
        sockets = [handle.socket for handle in self.handles if handle.in_flight > 0]
        if not sockets:
            return []
        readable, _, _ = select.select(sockets, [], [], self.socket_timeout)
        if not readable:
            raise ProtocolError("等待执行器响应超时", field="select")
        socket_to_handle = {handle.socket: handle for handle in self.handles}
        frames: List[Dict[str, Any]] = []
        for sock in readable:
            handle = socket_to_handle[sock]
            frame = recv_frame(handle.socket)
            if frame.get("type") == "error":
                raise ProtocolError(
                    frame.get("message", "执行器错误"),
                    object_id=handle.executor_id,
                    field=frame.get("code"),
                )
            handle.in_flight -= 1
            frames.append(frame)
        return frames

    def _acquire_handle(
        self, shard_id: str, waits: List[WaitEvent]
    ) -> Tuple[ExecutorHandle, List[Dict[str, Any]]]:
        wait_start: Optional[int] = None
        reason = None
        drained: List[Dict[str, Any]] = []
        while True:
            if self._global_busy() >= self.max_in_flight:
                if wait_start is None:
                    wait_start = now_ms()
                reason = "global_in_flight_limit"
            else:
                available = [
                    handle
                    for handle in self.handles
                    if handle.in_flight < self.per_executor_slots
                ]
                if available:
                    handle = sorted(
                        available, key=lambda item: item.executor_id
                    )[0]
                    if wait_start is not None:
                        waits.append(
                            WaitEvent(shard_id, reason, wait_start, now_ms())
                        )
                    return handle, drained
                if wait_start is None:
                    wait_start = now_ms()
                reason = "executor_slot_limit"
            drained.extend(self._drain_ready())

    def run(self, requests: List[Dict[str, Any]]) -> ScheduleReport:
        report = ScheduleReport()
        pending = list(requests)
        while pending or any(handle.in_flight for handle in self.handles):
            if pending:
                request = pending.pop(0)
                handle, drained = self._acquire_handle(
                    request["batch"]["shard_id"], report.waits
                )
                report.receipts.extend(drained)
                send_frame(handle.socket, request)
                handle.in_flight += 1
                handle.dispatched += 1
                report.max_global_in_flight = max(
                    report.max_global_in_flight, self._global_busy()
                )
                report.max_executor_in_flight[handle.executor_id] = max(
                    report.max_executor_in_flight.get(handle.executor_id, 0),
                    handle.in_flight,
                )
                continue
            report.receipts.extend(self._drain_ready())
        report.receipts.sort(key=lambda item: item["shard_id"])
        return report

    def stop(self) -> None:
        for handle in self.handles:
            try:
                send_frame(handle.socket, {"type": "shutdown"})
                recv_frame(handle.socket)
            except Exception:
                pass
            try:
                handle.socket.close()
            except Exception:
                pass
            handle.process.wait(timeout=self.socket_timeout)
        self.handles.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()
        return False
