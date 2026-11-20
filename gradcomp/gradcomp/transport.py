"""环形传输层：本地回环 TCP 端点与进程内回环传输。

TCP 端点每个工作进程监听 127.0.0.1 上唯一端口 base_port+rank，
并与环上的下一个 rank 建立单条长连接；绑定/连接失败报 E_PORT，
握手与帧违规报 E_PROTO。所有 rank 先完成监听再互联，避免启动死锁。
"""
from __future__ import annotations

import queue
import socket
import time
from typing import Protocol

import numpy as np

from . import protocol as p
from .config import Config
from .errors import E_PORT, E_PROTO, GradCompError

LOOPBACK = "127.0.0.1"
CONNECT_RETRIES = 100
RETRY_INTERVAL = 0.05
SOCKET_TIMEOUT = 20.0


class RingTransport(Protocol):
    def send(self, rank: int, msg: p.Message) -> None: ...
    def recv(self, rank: int) -> p.Message: ...
    def close(self) -> None: ...


class TcpEndpoint:
    """单个工作进程视角的环端点：一条出连接、一条入连接。"""

    def __init__(self, rank: int, cfg: Config):
        self.rank = rank
        self.cfg = cfg
        self.n = cfg.workers
        self.next_rank = (rank + 1) % self.n
        self.prev_rank = (rank - 1) % self.n
        self._server: socket.socket | None = None
        self._out: socket.socket | None = None
        self._in: socket.socket | None = None

    def _bind(self) -> None:
        port = self.cfg.base_port + self.rank
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((LOOPBACK, port))
            server.listen(1)
            server.settimeout(SOCKET_TIMEOUT)
        except OSError as exc:
            server.close()
            raise GradCompError(
                E_PORT, f"回环端口 {port} 占用或不可绑定: {exc}",
                field="base_port", details={"port": port, "rank": self.rank})
        self._server = server

    def _connect_next(self) -> None:
        port = self.cfg.base_port + self.next_rank
        last_exc: Exception | None = None
        for _ in range(CONNECT_RETRIES):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((LOOPBACK, port))
                sock.settimeout(SOCKET_TIMEOUT)
                self._out = sock
                return
            except OSError as exc:
                last_exc = exc
                sock.close()
                time.sleep(RETRY_INTERVAL)
        raise GradCompError(
            E_PORT, f"无法连接环上对端端口 {port}: {last_exc}",
            field="base_port",
            details={"port": port, "rank": self.rank})

    def setup(self) -> None:
        # 1) 所有 rank 先绑定监听（无阻塞连接）。
        self._bind()
        # 2) 连接下一个 rank（对端已在监听，必要时有限重试）。
        self._connect_next()
        # 3) 接受上一个 rank 的入连接。
        try:
            self._in, _ = self._server.accept()
            self._in.settimeout(SOCKET_TIMEOUT)
        except OSError as exc:
            raise GradCompError(E_PORT, f"接收入连接失败: {exc}",
                                field="base_port")
        self._handshake()

    def _handshake(self) -> None:
        hs = p.Message(
            msg_type=p.MSG_HANDSHAKE, phase=p.PHASE_HANDSHAKE, step=0,
            src=self.rank, dst=self.next_rank, chunk=0, round_index=0,
            block=0, payload=np.int64([self.rank]))
        p.send_message(self._out, hs)
        reply = p.recv_message(self._in)
        if reply.msg_type != p.MSG_HANDSHAKE:
            raise GradCompError(E_PROTO, "握手阶段收到非握手消息",
                                details={"msg_type": reply.msg_type})
        if reply.src != self.prev_rank or reply.dst != self.rank:
            raise GradCompError(
                E_PROTO, "环握手身份不匹配",
                details={"expected_src": self.prev_rank, "got_src": reply.src,
                         "expected_dst": self.rank, "got_dst": reply.dst})

    def send(self, rank: int, msg: p.Message) -> None:
        if rank != self.next_rank:
            raise GradCompError(E_PROTO, "TCP 环只能向下一个 rank 发送",
                                details={"dst": rank,
                                         "next_rank": self.next_rank})
        if self._out is None:
            raise GradCompError(E_PROTO, "端点尚未建立连接即发送")
        try:
            p.send_message(self._out, msg)
        except OSError as exc:
            raise GradCompError(E_PROTO, f"发送失败: {exc}")

    def recv(self, rank: int | None = None) -> p.Message:
        if rank is not None and rank != self.rank:
            raise GradCompError(E_PROTO, "TCP 环只能接收本 rank 的消息",
                                details={"rank": rank, "self": self.rank})
        if self._in is None:
            raise GradCompError(E_PROTO, "端点尚未建立连接即接收")
        try:
            return p.recv_message(self._in)
        except GradCompError:
            raise
        except OSError as exc:
            raise GradCompError(E_PROTO, f"接收失败: {exc}")

    def accept_control(self):
        """接受协调进程的短生命周期控制连接。"""
        assert self._server is not None
        self._server.settimeout(SOCKET_TIMEOUT)
        conn, _addr = self._server.accept()
        conn.settimeout(SOCKET_TIMEOUT)
        return conn, _addr

    def close(self) -> None:
        for sock in (self._out, self._in, self._server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._out = self._in = self._server = None


class LoopbackTransport:
    """进程内（多线程）环传输，用队列模拟回环链路。

    可选 reorder_seed：给定时，broker 会在 rank 队列内随机插位制造乱序，
    用于验证“消息到达顺序不影响最终结果”。
    """

    def __init__(self, n: int, reorder_seed: int | None = None):
        self.n = n
        self._queues: list[queue.Queue] = [queue.Queue() for _ in range(n)]
        # 每个目标 rank 的“同阶段乱序窗口”：仅在相同 (step, phase) 内洗牌，
        # 绝不跨越阶段边界，因此既制造乱序到达又不破坏阶段协议。
        self._windows: dict[int, tuple[int, int, list]] = {}
        self._closed = False
        self._rng = (np.random.default_rng(reorder_seed)
                     if reorder_seed is not None else None)

    def _flush_window(self, dst: int) -> None:
        window = self._windows.pop(dst, None)
        if window is None:
            return
        _step, _phase, items = window
        if self._rng is not None and len(items) > 1:
            # 随机选一个相邻交换点：确定性可复现，且不会无限滞留消息。
            idx = int(self._rng.integers(0, len(items) - 1))
            items[idx], items[idx + 1] = items[idx + 1], items[idx]
        for item in items:
            self._queues[dst].put(item)

    def send(self, rank: int, msg: p.Message) -> None:
        if self._closed:
            raise GradCompError(E_PROTO, "传输已关闭后仍尝试发送")
        if not (0 <= msg.dst < self.n):
            raise GradCompError(E_PROTO, "消息目标 rank 越界",
                                details={"dst": msg.dst})
        if self._rng is None or msg.msg_type != p.MSG_DATA:
            self._flush_window(msg.dst)
            self._queues[msg.dst].put(msg)
            return
        key = (msg.step, msg.phase)
        window = self._windows.get(msg.dst)
        if window is None:
            self._windows[msg.dst] = (key[0], key[1], [msg])
        elif (window[0], window[1]) == key:
            window[2].append(msg)
        else:
            self._flush_window(msg.dst)
            self._windows[msg.dst] = (key[0], key[1], [msg])

    def recv(self, rank: int) -> p.Message:
        if self._queues[rank].empty():
            self._flush_window(rank)
        return self._queues[rank].get()

    def close(self) -> None:
        self._closed = True
