"""隔离/恢复指令下发调度器。

- 每个实例对应一个确定性模拟服务子进程(Windows 上为 python 子进程),
  通过 127.0.0.1 回环 TCP 的 JSON 行协议通讯。
- 全局并发上限 + 每故障域并发配额, 耗尽的指令排队, 记录等待开始
  (enqueue)与结束(start)的单调时钟毫秒。
- 握手与单次下发超时只来自真实套接字超时; 端口占用、握手失败、
  回执缺失分别以 E_PORT / E_HANDSHAKE / E_REPLY 报告。
- 失败指令重试一次, 重试回执以 attempt=2 与首次区分记录。
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DiagError
from .state import CommandIntent
from .util import monotonic_ms

MOCK_SERVER = str(Path(__file__).with_name("mock_server.py"))
MAX_ATTEMPTS = 2


@dataclass
class CommandResult:
    cmd_seq: int
    instance: str
    domain: str
    action: str
    event_seq: int
    attempt: int
    status: str            # "ok" | "unknown" | 错误码
    receipt: dict | None
    error: str | None
    enqueue_mono_ms: int
    start_mono_ms: int
    end_mono_ms: int


class Dispatcher:
    def __init__(self, instances, base_port: int = 24100, host: str = "127.0.0.1",
                 global_limit: int = 4, domain_quota: int = 2,
                 handshake_timeout: float = 2.0, command_timeout: float = 2.0,
                 modes: dict | None = None, launch: bool = True):
        self.instances = instances
        self.host = host
        self.ports = {iid: base_port + idx for idx, iid in enumerate(sorted(instances))}
        self.global_limit = global_limit
        self.domain_quota = domain_quota
        self.handshake_timeout = handshake_timeout
        self.command_timeout = command_timeout
        self.modes = modes or {}
        self.launch = launch
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._running = 0
        self.max_concurrent = 0
        self.domain_max: dict[str, int] = {}
        self._domain_running: dict[str, int] = {}

    # ---- 子进程生命周期 ----
    def _check_port(self, iid: str) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((self.host, self.ports[iid]))
        except OSError as exc:
            raise DiagError("E_PORT",
                            f"instance {iid!r}: port {self.ports[iid]} unavailable: {exc}",
                            {"instance": iid, "port": self.ports[iid]})
        finally:
            probe.close()

    def start_services(self) -> None:
        if not self.launch:
            return
        for iid in sorted(self.instances):
            self._check_port(iid)
        for iid in sorted(self.instances):
            mode = self.modes.get(iid, "normal")
            proc = subprocess.Popen(
                [sys.executable, MOCK_SERVER, "--id", iid, "--host", self.host,
                 "--port", str(self.ports[iid]), "--mode", mode],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._procs[iid] = proc
            if proc.poll() is not None:
                raise DiagError("E_PORT",
                                f"instance {iid!r}: mock service exited at start",
                                {"instance": iid, "port": self.ports[iid]})
            self._wait_ready(iid)

    def _wait_ready(self, iid: str, attempts: int = 100) -> None:
        """等待子进程完成监听(TCP 连接成功即就绪, 不依赖应用层应答)。"""
        import time
        for _ in range(attempts):
            try:
                probe = socket.create_connection(
                    (self.host, self.ports[iid]), timeout=0.2)
                probe.close()
                return
            except ConnectionRefusedError:
                time.sleep(0.02)
            except OSError:
                time.sleep(0.02)
        raise DiagError("E_PORT",
                        f"instance {iid!r}: service not listening on "
                        f"{self.ports[iid]}",
                        {"instance": iid, "port": self.ports[iid]})

    def stop_services(self) -> None:
        for proc in self._procs.values():
            proc.terminate()
        for proc in self._procs.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._procs.clear()

    # ---- 单次下发 ----
    def _run_once(self, intent: CommandIntent, cmd_seq: int) -> dict:
        iid = intent.instance
        port = self.ports[iid]
        try:
            sock = socket.create_connection((self.host, port),
                                            timeout=self.handshake_timeout)
        except OSError as exc:
            raise DiagError("E_PORT",
                            f"instance {iid!r}: cannot connect to {self.host}:{port}: {exc}",
                            {"instance": iid, "port": port})
        fh = None
        try:
            sock.settimeout(self.handshake_timeout)
            fh = sock.makefile("r", encoding="utf-8")
            try:
                hello_line = fh.readline()
            except (socket.timeout, TimeoutError):
                raise DiagError("E_HANDSHAKE",
                                f"instance {iid!r}: handshake timeout",
                                {"instance": iid, "port": port})
            try:
                hello = json.loads(hello_line) if hello_line else None
            except json.JSONDecodeError:
                hello = None
            if not isinstance(hello, dict) or hello.get("hello") != iid:
                raise DiagError("E_HANDSHAKE",
                                f"instance {iid!r}: invalid handshake {hello_line!r}",
                                {"instance": iid, "port": port})
            sock.settimeout(self.command_timeout)
            sock.sendall((json.dumps({"cmd_seq": cmd_seq, "action": intent.action,
                                      "instance": iid}) + "\n").encode())
            try:
                reply_line = fh.readline()
            except (socket.timeout, TimeoutError):
                raise DiagError("E_REPLY",
                                f"instance {iid!r}: reply timeout for cmd {cmd_seq}",
                                {"instance": iid, "cmd_seq": cmd_seq})
            try:
                reply = json.loads(reply_line) if reply_line else None
            except json.JSONDecodeError:
                reply = None
            if not isinstance(reply, dict):
                raise DiagError("E_REPLY",
                                f"instance {iid!r}: malformed reply {reply_line!r}",
                                {"instance": iid, "cmd_seq": cmd_seq})
            for field_name in ("instance", "cmd_seq", "isolated", "elapsed_ms"):
                if field_name not in reply:
                    raise DiagError("E_REPLY",
                                    f"instance {iid!r}: reply missing field {field_name!r}",
                                    {"instance": iid, "cmd_seq": cmd_seq,
                                     "missing": field_name})
            if reply["instance"] != iid or reply["cmd_seq"] != cmd_seq:
                raise DiagError("E_REPLY",
                                f"instance {iid!r}: reply out of order "
                                f"(want cmd {cmd_seq}, got {reply['cmd_seq']!r})",
                                {"instance": iid, "cmd_seq": cmd_seq,
                                 "got_cmd_seq": reply["cmd_seq"]})
            if not isinstance(reply["isolated"], bool) \
                    or not isinstance(reply["elapsed_ms"], int):
                raise DiagError("E_REPLY",
                                f"instance {iid!r}: reply field type mismatch",
                                {"instance": iid, "cmd_seq": cmd_seq})
            return reply
        finally:
            if fh is not None:
                fh.close()
            sock.close()

    def _run_guarded(self, intent: CommandIntent, cmd_seq: int,
                     global_sem: threading.Semaphore,
                     domain_sems: dict[str, threading.Semaphore]) -> list[CommandResult]:
        enqueue_ms = monotonic_ms()
        with global_sem, domain_sems[intent.domain]:
            start_ms = monotonic_ms()
            with self._lock:
                self._running += 1
                self.max_concurrent = max(self.max_concurrent, self._running)
                running = self._domain_running.get(intent.domain, 0) + 1
                self._domain_running[intent.domain] = running
                self.domain_max[intent.domain] = max(
                    self.domain_max.get(intent.domain, 0), running)
            results: list[CommandResult] = []
            try:
                last_error: DiagError | None = None
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        receipt = self._run_once(intent, cmd_seq)
                        results.append(CommandResult(
                            cmd_seq, intent.instance, intent.domain, intent.action,
                            intent.event_seq, attempt, "ok", receipt, None,
                            enqueue_ms, start_ms, monotonic_ms()))
                        return results
                    except DiagError as exc:
                        last_error = exc
                        results.append(CommandResult(
                            cmd_seq, intent.instance, intent.domain, intent.action,
                            intent.event_seq, attempt, exc.code, None, exc.message,
                            enqueue_ms, start_ms, monotonic_ms()))
                        if exc.code not in ("E_HANDSHAKE", "E_REPLY"):
                            return results
                return results
            finally:
                with self._lock:
                    self._running -= 1
                    self._domain_running[intent.domain] -= 1

    def submit(self, intents: list[CommandIntent], start_cmd_seq: int) -> list[CommandResult]:
        """并发下发一批指令, 返回所有尝试(含重试)的结果记录。"""
        if not intents:
            return []
        global_sem = threading.Semaphore(self.global_limit)
        domains = {intent.domain for intent in intents}
        domain_sems = {d: threading.Semaphore(self.domain_quota) for d in domains}
        results: list[CommandResult] = []
        with ThreadPoolExecutor(max_workers=self.global_limit) as pool:
            futures = [pool.submit(self._run_guarded, intent, start_cmd_seq + idx,
                                   global_sem, domain_sems)
                       for idx, intent in enumerate(intents)]
            for future in futures:
                results.extend(future.result())
        return results
