"""节点子进程编排：启动、端口预检、握手、JSON 行收发。"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import (
    HandshakeTimeoutError,
    NodeStartError,
    PortInUseError,
)
from .structured_log import StructuredLogger

NODE_SCRIPT = Path(__file__).with_name("node_simulator.py")


def is_port_free(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
    except OSError:
        return False
    finally:
        probe.close()
    return True


class NodeHandle:
    """单个节点的进程与连接句柄。"""

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        process: subprocess.Popen,
        conn: socket.socket,
        scale: int,
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.process = process
        self.conn = conn
        self.scale = scale
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def _recv_line(self) -> str:
        while b"\n" not in self._buffer:
            chunk = self.conn.recv(65536)
            if not chunk:
                raise ConnectionError(f"节点 {self.node_id} 连接关闭")
            self._buffer.extend(chunk)
        line, _, rest = self._buffer.partition(b"\n")
        self._buffer[:] = rest
        return line.decode("utf-8")

    def send(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.conn.sendall(
                (json.dumps(payload, ensure_ascii=True) + "\n").encode("ascii")
            )

    def recv(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(self._recv_line())

    def close(self) -> None:
        try:
            self.send({"type": "shutdown"})
            try:
                self.recv()
            except (ConnectionError, OSError, json.JSONDecodeError):
                pass
        except OSError:
            pass
        try:
            self.conn.close()
        except OSError:
            pass
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=5)
            for pipe in (self.process.stdout, self.process.stderr, self.process.stdin):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass


class NodeOrchestrator:
    def __init__(
        self,
        node_specs: Dict[str, "object"],
        scale: int,
        logger: StructuredLogger,
        handshake_timeout_ms: int = 5000,
        startup_timeout_ms: int = 8000,
        workdir: Optional[Path] = None,
        skip_prebind_check: bool = False,
        fault_env: Optional[Dict[str, str]] = None,
    ):
        self.node_specs = node_specs
        self.scale = scale
        self.logger = logger
        self.handshake_timeout_ms = handshake_timeout_ms
        self.startup_timeout_ms = startup_timeout_ms
        self.workdir = workdir
        # 测试专用：外部已有哑服务监听时跳过端口预检与子进程拉起
        self.skip_prebind_check = skip_prebind_check
        self.fault_env = fault_env or {}
        self.handles: Dict[str, NodeHandle] = {}

    def start_all(self) -> Dict[str, NodeHandle]:
        for node_id in sorted(self.node_specs):
            self.handles[node_id] = self._start_one(self.node_specs[node_id])
        return self.handles

    def _start_one(self, spec: "object") -> NodeHandle:
        node_id, host, port = spec.node_id, spec.host, spec.port
        t0 = time.monotonic()
        if self.skip_prebind_check:
            return self._connect_external(spec, t0)
        # 1) 端口预检 -> E_PORT
        if not is_port_free(host, port):
            self.logger.log(
                "node.port_in_use",
                code="E_PORT",
                node=node_id,
                host=host,
                port=port,
            )
            raise PortInUseError(
                f"回环端口 {host}:{port} 已被占用",
                {"node": node_id, "host": host, "port": port},
            )
        cmd = [
            sys.executable,
            "-u",
            str(NODE_SCRIPT),
            "--node",
            node_id,
            "--host",
            host,
            "--port",
            str(port),
            "--scale",
            str(self.scale),
        ]
        env = None
        if self.fault_env:
            env = dict(__import__("os").environ)
            env.update(self.fault_env)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.workdir) if self.workdir else None,
                text=True,
                env=env,
            )
        except OSError as exc:
            self.logger.log(
                "node.spawn_failed",
                code="E_NODE",
                node=node_id,
                error=str(exc),
            )
            raise NodeStartError(
                f"节点 {node_id} 启动失败: {exc}",
                {"node": node_id, "error": str(exc)},
            ) from exc

        # 2) 等待 READY 或进程退出 -> E_NODE
        deadline = t0 + self.startup_timeout_ms / 1000.0
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read() if proc.stderr else ""
                code = "E_PORT" if "BIND_ERROR" in err else "E_NODE"
                self.logger.log(
                    "node.exited_early",
                    code=code,
                    node=node_id,
                    returncode=proc.returncode,
                    stderr=err.strip(),
                )
                if code == "E_PORT":
                    raise PortInUseError(
                        f"节点 {node_id} 绑定端口失败",
                        {"node": node_id, "stderr": err.strip()},
                    )
                raise NodeStartError(
                    f"节点 {node_id} 提前退出",
                    {"node": node_id, "returncode": proc.returncode, "stderr": err},
                )
            line = proc.stdout.readline() if proc.stdout else ""
            if line.startswith("READY"):
                ready = True
                break
            time.sleep(0.01)
        if not ready:
            proc.kill()
            self.logger.log(
                "node.ready_timeout", code="E_NODE", node=node_id,
                timeout_ms=self.startup_timeout_ms,
            )
            raise NodeStartError(
                f"节点 {node_id} 就绪超时",
                {"node": node_id, "timeout_ms": self.startup_timeout_ms},
            )

        # 3) 连接并握手 -> E_HANDSHAKE
        conn = socket.create_connection((host, port), timeout=self.handshake_timeout_ms / 1000.0)
        conn.settimeout(self.handshake_timeout_ms / 1000.0)
        try:
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    raise ConnectionError("握手期间连接关闭")
                data += chunk
            hello = json.loads(data.decode("utf-8").splitlines()[0])
            if hello.get("type") != "hello" or hello.get("node") != node_id:
                raise HandshakeTimeoutError("握手报文非法", {"hello": hello})
        except (socket.timeout, OSError, ConnectionError, json.JSONDecodeError) as exc:
            conn.close()
            proc.kill()
            self.logger.log(
                "node.handshake_failed",
                code="E_HANDSHAKE",
                node=node_id,
                error=str(exc),
                timeout_ms=self.handshake_timeout_ms,
            )
            raise HandshakeTimeoutError(
                f"节点 {node_id} 握手超时/失败",
                {"node": node_id, "error": str(exc)},
            ) from exc
        conn.settimeout(None)
        self.logger.log(
            "node.started",
            node=node_id,
            host=host,
            port=port,
            pid=proc.pid,
            elapsed_ms=int(round((time.monotonic() - t0) * 1000)),
        )
        return NodeHandle(node_id, host, port, proc, conn, self.scale)

    def _connect_external(self, spec, t0):
        """对接已在监听的外部（哑）服务，仅走握手路径，测试用。"""
        node_id, host, port = spec.node_id, spec.host, spec.port
        conn = socket.create_connection(
            (host, port), timeout=self.handshake_timeout_ms / 1000.0
        )
        conn.settimeout(self.handshake_timeout_ms / 1000.0)
        try:
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    raise ConnectionError("握手期间连接关闭")
                data += chunk
            hello = json.loads(data.decode("utf-8").splitlines()[0])
            if hello.get("type") != "hello" or hello.get("node") != node_id:
                raise HandshakeTimeoutError("握手报文非法", {"hello": hello})
        except (socket.timeout, OSError, ConnectionError, json.JSONDecodeError) as exc:
            conn.close()
            self.logger.log(
                "node.handshake_failed", code="E_HANDSHAKE", node=node_id,
                error=str(exc), timeout_ms=self.handshake_timeout_ms,
            )
            raise HandshakeTimeoutError(
                f"节点 {node_id} 握手超时/失败",
                {"node": node_id, "error": str(exc)},
            ) from exc
        conn.settimeout(None)
        return NodeHandle(node_id, host, port, process=None, conn=conn, scale=self.scale)

    def shutdown_all(self) -> None:
        for node_id in sorted(self.handles, reverse=True):
            try:
                self.handles[node_id].close()
                self.logger.log("node.stopped", node=node_id)
            except Exception as exc:  # pragma: no cover - 清理路径
                self.logger.log(
                    "node.stop_error", node=node_id, error=str(exc)
                )
        self.handles.clear()
