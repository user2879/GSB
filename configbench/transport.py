"""平台侧节点子进程编排与 JSON 行收发。"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time

from .errors import FieldError, HandshakeError, PortError, TimeoutError

REQUIRED_ACK_FIELDS = ("type", "request_id", "config_version", "summary",
                       "summary_digest", "elapsed_ms")


def mono_ms():
    return int(time.monotonic() * 1000)


class NodeProcess:
    """按注册表启动的一个节点子进程。"""

    def __init__(self, node, startup_timeout_ms=4000):
        self.node = node
        self.process = None
        self.spec_file = None
        self.startup_timeout_ms = startup_timeout_ms

    def start(self):
        fd, self.spec_file = tempfile.mkstemp(prefix=f"node-{self.node.node_id}-",
                                              suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({
                "node_id": self.node.node_id, "host": self.node.host,
                "port": self.node.port, "group_key": self.node.group_key,
                "scale": self.node.scale, "fail_inject": self.node.fail_inject,
            }, f)
        cmd = [sys.executable, "-m", "configbench.node_main", self.spec_file]
        popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
                        "text": True}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.process = subprocess.Popen(cmd, **popen_kwargs)
        try:
            self._await_ready()
        except Exception:
            # 启动/握手失败也要回收子进程，避免泄漏。
            self.stop()
            raise
        return self

    def _await_ready(self):
        deadline = mono_ms() + self.startup_timeout_ms
        if self.node.fail_inject == "port":
            # 节点会稳定绑定失败并退出；读取其退出与 bind_error 输出。
            while mono_ms() < deadline:
                if self.process.poll() is not None:
                    line = self.process.stdout.readline() or ""
                    msg = json.loads(line) if line.strip() else {}
                    raise PortError(
                        f"节点 {self.node.node_id} 端口 {self.node.port} 占用",
                        field=f"registry.nodes[{self.node.node_id}].port",
                        basis=msg.get("error",
                                      f"进程已退出 code={self.process.returncode}"))
                time.sleep(0.02)
            raise PortError("节点未就绪且未报告绑定结果",
                            field=f"node[{self.node.node_id}]",
                            basis="启动超时")
        while mono_ms() < deadline:
            if self.process.poll() is not None:
                err = self.process.stderr.read() if self.process.stderr else ""
                raise PortError(
                    f"节点进程提前退出 code={self.process.returncode}",
                    field=f"node[{self.node.node_id}]",
                    basis=err.strip()[:300])
            line = self.process.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            msg = json.loads(line)
            if msg.get("type") == "ready":
                return
            if msg.get("type") == "bind_error":
                raise PortError(
                    f"节点端口 {self.node.port} 占用",
                    field=f"registry.nodes[{self.node.node_id}].port",
                    basis=msg.get("error", "bind 失败"))
        raise PortError("节点启动超时", field=f"node[{self.node.node_id}]",
                        basis=f">{self.startup_timeout_ms}ms 未输出 ready")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process is not None:
            for stream in (self.process.stdout, self.process.stderr):
                if stream is not None:
                    stream.close()
        try:
            if self.spec_file:
                os.remove(self.spec_file)
        except OSError:
            pass

    def kill_hard(self):
        """强制终止（模拟崩溃），不做任何清理握手。"""
        if self.process and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        if self.process is not None:
            for stream in (self.process.stdout, self.process.stderr):
                if stream is not None:
                    stream.close()


def _send_line(sock, obj):
    sock.sendall(json.dumps(obj, sort_keys=True).encode("utf-8") + b"\n")


def _recv_line(sock):
    buf = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("连接被对端关闭")
        buf.extend(chunk)
        if b"\n" in buf:
            line, _, _ = buf.partition(b"\n")
            return json.loads(line.decode("utf-8"))


def dispatch_once(node, request, *, connect_timeout_ms=1500,
                  handshake_timeout_ms=1500, ack_timeout_ms=2000):
    """单次下发（不重试）。失败抛带 E_* 码的异常，由调度器决定重试。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(connect_timeout_ms / 1000.0)
    try:
        try:
            sock.connect((node.host, node.port))
        except ConnectionRefusedError:
            raise PortError(f"连接被拒 {node.host}:{node.port}",
                            field=f"node[{node.node_id}].port",
                            basis="connect 返回 ECONNREFUSED")
        except socket.timeout:
            raise TimeoutError(
                "连接超时", field=f"node[{node.node_id}]",
                basis=f"connect > {connect_timeout_ms}ms")
        # 真实握手超时
        sock.settimeout(handshake_timeout_ms / 1000.0)
        _send_line(sock, {"type": "hello", "node_id": node.node_id})
        try:
            hello_ack = _recv_line(sock)
        except socket.timeout:
            raise HandshakeError(
                f"握手超时 node={node.node_id}",
                field=f"node[{node.node_id}].hello",
                basis=f">{handshake_timeout_ms}ms 未收到 hello_ack")
        if hello_ack.get("type") != "hello_ack" or \
                hello_ack.get("node_id") != node.node_id:
            raise HandshakeError("握手响应非法",
                                 field=f"node[{node.node_id}].hello_ack",
                                 basis=str(hello_ack)[:200])
        # 回执超时
        sock.settimeout(ack_timeout_ms / 1000.0)
        _send_line(sock, request)
        try:
            ack = _recv_line(sock)
        except socket.timeout:
            raise TimeoutError(
                f"回执超时 node={node.node_id}",
                field=f"node[{node.node_id}].slice_ack",
                basis=f">{ack_timeout_ms}ms 未收到回执")
        if ack.get("type") == "error":
            raise FieldError("节点返回错误帧",
                             field=f"node[{node.node_id}].ack",
                             basis=str(ack)[:200])
        missing = [f for f in REQUIRED_ACK_FIELDS if f not in ack]
        if missing:
            raise FieldError(
                f"回执字段缺失: {missing}",
                field=f"node[{node.node_id}].slice_ack",
                basis=f"必填字段 {REQUIRED_ACK_FIELDS}")
        if ack["request_id"] != request["request_id"]:
            raise FieldError("回执 request_id 不匹配",
                             field=f"node[{node.node_id}].request_id",
                             basis=f"发 {request['request_id']} 收 {ack['request_id']}")
        return ack
    finally:
        sock.close()
