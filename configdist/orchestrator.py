"""节点子进程编排与回执协议校验。"""

import json
import socket
import subprocess
import sys
import time

from .canon import canon
from .errors import PortError, ProtoError, TimeoutError_
from .node import EXIT_PORT_BUSY

HANDSHAKE_TIMEOUT_MS = 5000


def _port_free(port):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def validate_receipt(receipt, node_id, view):
    """按 s1/s2 视图强校验回执，非法抛 ProtoError(E_PROTO)。"""
    if not isinstance(receipt, dict):
        raise ProtoError(node_id, "回执不是 JSON 对象")
    if receipt.get("op") != "receipt":
        raise ProtoError(node_id, f"op 非法: {receipt.get('op')!r}")
    if receipt.get("node") != node_id:
        raise ProtoError(node_id, "回执节点标识不匹配")
    if receipt.get("view") != view:
        raise ProtoError(node_id, f"视图应为 {view}")
    if not isinstance(receipt.get("manifest_version"), int):
        raise ProtoError(node_id, "manifest_version 缺失或类型错误")
    digest = receipt.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ProtoError(node_id, "digest 缺失或不是 64 位十六进制")
    if not isinstance(receipt.get("elapsed_ms"), int) \
            or receipt["elapsed_ms"] < 0:
        raise ProtoError(node_id, "elapsed_ms 缺失或为负")
    if not isinstance(receipt.get("key"), str):
        raise ProtoError(node_id, "key 缺失")
    if view == "s2":
        if not isinstance(receipt.get("summary"), str):
            raise ProtoError(node_id, "s2 回执缺少 summary")
        if not isinstance(receipt.get("key_version"), int):
            raise ProtoError(node_id, "s2 回执缺少 key_version")
        if not isinstance(receipt.get("profile"), str):
            raise ProtoError(node_id, "s2 回执缺少 profile")
    return receipt


class NodeManager:
    def __init__(self, specs, data_path, state_dir):
        self.specs = specs
        self.data_path = data_path
        self.state_dir = state_dir
        self.procs = {}

    def start(self, handshake_timeout_ms=HANDSHAKE_TIMEOUT_MS):
        for node_id, spec in self.specs.items():
            self._start_one(spec, handshake_timeout_ms)

    def _start_one(self, spec, handshake_timeout_ms):
        if not _port_free(spec.port):
            raise PortError(spec.id, spec.port)
        proc = subprocess.Popen(
            [sys.executable, "-m", "configdist.node",
             "--id", spec.id, "--port", str(spec.port),
             "--view", spec.view, "--profile", spec.profile,
             "--data", self.data_path, "--state-dir", self.state_dir,
             "--fudge", str(spec.fudge)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs[spec.id] = proc
        deadline = time.monotonic() + handshake_timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                if proc.returncode == EXIT_PORT_BUSY or "E_PORT" in out:
                    raise PortError(spec.id, spec.port)
                raise ProtoError(spec.id, f"节点进程异常退出: {out.strip()}")
            try:
                ack = self._rpc(spec, {"op": "hello"},
                                timeout_ms=max(200, handshake_timeout_ms // 10))
                if ack.get("op") == "hello_ack" and ack.get("node") == spec.id:
                    return ack
                raise ProtoError(spec.id, f"握手应答非法: {ack!r}")
            except (ConnectionError, OSError):
                time.sleep(0.05)
        raise TimeoutError_(spec.id, handshake_timeout_ms, "handshake")

    def restart(self, node_id, handshake_timeout_ms=HANDSHAKE_TIMEOUT_MS):
        """强制终止后按持久化状态重启节点，返回握手应答。"""
        self.kill(node_id)
        return self._start_one(self.specs[node_id], handshake_timeout_ms)

    def _rpc(self, spec, message, timeout_ms):
        with socket.create_connection(("127.0.0.1", spec.port),
                                      timeout=timeout_ms / 1000.0) as conn:
            conn.settimeout(timeout_ms / 1000.0)
            f = conn.makefile("rwb")
            f.write((canon(message) + "\n").encode("utf-8"))
            f.flush()
            line = f.readline()
        if not line:
            raise ProtoError(spec.id, "空应答")
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            raise ProtoError(spec.id, "应答不是合法 JSON 行")

    def dispatch(self, task):
        """执行一次切片下发，带真实套接字超时。"""
        spec = self.specs[task.node]
        try:
            receipt = self._rpc(spec, task.payload, task.timeout_ms)
        except socket.timeout:
            raise TimeoutError_(task.node, task.timeout_ms, "dispatch")
        except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError):
            raise TimeoutError_(task.node, task.timeout_ms, "dispatch")
        return validate_receipt(receipt, task.node, spec.view)

    def kill(self, node_id):
        proc = self.procs.pop(node_id, None)
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    def stop_all(self):
        for node_id, spec in self.specs.items():
            proc = self.procs.get(node_id)
            if proc and proc.poll() is None:
                try:
                    self._rpc(spec, {"op": "shutdown"}, timeout_ms=1000)
                except Exception:
                    pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.procs.clear()
