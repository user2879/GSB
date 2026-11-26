"""节点子进程：在声明端口上真实监听 127.0.0.1 的 TCP JSON 行服务。"""
from __future__ import annotations

import socket
import sys
from typing import Any

from .clock import MonotonicClock
from .errors import FieldError, PortError
from .fixedpoint import validate_scale
from .protocol import (DISPATCH, HANDSHAKE_ACK,
                       HANDSHAKE_REQUEST, LEASE_CHECK, REQUIRED_RECEIPT_FIELDS,
                       SHUTDOWN, decode_frame, encode_frame)
from .sample import aggregate_batch
from .spec import NodeSpec


class NodeWorker:
    """单节点服务端逻辑；可在子进程入口中运行。"""

    def __init__(self, node: NodeSpec, scale: int) -> None:
        validate_scale(scale)
        self.node = node
        self.scale = scale
        self.clock = MonotonicClock()
        # 节点侧记录当前已知令牌版本，供乐观续约校验。
        self.known_token_versions: dict[str, int] = {}
        self._server: socket.socket | None = None

    def bind(self) -> socket.socket:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.node.host, self.node.port))
        except OSError as exc:
            server.close()
            raise PortError(
                f"bind failed on {self.node.host}:{self.node.port}: {exc}",
                location=f"nodes[{self.node.node_id}]", field="port")
        server.listen(8)
        self._server = server
        return server

    def serve_forever(self) -> None:
        server = self.bind()
        try:
            while True:
                conn, _ = server.accept()
                with conn:
                    self._handle(conn)
        finally:
            server.close()

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(5.0)
        buffer = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    reply, should_stop = self._route(decode_frame(line))
                except FieldError as exc:
                    reply = {"type": "DT_ERROR", "code": exc.code,
                             "node_id": self.node.node_id,
                             "message": exc.message, "field": exc.field}
                    should_stop = False
                conn.sendall(encode_frame(reply))
                if should_stop:
                    # 先确保 ACK 已刷出再半关闭，避免客户端读到 RST。
                    conn.shutdown(socket.SHUT_WR)
                    return

    def _route(self, frame: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        message_type = frame.get("type")
        if message_type == HANDSHAKE_REQUEST:
            return {"type": HANDSHAKE_ACK, "node_id": self.node.node_id,
                    "port": self.node.port}, False
        if message_type == DISPATCH:
            return self._dispatch(frame), False
        if message_type == LEASE_CHECK:
            return self._lease_check(frame), False
        if message_type == SHUTDOWN:
            return {"type": "DT_SHUTDOWN_ACK",
                    "node_id": self.node.node_id}, True
        raise FieldError(f"unknown frame type {message_type!r}",
                         location=f"nodes[{self.node.node_id}]", field="type")

    def _dispatch(self, frame: dict[str, Any]) -> dict[str, Any]:
        for required in ("shard_id", "param_version", "batch", "token_id",
                         "token_version"):
            if required not in frame:
                raise FieldError(
                    f"dispatch frame missing field {required!r}",
                    location=f"nodes[{self.node.node_id}]", field=required)
        token_id = frame["token_id"]
        version = int(frame["token_version"])
        previous = self.known_token_versions.get(token_id)
        if previous is not None and version <= previous:
            return {"type": "DT_ERROR", "code": "E_LEASE",
                    "node_id": self.node.node_id,
                    "message": "stale token version on node side"}
        self.known_token_versions[token_id] = version
        start = self.clock.now_ms()
        summary = aggregate_batch(frame["batch"], self.node.node_id,
                                  self.scale).to_dict()
        elapsed = self.clock.now_ms() - start
        summary["elapsed_ms"] = int(elapsed)
        receipt = {
            "type": "DT_RECEIPT",
            "node_id": self.node.node_id,
            "shard_id": frame["shard_id"],
            "param_version": frame["param_version"],
            "summary": summary,
            "elapsed_ms": int(elapsed),
            "token_id": token_id,
            "token_version": version,
        }
        for field_name in REQUIRED_RECEIPT_FIELDS:
            if field_name not in receipt:
                raise FieldError(
                    f"receipt missing field {field_name!r}",
                    location=f"nodes[{self.node.node_id}]", field=field_name)
        return receipt

    def _lease_check(self, frame: dict[str, Any]) -> dict[str, Any]:
        token_id = frame.get("token_id")
        expected = frame.get("token_version")
        known = self.known_token_versions.get(token_id)
        ok = known is not None and known == expected
        return {"type": "DT_LEASE_CHECK_ACK", "node_id": self.node.node_id,
                "token_id": token_id, "token_version": expected,
                "accepted": bool(ok)}


def worker_main(node: NodeSpec, scale: int) -> int:
    """子进程入口；异常以非零退出码与单行 JSON 错误上报。"""
    try:
        NodeWorker(node, scale).serve_forever()
    except PortError as exc:
        sys.stderr.write(exc.to_dict().__repr__() + "\n")
        return 2
    except Exception as exc:  # pragma: no cover - 防御性兜底
        sys.stderr.write(f"worker fatal: {exc!r}\n")
        return 1
    return 0
