"""主进程侧真实 TCP 客户端：握手、下发、续约乐观校验与关停。

所有等待都设置在真实套接字上（``socket.settimeout``），不使用线程超时；
端口占用映射为 E_PORT，读/握手超时映射为 E_TIMEOUT，回执缺字段映射为
E_FIELD。
"""
from __future__ import annotations

import json
import socket
from dataclasses import asdict
from typing import Any

from .errors import FieldError, PortError, TimeoutError
from .protocol import (DISPATCH, HANDSHAKE_ACK, HANDSHAKE_REQUEST,
                       LEASE_CHECK, REQUIRED_RECEIPT_FIELDS, SHUTDOWN,
                       decode_frame, encode_frame)
from .spec import NodeSpec


class NodeClient:
    def __init__(self, node: NodeSpec, connect_timeout: float = 3.0,
                 read_timeout: float = 5.0) -> None:
        self.node = node
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout)
        try:
            sock.connect((self.node.host, self.node.port))
        except ConnectionRefusedError as exc:
            sock.close()
            raise PortError(
                f"no listener on {self.node.host}:{self.node.port}: {exc}",
                location=f"nodes[{self.node.node_id}]", field="port")
        except socket.timeout as exc:
            sock.close()
            raise TimeoutError(
                f"connect timeout on {self.node.host}:{self.node.port}",
                location=f"nodes[{self.node.node_id}]", field="socket")
        except OSError as exc:
            sock.close()
            raise PortError(
                f"socket error on {self.node.host}:{self.node.port}: {exc}",
                location=f"nodes[{self.node.node_id}]", field="port")
        # 真实读取超时同样挂在套接字上。
        sock.settimeout(self.read_timeout)
        self._sock = sock
        self._handshake()

    def _handshake(self) -> None:
        assert self._sock is not None
        try:
            self._sock.sendall(encode_frame(
                {"type": HANDSHAKE_REQUEST, "node_id": self.node.node_id}))
            reply = self._read_line()
        except socket.timeout as exc:
            raise TimeoutError(
                f"handshake timeout on node {self.node.node_id}",
                location=f"nodes[{self.node.node_id}]",
                field="handshake") from exc
        if reply.get("type") != HANDSHAKE_ACK:
            raise FieldError(
                f"bad handshake ack: {reply.get('type')!r}",
                location=f"nodes[{self.node.node_id}]", field="type")

    def dispatch(self, shard_id: str, param_version: int,
                 batch: dict[str, Any], token_id: str,
                 token_version: int) -> dict[str, Any]:
        assert self._sock is not None
        frame = {"type": DISPATCH, "shard_id": shard_id,
                 "param_version": param_version, "batch": batch,
                 "token_id": token_id, "token_version": token_version}
        try:
            self._sock.sendall(encode_frame(frame))
            receipt = self._read_line()
        except socket.timeout as exc:
            raise TimeoutError(
                f"dispatch read timeout on node {self.node.node_id}",
                location=f"nodes[{self.node.node_id}]",
                field="receipt") from exc
        if receipt.get("type") == "DT_ERROR" and receipt.get("code") == "E_LEASE":
            from .errors import LeaseError
            raise LeaseError(receipt.get("message", "node rejected token"),
                             location=f"nodes[{self.node.node_id}]",
                             field="token_version")
        for field_name in REQUIRED_RECEIPT_FIELDS:
            if field_name not in receipt:
                raise FieldError(
                    f"receipt missing field {field_name!r}",
                    location=f"nodes[{self.node.node_id}]", field=field_name)
        if receipt["node_id"] != self.node.node_id:
            raise FieldError(
                f"receipt node_id mismatch: {receipt['node_id']!r}",
                location=f"nodes[{self.node.node_id}]", field="node_id")
        return receipt

    def lease_check(self, token_id: str, token_version: int) -> bool:
        assert self._sock is not None
        frame = {"type": LEASE_CHECK, "token_id": token_id,
                 "token_version": token_version}
        try:
            self._sock.sendall(encode_frame(frame))
            reply = self._read_line()
        except socket.timeout as exc:
            raise TimeoutError(
                f"lease check timeout on node {self.node.node_id}",
                location=f"nodes[{self.node.node_id}]",
                field="lease_check") from exc
        return bool(reply.get("accepted"))

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.sendall(encode_frame(
                {"type": SHUTDOWN, "node_id": self.node.node_id}))
            self._read_line()
        except (OSError, FieldError):
            # 服务端可能在 ACK 后立即关闭，优雅关停容忍该竞态。
            pass
        finally:
            self._sock.close()
            self._sock = None

    def _read_line(self) -> dict[str, Any]:
        assert self._sock is not None
        buffer = b""
        while b"\n" not in buffer:
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout as exc:
                raise TimeoutError(
                    f"socket read timeout on node {self.node.node_id}",
                    location=f"nodes[{self.node.node_id}]",
                    field="socket") from exc
            if not chunk:
                raise FieldError(
                    "connection closed before complete frame",
                    location=f"nodes[{self.node.node_id}]", field="frame")
            buffer += chunk
        line, _ = buffer.split(b"\n", 1)
        return decode_frame(line)
