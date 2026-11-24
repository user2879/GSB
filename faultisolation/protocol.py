"""TCP 回环 JSON 行协议。

每条消息为一行 UTF-8 编码的紧凑 JSON，以 ``\\n`` 结尾，字段顺序固定，
便于主流程与校验器逐字节比较。请求与回执的 ``type`` 取值：

请求: ``acquire`` / ``renew`` / ``release`` / ``snapshot`` / ``shutdown``
回执: ``ack``（含 ``status``）或 ``error``。
"""

from __future__ import annotations

import json
import socket
from typing import Any

from .errors import ProtocolError

ACTIONS = ("acquire", "renew", "release")
MESSAGE_TYPES = ("acquire", "renew", "release", "snapshot", "shutdown")
STATUSES = (
    "granted",
    "renewed",
    "released",
    "expired",
    "denied_cap",
    "denied_quota",
    "denied_lease",
)


def encode_line(payload: dict[str, Any]) -> bytes:
    """字段排序固定、无空白的 JSON 行，保证双端逐字节一致。"""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def decode_line(line: bytes) -> dict[str, Any]:
    if not line:
        raise ProtocolError("收到空协议行", raw=line)
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("JSON 行解析失败", raw=line[:64]) from exc
    if not isinstance(payload, dict):
        raise ProtocolError("协议消息必须为 JSON 对象")
    return payload


def send_message(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(encode_line(payload))


def recv_message(sock: socket.socket) -> dict[str, Any]:
    """从已连接套接字读取一行 JSON（消息长度有界，防止异常对端撑爆内存）。"""
    chunks: list[bytes] = []
    total = 0
    while True:
        byte = sock.recv(1)
        if not byte:
            if not chunks:
                raise ProtocolError("对端在响应前关闭连接")
            break
        if byte == b"\n":
            break
        chunks.append(byte)
        total += 1
        if total > 8 * 1024 * 1024:
            raise ProtocolError("单行协议消息超过 8MiB 上限")
    return decode_line(b"".join(chunks))


def request(host: str, port: int, payload: dict[str, Any], timeout_ms: int = 5000) -> dict[str, Any]:
    """同步发起一次回环 JSON 行请求并返回回执，超时来自真实往返。"""
    with socket.create_connection((host, port), timeout=timeout_ms / 1000.0) as sock:
        sock.settimeout(timeout_ms / 1000.0)
        send_message(sock, payload)
        return recv_message(sock)
