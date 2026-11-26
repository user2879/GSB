from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any

from .errors import (
    LeaseConflictError,
    NodeUnavailableError,
    PortInUseError,
    ProtocolError,
    SocketTimeoutError,
)


def assert_loopback_port_free(port: int, host: str = "127.0.0.1") -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    except OSError as exc:
        raise PortInUseError(
            f"回环端口 {host}:{port} 已被占用或不可绑定",
            entry=f"port:{port}",
            field="port",
            details={"host": host, "port": port, "errno": getattr(exc, "errno", None)},
        ) from exc
    finally:
        probe.close()


def wait_for_tcp_port(host: str, port: int, timeout_ms: int) -> None:
    deadline = time.monotonic() + max(0.001, timeout_ms / 1000.0)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.02)
    raise NodeUnavailableError(
        f"节点 {host}:{port} 在 {timeout_ms}ms 内未就绪",
        entry=f"{host}:{port}",
        field="port",
        details={"last_error": repr(last_error)},
    )


class JsonLineConnection:
    def __init__(self, sock: socket.socket, timeout_seconds: float) -> None:
        self.sock = sock
        self.timeout_seconds = timeout_seconds
        self.reader = sock.makefile("rb")

    @classmethod
    def connect(cls, host: str, port: int, timeout_ms: int) -> "JsonLineConnection":
        seconds = max(0.001, timeout_ms / 1000.0)
        try:
            sock = socket.create_connection((host, port), timeout=seconds)
        except TimeoutError as exc:
            raise SocketTimeoutError(
                f"连接 {host}:{port} 超时",
                entry=f"{host}:{port}",
                field="connect",
            ) from exc
        except OSError as exc:
            raise NodeUnavailableError(
                f"无法连接 {host}:{port}: {exc}",
                entry=f"{host}:{port}",
                field="connect",
            ) from exc
        sock.settimeout(seconds)
        return cls(sock, seconds)

    def send(self, message: dict[str, Any]) -> None:
        try:
            self.sock.sendall(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
        except OSError as exc:
            raise NodeUnavailableError(f"TCP 发送失败: {exc}", field="send") from exc

    def read(self) -> dict[str, Any]:
        self.sock.settimeout(self.timeout_seconds)
        try:
            line = self.reader.readline()
        except TimeoutError as exc:
            raise SocketTimeoutError(
                "套接字 readline 超时",
                field="readline",
                details={"timeout_seconds": self.timeout_seconds},
            ) from exc
        except OSError as exc:
            raise NodeUnavailableError(f"TCP 读取失败: {exc}", field="read") from exc
        if not line:
            raise ProtocolError("节点关闭了连接且未返回消息", field="readline")
        try:
            message = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"节点返回非法 JSON 行: {exc}", field="readline") from exc
        if not isinstance(message, dict):
            raise ProtocolError("节点返回 JSON 不是对象", field="readline")
        if message.get("type") == "error":
            code = message.get("code")
            error_cls = {
                "E_TIMEOUT": SocketTimeoutError,
                "E_LEASE": LeaseConflictError,
                "E_PORT": PortInUseError,
            }.get(code, ProtocolError)
            raise error_cls(str(message.get("message", "节点返回错误")), details=message)
        return message

    def close(self) -> None:
        try:
            self.reader.close()
        finally:
            self.sock.close()

    def __enter__(self) -> "JsonLineConnection":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def chunk_rows(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        return [rows]
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def dispatch_job(
    *,
    host: str,
    port: int,
    node_id: str,
    shard: dict[str, Any],
    quota_version: int,
    rows: list[dict[str, Any]],
    lease_version: int,
    batch_size: int = 4,
    timeout_ms: int = 2000,
    delay_ms: int = 0,
) -> dict[str, Any]:
    batches = chunk_rows(rows, batch_size)
    with JsonLineConnection.connect(host, port, timeout_ms) as conn:
        conn.send({"type": "hello", "protocol_version": 1, "node_id": node_id})
        ack = conn.read()
        if ack.get("type") != "hello_ack" or ack.get("node_id") != node_id:
            raise ProtocolError("握手回执不匹配", entry=node_id, field="hello_ack", details=ack)

        conn.send(
            {
                "type": "start",
                "job_id": f"{shard['shard_id']}:v{lease_version}",
                "shard_id": shard["shard_id"],
                "quota_version": int(quota_version),
                "lease_version": int(lease_version),
                "row_count": len(rows),
                "batch_count": len(batches),
                "delay_ms": int(delay_ms),
            }
        )
        start_ack = conn.read()
        if start_ack.get("type") != "start_ack":
            raise ProtocolError("缺少 start_ack", entry=shard["shard_id"], field="start_ack", details=start_ack)

        received = 0
        for index, batch in enumerate(batches):
            conn.send(
                {
                    "type": "batch",
                    "job_id": f"{shard['shard_id']}:v{lease_version}",
                    "batch_index": index,
                    "rows": batch,
                }
            )
            batch_ack = conn.read()
            if batch_ack.get("type") != "batch_ack" or batch_ack.get("batch_index") != index:
                raise ProtocolError("批次回执不匹配", entry=shard["shard_id"], field=f"batch_ack[{index}]")
            received += int(batch_ack.get("accepted_rows", len(batch)))

        conn.send({"type": "finish", "job_id": f"{shard['shard_id']}:v{lease_version}", "received_rows": received})
        receipt = conn.read()
        if receipt.get("type") != "receipt":
            raise ProtocolError("缺少 receipt", entry=shard["shard_id"], field="receipt", details=receipt)
        required = {"node_id", "shard_id", "quota_version", "loss_summary", "monotonic_ms"}
        missing = sorted(required - set(receipt))
        if missing:
            raise ProtocolError("回执缺少必填字段", entry=shard["shard_id"], field="receipt", details={"missing": missing})
        return receipt


def renew_lease_over_tcp(
    *,
    host: str,
    port: int,
    node_id: str,
    shard_id: str,
    expected_version: int,
    ttl_ms: int,
    timeout_ms: int = 1000,
) -> dict[str, Any]:
    with JsonLineConnection.connect(host, port, timeout_ms) as conn:
        conn.send({"type": "hello", "protocol_version": 1, "node_id": node_id})
        conn.read()
        conn.send(
            {
                "type": "renew",
                "shard_id": shard_id,
                "expected_version": int(expected_version),
                "ttl_ms": int(ttl_ms),
            }
        )
        response = conn.read()
        if response.get("type") != "renew_ack":
            raise LeaseConflictError("节点拒绝续约", entry=shard_id, field="lease_version", details=response)
        return response


def read_blackhole_for_timeout_test(port: int, timeout_ms: int = 100) -> None:
    """Helper used by tests against a local socket that never replies."""

    with JsonLineConnection.connect("127.0.0.1", port, timeout_ms) as conn:
        conn.send({"type": "hello", "node_id": "timeout-probe"})
        conn.read()
