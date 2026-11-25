"""回环 TCP 上的 JSONL 分帧协议与真实套接字超时。

帧格式：单条 JSON 文本 + 换行符；握手由执行器先发送 HELLO，
主进程回复 READY 后进入请求/响应循环。所有收发均设置真实 socket timeout。
"""

import json
import socket
from typing import Any, Dict

from .errors import PortError, ProtocolError

HELLO = {"type": "hello", "protocol": "coljoin-tcp-v1"}
READY = {"type": "ready"}
BYE = {"type": "shutdown"}


def find_free_port() -> int:
    """探测 127.0.0.1 上的空闲高位端口；失败按 E_PORT 报告。"""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    except OSError as exc:
        raise PortError(f"空闲端口探测失败: {exc}", field="port") from exc
    finally:
        probe.close()
    return port


def send_frame(sock: socket.socket, payload: Dict[str, Any]) -> None:
    sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


def recv_frame(sock: socket.socket) -> Dict[str, Any]:
    buffer = bytearray()
    while True:
        try:
            chunk = sock.recv(65536)
        except socket.timeout as exc:
            raise ProtocolError("套接字读取超时", field="recv") from exc
        except OSError as exc:
            raise ProtocolError(f"套接字读取失败: {exc}", field="recv") from exc
        if not chunk:
            raise ProtocolError("对端关闭连接", field="recv")
        buffer.extend(chunk)
        if buffer.endswith(b"\n"):
            break
    try:
        return json.loads(buffer.decode("utf-8").strip())
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"非法 JSON 帧: {exc.msg}", field="frame") from exc


def is_port_in_use(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        probe.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()
