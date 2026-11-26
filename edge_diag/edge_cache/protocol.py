"""JSON 行协议的收发。

- 每条消息为单行 UTF-8 JSON，以 '\\n' 结束。
- 所有握手/读写均设置真实套接字超时，超时抛 TimeoutError(E_TIMEOUT)，
  不使用任何人为 sleep 模拟超时。
- 端口占用由连接拒绝区分，抛 PortError(E_PORT)。
"""

import json
import socket

from .errors import PortError, TimeoutError

DEFAULT_CONNECT_TIMEOUT_MS = 3000
DEFAULT_IO_TIMEOUT_MS = 8000
MAX_LINE_BYTES = 16 * 1024 * 1024


def connect_loopback(host: str, port: int,
                     connect_timeout_ms: int = DEFAULT_CONNECT_TIMEOUT_MS):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(connect_timeout_ms / 1000.0)
    try:
        sock.connect((host, port))
    except ConnectionRefusedError as exc:
        sock.close()
        raise PortError(f"端口 {host}:{port} 连接被拒绝（节点未监听或端口占用）",
                        f"{host}:{port}") from exc
    except socket.timeout as exc:
        sock.close()
        raise TimeoutError(f"连接 {host}:{port} 超时", f"{host}:{port}") from exc
    except OSError as exc:
        sock.close()
        raise PortError(f"套接字错误 {host}:{port}: {exc}",
                        f"{host}:{port}") from exc
    return sock


class LineConnection:
    def __init__(self, sock, io_timeout_ms=DEFAULT_IO_TIMEOUT_MS):
        self.sock = sock
        self.io_timeout_ms = io_timeout_ms
        self._buf = b""
        sock.settimeout(io_timeout_ms / 1000.0)

    def send_message(self, message: dict):
        line = json.dumps(message, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8") + b"\n"
        try:
            self.sock.sendall(line)
        except socket.timeout as exc:
            raise TimeoutError("发送消息超时", None) from exc
        except OSError as exc:
            raise PortError(f"套接字写入失败: {exc}", None) from exc

    def read_message(self) -> dict:
        while b"\n" not in self._buf:
            if len(self._buf) > MAX_LINE_BYTES:
                raise PortError("消息行超过上限", None)
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout as exc:
                raise TimeoutError(
                    f"等待节点回执超时（{self.io_timeout_ms}ms）", None) from exc
            except OSError as exc:
                raise PortError(f"套接字读取失败: {exc}", None) from exc
            if not chunk:
                raise PortError("节点在响应前关闭连接", None)
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise PortError(f"节点回执不是合法 JSON: {exc}", None) from exc

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def request_reply(host, port, message, connect_timeout_ms=DEFAULT_CONNECT_TIMEOUT_MS,
                  io_timeout_ms=DEFAULT_IO_TIMEOUT_MS):
    sock = connect_loopback(host, port, connect_timeout_ms)
    conn = LineConnection(sock, io_timeout_ms)
    try:
        conn.send_message(message)
        return conn.read_message()
    finally:
        conn.close()
