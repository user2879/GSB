"""回环 TCP 上的 JSON 行协议（真实套接字 + 真实超时）。"""

import json
import socket

from .errors import PortError, TimeoutError_


def connect_loopback(port, connect_timeout_ms):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(connect_timeout_ms / 1000.0)
    try:
        sock.connect(("127.0.0.1", int(port)))
    except ConnectionRefusedError as exc:
        sock.close()
        raise PortError(f"连接 127.0.0.1:{port} 被拒绝（端口未监听或被占用）",
                        location=f"127.0.0.1:{port}", field="port") from exc
    except socket.timeout as exc:
        sock.close()
        raise TimeoutError_(f"连接 127.0.0.1:{port} 握手超时",
                            location=f"127.0.0.1:{port}", field="connect") from exc
    except OSError as exc:
        sock.close()
        raise PortError(str(exc), location=f"127.0.0.1:{port}", field="port") from exc
    return sock


def send_line(sock, message):
    data = json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sock.sendall(data + b"\n")


def recv_line(sock, read_timeout_ms, location="read"):
    sock.settimeout(read_timeout_ms / 1000.0)
    buffer = bytearray()
    try:
        while not buffer.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("对端在消息结束前关闭连接")
            buffer.extend(chunk)
    except socket.timeout as exc:
        raise TimeoutError_("读取工作进程回执超时", location=location, field="read") from exc
    return json.loads(buffer.decode("utf-8"))


class WorkerClient:
    """单次连接的同步客户端，上下文管理保证关闭套接字。"""

    def __init__(self, port, connect_timeout_ms, read_timeout_ms):
        self.port = port
        self.connect_timeout_ms = connect_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.sock = None

    def __enter__(self):
        self.sock = connect_loopback(self.port, self.connect_timeout_ms)
        hello = recv_line(self.sock, self.connect_timeout_ms, location="handshake")
        if hello.get("type") != "hello" or hello.get("host") != "127.0.0.1":
            raise PortError("工作进程握手报文非法", location=f"127.0.0.1:{self.port}",
                            field="handshake")
        return self

    def request(self, message):
        send_line(self.sock, message)
        return recv_line(self.sock, self.read_timeout_ms,
                         location=f"127.0.0.1:{self.port}")

    def __exit__(self, exc_type, exc, tb):
        if self.sock is not None:
            self.sock.close()
        return False
