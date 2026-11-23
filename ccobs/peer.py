"""127.0.0.1 回环确定性模拟对端。

对端在 127.0.0.1 高位端口监听, 按本地 JSON 脚本逐条处理客户端报文:
  ack      正常确认
  dup_ack  重复确认
  loss     丢包(不应答)
  delay    延迟 delay_ms 后确认(逻辑时钟)
  reorder  乱序应答(挂起, 在下一条应答之后发出)
  gap      脚本执行故障注入: 跳过本条目及其后 skip-1 条, 均不应答

线上传输仅为载体; 所有时间量均由逻辑时钟决定, 保证同一脚本在任意
执行次数下产出逐字节一致的批次边界与应答归属。
"""
import json
import socket
import struct
import threading


def _send_msg(conn, obj):
    payload = json.dumps(obj, sort_keys=True).encode("utf-8")
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_msg(conn):
    header = b""
    while len(header) < 4:
        chunk = conn.recv(4 - len(header))
        if not chunk:
            return None
        header += chunk
    (length,) = struct.unpack(">I", header)
    body = b""
    while len(body) < length:
        chunk = conn.recv(length - len(body))
        if not chunk:
            return None
        body += chunk
    return json.loads(body.decode("utf-8"))


class PeerServer:
    """确定性模拟对端(单连接, 脚本驱动)。"""

    def __init__(self, script, host, port):
        self.script = script
        self.host = host
        self.port = port
        self.bound_port = None
        self._ready = threading.Event()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="ccobs-peer")
        self.error = None

    def start(self):
        self._thread.start()
        self._ready.wait(timeout=5.0)
        if self.error is not None:
            raise self.error
        return self.bound_port

    def _serve(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self.host, self.port))
                srv.listen(1)
                self.bound_port = srv.getsockname()[1]
                self._ready.set()
                conn, _addr = srv.accept()
                with conn:
                    self._drive(conn)
        except Exception as exc:  # noqa: BLE001 - 记录并向上传递
            self.error = exc
            self._ready.set()
        finally:
            self._done.set()

    def _drive(self, conn):
        by_seq = self.script.by_seq()
        held = None  # reorder 挂起的应答
        skip_remaining = 0  # gap 注入: 后续不应答的条目数
        while True:
            msg = _recv_msg(conn)
            if msg is None:
                break
            entry = by_seq.get(msg.get("seq"))
            if entry is None:
                continue
            if entry.action == "gap":
                skip_remaining = entry.skip - 1
                continue  # 本条目也不应答
            if skip_remaining > 0:
                skip_remaining -= 1
                continue
            reply = self._reply_for(entry)
            if reply is None:
                continue  # loss: 不应答
            if entry.action == "reorder":
                held = reply
                continue
            _send_msg(conn, reply)
            if held is not None:
                _send_msg(conn, held)
                held = None
        if held is not None:
            try:
                _send_msg(conn, held)
            except OSError:
                pass

    @staticmethod
    def _reply_for(entry):
        if entry.action == "loss":
            return None
        if entry.action == "dup_ack":
            return {"type": "dup_ack", "event_seq": entry.event_seq,
                    "delay_ms": entry.delay_ms}
        return {"type": "ack", "event_seq": entry.event_seq,
                "delay_ms": entry.delay_ms}

    def wait_done(self, timeout=10.0):
        self._done.wait(timeout=timeout)
