"""127.0.0.1 回环 TCP 上的确定性重放执行器（业务要求 3）。

协议为 JSON 行（每行一个 JSON 对象，``\\n`` 分隔）：

* 子进程启动后向 stdout 打印握手行
  ``{"event":"ready","port":<int>,"rule_version":...}``；
* 父进程连接后发送 ``{"type":"hello","rule_version":...}``；
* 每个批次发送
  ``{"type":"batch","batch_id":...,"snapshot_seq":...,"reader_txn_id":...,
     "query_ids":[...],"row_keys":[...]}``，
  执行器逐行返回含 ``batch_id``、``query_ids``、``summaries``、
  ``rule_version`` 的回执。

所有握手与读取均设置真实 socket 超时；不使用任何人为 sleep 来伪装超时。
"""

import json
import socket
import subprocess
import sys
import threading
from pathlib import Path

from . import RULE_VERSION
from .digest import canonical
from .errors import FieldError_, PortError, TimeoutError_
from .loader import load_spec
from .visibility import decide_query

HANDSHAKE_TIMEOUT = 2.0
READ_TIMEOUT = 2.0
CONNECT_TIMEOUT = 2.0
REQUIRED_RECEIPT_FIELDS = ("batch_id", "query_ids", "summaries", "rule_version")


# ----------------------------- 子进程服务端 -----------------------------

def _readline(conn: socket.socket, timeout: float) -> dict:
    conn.settimeout(timeout)
    buffer = bytearray()
    while not buffer.endswith(b"\n"):
        chunk = conn.recv(4096)
        if not chunk:
            raise ConnectionError("连接被对端关闭")
        buffer.extend(chunk)
    return json.loads(buffer.decode("utf-8"))


def _sendline(conn: socket.socket, payload: dict) -> None:
    conn.sendall((canonical(payload) + "\n").encode("utf-8"))


def run_server(spec_path: str, injection: str = "none",
               fixed_port: int = 0) -> int:
    """子进程入口：绑定回环端口并处理批次，返回进程退出码。"""
    spec = load_spec(spec_path)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", fixed_port))
    except OSError as exc:
        print(json.dumps({"event": "error", "code": "E_PORT",
                          "message": str(exc)}), flush=True)
        return 2
    sock.listen(8)
    port = sock.getsockname()[1]
    # 握手公告行：父进程据此获得端口。
    print(json.dumps({"event": "ready", "port": port,
                      "rule_version": RULE_VERSION}), flush=True)

    if injection == "silent":
        # 不 accept、不回任何数据，用于触发真实握手/读取超时。
        sock.settimeout(None)
        try:
            while True:
                sock.accept()
        except KeyboardInterrupt:  # pragma: no cover
            return 0
    if injection == "silent_hold":
        # 接受连接但对任何数据都不响应：触发真实 socket 读超时（无 sleep）。
        held, _ = sock.accept()
        try:
            held.settimeout(None)
            while True:
                held.recv(4096)
        except OSError:
            return 0

    conn, _ = sock.accept()
    with conn:
        hello = _readline(conn, HANDSHAKE_TIMEOUT)
        if hello.get("type") != "hello":
            _sendline(conn, {"event": "error", "code": "E_FIELD",
                             "message": "缺少 hello 握手"})
            return 3
        _sendline(conn, {"event": "hello_ack", "rule_version": RULE_VERSION})
        while True:
            try:
                message = _readline(conn, READ_TIMEOUT)
            except (ConnectionError, json.JSONDecodeError):
                return 0
            if message.get("type") == "shutdown":
                _sendline(conn, {"event": "bye"})
                return 0
            if message.get("type") != "batch":
                _sendline(conn, {"event": "error", "code": "E_FIELD",
                                 "message": "缺少 batch 类型"})
                return 3
            summaries = []
            for qid, row_key in zip(message["query_ids"],
                                    message["row_keys"]):
                query = next(q for q in spec.queries if q.query_id == qid)
                decision = decide_query(
                    spec, query,
                    forced_rule_conflict=(injection == "rule_conflict"),
                )
                summaries.append(decision.summary())
            receipt = {
                "batch_id": message["batch_id"],
                "query_ids": list(message["query_ids"]),
                "summaries": summaries,
                "rule_version": RULE_VERSION,
            }
            if injection == "missing_field":
                # 回执字段缺失注入：删除约定必填字段。
                receipt.pop("rule_version", None)
            _sendline(conn, receipt)


# ------------------------------- 客户端 --------------------------------

class ReplayExecutor:
    """以子进程托管执行器，并通过回环 TCP 收发 JSON 行。"""

    def __init__(self, spec_path: str | Path, injection: str = "none",
                 python_executable: str | None = None):
        self.spec_path = str(spec_path)
        self.injection = injection
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._python = python_executable or sys.executable
        # 单套接字上的批次请求必须串行，避免多线程写交错破坏 JSON 行协议；
        # 并发上限仍由调度器信号量真实约束，该锁只保护套接字事务原子性。
        self._io_lock = threading.Lock()

    def start(self) -> None:
        if self.injection == "port_busy":
            # 真实占用一个端口，再让执行器子进程绑定同一端口 -> E_PORT。
            holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            holder.bind(("127.0.0.1", 0))
            holder.listen(1)
            busy_port = holder.getsockname()[1]
            self._holder = holder
            self._proc = subprocess.Popen(
                [self._python, "-m", "mvcc_lab.executor", self.spec_path,
                 "--inject", "none", "--port", str(busy_port)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            line = self._read_ready_line()
            if isinstance(line, dict) and line.get("code") == "E_PORT":
                raise PortError(
                    f"回环端口 {busy_port} 已被占用",
                    object_id=f"127.0.0.1:{busy_port}",
                    field="port",
                )
            raise PortError("端口占用注入未按预期返回 E_PORT",
                            object_id=str(busy_port), field="port")

        self._proc = subprocess.Popen(
            [self._python, "-m", "mvcc_lab.executor", self.spec_path,
             "--inject", self.injection],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        line = self._read_ready_line()
        if not isinstance(line, dict) or line.get("event") != "ready":
            code = line.get("code") if isinstance(line, dict) else None
            if code == "E_PORT":
                raise PortError("执行器报告端口占用", field="port")
            raise PortError(f"执行器握手公告非法: {line!r}", field="ready")
        port = line["port"]
        self._connect(port)
        self._handshake()

    def _read_ready_line(self) -> dict:
        # 真实超时：用独立线程阻塞读一行，Event 等待实施真实超时，
        # 不依赖对缓冲管道不可靠的 select。
        if self._proc is None or self._proc.stdout is None:
            raise PortError("执行器子进程未启动", field="process")
        box: dict = {}
        event = threading.Event()

        def _reader():
            box["line"] = self._proc.stdout.readline()  # type: ignore[union-attr]
            event.set()

        worker = threading.Thread(target=_reader, daemon=True)
        worker.start()
        if not event.wait(HANDSHAKE_TIMEOUT):
            self.close()
            raise TimeoutError_("等待执行器握手公告超时", field="ready")
        line = box.get("line", "")
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            self.close()
            raise PortError(f"执行器提前退出: {err.strip()}", field="process")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise PortError(f"握手公告不是合法 JSON: {line!r}",
                            field="ready") from exc

    def _connect(self, port: int) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(CONNECT_TIMEOUT)
        try:
            if self.injection == "silent":
                # 真实连接成功但服务端不响应 -> 握手读取超时（无 sleep）。
                sock.connect(("127.0.0.1", port))
            else:
                sock.connect(("127.0.0.1", port))
        except socket.timeout as exc:
            sock.close()
            raise TimeoutError_("连接执行器超时",
                                object_id=f"127.0.0.1:{port}",
                                field="connect") from exc
        self._sock = sock

    def _handshake(self) -> None:
        assert self._sock is not None
        self._sock.settimeout(HANDSHAKE_TIMEOUT)
        try:
            self._sock.sendall(
                (canonical({"type": "hello",
                            "rule_version": RULE_VERSION}) + "\n").encode()
            )
            data = self._recv_line(HANDSHAKE_TIMEOUT)
        except socket.timeout as exc:
            raise TimeoutError_("执行器握手响应超时",
                                field="hello_ack") from exc
        except (ConnectionError, OSError) as exc:
            # 对端不响应或重置：真实 I/O 失败，归并为握手超时类业务码。
            raise TimeoutError_(
                f"握手期间连接异常: {exc}", field="hello_ack"
            ) from exc
        if data.get("event") != "hello_ack":
            raise FieldError_("握手回执缺少 hello_ack", field="event")

    def _recv_line(self, timeout: float) -> dict:
        assert self._sock is not None
        self._sock.settimeout(timeout)
        buffer = bytearray()
        while not buffer.endswith(b"\n"):
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                raise
            except (ConnectionError, OSError) as exc:
                raise TimeoutError_(
                    f"读取回执时连接异常: {exc}", field="receipt"
                ) from exc
            if not chunk:
                raise TimeoutError_("执行器关闭了连接且未返回完整回执",
                                    field="receipt")
            buffer.extend(chunk)
        return json.loads(buffer.decode("utf-8"))

    def send_batch(self, batch_id: str, snapshot_seq: int,
                   reader_txn_id: str, query_ids, row_keys) -> dict:
        if self._sock is None:
            raise PortError("执行器尚未启动", field="socket")
        message = {
            "type": "batch",
            "batch_id": batch_id,
            "snapshot_seq": snapshot_seq,
            "reader_txn_id": reader_txn_id,
            "query_ids": list(query_ids),
            "row_keys": list(row_keys),
        }
        with self._io_lock:
            try:
                self._sock.sendall(
                    (canonical(message) + "\n").encode()
                )
                receipt = self._recv_line(READ_TIMEOUT)
            except socket.timeout as exc:
                raise TimeoutError_(
                    f"批次 {batch_id} 回执读取超时",
                    object_id=batch_id, field="receipt",
                ) from exc
            for name in REQUIRED_RECEIPT_FIELDS:
                if name not in receipt:
                    raise FieldError_(
                        f"批次 {batch_id} 回执缺少必填字段 {name}",
                        object_id=batch_id, field=name,
                    )
            if list(receipt["query_ids"]) != list(query_ids):
                raise FieldError_(
                    f"批次 {batch_id} 回执 query_ids 与请求不一致",
                    object_id=batch_id, field="query_ids",
                )
            return receipt

    def close(self) -> None:
        try:
            if self._sock is not None:
                try:
                    self._sock.sendall(
                        (canonical({"type": "shutdown"}) + "\n").encode()
                    )
                except OSError:
                    pass
                self._sock.close()
                self._sock = None
        finally:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=3)
            self._proc = None
            holder = getattr(self, "_holder", None)
            if holder is not None:
                holder.close()
                self._holder = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    injection = "none"
    fixed_port = 0
    if "--inject" in argv:
        injection = argv[argv.index("--inject") + 1]
    if "--port" in argv:
        fixed_port = int(argv[argv.index("--port") + 1])
    spec_path = next(a for a in argv if not a.startswith("--") and
                     a not in ("--inject", injection, "--port", str(fixed_port)))
    return run_server(spec_path, injection, fixed_port=fixed_port)


if __name__ == "__main__":
    raise SystemExit(main())
