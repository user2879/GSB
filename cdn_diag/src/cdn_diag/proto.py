"""JSON 行协议：每行一个完整 JSON 对象，UTF-8 编码。"""

import json
import socket

from .errors import ProtoError

REQUIRED_HELLO_ACK_FIELDS = ("type", "protocol", "server_id")


def send_obj(sock, obj):
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    sock.sendall(data)


def recv_obj(sock_file):
    line = sock_file.readline()
    if not line:
        raise ProtoError("连接对端关闭，未收到回执")
    try:
        obj = json.loads(line.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProtoError("回执不是合法 JSON 行", error=str(exc))
    if not isinstance(obj, dict) or "type" not in obj:
        raise ProtoError("回执缺少 type 字段")
    return obj


def validate_hello_ack(obj, expected_server_id):
    """握手回执字段校验，缺字段 -> E_PROTO。"""
    for f in REQUIRED_HELLO_ACK_FIELDS:
        if f not in obj:
            raise ProtoError("握手回执缺少字段", missing_field=f)
    if obj["type"] != "hello_ack":
        raise ProtoError("握手回执类型错误", got=obj["type"])
    if obj["server_id"] != expected_server_id:
        raise ProtoError(
            "握手回执 server_id 不符",
            expected=expected_server_id,
            got=obj["server_id"],
        )


def serve(sock, handler, on_close=None):
    """单连接服务循环：逐行读取 JSON 并分派。"""
    f = sock.makefile("rb")
    try:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                obj = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                send_obj(sock, {"type": "error", "code": "E_PROTO", "message": "非法 JSON 行"})
                continue
            try:
                reply = handler(obj)
            except Exception as exc:  # noqa: BLE001 - 子进程边界统一转错误回执
                code = getattr(exc, "code", "E_INTERNAL")
                reply = {
                    "type": "error",
                    "code": code,
                    "message": str(exc),
                    "details": getattr(exc, "details", {}),
                }
            if reply is None:
                break
            send_obj(sock, reply)
    finally:
        try:
            sock.close()
        except OSError:
            pass
        if on_close:
            on_close()
