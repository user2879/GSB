"""组件节点子进程入口。

协议（127.0.0.1 回环 TCP，每行一个 JSON 帧，\\n 分隔）：

1. 平台 -> 节点 ``{"type":"hello","node_id":...}``
2. 节点 -> 平台 ``{"type":"hello_ack","node_id":...,"pid":...}``
3. 平台 -> 节点 ``{"type":"slice","config_version":N,"items":[...],
   "samples":{...},"request_id":...}``
4. 节点 -> 平台 ``{"type":"slice_ack","request_id":...,"config_version":N,
   "summary":...,"summary_digest":...,"elapsed_ms":...}``

所有套接字操作设置真实超时；同一连接顺序处理多条 slice 请求，
节点对相同切片+种子返回逐字节一致的 summary。
"""

import json
import socket
import sys
import threading
import time

from .aggregate import aggregate_slice
from .manifest import ItemSpec, DispatchPolicy, DeriveEdge, SampleDecl

E_FIELD = "E_FIELD"

REQUIRED_SLASH_FIELDS = ("type", "config_version", "items", "samples", "request_id")


def _mono_ms():
    return int(time.monotonic() * 1000)


def _readline(conn):
    buf = bytearray()
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            return None
        buf.extend(chunk)
        if b"\n" in chunk:
            line, _, _ = buf.partition(b"\n")
            return line


def _item_from_dict(raw):
    return ItemSpec(
        key=raw["key"],
        value_type=raw["value_type"],
        value=raw["value"],
        scale=raw["scale"],
        targets=tuple(raw["targets"]),
        dispatch=DispatchPolicy(1, 0, "fail_fast"),
        derives_from=tuple(
            DeriveEdge(e["key"], e["coef_raw"], e["coef_scale"])
            for e in raw.get("derives_from", [])
        ),
    )


def _handle_conn(conn, node, fail_mode):
    conn.settimeout(5.0)
    line = _readline(conn)
    if line is None:
        return
    hello = json.loads(line.decode("utf-8"))
    if hello.get("type") != "hello" or hello.get("node_id") != node.node_id:
        conn.sendall(json.dumps({"type": "error", "code": "E_HANDSHAKE"}).encode()
                     + b"\n")
        return
    if fail_mode == "handshake":
        # 故意不回握手，让平台触发真实握手超时。
        time.sleep(8.0)
        return
    conn.sendall(json.dumps({
        "type": "hello_ack", "node_id": node.node_id, "pid": _mono_ms(),
    }).encode() + b"\n")

    while True:
        line = _readline(conn)
        if line is None:
            return
        try:
            msg = json.loads(line.decode("utf-8"))
            missing = [f for f in REQUIRED_SLASH_FIELDS if f not in msg]
            if missing:
                conn.sendall(json.dumps({
                    "type": "error", "code": E_FIELD, "missing": missing,
                }).encode() + b"\n")
                continue
            if msg.get("type") != "slice":
                conn.sendall(json.dumps({
                    "type": "error", "code": E_FIELD,
                    "missing": ["type=slice"],
                }).encode() + b"\n")
                continue
            start = _mono_ms()
            items = [_item_from_dict(i) for i in msg["items"]]
            sd = msg["samples"]
            sample_decl = SampleDecl(
                sd["seed"], sd["n_samples"], sd["n_groups"],
                sd["value_range"], sd["declared_scale"])
            summary, digest = aggregate_slice(
                node, msg["config_version"], items, sample_decl)
            elapsed = _mono_ms() - start
            ack = {
                "type": "slice_ack",
                "request_id": msg["request_id"],
                "config_version": msg["config_version"],
                "summary": summary,
                "summary_digest": digest,
                "elapsed_ms": elapsed,
            }
            if fail_mode == "field":
                # 模拟回执字段缺失。
                del ack["summary_digest"]
            conn.sendall(json.dumps(ack, sort_keys=True).encode() + b"\n")
        except Exception as exc:  # 节点侧任何异常都回错误帧而非挂死连接
            try:
                conn.sendall(json.dumps({
                    "type": "error", "code": E_FIELD,
                    "error": type(exc).__name__,
                }).encode() + b"\n")
            except OSError:
                return


def serve(node):
    fail_mode = node.fail_inject
    if fail_mode == "port":
        # 先占用自身端口后再尝试绑定同一端口，稳定触发 E_PORT。
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        holder.bind((node.host, node.port))
        holder.listen(1)
        def _serve_holder():
            while True:
                try:
                    c, _ = holder.accept()
                    c.close()
                except OSError:
                    return
        threading.Thread(target=_serve_holder, daemon=True).start()

    if fail_mode == "flaky":
        # flaky：门闩套接字对首个连接立即 RST（产生可重试错误），其后的
        # 连接正常接受并走同一协议处理；验证平台重试且重发语义不变。
        gate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        gate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        gate.bind((node.host, node.port))
        gate.listen(8)
        gate.settimeout(30.0)
        rejected = {"n": 0}

        def _serve_gate():
            import struct
            while True:
                try:
                    conn, _ = gate.accept()
                except OSError:
                    return
                if rejected["n"] < 1:
                    rejected["n"] += 1
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                    struct.pack("ii", 1, 0))  # 关闭即 RST
                    conn.close()
                    continue
                threading.Thread(target=_handle_conn,
                                 args=(conn, node, fail_mode),
                                 daemon=True).start()

        threading.Thread(target=_serve_gate, daemon=True).start()
        print(json.dumps({"type": "ready", "node_id": node.node_id,
                          "port": node.port, "flaky": True}), flush=True)
        while True:
            time.sleep(3600)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,
                   1 if fail_mode != "port" else 0)
    try:
        srv.bind((node.host, node.port))
    except OSError as exc:
        print(json.dumps({"type": "bind_error", "code": "E_PORT",
                          "node_id": node.node_id, "error": str(exc)}),
              flush=True)
        return 2
    srv.listen(8)
    srv.settimeout(30.0)
    print(json.dumps({"type": "ready", "node_id": node.node_id,
                      "port": node.port}), flush=True)
    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        threading.Thread(target=_handle_conn, args=(conn, node, fail_mode),
                         daemon=True).start()


def main():
    spec_path = sys.argv[1]
    with open(spec_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    from .manifest import RegistryNode

    node = RegistryNode(
        node_id=raw["node_id"], host=raw["host"], port=raw["port"],
        group_key=raw["group_key"], scale=raw["scale"],
        fail_inject=raw.get("fail_inject"),
    )
    sys.exit(serve(node) or 0)


if __name__ == "__main__":
    main()
