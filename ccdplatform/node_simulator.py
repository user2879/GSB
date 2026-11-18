"""本地模拟组件节点：独立子进程 + 回环 TCP JSON 行协议。

协议
----
1. 平台启动进程后连接 ``host:port``，节点在 socket accept 后先发送
   ``{"type":"hello","node":...,"seq":0}`` 完成握手。
2. 平台每批下发一行 JSON：
   ``{"type":"dispatch","node":...,"version":...,"seq":N,"items":[...]}``
3. 节点按同一份纯整数口径重算派生格点值，生成派生状态摘要并回执：
   ``{"type":"receipt","node":...,"version":...,"seq":N,"digest":...,"attempt":A}``
4. 平台发送 ``{"type":"shutdown"}`` 后节点优雅退出。

依赖感知切片允许同一个基础键在多批中重复出现，节点以幂等覆盖方式合并，
派生项携带完整闭包，因此每批都能独立完成重算。节点只依赖标准库。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from typing import Any, Dict, List


def recv_line(conn: socket.socket, buffer: bytearray) -> str:
    while b"\n" not in buffer:
        chunk = conn.recv(65536)
        if not chunk:
            raise ConnectionError("连接已关闭")
        buffer.extend(chunk)
    line, _, rest = buffer.partition(b"\n")
    buffer[:] = rest
    return line.decode("utf-8")


def send_line(conn: socket.socket, payload: Dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload, ensure_ascii=True) + "\n").encode("ascii"))


def trunc_zero(numerator: int, denominator: int = 1) -> int:
    q, _ = divmod(abs(numerator), denominator)
    return q if numerator >= 0 else -q


def recompute_derived(
    items: List[Dict[str, Any]], scale: int, known_q: Dict[str, int] | None = None
) -> Dict[str, int]:
    """与平台同口径的整数乘加重算，返回每个键的格点值。

    ``known_q`` 提供此前批次已收到的键值，本批重复键以新值幂等覆盖。
    派生项按拓扑序求解；只累计本节点实际可达的上游，与平台
    ``Dispatcher._payload_for`` / ``convergence`` 的过滤口径一致。
    """
    qmap: Dict[str, int] = dict(known_q or {})
    derived = {it["key"]: it for it in items if it.get("derivation")}
    for it in items:
        if not it.get("derivation"):
            qmap[it["key"]] = int(it["quantized"])

    remaining = dict(derived)
    while remaining:
        progressed = False
        for key in list(remaining):
            refs = remaining[key]["derivation"]["refs"]
            reachable = [(rk, cq) for rk, cq in refs if rk in qmap]
            # 等待所有可达上游就绪（不可达上游已被平台过滤，永不出现）
            if any(rk not in qmap for rk, _ in reachable):
                continue
            total = 0
            for ref_key, coef_q in reachable:
                total += int(coef_q) * qmap[ref_key]
            # 同尺度乘加后除以 scale 回到同一格度，向零截断
            qmap[key] = trunc_zero(total, scale)
            del remaining[key]
            progressed = True
        if not progressed:
            raise ValueError(
                f"节点侧重算失败，缺失上游或存在环: {sorted(remaining)}"
            )
    return qmap


def state_digest(node_id: str, version: int, qmap: Dict[str, int]) -> str:
    material = json.dumps(
        {"node": node_id, "version": version, "q": qmap},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(material).hexdigest()


def run(node_id: str, host: str, port: int, scale: int) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        # 明确的端口占用信号，编排器据此报 E_PORT
        sys.stderr.write(f"BIND_ERROR {node_id} {host}:{port} {exc}\n")
        sys.stderr.flush()
        return 2
    sock.listen(4)
    sock.settimeout(30.0)
    sys.stdout.write(f"READY {node_id} {host}:{port}\n")
    sys.stdout.flush()

    conn, _addr = sock.accept()
    conn.settimeout(None)
    buffer = bytearray()
    cumulative_q: Dict[str, int] = {}
    try:
        send_line(conn, {"type": "hello", "node": node_id, "seq": 0})
        while True:
            message = json.loads(recv_line(conn, buffer))
            mtype = message.get("type")
            if mtype == "shutdown":
                send_line(conn, {"type": "bye", "node": node_id})
                return 0
            if mtype != "dispatch":
                continue
            seq = int(message["seq"])
            attempt = int(message.get("attempt", 1))
            # 跨批累积；摘要覆盖本节点截至当前批次的全部已知键。
            # 同一 seq 重试时以幂等覆盖方式重算，结果确定性不变。
            cumulative_q = recompute_derived(
                message["items"], scale, known_q=cumulative_q
            )
            digest = state_digest(
                node_id, int(message["version"]), cumulative_q
            )
            receipt = {
                "type": "receipt",
                "node": node_id,
                "version": int(message["version"]),
                "seq": seq,
                "attempt": attempt,
                "digest": digest,
            }
            # 故障注入：仅在“首个批次的首次尝试”发送回执前延迟，
            # 平台侧因 socket 读超时判定失败并进入重试；
            # 迟到的 attempt=1 回执随后到达，由平台排空逻辑丢弃。
            fault = os.environ.get("CCD_FAULT_FIRST_DELAY_MS")
            if fault and seq == 1 and attempt == 1:
                time.sleep(int(fault) / 1000.0)
            send_line(conn, receipt)
    except (ConnectionError, BrokenPipeError):
        return 0
    finally:
        try:
            conn.close()
        except OSError:
            pass
        sock.close()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="模拟组件节点")
    parser.add_argument("--node", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--scale", type=int, default=1000)
    args = parser.parse_args(argv)
    return run(args.node, args.host, args.port, args.scale)


if __name__ == "__main__":
    raise SystemExit(main())
