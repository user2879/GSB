"""本地确定性上游解析器（127.0.0.1 TCP）。

依据本地 JSON 应答表返回 DNS 线格式响应：
- 默认按 u2（EDNS + DO=1，附加 OPT 伪记录）；
--edns0 关闭时退化为 u1；
--bad-txn 注入事务标识错误（E_TXN）；
--bad-chain 注入应答名链断裂（E_CHAIN）。
不访问任何真实域名或线上解析器。
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dns_workbench.wire import (  # noqa: E402
    CLASS_IN,
    TYPE_A,
    TYPE_AAAA,
    TYPE_CNAME,
    TYPE_OPT,
    DnsMessage,
    ResourceRecord,
    build_response_flags,
    encode_tcp_frame,
    parse_message,
    read_tcp_frame,
)


def load_answer_table(path: Path) -> dict:
    table = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        table[(item["name"].lower(), int(item["qtype"]))] = item
    return table


def build_response(request: DnsMessage, table: dict, *, edns: bool, bad_txn: bool, bad_chain: bool) -> DnsMessage:
    question = request.first_question()
    key = (question.qname, question.qtype)
    item = table.get(key)
    txn_id = (request.txn_id ^ 0xFFFF) if bad_txn else request.txn_id
    if item is None:
        return DnsMessage(txn_id, build_response_flags(rcode=3, aa=False), [question])
    answers: list[ResourceRecord] = []
    for rr in item["answers"]:
        name = rr["name"]
        if bad_chain and answers:
            name = "broken-chain.example.test"
        rtype = int(rr["qtype"])
        rdata = bytes.fromhex(rr.get("rdata_hex", ""))
        if rtype == TYPE_CNAME and not rdata:
            rdata = b"".join(bytes([len(label)]) + label.encode("ascii") for label in rr["value"].split(".")) + b"\x00"
        answers.append(ResourceRecord(name, rtype, CLASS_IN, int(rr["ttl"]), rdata, rr.get("value", "")))
    additional: list[ResourceRecord] = []
    if edns:
        # OPT: 类字段承载 UDP payload size，TTL 字段承载 extended-rcode/version/DO
        opt_ttl = 0x00008000  # DO=1
        additional.append(ResourceRecord("", TYPE_OPT, 4096, opt_ttl, b"", ""))
    return DnsMessage(txn_id, build_response_flags(rcode=0, aa=False), [question], answers, [], additional)


def serve(table: dict, host: str, port: int, *, edns: bool, bad_txn: bool, bad_chain: bool) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(8)
        print(f"mock upstream listening on {host}:{port} edns={edns} bad_txn={bad_txn} bad_chain={bad_chain}", flush=True)
        while True:
            conn, _addr = listener.accept()
            threading.Thread(target=_handle, args=(conn, table, edns, bad_txn, bad_chain), daemon=True).start()


def _handle(conn: socket.socket, table: dict, edns: bool, bad_txn: bool, bad_chain: bool) -> None:
    with conn:
        try:
            payload = read_tcp_frame(conn)
            request = parse_message(payload)
            response = build_response(request, table, edns=edns, bad_txn=bad_txn, bad_chain=bad_chain)
            conn.sendall(encode_tcp_frame(response))
        except Exception as exc:  # 诊断模拟服务：打印异常并断开
            print("mock upstream error:", exc, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="本地确定性 DNS 上游模拟（TCP）")
    parser.add_argument("--answers", default="dns_workbench/samples/upstream_answers.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5354)
    parser.add_argument("--u1", action="store_true", help="使用 u1（无 EDNS），默认 u2")
    parser.add_argument("--bad-txn", action="store_true", help="注入事务标识错误")
    parser.add_argument("--bad-chain", action="store_true", help="注入应答名链断裂")
    args = parser.parse_args()
    table = load_answer_table(Path(args.answers))
    serve(table, args.host, args.port, edns=not args.u1, bad_txn=args.bad_txn, bad_chain=args.bad_chain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
