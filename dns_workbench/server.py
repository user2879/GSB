"""127.0.0.1 回环 UDP/TCP DNS 诊断服务。

处理流程：
1. 解析线格式查询（非法 -> E_WIRE，返回 SERVFAIL 并记录偏移）；
2. 先查 TTL 缓存（命中按单调时钟输出剩余 TTL）；
3. 未命中执行本地 JSON 区权威解析；
4. 需要上游佐证时通过 TCP 转发到本地模拟上游并做应答链校验；
5. 权威/上游结果写入带 TTL 缓存；UDP 超 512 字节截断置 TC。
"""
from __future__ import annotations

import socketserver
import struct
import threading
from dataclasses import dataclass

from .errors import WireError, WorkbenchError
from .ledger import CacheLedger
from .records import CacheEntry, SOURCE_AUTHORITATIVE, SOURCE_UPSTREAM
from .structured_log import log_error
from .upstream import TcpUpstreamClient
from .wire import (
    CLASS_IN,
    TYPE_A,
    TYPE_AAAA,
    DnsMessage,
    ResourceRecord,
    build_response_flags,
    encode_message,
    parse_message,
    truncate_for_udp,
)
from .zone import AuthorityResolver, ZoneRecord

RCODE_NO_ERROR = 0
RCODE_FORMERR = 1
RCODE_SERVFAIL = 2
RCODE_NXDOMAIN = 3
RCODE_NOTIMP = 4


def zone_record_to_rr(record: ZoneRecord) -> ResourceRecord:
    return ResourceRecord(
        name=record.name,
        rtype=record.qtype,
        rclass=record.rclass,
        ttl=record.ttl,
        rdata=bytes.fromhex(record.rdata_hex) if record.rdata_hex else b"",
        rdata_text=record.value,
    )


def summary_to_rr(name: str, qtype: int, ttl: int, value: str) -> ResourceRecord:
    """把缓存中的记录值摘要还原为可编码的资源记录。"""
    text = value.split(" ", 1)[1] if " " in value else value
    rdata = b""
    if qtype == TYPE_A and text and text[0].isdigit():
        rdata = bytes(int(part) for part in text.split("."))
        text = text
    elif qtype == TYPE_AAAA and ":" in text:
        rdata = b"".join(bytes.fromhex(pair) for pair in text.split(":"))
    return ResourceRecord(name=name, rtype=qtype, rclass=CLASS_IN, ttl=ttl, rdata=rdata, rdata_text=text)


@dataclass
class QueryReport:
    txn_id: int
    qname: str
    qtype: int
    source: str
    rcode: int
    cache_hit: bool
    remaining_ttl_s: int | None
    upstream_format: str | None
    consistent: bool | None
    attempts: int | None


class DnsDiagnosticEngine:
    def __init__(
        self,
        *,
        ledger: CacheLedger,
        resolver: AuthorityResolver,
        upstream_client: TcpUpstreamClient | None = None,
        logger=None,
        forward: bool = False,
    ) -> None:
        self.ledger = ledger
        self.resolver = resolver
        self.upstream_client = upstream_client
        self.logger = logger
        self.forward = forward
        self.reports: list[QueryReport] = []

    def handle_query_wire(self, wire_data: bytes, *, use_cache: bool = True, transport: str = "udp") -> tuple[bytes, bool]:
        """返回 (响应报文, 是否截断)。"""
        try:
            request = parse_message(wire_data)
        except WireError as exc:
            if self.logger:
                log_error(self.logger, exc)
            header = self._servfail_header(getattr(exc, "offset", 0))
            return header, False
        question = request.first_question()
        if question is None:
            response = DnsMessage(request.txn_id, build_response_flags(rcode=RCODE_FORMERR, aa=False))
            return encode_message(response), False
        message, truncated = self.build_response(request, question, use_cache=use_cache)
        if truncated and transport == "udp":
            payload, _ = truncate_for_udp(message)
        else:
            payload = encode_message(message)
        return payload, truncated and transport == "udp"

    def build_response(self, request: DnsMessage, question, *, use_cache: bool = True) -> tuple[DnsMessage, bool]:
        cache = self.ledger.cache()
        hit = cache.get(question.key()) if use_cache else None
        if hit is not None:
            record = hit.entry
            answers = [summary_to_rr(question.qname, question.qtype, hit.remaining_ttl_s, record.value)]
            message = DnsMessage(
                request.txn_id,
                build_response_flags(rcode=record.rcode, aa=record.aa),
                [question],
                answers,
            )
            self._report(request, question, "cache", record.rcode, True, hit.remaining_ttl_s, None, None, None)
            return message, False

        result = self.resolver.resolve(question.qname, question.qtype)
        answers = [zone_record_to_rr(record) for record in result.answers]
        chain = [zone_record_to_rr(record) for record in result.cname_chain]
        authority = [zone_record_to_rr(result.soa)] if result.soa is not None and result.rcode != RCODE_NO_ERROR else []
        upstream_format = None
        consistent = None
        attempts = None
        if self.forward and self.upstream_client is not None and result.rcode == RCODE_NO_ERROR:
            forwarded = self._forward(request)
            if forwarded is not None:
                upstream_format, consistent, attempts = forwarded

        message = DnsMessage(
            request.txn_id,
            build_response_flags(rcode=result.rcode, aa=True),
            [question],
            chain + answers,
            authority,
        )
        self._store_result(question, result, answers, chain, source=SOURCE_AUTHORITATIVE)
        self._report(
            request, question, SOURCE_AUTHORITATIVE, result.rcode, False,
            answers[0].ttl if answers else (result.soa.ttl if result.soa else 0),
            upstream_format, consistent, attempts,
        )
        truncated = len(encode_message(message)) > 512
        return message, truncated

    def _forward(self, request: DnsMessage):
        try:
            forward_result = self.upstream_client.forward(request)
        except WorkbenchError as exc:
            if self.logger:
                log_error(self.logger, exc, txn_id=request.txn_id)
            return None
        fmt = "u2" if forward_result.response.opt_record() is not None else "u1"
        for record in forward_result.response.answers:
            entry = CacheEntry(
                name=record.name,
                qtype=record.rtype,
                value=record.value_summary(),
                ttl=max(0, record.ttl),
                source_version="v2",
                content_digest="",
                rcode=forward_result.response.rcode,
                aa=forward_result.response.aa,
                source=SOURCE_UPSTREAM,
            )
            from .records import canonical_digest

            entry.content_digest = canonical_digest(entry.normalized_payload())
            self.ledger.cache().put(entry)
        return fmt, forward_result.consistent, forward_result.attempts

    def _store_result(self, question, result, answers: list[ResourceRecord], chain: list[ResourceRecord], *, source: str) -> None:
        cache = self.ledger.cache()
        if result.rcode == RCODE_NO_ERROR and answers:
            for record in answers:
                entry = CacheEntry(
                    name=question.qname,
                    qtype=question.qtype,
                    value=record.value_summary(),
                    ttl=record.ttl,
                    source_version="v2",
                    content_digest="",
                    rcode=0,
                    aa=True,
                    source=source,
                )
                from .records import canonical_digest

                entry.content_digest = canonical_digest(entry.normalized_payload())
                cache.put(entry)
            return
        ttl = cache.negative_ttl(self.resolver.zone.soa_minimum())
        negative = CacheEntry(
            name=question.qname,
            qtype=question.qtype,
            value=f"NEGATIVE rcode={result.rcode}",
            ttl=ttl,
            source_version="v2",
            content_digest="",
            rcode=result.rcode,
            aa=True,
            source=source,
        )
        from .records import canonical_digest

        negative.content_digest = canonical_digest(negative.normalized_payload())
        cache.put(negative, ttl_seconds=ttl)

    def _report(self, request, question, source, rcode, cache_hit, ttl, upstream_format, consistent, attempts) -> None:
        self.reports.append(
            QueryReport(
                txn_id=request.txn_id,
                qname=question.qname,
                qtype=question.qtype,
                source=source,
                rcode=rcode,
                cache_hit=cache_hit,
                remaining_ttl_s=ttl,
                upstream_format=upstream_format,
                consistent=consistent,
                attempts=attempts,
            )
        )

    @staticmethod
    def _servfail_header(offset: int) -> bytes:
        message = DnsMessage(0, build_response_flags(rcode=RCODE_SERVFAIL, aa=False))
        return encode_message(message)


class _UdpHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        data, sock = self.request
        engine: DnsDiagnosticEngine = self.server.engine  # type: ignore[attr-defined]
        payload, _truncated = engine.handle_query_wire(data, transport="udp")
        sock.sendto(payload, self.client_address)


class _TcpHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        from .wire import read_tcp_frame

        engine: DnsDiagnosticEngine = self.server.engine  # type: ignore[attr-defined]
        try:
            payload = read_tcp_frame(self.request)
        except WireError:
            return
        response_message, _ = engine.handle_query_wire(payload, use_cache=False, transport="tcp")
        self.request.sendall(struct.pack("!H", len(response_message)) + response_message)


class ThreadedUdpServer(socketserver.ThreadingMixIn, socketserver.UDPServer):
    allow_reuse_address = True
    daemon_threads = True


class ThreadedTcpServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class DnsServerPair:
    def __init__(self, engine: DnsDiagnosticEngine, *, host: str = "127.0.0.1", udp_port: int = 5353, tcp_port: int = 5353) -> None:
        self.engine = engine
        self.host = host
        self.udp_port = udp_port
        self.tcp_port = tcp_port
        self.udp: ThreadedUdpServer | None = None
        self.tcp: ThreadedTcpServer | None = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self.udp = ThreadedUdpServer((self.host, self.udp_port), _UdpHandler)
        self.udp.engine = self.engine  # type: ignore[attr-defined]
        self.tcp = ThreadedTcpServer((self.host, self.tcp_port), _TcpHandler)
        self.tcp.engine = self.engine  # type: ignore[attr-defined]
        self._threads = [
            threading.Thread(target=self.udp.serve_forever, name="dns-udp", daemon=True),
            threading.Thread(target=self.tcp.serve_forever, name="dns-tcp", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        if self.udp:
            self.udp.shutdown()
            self.udp.server_close()
        if self.tcp:
            self.tcp.shutdown()
            self.tcp.server_close()
