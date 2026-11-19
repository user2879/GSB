"""Loopback TCP upstream client, response-chain validator and simulator."""

from __future__ import annotations

import argparse
import socket
import socketserver
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import BusinessError
from .names import normalize_name
from .netutil import check_tcp_port, require_loopback_host
from .resolver import ZoneResolver
from .wire import (
    AA,
    CLASS_IN,
    QR,
    RA,
    RD,
    RCODE_NOERROR,
    RCODE_NXDOMAIN,
    RCODE_SERVFAIL,
    ResourceRecord,
    SOAData,
    TYPE_BY_NAME,
    DNSMessage,
    Question,
    build_error_response,
    encode_message,
    frame_tcp,
    parse_message,
    question_signature,
    receive_tcp_message,
)
from .zone import ZoneImage, ZoneManager


@dataclass(frozen=True)
class UpstreamResult:
    message: DNSMessage
    answers: tuple[ResourceRecord, ...]
    authority: tuple[ResourceRecord, ...]
    rcode: int
    upstream_version: int
    attribution: tuple[str, ...]


def _chain_error(message: str, **details: object) -> BusinessError:
    return BusinessError("E_CHAIN", message, details=dict(details))


def validate_upstream_response(
    requested: Question,
    response: DNSMessage,
    *,
    expected_transaction_id: int,
    max_cname_hops: int = 8,
) -> UpstreamResult:
    if response.header.transaction_id != expected_transaction_id:
        raise _chain_error(
            "upstream transaction ID mismatch",
            expected=expected_transaction_id,
            actual=response.header.transaction_id,
        )
    if not response.header.is_response:
        raise _chain_error("upstream packet does not have the QR response bit")
    if (response.header.flags >> 11) & 0xF:
        raise _chain_error("upstream response uses a non-query opcode")
    if response.header.qdcount != 1 or len(response.questions) != 1:
        raise _chain_error("upstream response must echo exactly one question")
    echoed = response.questions[0]
    if question_signature(echoed) != question_signature(requested):
        raise _chain_error("upstream question echo differs from the forwarded question")
    rcode = response.header.rcode
    if rcode not in (RCODE_NOERROR, RCODE_NXDOMAIN):
        raise _chain_error("upstream returned an unsupported response code", rcode=rcode)
    for record in response.answers + response.authority + response.additional:
        if record.rclass != CLASS_IN:
            raise _chain_error("upstream returned a non-IN record", record=record.canonical())
        if record.ttl < 0:
            raise _chain_error("upstream returned a negative TTL", record=record.canonical())

    answers = response.answers
    attribution: list[str] = []
    if requested.qtype == TYPE_BY_NAME["CNAME"]:
        for record in answers:
            if record.name != requested.name or record.rtype != TYPE_BY_NAME["CNAME"]:
                raise _chain_error("malformed CNAME answer chain", record=record.canonical())
        if len(answers) > 1:
            raise _chain_error("multiple CNAME records in direct CNAME answer")
    else:
        expected_name = requested.name
        seen: set[str] = {expected_name}
        hops = 0
        final_seen = False
        for record in answers:
            if final_seen:
                if record.name == expected_name and record.rtype == requested.qtype:
                    continue
                raise _chain_error("record appears after the terminal answer", record=record.canonical())
            if record.name != expected_name:
                raise _chain_error(
                    "answer owner does not follow the question/CNAME chain",
                    expected=expected_name,
                    actual=record.name,
                )
            if record.rtype == TYPE_BY_NAME["CNAME"]:
                target = record.target
                assert target is not None
                hops += 1
                if hops > max_cname_hops:
                    raise _chain_error("upstream CNAME hop limit exceeded", limit=max_cname_hops)
                if target in seen:
                    raise _chain_error("upstream CNAME loop", target=target)
                seen.add(target)
                attribution.append(f"cname={record.name} target={target} hop={hops}")
                expected_name = target
            elif record.rtype == requested.qtype:
                final_seen = True
                attribution.append(f"terminal={record.name} type={record.type_name}")
            else:
                raise _chain_error(
                    "answer type is inconsistent with the requested type chain",
                    expected=requested.type_name,
                    actual=record.type_name,
                    owner=record.name,
                )
        if rcode == RCODE_NOERROR and answers and not final_seen:
            raise _chain_error("positive upstream answer chain has no terminal answer")
        if rcode == RCODE_NXDOMAIN and final_seen:
            raise _chain_error("NXDOMAIN response contains a terminal positive answer")

    serials: list[int] = []
    for record in response.authority + response.additional:
        if record.rtype == TYPE_BY_NAME["SOA"] and isinstance(record.rdata, SOAData):
            serials.append(record.rdata.serial)
    if len(set(serials)) > 1:
        raise _chain_error("upstream response contains inconsistent SOA serials", serials=sorted(set(serials)))
    upstream_version = serials[0] if serials else 0
    if not answers and rcode == RCODE_NOERROR:
        attribution.append("negative=nodata")
    if rcode == RCODE_NXDOMAIN:
        attribution.append("negative=nxdomain")
    attribution.append(f"upstream_version={upstream_version}")
    return UpstreamResult(
        message=response,
        answers=answers,
        authority=response.authority,
        rcode=rcode,
        upstream_version=upstream_version,
        attribution=tuple(attribution),
    )


class UpstreamClient:
    def __init__(self, host: str, port: int, *, timeout: float = 2.0, max_cname_hops: int = 8) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_cname_hops = max_cname_hops

    def exchange(self, payload: bytes) -> bytes:
        require_loopback_host(self.host)
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                sock.settimeout(self.timeout)
                sock.sendall(frame_tcp(payload))
                return receive_tcp_message(sock)
        except (OSError, ConnectionError, struct.error) as exc:
            raise BusinessError(
                "E_UPSTREAM",
                f"upstream TCP resolver is unavailable: {exc}",
                details={"host": self.host, "port": self.port},
            ) from exc

    def fetch(self, question: Question, transaction_id: int) -> UpstreamResult:
        query = encode_message(transaction_id, RD, (question,))
        raw = self.exchange(query)
        try:
            response = parse_message(raw)
        except BusinessError as exc:
            raise _chain_error("upstream response wire is invalid", cause=exc.render()) from exc
        return validate_upstream_response(
            question,
            response,
            expected_transaction_id=transaction_id,
            max_cname_hops=self.max_cname_hops,
        )

    def probe_version(self, origin: str) -> int:
        question = Question(normalize_name(origin), TYPE_BY_NAME["SOA"], CLASS_IN)
        result = self.fetch(question, 0x5A5A)
        return result.upstream_version


class _UpstreamHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: UpstreamTCPServer = self.server  # type: ignore[assignment]
        self.request.settimeout(5.0)
        while True:
            try:
                raw = receive_tcp_message(self.request)
            except (ConnectionError, OSError, struct.error):
                return
            try:
                response = server.simulator.build_response(raw)
            except Exception:
                try:
                    response = build_error_response(raw, RCODE_SERVFAIL)
                except Exception:
                    return
            try:
                self.request.sendall(frame_tcp(response))
            except OSError:
                return


class UpstreamTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], simulator: "UpstreamSimulator") -> None:
        self.simulator = simulator
        super().__init__(address, _UpstreamHandler)


class UpstreamSimulator:
    """A deterministic loopback TCP resolver backed by a local JSON zone."""

    def __init__(
        self,
        zone_path: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 15354,
        mode: str = "valid",
        conflict_target: str = "www.lab.test.",
        conflict_address: str = "203.0.113.66",
    ) -> None:
        self.zone_path = Path(zone_path)
        self.host = host
        self.port = port
        self.mode = mode
        if mode not in ("valid", "bad-txid", "bad-question", "bad-chain"):
            raise BusinessError("E_ARGUMENT", f"unknown upstream simulator mode: {mode}")
        manager = ZoneManager()
        self.zone: ZoneImage = manager.load_file(self.zone_path).image
        self.resolver = ZoneResolver(max_cname_hops=8)
        self.conflict_target = normalize_name(conflict_target)
        self.conflict_address = conflict_address
        self._server: UpstreamTCPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def version(self) -> int:
        if isinstance(self.zone.soa.rdata, SOAData):
            return self.zone.soa.rdata.serial
        return 0

    def start(self) -> None:
        if self._server is not None:
            return
        check_tcp_port(self.host, self.port)
        try:
            self._server = UpstreamTCPServer((self.host, self.port), self)
        except OSError as exc:
            raise BusinessError(
                "E_PORT",
                f"cannot bind upstream TCP port: {exc}",
                details={"host": self.host, "port": self.port},
            ) from exc
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"upstream-{self.port}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._server = None
        self._thread = None

    def __enter__(self) -> "UpstreamSimulator":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.stop()

    def _resolve(self, question: Question) -> tuple[int, tuple[ResourceRecord, ...], tuple[ResourceRecord, ...]]:
        if self.zone.is_covered(question.name):
            try:
                resolution = self.resolver.resolve(self.zone, question.name, question.qtype)
            except BusinessError:
                return RCODE_SERVFAIL, (), (self.zone.soa,)
            authority = resolution.authority or (self.zone.soa,)
            answers = list(resolution.answers)
            if resolution.incomplete_target is not None and question.qtype == TYPE_BY_NAME["A"]:
                # The deterministic simulator pretends that its recursive
                # resolver resolved the external target and returned a stale
                # local-lab address. The service must arbitrate in favour of
                # its authoritative lab.test data.
                answers.append(
                    ResourceRecord(
                        resolution.incomplete_target,
                        TYPE_BY_NAME["A"],
                        CLASS_IN,
                        45,
                        self.conflict_address,
                    )
                )
            return resolution.rcode, tuple(answers), authority
        return RCODE_NXDOMAIN, (), (self.zone.soa,)

    def build_response(self, raw_query: bytes) -> bytes:
        query = parse_message(raw_query)
        if query.header.is_response or query.header.qdcount != 1:
            return build_error_response(raw_query, 1)
        question = query.questions[0]
        transaction_id = query.header.transaction_id
        rcode, answers, authority = self._resolve(question)
        answers = list(answers)
        authority = list(authority)

        if question.name == self.conflict_target and question.qtype == TYPE_BY_NAME["A"]:
            cname = ResourceRecord(
                question.name,
                TYPE_BY_NAME["CNAME"],
                CLASS_IN,
                60,
                self.conflict_target,
            )
            conflict = ResourceRecord(
                self.conflict_target,
                TYPE_BY_NAME["A"],
                CLASS_IN,
                60,
                self.conflict_address,
            )
            rcode, answers = RCODE_NOERROR, [cname, conflict]
            authority = [self.zone.soa]

        if self.mode == "bad-txid":
            transaction_id = (transaction_id + 1) & 0xFFFF
        if self.mode == "bad-question":
            question = Question("wrong.upstream.test.", question.qtype, question.qclass)
        if self.mode == "bad-chain" and answers:
            answers[0] = ResourceRecord(
                "wrong.upstream.test.",
                answers[0].rtype,
                answers[0].rclass,
                answers[0].ttl,
                answers[0].rdata,
            )
        flags = QR | AA | RA | (query.header.flags & RD) | (rcode & 0xF)
        return encode_message(transaction_id, flags, (question,), tuple(answers), tuple(authority))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Loopback deterministic DNS upstream simulator")
    parser.add_argument("--zone", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15354)
    parser.add_argument("--mode", choices=("valid", "bad-txid", "bad-question", "bad-chain"), default="valid")
    args = parser.parse_args(list(argv) if argv is not None else None)
    simulator = UpstreamSimulator(args.zone, host=args.host, port=args.port, mode=args.mode)
    simulator.start()
    print(
        f"UPSTREAM_READY host={simulator.host} port={simulator.port} "
        f"version={simulator.version} mode={simulator.mode}",
        flush=True,
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        simulator.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
