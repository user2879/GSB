"""Strict DNS wire-format codec with bounded compression-pointer handling."""

from __future__ import annotations

import ipaddress
import struct
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from .errors import BusinessError, wire_error
from .names import normalize_name


TYPE_BY_NAME = {"A": 1, "NS": 2, "CNAME": 5, "SOA": 6, "AAAA": 28}
NAME_BY_TYPE = {value: key for key, value in TYPE_BY_NAME.items()}
CLASS_IN = 1

QR = 0x8000
AA = 0x0400
TC = 0x0200
RD = 0x0100
RA = 0x0080
AD = 0x0020
CD = 0x0010
RCODE_MASK = 0x000F

RCODE_NOERROR = 0
RCODE_FORMERR = 1
RCODE_SERVFAIL = 2
RCODE_NXDOMAIN = 3
RCODE_NOTIMP = 4

RCODE_NAMES = {
    RCODE_NOERROR: "NOERROR",
    RCODE_FORMERR: "FORMERR",
    RCODE_SERVFAIL: "SERVFAIL",
    RCODE_NXDOMAIN: "NXDOMAIN",
    RCODE_NOTIMP: "NOTIMP",
}


@dataclass(frozen=True)
class SOAData:
    mname: str
    rname: str
    serial: int
    refresh: int
    retry: int
    expire: int
    minimum: int

    def text(self) -> str:
        return (
            f"{self.mname} {self.rname} {self.serial} {self.refresh} "
            f"{self.retry} {self.expire} {self.minimum}"
        )


@dataclass(frozen=True)
class DNSHeader:
    transaction_id: int
    flags: int
    qdcount: int
    ancount: int
    nscount: int
    arcount: int

    @property
    def rcode(self) -> int:
        return self.flags & RCODE_MASK

    @property
    def is_response(self) -> bool:
        return bool(self.flags & QR)


@dataclass(frozen=True)
class Question:
    name: str
    qtype: int
    qclass: int = CLASS_IN

    @property
    def type_name(self) -> str:
        return NAME_BY_TYPE.get(self.qtype, f"TYPE{self.qtype}")


@dataclass(frozen=True)
class ResourceRecord:
    name: str
    rtype: int
    rclass: int
    ttl: int
    rdata: str | SOAData | bytes

    @property
    def type_name(self) -> str:
        return NAME_BY_TYPE.get(self.rtype, f"TYPE{self.rtype}")

    @property
    def target(self) -> str | None:
        if self.rtype in (TYPE_BY_NAME["CNAME"], TYPE_BY_NAME["NS"]):
            assert isinstance(self.rdata, str)
            return self.rdata
        return None

    def rdata_text(self) -> str:
        if isinstance(self.rdata, bytes):
            return self.rdata.hex()
        if isinstance(self.rdata, SOAData):
            return self.rdata.text()
        return self.rdata

    def with_ttl(self, ttl: int) -> "ResourceRecord":
        return replace(self, ttl=max(0, int(ttl)))

    def canonical(self) -> str:
        return f"{self.name}|{self.type_name}|{self.rclass}|{self.ttl}|{self.rdata_text()}"


@dataclass(frozen=True)
class DNSMessage:
    header: DNSHeader
    questions: tuple[Question, ...]
    answers: tuple[ResourceRecord, ...]
    authority: tuple[ResourceRecord, ...]
    additional: tuple[ResourceRecord, ...]


def type_to_int(type_name_or_int: str | int) -> int:
    if isinstance(type_name_or_int, int):
        if type_name_or_int <= 0:
            raise BusinessError("E_ARGUMENT", "record type must be positive")
        return type_name_or_int
    upper = type_name_or_int.upper()
    if upper not in TYPE_BY_NAME:
        raise BusinessError("E_ARGUMENT", f"unsupported record type: {type_name_or_int}")
    return TYPE_BY_NAME[upper]


def _decode_name(data: bytes, start: int) -> tuple[str, int]:
    if start < 0 or start >= len(data):
        raise wire_error("name offset is outside the packet", start)
    labels: list[str] = []
    cursor = start
    next_offset: int | None = None
    jumps = 0
    visited_pointer_positions: set[int] = set()
    expanded_wire_len = 1

    while True:
        if cursor >= len(data):
            raise wire_error("name runs past the end of the packet", cursor)
        length = data[cursor]
        if length & 0xC0 == 0xC0:
            pointer_pos = cursor
            if cursor + 1 >= len(data):
                raise wire_error("truncated compression pointer", cursor)
            target = ((length & 0x3F) << 8) | data[cursor + 1]
            if target >= pointer_pos:
                raise wire_error(
                    "compression pointer must point strictly before its own position",
                    pointer_pos,
                    pointer_target=target,
                )
            if target >= len(data):
                raise wire_error("compression pointer target is outside the packet", pointer_pos, pointer_target=target)
            if target in visited_pointer_positions:
                raise wire_error("compression pointer loop", pointer_pos, pointer_target=target)
            visited_pointer_positions.add(target)
            jumps += 1
            if jumps > 64:
                raise wire_error("too many compression pointer jumps", pointer_pos)
            if next_offset is None:
                next_offset = cursor + 2
            cursor = target
            continue
        if length & 0xC0:
            raise wire_error("reserved label-length bits are set", cursor)
        if length == 0:
            if next_offset is None:
                next_offset = cursor + 1
            break
        if length > 63:
            raise wire_error("label is longer than 63 bytes", cursor)
        label_start = cursor + 1
        label_end = label_start + length
        if label_end > len(data):
            raise wire_error("label runs past the end of the packet", cursor)
        raw_label = data[label_start:label_end]
        try:
            label = raw_label.decode("ascii")
        except UnicodeDecodeError as exc:
            raise wire_error("non-ASCII label in local laboratory wire sample", label_start) from exc
        labels.append(label)
        expanded_wire_len += 1 + length
        if expanded_wire_len > 255:
            raise wire_error("expanded name exceeds 255 wire bytes", start)
        cursor = label_end

    assert next_offset is not None
    if not labels:
        return ".", next_offset
    try:
        return normalize_name(".".join(labels) + "."), next_offset
    except ValueError as exc:
        raise wire_error(f"invalid DNS name: {exc}", start) from exc


class _Parser:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def require(self, offset: int, length: int, what: str) -> None:
        if offset < 0 or length < 0 or offset + length > len(self.data):
            raise wire_error(f"truncated {what}", offset)

    def header(self) -> DNSHeader:
        self.require(0, 12, "DNS header")
        tid, flags, qd, an, ns, ar = struct.unpack_from("!HHHHHH", self.data, 0)
        if qd > 1000 or an > 4096 or ns > 4096 or ar > 4096:
            raise wire_error("unreasonable section count", 4, qdcount=qd, ancount=an, nscount=ns, arcount=ar)
        return DNSHeader(tid, flags, qd, an, ns, ar)

    def question(self, offset: int) -> tuple[Question, int]:
        name, offset = _decode_name(self.data, offset)
        self.require(offset, 4, "question fields")
        qtype, qclass = struct.unpack_from("!HH", self.data, offset)
        return Question(name, qtype, qclass), offset + 4

    def rr(self, offset: int) -> tuple[ResourceRecord, int]:
        name, offset = _decode_name(self.data, offset)
        self.require(offset, 10, "resource-record header")
        rtype, rclass, ttl, rdlength = struct.unpack_from("!HHIH", self.data, offset)
        if ttl > 0x7FFFFFFF:
            raise wire_error("record TTL has the reserved high bit set", offset + 4)
        rdata_start = offset + 10
        rdata_end = rdata_start + rdlength
        self.require(rdata_start, rdlength, "record data")

        if rtype == TYPE_BY_NAME["A"]:
            if rdlength != 4:
                raise wire_error("A record data length must be 4", offset + 8)
            rdata: str | SOAData | bytes = str(ipaddress.IPv4Address(self.data[rdata_start:rdata_end]))
        elif rtype == TYPE_BY_NAME["AAAA"]:
            if rdlength != 16:
                raise wire_error("AAAA record data length must be 16", offset + 8)
            rdata = str(ipaddress.IPv6Address(self.data[rdata_start:rdata_end]))
        elif rtype in (TYPE_BY_NAME["CNAME"], TYPE_BY_NAME["NS"]):
            target, consumed = _decode_name(self.data, rdata_start)
            if consumed != rdata_end:
                raise wire_error("compressed name does not exactly fill RDATA", rdata_start)
            rdata = target
        elif rtype == TYPE_BY_NAME["SOA"]:
            mname, cursor = _decode_name(self.data, rdata_start)
            rname, cursor = _decode_name(self.data, cursor)
            if cursor + 20 != rdata_end:
                raise wire_error("SOA data length is inconsistent", cursor)
            serial, refresh, retry, expire, minimum = struct.unpack_from("!IIIII", self.data, cursor)
            rdata = SOAData(mname, rname, serial, refresh, retry, expire, minimum)
        else:
            rdata = self.data[rdata_start:rdata_end]
        return ResourceRecord(name, rtype, rclass, ttl, rdata), rdata_end

    def message(self) -> DNSMessage:
        header = self.header()
        cursor = 12
        questions: list[Question] = []
        for _ in range(header.qdcount):
            question, cursor = self.question(cursor)
            questions.append(question)
        answers: list[ResourceRecord] = []
        authority: list[ResourceRecord] = []
        additional: list[ResourceRecord] = []
        for target, count in (
            (answers, header.ancount),
            (authority, header.nscount),
            (additional, header.arcount),
        ):
            for _ in range(count):
                record, cursor = self.rr(cursor)
                target.append(record)
        if cursor != len(self.data):
            raise wire_error("trailing bytes after DNS message", cursor, trailing=len(self.data) - cursor)
        return DNSMessage(header, tuple(questions), tuple(answers), tuple(authority), tuple(additional))


def parse_message(data: bytes) -> DNSMessage:
    if len(data) < 2:
        raise wire_error("packet is too short to contain a transaction ID", 0, length=len(data))
    return _Parser(data).message()


def peek_transaction_id(data: bytes) -> int | None:
    if len(data) < 2:
        return None
    return struct.unpack_from("!H", data, 0)[0]


def _write_name(buffer: bytearray, name: str, compression: dict[str, int] | None) -> None:
    normalized = normalize_name(name)
    if normalized == ".":
        buffer.append(0)
        return
    labels = normalized[:-1].split(".")
    for index, label in enumerate(labels):
        suffix = ".".join(labels[index:]) + "."
        current = len(buffer)
        if compression is not None:
            target = compression.get(suffix)
            if target is not None and target < current and target <= 0x3FFF:
                buffer.extend(struct.pack("!H", 0xC000 | target))
                return
            compression[suffix] = current
        encoded = label.encode("ascii")
        buffer.append(len(encoded))
        buffer.extend(encoded)
    buffer.append(0)


def _write_rr(buffer: bytearray, rr: ResourceRecord, compression: dict[str, int] | None) -> None:
    _write_name(buffer, rr.name, compression)
    rdata_length_at = len(buffer) + 8
    buffer.extend(struct.pack("!HHI", rr.rtype, rr.rclass, max(0, int(rr.ttl))))
    buffer.extend(b"\x00\x00")
    rdata_start = len(buffer)
    if rr.rtype == TYPE_BY_NAME["A"]:
        buffer.extend(ipaddress.IPv4Address(rr.rdata_text()).packed)  # type: ignore[arg-type]
    elif rr.rtype == TYPE_BY_NAME["AAAA"]:
        buffer.extend(ipaddress.IPv6Address(rr.rdata_text()).packed)  # type: ignore[arg-type]
    elif rr.rtype in (TYPE_BY_NAME["CNAME"], TYPE_BY_NAME["NS"]):
        _write_name(buffer, rr.rdata_text(), compression)
    elif rr.rtype == TYPE_BY_NAME["SOA"]:
        if not isinstance(rr.rdata, SOAData):
            raise TypeError("SOA record requires SOAData")
        _write_name(buffer, rr.rdata.mname, compression)
        _write_name(buffer, rr.rdata.rname, compression)
        buffer.extend(
            struct.pack(
                "!IIIII",
                rr.rdata.serial,
                rr.rdata.refresh,
                rr.rdata.retry,
                rr.rdata.expire,
                rr.rdata.minimum,
            )
        )
    elif isinstance(rr.rdata, bytes):
        buffer.extend(rr.rdata)
    else:
        raise TypeError(f"cannot encode RDATA for type {rr.type_name}")
    rdata_length = len(buffer) - rdata_start
    struct.pack_into("!H", buffer, rdata_length_at, rdata_length)


def encode_message(
    transaction_id: int,
    flags: int,
    questions: Sequence[Question] = (),
    answers: Sequence[ResourceRecord] = (),
    authority: Sequence[ResourceRecord] = (),
    additional: Sequence[ResourceRecord] = (),
    *,
    compress: bool = True,
) -> bytes:
    if not 0 <= transaction_id <= 0xFFFF:
        raise ValueError("transaction ID must fit in 16 bits")
    buffer = bytearray(
        struct.pack(
            "!HHHHHH",
            transaction_id,
            flags,
            len(questions),
            len(answers),
            len(authority),
            len(additional),
        )
    )
    compression: dict[str, int] | None = {} if compress else None
    for question in questions:
        _write_name(buffer, question.name, compression)
        buffer.extend(struct.pack("!HH", question.qtype, question.qclass))
    for records in (answers, authority, additional):
        for record in records:
            _write_rr(buffer, record, compression)
    return bytes(buffer)


def make_query(
    name: str,
    qtype: str | int,
    *,
    transaction_id: int = 0x1001,
    recursion_desired: bool = True,
) -> bytes:
    question = Question(normalize_name(name), type_to_int(qtype), CLASS_IN)
    flags = RD if recursion_desired else 0
    return encode_message(transaction_id, flags, (question,), compress=False)


def build_response(
    query: DNSMessage,
    *,
    answers: Iterable[ResourceRecord] = (),
    authority: Iterable[ResourceRecord] = (),
    additional: Iterable[ResourceRecord] = (),
    rcode: int = RCODE_NOERROR,
    authoritative: bool = False,
    truncated: bool = False,
) -> bytes:
    if query.header.is_response:
        raise ValueError("cannot build a response from a response")
    flags = QR | (query.header.flags & (RD | CD)) | RA | (rcode & RCODE_MASK)
    if authoritative:
        flags |= AA
    if truncated:
        flags |= TC
    return encode_message(
        query.header.transaction_id,
        flags,
        query.questions,
        tuple(answers),
        tuple(authority),
        tuple(additional),
    )


def build_error_response(
    data: bytes,
    rcode: int,
    *,
    question: Question | None = None,
    flags: int = 0,
) -> bytes:
    transaction_id = peek_transaction_id(data) or 0
    questions = (question,) if question is not None else ()
    response_flags = QR | RA | (flags & (RD | CD)) | (rcode & RCODE_MASK)
    return encode_message(transaction_id, response_flags, questions)


def maybe_truncate_udp(payload: bytes, limit: int = 512) -> tuple[bytes, bool]:
    if len(payload) <= limit:
        return payload, False
    message = parse_message(payload)
    truncated = encode_message(
        message.header.transaction_id,
        message.header.flags | TC,
        message.questions,
        (),
        (),
        (),
    )
    if len(truncated) > limit:
        truncated = encode_message(
            message.header.transaction_id,
            message.header.flags | TC,
            (),
            (),
            (),
            (),
        )
    return truncated, True


def frame_tcp(payload: bytes) -> bytes:
    if len(payload) > 0xFFFF:
        raise ValueError("TCP DNS payload exceeds 65535 bytes")
    return struct.pack("!H", len(payload)) + payload


def unframe_tcp(frame: bytes) -> bytes:
    if len(frame) < 2:
        raise wire_error("TCP DNS frame is missing its length prefix", 0)
    declared = struct.unpack_from("!H", frame, 0)[0]
    if declared != len(frame) - 2:
        raise wire_error("TCP DNS frame length does not match its payload", 0, declared=declared)
    return frame[2:]


def read_exact(sock, length: int) -> bytes:  # type: ignore[no-untyped-def]
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed the TCP stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_tcp_message(sock) -> bytes:  # type: ignore[no-untyped-def]
    prefix = read_exact(sock, 2)
    length = struct.unpack("!H", prefix)[0]
    return read_exact(sock, length)


def question_signature(question: Question) -> tuple[str, int, int]:
    return (question.name.lower(), question.qtype, question.qclass)
