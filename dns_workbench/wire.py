"""DNS 报文线格式（RFC 1035/2671/3225 子集）编解码。

仅实现诊断工作台需要的确定性子集：
- 头部 12 字节与各计数字段；
- 问题段、应答/权威/附加资源记录段；
- 名称压缩指针展开，显式防御指针环、指针向前引用与越界偏移；
- EDNS(0) OPT 伪记录（type=41）与 DNSSEC OK(DO) 标志按普通 RR 透传；
- UDP 512 字节截断置 TC，TCP 使用 2 字节长度前缀且不截断。
"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass, field

from .errors import WireError

UDP_PAYLOAD_LIMIT = 512
POINTER_MASK = 0xC0
POINTER_VALUE = 0xC0

TYPE_A = 1
TYPE_NS = 2
TYPE_CNAME = 5
TYPE_SOA = 6
TYPE_PTR = 12
TYPE_MX = 15
TYPE_TXT = 16
TYPE_AAAA = 28
TYPE_OPT = 41

TYPE_NAMES = {
    TYPE_A: "A",
    TYPE_NS: "NS",
    TYPE_CNAME: "CNAME",
    TYPE_SOA: "SOA",
    TYPE_PTR: "PTR",
    TYPE_MX: "MX",
    TYPE_TXT: "TXT",
    TYPE_AAAA: "AAAA",
    TYPE_OPT: "OPT",
}

CLASS_IN = 1


def type_name(rtype: int) -> str:
    return TYPE_NAMES.get(rtype, f"TYPE{rtype}")


@dataclass
class Question:
    qname: str
    qtype: int
    qclass: int = CLASS_IN

    def key(self) -> tuple[str, int]:
        return (self.qname, self.qtype)


@dataclass
class ResourceRecord:
    name: str
    rtype: int
    rclass: int
    ttl: int
    rdata: bytes = b""
    rdata_text: str = ""

    def value_summary(self) -> str:
        """稳定的记录值摘要，供缓存账本与展示使用。"""
        if self.rdata_text:
            return f"{type_name(self.rtype)} {self.rdata_text}"
        return f"{type_name(self.rtype)} 0x{self.rdata.hex()}"


@dataclass
class DnsMessage:
    txn_id: int
    flags: int
    questions: list[Question] = field(default_factory=list)
    answers: list[ResourceRecord] = field(default_factory=list)
    authority: list[ResourceRecord] = field(default_factory=list)
    additional: list[ResourceRecord] = field(default_factory=list)

    @property
    def qr(self) -> bool:
        return bool(self.flags & 0x8000)

    @property
    def opcode(self) -> int:
        return (self.flags >> 11) & 0x0F

    @property
    def aa(self) -> bool:
        return bool(self.flags & 0x0400)

    @property
    def tc(self) -> bool:
        return bool(self.flags & 0x0200)

    @property
    def rd(self) -> bool:
        return bool(self.flags & 0x0100)

    @property
    def ra(self) -> bool:
        return bool(self.flags & 0x0080)

    @property
    def rcode(self) -> int:
        return self.flags & 0x000F

    def first_question(self) -> Question | None:
        return self.questions[0] if self.questions else None

    def opt_record(self) -> ResourceRecord | None:
        for record in self.additional:
            if record.rtype == TYPE_OPT:
                return record
        return None

    def dnssec_ok(self) -> bool:
        opt = self.opt_record()
        if opt is None:
            return False
        return bool(struct.unpack("!H", opt.rdata[:2])[0] & 0x8000) if len(opt.rdata) >= 4 else False


def _decode_name(data: bytes, offset: int) -> tuple[str, int]:
    """展开可能含压缩指针的名称，返回 (小写名称, 名称结束后的下一偏移)。

    防御：越界标签/长度、指针越界、指针向前引用、指针环。
    """
    labels: list[str] = []
    original_offset = offset
    cursor = offset
    end_of_name: int | None = None
    seen_pointers: set[int] = set()
    data_len = len(data)

    while True:
        if cursor >= data_len:
            raise WireError(
                "名称解析越界：标签长度超出报文",
                offset=cursor,
                context={"start": original_offset},
            )
        length = data[cursor]
        if length == 0:
            if end_of_name is None:
                end_of_name = cursor + 1
            break
        kind = length & POINTER_MASK
        if kind == POINTER_VALUE:
            if cursor + 1 >= data_len:
                raise WireError("压缩指针不完整：缺少第二个字节", offset=cursor)
            pointer = ((length & 0x3F) << 8) | data[cursor + 1]
            if end_of_name is None:
                end_of_name = cursor + 2
            if pointer >= data_len:
                raise WireError(
                    "压缩指针越界",
                    offset=cursor,
                    context={"pointer": pointer, "length": data_len},
                )
            if pointer >= original_offset:
                raise WireError(
                    "压缩指针必须指向先前出现的名称（禁止向前/自引用）",
                    offset=cursor,
                    context={"pointer": pointer, "name_start": original_offset},
                )
            if pointer in seen_pointers:
                raise WireError(
                    "检测到压缩指针环",
                    offset=cursor,
                    context={"pointer": pointer, "chain": sorted(seen_pointers)},
                )
            seen_pointers.add(pointer)
            cursor = pointer
            continue
        if kind != 0:
            raise WireError(
                "名称标签含保留位（非法标签类型 0b10/0b01）",
                offset=cursor,
                context={"label_byte": length},
            )
        label_start = cursor + 1
        label_end = label_start + length
        if length > 63 or label_end > data_len:
            raise WireError(
                "普通标签越界或长度非法",
                offset=cursor,
                context={"length": length},
            )
        try:
            labels.append(data[label_start:label_end].decode("ascii").lower())
        except UnicodeDecodeError as exc:
            raise WireError("名称标签非 ASCII", offset=cursor) from exc
        cursor = label_end

    assert end_of_name is not None
    return ".".join(labels), end_of_name


def _decode_rdata(rtype: int, rdata: bytes, data: bytes, rdata_offset: int) -> str:
    try:
        if rtype == TYPE_A:
            if len(rdata) != 4:
                raise ValueError("A 记录长度必须为 4")
            return ".".join(str(octet) for octet in rdata)
        if rtype == TYPE_AAAA:
            if len(rdata) != 16:
                raise ValueError("AAAA 记录长度必须为 16")
            return ":".join(f"{rdata[i]:02x}{rdata[i + 1]:02x}" for i in range(0, 16, 2))
        if rtype in (TYPE_NS, TYPE_CNAME, TYPE_PTR):
            name, _ = _decode_name(data, rdata_offset)
            return name
        if rtype == TYPE_MX:
            preference = struct.unpack("!H", rdata[:2])[0]
            name, _ = _decode_name(data, rdata_offset + 2)
            return f"{preference} {name}"
        if rtype == TYPE_SOA:
            offset = rdata_offset
            mname, offset = _decode_name(data, offset)
            rname, offset = _decode_name(data, offset)
            tail_start = offset - rdata_offset
            serial, refresh, retry, expire, minimum = struct.unpack("!IIIII", rdata[tail_start:tail_start + 20])
            return f"{mname} {rname} {serial} {refresh} {retry} {expire} {minimum}"
        if rtype == TYPE_TXT:
            chunks: list[str] = []
            cursor = 0
            while cursor < len(rdata):
                length = rdata[cursor]
                cursor += 1
                chunks.append(rdata[cursor:cursor + length].decode("ascii"))
                cursor += length
            return " ".join(chunks)
    except (WireError, ValueError, struct.error, UnicodeDecodeError):
        return ""
    return ""


def parse_message(data: bytes) -> DnsMessage:
    if len(data) < 12:
        raise WireError("报文短于 12 字节头部", offset=len(data))
    txn_id, flags, qdcount, ancount, nscount, arcount = struct.unpack("!HHHHHH", data[:12])
    message = DnsMessage(txn_id=txn_id, flags=flags)
    cursor = 12
    try:
        for _ in range(qdcount):
            qname, cursor = _decode_name(data, cursor)
            if cursor + 4 > len(data):
                raise WireError("问题段 QTYPE/QCLASS 越界", offset=cursor)
            qtype, qclass = struct.unpack("!HH", data[cursor:cursor + 4])
            cursor += 4
            message.questions.append(Question(qname, qtype, qclass))
        for count, bucket in (
            (ancount, message.answers),
            (nscount, message.authority),
            (arcount, message.additional),
        ):
            for _ in range(count):
                record, cursor = _parse_rr(data, cursor)
                bucket.append(record)
    except struct.error as exc:
        raise WireError("报文结构解包失败", offset=cursor) from exc
    if cursor != len(data):
        raise WireError("报文存在未消费的尾部字节", offset=cursor, context={"parsed": cursor, "length": len(data)})
    return message


def _parse_rr(data: bytes, offset: int) -> tuple[ResourceRecord, int]:
    name, offset = _decode_name(data, offset)
    if offset + 10 > len(data):
        raise WireError("资源记录固定 10 字节越界", offset=offset)
    rtype, rclass, ttl, rdlength = struct.unpack("!HHIH", data[offset:offset + 10])
    offset += 10
    if offset + rdlength > len(data):
        raise WireError("资源记录 RDATA 越界", offset=offset, context={"rdlength": rdlength})
    rdata = data[offset:offset + rdlength]
    ttl_signed = struct.unpack("!i", struct.pack("!I", ttl))[0]
    if ttl_signed < 0:
        raise WireError("TTL 不得为负", offset=offset - 10, context={"ttl": ttl_signed})
    text = _decode_rdata(rtype, rdata, data, offset)
    record = ResourceRecord(name, rtype, rclass, ttl, bytes(rdata), text)
    return record, offset + rdlength


class _NameEncoder:
    """无状态名称压缩器：偏移表在整条报文编码期间共享，偏移由调用层显式传入。"""

    def __init__(self) -> None:
        self._offsets: dict[str, int] = {}

    def encode_at(self, name: str, cursor: int) -> tuple[bytes, int]:
        """返回 (编码字节, 编码结束后的绝对偏移)。"""
        normalized = name.rstrip(".").lower()
        labels = normalized.split(".") if normalized else []
        output = bytearray()
        position = cursor
        for index in range(len(labels)):
            suffix = ".".join(labels[index:])
            if suffix in self._offsets:
                output.extend(struct.pack("!H", 0xC000 | self._offsets[suffix]))
                return bytes(output), position + 2
            self._offsets[suffix] = position
            label = labels[index].encode("ascii")
            if len(label) > 63:
                raise WireError("标签长度超过 63", offset=position)
            output.append(len(label))
            output.extend(label)
            position += 1 + len(label)
        output.append(0)
        return bytes(output), position + 1


def _encode_name(name: str, encoder: "_NameEncoder", cursor: int) -> tuple[bytes, int]:
    return encoder.encode_at(name, cursor)


def _encode_rdata(record: ResourceRecord, encoder: "_NameEncoder", cursor: int) -> tuple[bytes, int]:
    rtype = record.rtype
    if rtype == TYPE_OPT:
        return record.rdata, cursor
    if rtype in (TYPE_NS, TYPE_CNAME, TYPE_PTR):
        return _encode_name(record.rdata_text or record.name, encoder, cursor)
    if rtype == TYPE_MX:
        preference, _, target = record.rdata_text.partition(" ")
        header = struct.pack("!H", int(preference))
        tail, cursor = _encode_name(target, encoder, cursor + 2)
        return header + tail, cursor
    if rtype == TYPE_SOA:
        tokens = record.rdata_text.split()
        if len(tokens) != 7:
            raise WireError("SOA RDATA 文本必须含 7 个字段")
        output = bytearray()
        chunk, cursor = _encode_name(tokens[0], encoder, cursor)
        output.extend(chunk)
        chunk, cursor = _encode_name(tokens[1], encoder, cursor)
        output.extend(chunk)
        fixed = struct.pack("!IIIII", *(int(token) for token in tokens[2:]))
        output.extend(fixed)
        return bytes(output), cursor + len(fixed)
    if rtype == TYPE_TXT:
        output = bytearray()
        for chunk in record.rdata_text.split():
            encoded = chunk.encode("ascii")
            output.append(len(encoded))
            output.extend(encoded)
        return bytes(output), cursor + len(output)
    if record.rdata:
        return record.rdata, cursor + len(record.rdata)
    if rtype == TYPE_A:
        raw = bytes(int(part) for part in record.rdata_text.split("."))
        return raw, cursor + len(raw)
    if rtype == TYPE_AAAA:
        raw = b"".join(bytes.fromhex(pair) for pair in record.rdata_text.split(":"))
        return raw, cursor + len(raw)
    raise WireError(f"类型 {rtype} 缺少可编码的 RDATA")


def _encode_rr(record: ResourceRecord, encoder: "_NameEncoder", cursor: int) -> tuple[bytes, int]:
    output = bytearray()
    chunk, cursor = _encode_name(record.name, encoder, cursor)
    output.extend(chunk)
    fixed = struct.pack("!HHIH", record.rtype, record.rclass, record.ttl, 0)
    output.extend(fixed)
    cursor += len(fixed)
    rdata, cursor = _encode_rdata(record, encoder, cursor)
    output[-2:] = struct.pack("!H", len(rdata))
    output.extend(rdata)
    return bytes(output), cursor


def encode_message(message: DnsMessage) -> bytes:
    encoder = _NameEncoder()
    header = struct.pack(
        "!HHHHHH",
        message.txn_id,
        message.flags,
        len(message.questions),
        len(message.answers),
        len(message.authority),
        len(message.additional),
    )
    output = bytearray(header)
    cursor = len(output)
    for question in message.questions:
        chunk, cursor = _encode_name(question.qname, encoder, cursor)
        output.extend(chunk)
        fixed = struct.pack("!HH", question.qtype, question.qclass)
        output.extend(fixed)
        cursor += len(fixed)
    for bucket in (message.answers, message.authority, message.additional):
        for record in bucket:
            chunk, cursor = _encode_rr(record, encoder, cursor)
            output.extend(chunk)
    return bytes(output)


def build_response_flags(*, rcode: int, aa: bool, tc: bool = False, ra: bool = True) -> int:
    flags = 0x8000 | (rcode & 0x000F)
    if aa:
        flags |= 0x0400
    if tc:
        flags |= 0x0200
    if ra:
        flags |= 0x0080
    return flags


def truncate_for_udp(message: DnsMessage, limit: int = UDP_PAYLOAD_LIMIT) -> tuple[bytes, bool]:
    """确定性 UDP 截断：问题段始终保留，资源记录按段内顺序逐条加入直到超限。

    超限时丢弃剩余记录并置 TC，保证结果不超过 512 字节。
    """
    full = encode_message(message)
    if len(full) <= limit:
        return full, False

    def rebuild(answers: list[ResourceRecord], authority: list[ResourceRecord], additional: list[ResourceRecord]) -> bytes:
        truncated = DnsMessage(
            txn_id=message.txn_id,
            flags=message.flags | 0x0200,
            questions=message.questions,
            answers=answers,
            authority=authority,
            additional=additional,
        )
        return encode_message(truncated)

    kept_answers: list[ResourceRecord] = []
    for record in message.answers:
        candidate = rebuild(kept_answers + [record], [], [])
        if len(candidate) > limit:
            return rebuild(kept_answers, [], []), True
        kept_answers.append(record)
    kept_authority: list[ResourceRecord] = []
    for record in message.authority:
        candidate = rebuild(kept_answers, kept_authority + [record], [])
        if len(candidate) > limit:
            return rebuild(kept_answers, kept_authority, []), True
        kept_authority.append(record)
    kept_additional: list[ResourceRecord] = []
    for record in message.additional:
        candidate = rebuild(kept_answers, kept_authority, kept_additional + [record])
        if len(candidate) > limit:
            return rebuild(kept_answers, kept_authority, kept_additional), True
        kept_additional.append(record)
    return rebuild(kept_answers, kept_authority, kept_additional), True


def encode_tcp_frame(message: DnsMessage) -> bytes:
    payload = encode_message(message)
    return struct.pack("!H", len(payload)) + payload


def read_tcp_frame(sock: socket.socket) -> bytes:
    header = _recv_exactly(sock, 2)
    if header is None:
        raise WireError("TCP 流在长度前缀处结束", offset=0)
    (length,) = struct.unpack("!H", header)
    payload = _recv_exactly(sock, length)
    if payload is None:
        raise WireError("TCP 报文体不完整", offset=2, context={"expected": length})
    return payload


def _recv_exactly(sock: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
