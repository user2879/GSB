"""Small, strict DER encoder/decoder used for the local certificate family."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


class DERError(ValueError):
    pass


@dataclass(frozen=True)
class TLV:
    tag: int
    value: bytes
    start: int
    header_end: int
    end: int

    @property
    def raw(self) -> bytes:
        return b""


def encode_length(length: int) -> bytes:
    if length < 0:
        raise DERError("negative length")
    if length < 0x80:
        return bytes([length])
    raw = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(raw)]) + raw


def tlv(tag: int, value: bytes) -> bytes:
    if not (0 <= tag <= 0xFF):
        raise DERError("single-byte tags only")
    return bytes([tag]) + encode_length(len(value)) + value


def sequence(*items: bytes) -> bytes:
    return tlv(0x30, b"".join(items))


def set_of(*items: bytes) -> bytes:
    return tlv(0x31, b"".join(items))


def explicit(tag_number: int, value: bytes) -> bytes:
    if not (0 <= tag_number <= 30):
        raise DERError("explicit tag out of range")
    return tlv(0xA0 | tag_number, value)


def integer(value: int) -> bytes:
    if value < 0:
        raise DERError("only non-negative integers are encoded")
    if value == 0:
        raw = b"\x00"
    else:
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        if raw[0] & 0x80:
            raw = b"\x00" + raw
    return tlv(0x02, raw)


def boolean(value: bool) -> bytes:
    return tlv(0x01, b"\xff" if value else b"\x00")


def null() -> bytes:
    return b"\x05\x00"


def octet_string(value: bytes) -> bytes:
    return tlv(0x04, value)


def bit_string(value: bytes, unused_bits: int = 0) -> bytes:
    if not (0 <= unused_bits <= 7):
        raise DERError("unused bits out of range")
    return tlv(0x03, bytes([unused_bits]) + value)


def utf8_string(value: str) -> bytes:
    return tlv(0x0C, value.encode("utf-8"))


def printable_string(value: str) -> bytes:
    return tlv(0x13, value.encode("ascii"))


def _base128(value: int) -> bytes:
    if value < 0:
        raise DERError("negative OID component")
    parts = [value & 0x7F]
    value >>= 7
    while value:
        parts.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(parts))


def oid(value: str) -> bytes:
    parts = [int(p) for p in value.split(".")]
    if len(parts) < 2 or parts[0] not in (0, 1, 2):
        raise DERError("invalid OID")
    if parts[1] > 39 and parts[0] < 2:
        raise DERError("invalid second OID component")
    packed = _base128(parts[0] * 40 + parts[1])
    for part in parts[2:]:
        packed += _base128(part)
    return tlv(0x06, packed)


def algorithm_identifier(oid_value: str, with_null: bool = True) -> bytes:
    return sequence(oid(oid_value), null() if with_null else b"")


def utc_time(timestamp_ms: int) -> bytes:
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    if 1950 <= dt.year <= 2049:
        return tlv(0x17, dt.strftime("%y%m%d%H%M%SZ").encode("ascii"))
    return tlv(0x18, dt.strftime("%Y%m%d%H%M%SZ").encode("ascii"))


def name(cn: str) -> bytes:
    cn_oid = oid("2.5.4.3")
    return sequence(set_of(sequence(cn_oid, utf8_string(cn))))


def parse_tlv(data: bytes, offset: int = 0) -> tuple[TLV, int]:
    if offset >= len(data):
        raise DERError("truncated tag")
    start = offset
    tag = data[offset]
    offset += 1
    if tag & 0x1F == 0x1F:
        raise DERError("high-tag-number form is not supported")
    if offset >= len(data):
        raise DERError("truncated length")
    first = data[offset]
    offset += 1
    if first < 0x80:
        length = first
    else:
        count = first & 0x7F
        if count == 0 or count > 8:
            raise DERError("unsupported DER length")
        if offset + count > len(data):
            raise DERError("truncated long length")
        if data[offset] == 0:
            raise DERError("non-minimal DER length")
        length = int.from_bytes(data[offset : offset + count], "big")
        offset += count
        if length < 0x80:
            raise DERError("non-minimal DER length")
    end = offset + length
    if end > len(data):
        raise DERError("truncated value")
    return TLV(tag=tag, value=data[offset:end], start=start, header_end=offset, end=end), end


def parse_one(data: bytes) -> TLV:
    node, end = parse_tlv(data, 0)
    if end != len(data):
        raise DERError("trailing data after DER value")
    return node


def children(node: TLV) -> list[TLV]:
    result: list[TLV] = []
    offset = 0
    while offset < len(node.value):
        child, offset = parse_tlv(node.value, offset)
        result.append(child)
    return result


def require_tag(node: TLV, tag: int, label: str = "value") -> TLV:
    if node.tag != tag:
        raise DERError(f"{label}: expected tag {tag:#x}, got {node.tag:#x}")
    return node


def decode_integer(node: TLV, signed: bool = True) -> int:
    require_tag(node, 0x02, "INTEGER")
    if not node.value:
        raise DERError("empty INTEGER")
    raw = node.value
    if len(raw) > 1 and raw[0] == 0 and not (raw[1] & 0x80):
        raise DERError("non-minimal positive INTEGER")
    return int.from_bytes(raw, "big", signed=signed)


def decode_oid(node: TLV) -> str:
    require_tag(node, 0x06, "OID")
    data = node.value
    if not data:
        raise DERError("empty OID")
    values: list[int] = []
    current = 0
    for byte in data:
        current = (current << 7) | (byte & 0x7F)
        if not byte & 0x80:
            values.append(current)
            current = 0
    if data[-1] & 0x80:
        raise DERError("truncated OID")
    first = values[0]
    if first < 40:
        arcs = [0, first]
    elif first < 80:
        arcs = [1, first - 40]
    else:
        arcs = [2, first - 80]
    arcs.extend(values[1:])
    return ".".join(str(v) for v in arcs)


def decode_time(node: TLV) -> int:
    text = node.value.decode("ascii")
    if node.tag == 0x17:
        if not text.endswith("Z") or len(text) != 13:
            raise DERError("invalid UTCTime")
        dt = datetime.strptime(text, "%y%m%d%H%M%SZ").replace(tzinfo=timezone.utc)
    elif node.tag == 0x18:
        if not text.endswith("Z") or len(text) != 15:
            raise DERError("invalid GeneralizedTime")
        dt = datetime.strptime(text, "%Y%m%d%H%M%SZ").replace(tzinfo=timezone.utc)
    else:
        raise DERError("not a DER time")
    return int(dt.timestamp() * 1000)


def decode_bit_string(node: TLV) -> tuple[int, bytes]:
    require_tag(node, 0x03, "BIT STRING")
    if not node.value:
        raise DERError("empty BIT STRING")
    unused = node.value[0]
    if unused > 7:
        raise DERError("invalid BIT STRING padding")
    payload = node.value[1:]
    if unused and payload and payload[-1] & ((1 << unused) - 1):
        raise DERError("non-zero BIT STRING padding")
    return unused, payload


def decode_name_cn(node: TLV) -> str:
    require_tag(node, 0x30, "Name")
    for rdn in children(node):
        for attribute in children(rdn):
            parts = children(attribute)
            if len(parts) >= 2 and decode_oid(parts[0]) == "2.5.4.3":
                value = parts[1].value
                try:
                    return value.decode("utf-8")
                except UnicodeDecodeError:
                    return value.decode("ascii", "replace")
    return ""


def join_der(items: Iterable[bytes]) -> bytes:
    return b"".join(items)

