"""A deliberately small HPACK implementation for the static table only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .errors import FrameError


STATIC_TABLE: tuple[tuple[bytes, bytes], ...] = (
    (b":authority", b""),
    (b":method", b"GET"),
    (b":method", b"POST"),
    (b":path", b"/"),
    (b":path", b"/index.html"),
    (b":scheme", b"http"),
    (b":scheme", b"https"),
    (b":status", b"200"),
    (b":status", b"204"),
    (b":status", b"206"),
    (b":status", b"304"),
    (b":status", b"400"),
    (b":status", b"404"),
    (b":status", b"500"),
    (b"accept-charset", b""),
    (b"accept-encoding", b"gzip, deflate"),
    (b"accept-language", b""),
    (b"accept-ranges", b""),
    (b"accept", b""),
    (b"access-control-allow-origin", b""),
    (b"age", b""),
    (b"allow", b""),
    (b"authorization", b""),
    (b"cache-control", b""),
    (b"content-disposition", b""),
    (b"content-encoding", b""),
    (b"content-language", b""),
    (b"content-length", b""),
    (b"content-location", b""),
    (b"content-range", b""),
    (b"content-type", b""),
    (b"cookie", b""),
    (b"date", b""),
    (b"etag", b""),
    (b"expect", b""),
    (b"expires", b""),
    (b"from", b""),
    (b"host", b""),
    (b"if-match", b""),
    (b"if-modified-since", b""),
    (b"if-none-match", b""),
    (b"if-range", b""),
    (b"if-unmodified-since", b""),
    (b"last-modified", b""),
    (b"link", b""),
    (b"location", b""),
    (b"max-forwards", b""),
    (b"proxy-authenticate", b""),
    (b"proxy-authorization", b""),
    (b"range", b""),
    (b"referer", b""),
    (b"refresh", b""),
    (b"retry-after", b""),
    (b"server", b""),
    (b"set-cookie", b""),
    (b"strict-transport-security", b""),
    (b"transfer-encoding", b""),
    (b"user-agent", b""),
    (b"vary", b""),
    (b"via", b""),
    (b"www-authenticate", b""),
)

STATIC_TABLE_COUNT = len(STATIC_TABLE)


@dataclass(frozen=True)
class HeaderField:
    """One supported static-table HPACK field.

    Exactly one of ``index`` and ``name_index``/``name`` is used. Index 0 in
    ``name_index`` means that ``name`` carries a literal name.
    """

    value: bytes
    index: int | None = None
    name_index: int | None = None
    name: bytes | None = None

    @classmethod
    def indexed(cls, index: int) -> "HeaderField":
        _validate_static_index(index, base_offset=0)
        return cls(index=index, value=STATIC_TABLE[index - 1][1])

    @classmethod
    def literal(cls, name: bytes, value: bytes, name_index: int = 0) -> "HeaderField":
        return cls(name=name, name_index=name_index, value=value)

    @classmethod
    def literal_indexed_name(cls, name_index: int, value: bytes) -> "HeaderField":
        return cls(name_index=name_index, value=value)

    def resolved(self) -> tuple[bytes, bytes]:
        if self.index is not None:
            return STATIC_TABLE[self.index - 1]
        if self.name_index:
            return STATIC_TABLE[self.name_index - 1][0], self.value
        return self.name or b"", self.value

    def to_dict(self) -> dict[str, object]:
        name, value = self.resolved()
        if self.index is not None:
            return {"index": self.index, "name": name.decode("utf-8", "replace"), "value": value.decode("utf-8", "replace")}
        return {
            "name_index": self.name_index or 0,
            "name": name.decode("utf-8", "replace"),
            "value": value.decode("utf-8", "replace"),
        }


def encode_integer(value: int, prefix_bits: int, prefix: int = 0) -> bytes:
    if value < 0:
        raise ValueError("HPACK integers are unsigned")
    limit = (1 << prefix_bits) - 1
    if value < limit:
        return bytes([prefix | value])
    out = bytearray([prefix | limit])
    value -= limit
    while value >= 128:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def decode_integer(
    data: bytes,
    offset: int,
    prefix_bits: int,
    *,
    base_offset: int = 0,
) -> tuple[int, int]:
    if offset >= len(data):
        raise FrameError(
            "truncated HPACK integer",
            frame_type="HEADERS",
            offset=base_offset + offset,
        )
    limit = (1 << prefix_bits) - 1
    value = data[offset] & limit
    offset += 1
    if value < limit:
        return value, offset
    shift = 0
    while True:
        if offset >= len(data):
            raise FrameError(
                "truncated HPACK integer continuation",
                frame_type="HEADERS",
                offset=base_offset + offset,
            )
        octet = data[offset]
        offset += 1
        value += (octet & 0x7F) << shift
        if not octet & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise FrameError(
                "HPACK integer is too large",
                frame_type="HEADERS",
                offset=base_offset + offset - 1,
            )


def _encode_string(value: bytes) -> bytes:
    return encode_integer(len(value), 7) + value


def _decode_string(
    data: bytes,
    offset: int,
    *,
    base_offset: int,
) -> tuple[bytes, int]:
    if offset >= len(data):
        raise FrameError("truncated HPACK string", frame_type="HEADERS", offset=base_offset + offset)
    huffman = bool(data[offset] & 0x80)
    if huffman:
        raise FrameError(
            "Huffman strings are outside the supported simplified HPACK profile",
            frame_type="HEADERS",
            offset=base_offset + offset,
        )
    length, offset = decode_integer(data, offset, 7, base_offset=base_offset)
    end = offset + length
    if end > len(data):
        raise FrameError(
            "truncated HPACK string payload",
            frame_type="HEADERS",
            offset=base_offset + offset,
        )
    return data[offset:end], end


def _validate_static_index(index: int, *, base_offset: int) -> None:
    if not 1 <= index <= STATIC_TABLE_COUNT:
        raise FrameError(
            f"static table index {index} is outside 1..{STATIC_TABLE_COUNT}",
            frame_type="HEADERS",
            offset=base_offset,
        )


def decode_header_block(data: bytes, *, base_offset: int = 0) -> tuple[HeaderField, ...]:
    fields: list[HeaderField] = []
    offset = 0
    while offset < len(data):
        first = data[offset]
        if first & 0x80:
            index, offset = decode_integer(data, offset, 7, base_offset=base_offset)
            _validate_static_index(index, base_offset=base_offset)
            fields.append(HeaderField.indexed(index))
            continue
        if first & 0x40:
            raise FrameError(
                "HPACK dynamic table insertion is not supported",
                frame_type="HEADERS",
                offset=base_offset + offset,
            )
        if first & 0x20:
            raise FrameError(
                "HPACK dynamic table size updates are not supported",
                frame_type="HEADERS",
                offset=base_offset + offset,
            )
        if first & 0x10:
            raise FrameError(
                "HPACK never-indexed literals are outside the supported simplified profile",
                frame_type="HEADERS",
                offset=base_offset + offset,
            )
        name_index, offset = decode_integer(data, offset, 4, base_offset=base_offset)
        if name_index:
            _validate_static_index(name_index, base_offset=base_offset)
            name = None
        else:
            name, offset = _decode_string(data, offset, base_offset=base_offset)
        value, offset = _decode_string(data, offset, base_offset=base_offset)
        fields.append(HeaderField(name=name, name_index=name_index, value=value))
    return tuple(fields)


def encode_header_block(fields: Iterable[HeaderField]) -> bytes:
    out = bytearray()
    for field in fields:
        if field.index is not None:
            _validate_static_index(field.index, base_offset=0)
            out.extend(encode_integer(field.index, 7, 0x80))
            continue
        name_index = field.name_index or 0
        if name_index:
            _validate_static_index(name_index, base_offset=0)
        out.extend(encode_integer(name_index, 4, 0x00))
        if not name_index:
            out.extend(_encode_string(field.name or b""))
        out.extend(_encode_string(field.value))
    return bytes(out)


def static_name(index: int) -> bytes:
    _validate_static_index(index, base_offset=0)
    return STATIC_TABLE[index - 1][0]


def find_name_index(name: bytes, fallback_index: int) -> int:
    for index, (candidate, _) in enumerate(STATIC_TABLE, start=1):
        if candidate == name:
            return index
    return fallback_index
