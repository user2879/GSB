from __future__ import annotations


def encode_uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError("uvarint cannot encode a negative value")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode_uvarint(data: bytes, offset: int = 0) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated uvarint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 70:
            raise ValueError("uvarint is too long")


def encode_sequence(values: list[int] | tuple[int, ...]) -> bytes:
    return b"".join(encode_uvarint(value) for value in values)


def decode_sequence(data: bytes, count: int, offset: int = 0) -> tuple[list[int], int]:
    values = []
    for _ in range(count):
        value, offset = decode_uvarint(data, offset)
        values.append(value)
    return values, offset

