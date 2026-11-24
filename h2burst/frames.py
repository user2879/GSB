"""HTTP/2-inspired frame and static-header codec used by the workbench.

The wire header is exactly the nine-byte HTTP/2 frame header. Header blocks
use a deliberately small, deterministic static-table format: a count followed
by (static index, UTF-8 value length, value) records. There is no Huffman or
dynamic table, making byte-for-byte round trips straightforward to audit.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .errors import WorkbenchError


FRAME_DATA = 0x0
FRAME_HEADERS = 0x1
FRAME_RST_STREAM = 0x3
FRAME_SETTINGS = 0x4
FRAME_GOAWAY = 0x7
FRAME_WINDOW_UPDATE = 0x8

FRAME_NAMES = {
    FRAME_DATA: "DATA",
    FRAME_HEADERS: "HEADERS",
    FRAME_RST_STREAM: "RST_STREAM",
    FRAME_SETTINGS: "SETTINGS",
    FRAME_GOAWAY: "GOAWAY",
    FRAME_WINDOW_UPDATE: "WINDOW_UPDATE",
}
SUPPORTED_FRAME_TYPES = frozenset(FRAME_NAMES)

FLAG_END_STREAM = 0x1
FLAG_ACK = 0x1
FLAG_END_HEADERS = 0x4

MAX_FRAME_PAYLOAD = (1 << 24) - 1
MAX_STREAM_ID = (1 << 31) - 1


def frame_name(frame_type: int) -> str:
    return FRAME_NAMES.get(frame_type, f"UNKNOWN_{frame_type}")


def _frame_error(
    message: str,
    field: str,
    frame_seq: int | None = None,
    **details: object,
) -> WorkbenchError:
    return WorkbenchError(
        code="E_FRAME",
        message=message,
        field=field,
        frame_seq=frame_seq,
        details={key: value for key, value in details.items() if value is not None},
    )


# RFC 7541 Appendix A. The tuple position is the one-based static index used on
# the local wire format. Keeping this as an ordered tuple (not a dict) is an
# explicit determinism requirement of the workbench.
STATIC_TABLE: tuple[tuple[str, str], ...] = (
    (":authority", ""),
    (":method", "GET"),
    (":method", "POST"),
    (":path", "/"),
    (":path", "/index.html"),
    (":scheme", "http"),
    (":scheme", "https"),
    (":status", "200"),
    (":status", "204"),
    (":status", "206"),
    (":status", "304"),
    (":status", "400"),
    (":status", "404"),
    (":status", "500"),
    ("accept-charset", ""),
    ("accept-encoding", "gzip, deflate"),
    ("accept-language", ""),
    ("accept-ranges", ""),
    ("accept", ""),
    ("access-control-allow-origin", ""),
    ("age", ""),
    ("allow", ""),
    ("authorization", ""),
    ("cache-control", ""),
    ("content-disposition", ""),
    ("content-encoding", ""),
    ("content-language", ""),
    ("content-length", ""),
    ("content-location", ""),
    ("content-range", ""),
    ("content-type", ""),
    ("cookie", ""),
    ("date", ""),
    ("etag", ""),
    ("expect", ""),
    ("expires", ""),
    ("from", ""),
    ("host", ""),
    ("if-match", ""),
    ("if-modified-since", ""),
    ("if-none-match", ""),
    ("if-range", ""),
    ("if-unmodified-since", ""),
    ("last-modified", ""),
    ("link", ""),
    ("location", ""),
    ("max-forwards", ""),
    ("proxy-authenticate", ""),
    ("proxy-authorization", ""),
    ("range", ""),
    ("referer", ""),
    ("refresh", ""),
    ("retry-after", ""),
    ("server", ""),
    ("set-cookie", ""),
    ("strict-transport-security", ""),
    ("transfer-encoding", ""),
    ("user-agent", ""),
    ("vary", ""),
    ("via", ""),
    ("www-authenticate", ""),
)

STATIC_INDEX: dict[tuple[str, str], int] = {
    entry: index for index, entry in enumerate(STATIC_TABLE, start=1)
}
STATIC_NAME_FIRST_INDEX: dict[str, int] = {}
for _index, (_name, _value) in enumerate(STATIC_TABLE, start=1):
    STATIC_NAME_FIRST_INDEX.setdefault(_name, _index)


@dataclass(frozen=True, slots=True)
class Frame:
    frame_type: int
    flags: int
    stream_id: int
    payload: bytes = b""

    @property
    def length(self) -> int:
        return len(self.payload)

    @property
    def type_name(self) -> str:
        return frame_name(self.frame_type)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_type": self.frame_type,
            "type_name": self.type_name,
            "flags": self.flags,
            "stream_id": self.stream_id,
            "length": self.length,
            "payload_hex": self.payload.hex(),
        }


def _validate_uint(value: object, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _frame_error(f"{field} must be an integer", field)
    if value < 0 or value > maximum:
        raise _frame_error(f"{field} must be in [0, {maximum}]", field)
    return value


def _canonical_header_items(
    headers: Mapping[str, str] | Iterable[tuple[str, str]],
    frame_seq: int | None = None,
) -> list[tuple[str, str]]:
    items = list(headers.items()) if isinstance(headers, Mapping) else list(headers)
    canonical: list[tuple[str, str]] = []
    for index, item in enumerate(items):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise _frame_error(
                "header entry must be a (name, value) pair",
                f"headers[{index}]",
                frame_seq,
            )
        name, value = item
        if not isinstance(name, str) or not isinstance(value, str):
            raise _frame_error(
                "header name and value must be strings",
                f"headers[{index}]",
                frame_seq,
            )
        if name not in STATIC_NAME_FIRST_INDEX:
            raise _frame_error(
                f"header name {name!r} is absent from the static table",
                f"headers[{index}].name",
                frame_seq,
            )
        try:
            value_bytes = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise _frame_error(
                "header value is not valid UTF-8",
                f"headers[{index}].value",
                frame_seq,
            ) from exc
        if len(value_bytes) > 0xFFFF:
            raise _frame_error(
                "header value exceeds 65535 bytes",
                f"headers[{index}].value",
                frame_seq,
            )
        canonical.append((name, value))
    # Static index is the primary key; value is secondary. This makes a mapping
    # and any equivalent pair sequence encode identically.
    canonical.sort(key=lambda pair: (STATIC_NAME_FIRST_INDEX[pair[0]], pair[1]))
    if len(canonical) > 0xFFFF:
        raise _frame_error("too many header fields", "headers", frame_seq)
    return canonical


def encode_header_block(
    headers: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    frame_seq: int | None = None,
) -> bytes:
    items = _canonical_header_items(headers, frame_seq)
    parts = [struct.pack(">H", len(items))]
    for name, value in items:
        index = STATIC_NAME_FIRST_INDEX[name]
        value_bytes = value.encode("utf-8")
        parts.append(struct.pack(">BH", index, len(value_bytes)))
        parts.append(value_bytes)
    return b"".join(parts)


def decode_header_block(block: bytes, *, frame_seq: int | None = None) -> tuple[tuple[str, str], ...]:
    if len(block) < 2:
        raise _frame_error("header block is shorter than its count", "payload", frame_seq)
    count = struct.unpack(">H", block[:2])[0]
    offset = 2
    result: list[tuple[str, str]] = []
    for index in range(count):
        if len(block) - offset < 3:
            raise _frame_error("truncated header entry", f"headers[{index}]", frame_seq)
        static_index, value_length = struct.unpack(">BH", block[offset : offset + 3])
        offset += 3
        if static_index < 1 or static_index > len(STATIC_TABLE):
            raise _frame_error(
                f"static table index {static_index} is out of range",
                f"headers[{index}].index",
                frame_seq,
            )
        if len(block) - offset < value_length:
            raise _frame_error("truncated header value", f"headers[{index}].value", frame_seq)
        raw_value = block[offset : offset + value_length]
        offset += value_length
        try:
            value = raw_value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _frame_error("header value is not valid UTF-8", f"headers[{index}].value", frame_seq) from exc
        result.append((STATIC_TABLE[static_index - 1][0], value))
    if offset != len(block):
        raise _frame_error("header block has trailing bytes", "payload", frame_seq)
    return tuple(result)


def validate_frame(
    frame: Frame,
    *,
    frame_seq: int | None = None,
    max_frame_size: int | None = None,
) -> Frame:
    if frame.frame_type not in SUPPORTED_FRAME_TYPES:
        raise _frame_error(
            f"unsupported frame type {frame.frame_type}",
            "frame_type",
            frame_seq,
            frame_type=frame.frame_type,
        )
    if isinstance(frame.flags, bool) or not isinstance(frame.flags, int) or not 0 <= frame.flags <= 0xFF:
        raise _frame_error("flags must be an unsigned byte", "flags", frame_seq)
    if isinstance(frame.stream_id, bool) or not isinstance(frame.stream_id, int):
        raise _frame_error("stream_id must be an integer", "stream_id", frame_seq)
    if frame.stream_id < 0 or frame.stream_id > MAX_STREAM_ID:
        raise _frame_error("stream_id is outside the 31-bit range", "stream_id", frame_seq)
    if len(frame.payload) > MAX_FRAME_PAYLOAD:
        raise _frame_error("payload exceeds the 24-bit length field", "length", frame_seq)
    if max_frame_size is not None and frame.frame_type in {FRAME_DATA, FRAME_HEADERS}:
        if len(frame.payload) > max_frame_size:
            raise _frame_error(
                f"payload exceeds negotiated max frame size {max_frame_size}",
                "length",
                frame_seq,
            )

    if frame.frame_type in {FRAME_SETTINGS, FRAME_GOAWAY} and frame.stream_id != 0:
        raise _frame_error("connection-level frame must use stream 0", "stream_id", frame_seq)
    if frame.frame_type in {FRAME_HEADERS, FRAME_DATA, FRAME_RST_STREAM} and frame.stream_id == 0:
        raise _frame_error("stream-level frame must use a non-zero stream", "stream_id", frame_seq)

    if frame.frame_type == FRAME_SETTINGS:
        if frame.flags & FLAG_ACK and frame.payload:
            raise _frame_error("SETTINGS ACK must have an empty payload", "payload", frame_seq)
        if len(frame.payload) % 6 != 0:
            raise _frame_error("SETTINGS payload length must be a multiple of 6", "length", frame_seq)
        seen: set[int] = set()
        for offset in range(0, len(frame.payload), 6):
            setting_id = struct.unpack(">H", frame.payload[offset : offset + 2])[0]
            if setting_id in seen:
                raise _frame_error(
                    f"duplicate SETTINGS identifier {setting_id}",
                    "payload.identifier",
                    frame_seq,
                )
            seen.add(setting_id)
    elif frame.frame_type == FRAME_HEADERS:
        decode_header_block(frame.payload, frame_seq=frame_seq)
    elif frame.frame_type == FRAME_WINDOW_UPDATE:
        if len(frame.payload) != 4:
            raise _frame_error("WINDOW_UPDATE payload must be exactly 4 bytes", "length", frame_seq)
        increment = struct.unpack(">I", frame.payload)[0]
        if increment > MAX_STREAM_ID:
            raise _frame_error("window increment reserve bit must be zero", "increment", frame_seq)
    elif frame.frame_type == FRAME_RST_STREAM:
        if len(frame.payload) != 4:
            raise _frame_error("RST_STREAM payload must be exactly 4 bytes", "length", frame_seq)
    elif frame.frame_type == FRAME_GOAWAY:
        if len(frame.payload) < 8:
            raise _frame_error("GOAWAY payload must be at least 8 bytes", "length", frame_seq)
        last_stream_id = struct.unpack(">I", frame.payload[:4])[0]
        if last_stream_id > MAX_STREAM_ID:
            raise _frame_error("last_stream_id reserve bit must be zero", "last_stream_id", frame_seq)
    return frame


def encode_frame(frame: Frame) -> bytes:
    validate_frame(frame)
    return (
        struct.pack(">I", len(frame.payload))[1:]
        + bytes((frame.frame_type, frame.flags))
        + struct.pack(">I", frame.stream_id)
        + frame.payload
    )


def decode_frame(data: bytes, *, frame_seq: int | None = None, max_frame_size: int | None = None) -> Frame:
    if len(data) < 9:
        raise _frame_error("frame is shorter than the fixed header", "length", frame_seq)
    length = int.from_bytes(data[:3], "big")
    frame_type = data[3]
    flags = data[4]
    raw_stream_id = struct.unpack(">I", data[5:9])[0]
    if raw_stream_id & 0x80000000:
        raise _frame_error("reserved stream identifier bit must be zero", "stream_id", frame_seq)
    if len(data) != 9 + length:
        raise _frame_error(
            f"declared length {length} does not match remaining bytes {len(data) - 9}",
            "length",
            frame_seq,
            declared=length,
            actual=len(data) - 9,
        )
    frame = Frame(frame_type, flags, raw_stream_id, data[9:])
    validate_frame(frame, frame_seq=frame_seq, max_frame_size=max_frame_size)
    return frame


def encode_frames(frames: Sequence[Frame]) -> bytes:
    return b"".join(encode_frame(frame) for frame in frames)


def decode_frames(data: bytes, *, max_frame_size: int | None = None) -> tuple[Frame, ...]:
    frames: list[Frame] = []
    offset = 0
    frame_seq = 1
    while offset < len(data):
        if len(data) - offset < 9:
            raise _frame_error("truncated frame header", "length", frame_seq)
        length = int.from_bytes(data[offset : offset + 3], "big")
        end = offset + 9 + length
        if end > len(data):
            raise _frame_error(
                f"declared length {length} exceeds remaining bytes {len(data) - offset - 9}",
                "length",
                frame_seq,
            )
        frames.append(
            decode_frame(
                data[offset:end],
                frame_seq=frame_seq,
                max_frame_size=max_frame_size,
            )
        )
        offset = end
        frame_seq += 1
    return tuple(frames)


def encode_settings(settings: Mapping[int, int] | Iterable[tuple[int, int]]) -> bytes:
    items = list(settings.items()) if isinstance(settings, Mapping) else list(settings)
    items.sort(key=lambda pair: pair[0])
    parts: list[bytes] = []
    for index, (setting_id, value) in enumerate(items):
        setting_id = _validate_uint(setting_id, f"settings[{index}].identifier", 0xFFFF)
        value = _validate_uint(value, f"settings[{index}].value", 0xFFFFFFFF)
        parts.append(struct.pack(">HI", setting_id, value))
    return b"".join(parts)


def decode_settings(payload: bytes, *, frame_seq: int | None = None) -> tuple[tuple[int, int], ...]:
    if len(payload) % 6 != 0:
        raise _frame_error("SETTINGS payload length must be a multiple of 6", "length", frame_seq)
    return tuple(
        struct.unpack(">HI", payload[offset : offset + 6])
        for offset in range(0, len(payload), 6)
    )


def make_settings(settings: Mapping[int, int] | Iterable[tuple[int, int]], flags: int = 0) -> Frame:
    return Frame(FRAME_SETTINGS, flags, 0, encode_settings(settings))


def make_headers(
    stream_id: int,
    headers: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    end_stream: bool = False,
) -> Frame:
    flags = FLAG_END_HEADERS | (FLAG_END_STREAM if end_stream else 0)
    return Frame(FRAME_HEADERS, flags, stream_id, encode_header_block(headers))


def make_data(stream_id: int, payload: bytes, *, end_stream: bool = False) -> Frame:
    return Frame(FRAME_DATA, FLAG_END_STREAM if end_stream else 0, stream_id, payload)


def make_window_update(stream_id: int, increment: int) -> Frame:
    increment = _validate_uint(increment, "increment", MAX_STREAM_ID)
    return Frame(FRAME_WINDOW_UPDATE, 0, stream_id, struct.pack(">I", increment))


def make_rst_stream(stream_id: int, error_code: int = 0) -> Frame:
    error_code = _validate_uint(error_code, "error_code", 0xFFFFFFFF)
    return Frame(FRAME_RST_STREAM, 0, stream_id, struct.pack(">I", error_code))


def make_goaway(last_stream_id: int, error_code: int = 0, debug_data: bytes = b"") -> Frame:
    last_stream_id = _validate_uint(last_stream_id, "last_stream_id", MAX_STREAM_ID)
    error_code = _validate_uint(error_code, "error_code", 0xFFFFFFFF)
    if len(debug_data) > MAX_FRAME_PAYLOAD - 8:
        raise _frame_error("GOAWAY debug data is too large", "debug_data")
    return Frame(
        FRAME_GOAWAY,
        0,
        0,
        struct.pack(">II", last_stream_id, error_code) + debug_data,
    )


def default_request_headers(body_bytes: int, stream_id: int) -> tuple[tuple[str, str], ...]:
    method = "POST" if body_bytes else "GET"
    return tuple(
        sorted(
            (
                (":method", method),
                (":path", f"/requests/{stream_id}"),
                (":scheme", "http"),
                (":authority", "local.h2burst"),
                ("content-length", str(body_bytes)),
            ),
            key=lambda pair: (STATIC_NAME_FIRST_INDEX[pair[0]], pair[1]),
        )
    )
