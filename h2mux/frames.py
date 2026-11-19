"""HTTP/2 wire frame model and codec for the six supported frame types."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .errors import FrameError
from .hpack import HeaderField, decode_header_block, encode_header_block


FRAME_TYPE_CODES: dict[str, int] = {
    "DATA": 0x0,
    "HEADERS": 0x1,
    "RST_STREAM": 0x3,
    "SETTINGS": 0x4,
    "GOAWAY": 0x7,
    "WINDOW_UPDATE": 0x8,
}
FRAME_TYPE_NAMES: dict[int, str] = {value: key for key, value in FRAME_TYPE_CODES.items()}

FLAG_BITS: dict[str, dict[str, int]] = {
    "DATA": {"END_STREAM": 0x1, "PADDED": 0x8},
    "HEADERS": {"END_STREAM": 0x1, "END_HEADERS": 0x4, "PADDED": 0x8, "PRIORITY": 0x20},
    "SETTINGS": {"ACK": 0x1},
}

MAX_WINDOW = (1 << 31) - 1
MAX_STREAM_ID = (1 << 31) - 1
MAX_FRAME_PAYLOAD = (1 << 24) - 1


@dataclass(frozen=True)
class PrioritySpec:
    depends_on: int = 0
    weight: int = 16
    exclusive: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.depends_on <= MAX_STREAM_ID:
            raise ValueError("priority dependency outside 31-bit range")
        if not 1 <= self.weight <= 256:
            raise ValueError("priority weight outside 1..256")


@dataclass(frozen=True)
class Setting:
    identifier: int
    value: int


@dataclass
class Frame:
    type: str
    stream_id: int
    flags: int = 0
    body: bytes = b""
    settings: tuple[Setting, ...] = ()
    headers: tuple[HeaderField, ...] = ()
    priority: PrioritySpec | None = None
    increment: int | None = None
    error_code: int | None = None
    last_stream_id: int | None = None
    debug_data: bytes = b""
    raw_bytes: bytes | None = field(default=None, repr=False, compare=False)

    @classmethod
    def data_frame(cls, stream_id: int, body: bytes, *, end_stream: bool = False) -> "Frame":
        return cls(
            type="DATA",
            stream_id=stream_id,
            flags=FLAG_BITS["DATA"]["END_STREAM"] if end_stream else 0,
            body=body,
        )

    @classmethod
    def headers_frame(
        cls,
        stream_id: int,
        headers: Iterable[HeaderField],
        *,
        end_stream: bool = False,
        end_headers: bool = True,
        priority: PrioritySpec | None = None,
    ) -> "Frame":
        flags = 0
        if end_stream:
            flags |= FLAG_BITS["HEADERS"]["END_STREAM"]
        if end_headers:
            flags |= FLAG_BITS["HEADERS"]["END_HEADERS"]
        if priority is not None:
            flags |= FLAG_BITS["HEADERS"]["PRIORITY"]
        return cls(
            type="HEADERS",
            stream_id=stream_id,
            flags=flags,
            headers=tuple(headers),
            priority=priority,
        )

    @classmethod
    def settings_frame(
        cls,
        settings: Iterable[Setting] = (),
        *,
        ack: bool = False,
    ) -> "Frame":
        return cls(
            type="SETTINGS",
            stream_id=0,
            flags=FLAG_BITS["SETTINGS"]["ACK"] if ack else 0,
            settings=() if ack else tuple(settings),
        )

    @classmethod
    def window_update_frame(cls, stream_id: int, increment: int) -> "Frame":
        return cls(type="WINDOW_UPDATE", stream_id=stream_id, increment=increment)

    @classmethod
    def rst_stream_frame(cls, stream_id: int, error_code: int) -> "Frame":
        return cls(type="RST_STREAM", stream_id=stream_id, error_code=error_code)

    @classmethod
    def goaway_frame(cls, last_stream_id: int, error_code: int, debug_data: bytes = b"") -> "Frame":
        return cls(
            type="GOAWAY",
            stream_id=0,
            last_stream_id=last_stream_id,
            error_code=error_code,
            debug_data=debug_data,
        )

    def flag(self, name: str) -> bool:
        return bool(self.flags & FLAG_BITS.get(self.type, {}).get(name, 0))

    def flag_names(self) -> list[str]:
        return [
            name
            for name, bit in sorted(FLAG_BITS.get(self.type, {}).items())
            if self.flags & bit
        ]

    @property
    def payload(self) -> bytes:
        return canonical_payload(self)

    @property
    def wire_length(self) -> int:
        return 9 + len(self.payload)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "stream_id": self.stream_id,
            "flags": self.flag_names(),
            "payload_bytes": len(self.payload),
        }
        if self.type == "SETTINGS":
            result["settings"] = [{"id": item.identifier, "value": item.value} for item in self.settings]
        elif self.type == "HEADERS":
            result["headers"] = [item.to_dict() for item in self.headers]
            if self.priority is not None:
                result["priority"] = {
                    "depends_on": self.priority.depends_on,
                    "weight": self.priority.weight,
                    "exclusive": self.priority.exclusive,
                }
        elif self.type == "DATA":
            result["data_bytes"] = len(self.body)
        elif self.type == "WINDOW_UPDATE":
            result["increment"] = self.increment
        elif self.type == "RST_STREAM":
            result["error_code"] = self.error_code
        elif self.type == "GOAWAY":
            result["last_stream_id"] = self.last_stream_id
            result["error_code"] = self.error_code
            result["debug_hex"] = self.debug_data.hex()
        return result

    def to_bytes(
        self,
        max_frame_size: int | None = MAX_FRAME_PAYLOAD,
        *,
        preserve_raw: bool = True,
    ) -> bytes:
        return encode_frame(self, max_frame_size=max_frame_size, preserve_raw=preserve_raw)

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        max_frame_size: int = MAX_FRAME_PAYLOAD,
        *,
        base_offset: int = 0,
    ) -> "Frame":
        return decode_frame(raw, max_frame_size, base_offset=base_offset)


def _frame_type_name(code: int) -> str:
    return FRAME_TYPE_NAMES.get(code, f"UNKNOWN(0x{code:02x})")


def _validate_stream_id(frame_type: str, stream_id: int, *, offset: int) -> None:
    if frame_type in {"SETTINGS", "GOAWAY"}:
        if stream_id != 0:
            raise FrameError(
                f"{frame_type} must use stream 0",
                frame_type=frame_type,
                offset=offset,
                stream_id=stream_id,
            )
        return
    if frame_type == "WINDOW_UPDATE":
        if stream_id != 0 and (stream_id <= 0 or stream_id % 2 == 0):
            raise FrameError(
                "WINDOW_UPDATE stream id must be 0 or a positive odd client stream",
                frame_type=frame_type,
                offset=offset,
                stream_id=stream_id,
            )
        return
    if stream_id <= 0 or stream_id % 2 == 0:
        raise FrameError(
            f"{frame_type} must target a positive odd client stream",
            frame_type=frame_type,
            offset=offset,
            stream_id=stream_id,
        )


def canonical_payload(frame: Frame) -> bytes:
    frame_type = frame.type
    if frame_type == "DATA":
        return frame.body
    if frame_type == "HEADERS":
        prefix = b""
        if frame.priority is not None:
            depends = frame.priority.depends_on & 0x7FFFFFFF
            if frame.priority.exclusive:
                depends |= 0x80000000
            prefix = depends.to_bytes(4, "big") + bytes([frame.priority.weight - 1])
        return prefix + encode_header_block(frame.headers)
    if frame_type == "SETTINGS":
        return b"".join(
            item.identifier.to_bytes(2, "big") + item.value.to_bytes(4, "big")
            for item in frame.settings
        )
    if frame_type == "WINDOW_UPDATE":
        return int(frame.increment or 0).to_bytes(4, "big")
    if frame_type == "RST_STREAM":
        return int(frame.error_code or 0).to_bytes(4, "big")
    if frame_type == "GOAWAY":
        last = int(frame.last_stream_id or 0) & 0x7FFFFFFF
        return last.to_bytes(4, "big") + int(frame.error_code or 0).to_bytes(4, "big") + frame.debug_data
    raise FrameError(f"unsupported frame type {frame_type!r}", frame_type=str(frame_type), offset=0)


def encode_frame(
    frame: Frame,
    max_frame_size: int | None = MAX_FRAME_PAYLOAD,
    *,
    preserve_raw: bool = True,
) -> bytes:
    if frame.raw_bytes is not None and preserve_raw:
        return frame.raw_bytes
    if frame.type not in FRAME_TYPE_CODES:
        raise FrameError(f"unsupported frame type {frame.type!r}", frame_type=frame.type, offset=0)
    if not 0 <= frame.flags <= 255:
        raise FrameError(
            "frame flags must fit in one byte",
            frame_type=frame.type,
            offset=4,
            stream_id=frame.stream_id,
        )
    if frame.type in {"DATA", "HEADERS"} and frame.flags & 0x8:
        raise FrameError(
            "padded DATA/HEADERS frames are outside the supported simplified profile",
            frame_type=frame.type,
            offset=4,
            stream_id=frame.stream_id,
        )
    if frame.type == "SETTINGS":
        for setting in frame.settings:
            if not 0 <= setting.identifier <= 65535:
                raise FrameError("SETTINGS identifier must be uint16", frame_type="SETTINGS", offset=9)
            if not 0 <= setting.value <= 0xFFFFFFFF:
                raise FrameError("SETTINGS value must be uint32", frame_type="SETTINGS", offset=9)
    if frame.type == "WINDOW_UPDATE":
        if frame.increment is None or not 1 <= frame.increment <= MAX_WINDOW:
            raise FrameError(
                "WINDOW_UPDATE increment must be 1..2^31-1",
                frame_type="WINDOW_UPDATE",
                offset=9,
                stream_id=frame.stream_id,
            )
    if frame.type == "RST_STREAM":
        if frame.error_code is None or not 0 <= frame.error_code <= 0xFFFFFFFF:
            raise FrameError("RST_STREAM error code must be uint32", frame_type="RST_STREAM", offset=9)
    if frame.type == "GOAWAY":
        if frame.last_stream_id is None or not 0 <= frame.last_stream_id <= MAX_STREAM_ID:
            raise FrameError("GOAWAY last_stream_id must be 31-bit", frame_type="GOAWAY", offset=9)
        if frame.error_code is None or not 0 <= frame.error_code <= 0xFFFFFFFF:
            raise FrameError("GOAWAY error code must be uint32", frame_type="GOAWAY", offset=9)
    payload = canonical_payload(frame)
    if len(payload) > MAX_FRAME_PAYLOAD:
        raise FrameError(
            "frame payload exceeds the 24-bit wire length limit",
            frame_type=frame.type,
            offset=0,
            stream_id=frame.stream_id,
        )
    if max_frame_size is not None and len(payload) > max_frame_size:
        raise FrameError(
            f"frame payload {len(payload)} exceeds configured frame size {max_frame_size}",
            frame_type=frame.type,
            offset=0,
            stream_id=frame.stream_id,
        )
    _validate_stream_id(frame.type, frame.stream_id, offset=5)
    if frame.type == "SETTINGS" and frame.flag("ACK") and payload:
        raise FrameError("SETTINGS ACK must have an empty payload", frame_type="SETTINGS", offset=9)
    header = (
        len(payload).to_bytes(3, "big")
        + bytes([FRAME_TYPE_CODES[frame.type], frame.flags & 0xFF])
        + (frame.stream_id & 0x7FFFFFFF).to_bytes(4, "big")
    )
    return header + payload


def decode_frame(
    raw: bytes,
    max_frame_size: int = MAX_FRAME_PAYLOAD,
    *,
    base_offset: int = 0,
) -> Frame:
    if len(raw) < 9:
        raise FrameError("truncated 9-byte frame header", frame_type="UNKNOWN", offset=base_offset + len(raw))
    payload_length = int.from_bytes(raw[0:3], "big")
    type_code = raw[3]
    frame_type = _frame_type_name(type_code)
    flags = raw[4]
    stream_id = int.from_bytes(raw[5:9], "big") & 0x7FFFFFFF
    if payload_length > MAX_FRAME_PAYLOAD:
        raise FrameError(
            "frame payload exceeds 24-bit limit",
            frame_type=frame_type,
            offset=base_offset,
            stream_id=stream_id,
        )
    if payload_length > max_frame_size:
        raise FrameError(
            f"frame payload {payload_length} exceeds configured frame size {max_frame_size}",
            frame_type=frame_type,
            offset=base_offset,
            stream_id=stream_id,
        )
    total = 9 + payload_length
    if len(raw) < total:
        raise FrameError(
            "truncated frame payload",
            frame_type=frame_type,
            offset=base_offset + len(raw),
            stream_id=stream_id,
        )
    if len(raw) > total:
        raise FrameError(
            "trailing bytes after declared frame payload",
            frame_type=frame_type,
            offset=base_offset + total,
            stream_id=stream_id,
        )
    if frame_type not in FRAME_TYPE_CODES:
        raise FrameError(
            "unsupported frame type",
            frame_type=frame_type,
            offset=base_offset + 3,
            stream_id=stream_id,
        )
    _validate_stream_id(frame_type, stream_id, offset=base_offset + 5)
    if frame_type in {"DATA", "HEADERS"} and flags & 0x8:
        raise FrameError(
            "padded DATA/HEADERS frames are outside the supported simplified profile",
            frame_type=frame_type,
            offset=base_offset + 4,
            stream_id=stream_id,
        )
    payload = raw[9:]
    common = {"type": frame_type, "stream_id": stream_id, "flags": flags, "raw_bytes": bytes(raw)}
    if frame_type == "DATA":
        return Frame(body=bytes(payload), **common)
    if frame_type == "HEADERS":
        offset = 0
        priority: PrioritySpec | None = None
        if flags & FLAG_BITS["HEADERS"]["PRIORITY"]:
            if len(payload) < 5:
                raise FrameError(
                    "HEADERS priority prefix is truncated",
                    frame_type="HEADERS",
                    offset=base_offset + 9 + len(payload),
                    stream_id=stream_id,
                )
            dependency = int.from_bytes(payload[:4], "big")
            weight = payload[4] + 1
            priority = PrioritySpec(
                depends_on=dependency & 0x7FFFFFFF,
                weight=weight,
                exclusive=bool(dependency & 0x80000000),
            )
            offset = 5
        headers = decode_header_block(payload[offset:], base_offset=base_offset + 9 + offset)
        return Frame(headers=headers, priority=priority, **common)
    if frame_type == "SETTINGS":
        ack = bool(flags & FLAG_BITS["SETTINGS"]["ACK"])
        if ack and payload:
            raise FrameError(
                "SETTINGS ACK must be empty",
                frame_type="SETTINGS",
                offset=base_offset + 9,
                stream_id=stream_id,
            )
        if not ack and len(payload) % 6:
            raise FrameError(
                "SETTINGS payload length must be a multiple of 6",
                frame_type="SETTINGS",
                offset=base_offset + 9 + (len(payload) // 6) * 6,
                stream_id=stream_id,
            )
        settings = tuple(
            Setting(
                identifier=int.from_bytes(payload[index : index + 2], "big"),
                value=int.from_bytes(payload[index + 2 : index + 6], "big"),
            )
            for index in range(0, len(payload), 6)
        )
        return Frame(settings=settings, **common)
    if frame_type == "WINDOW_UPDATE":
        if len(payload) != 4:
            raise FrameError(
                "WINDOW_UPDATE payload must be exactly 4 bytes",
                frame_type="WINDOW_UPDATE",
                offset=base_offset + 9,
                stream_id=stream_id,
            )
        increment = int.from_bytes(payload, "big") & 0x7FFFFFFF
        if increment == 0:
            raise FrameError(
                "WINDOW_UPDATE increment must be positive",
                frame_type="WINDOW_UPDATE",
                offset=base_offset + 9,
                stream_id=stream_id,
            )
        return Frame(increment=increment, **common)
    if frame_type == "RST_STREAM":
        if len(payload) != 4:
            raise FrameError(
                "RST_STREAM payload must be exactly 4 bytes",
                frame_type="RST_STREAM",
                offset=base_offset + 9,
                stream_id=stream_id,
            )
        return Frame(error_code=int.from_bytes(payload, "big"), **common)
    if frame_type == "GOAWAY":
        if len(payload) < 8:
            raise FrameError(
                "GOAWAY payload must be at least 8 bytes",
                frame_type="GOAWAY",
                offset=base_offset + 9 + len(payload),
                stream_id=stream_id,
            )
        return Frame(
            last_stream_id=int.from_bytes(payload[:4], "big") & 0x7FFFFFFF,
            error_code=int.from_bytes(payload[4:8], "big"),
            debug_data=bytes(payload[8:]),
            **common,
        )
    raise AssertionError("unreachable frame type")


def decode_frame_from_buffer(
    data: bytes,
    offset: int = 0,
    *,
    max_frame_size: int = MAX_FRAME_PAYLOAD,
) -> tuple[Frame, int]:
    if offset + 9 > len(data):
        raise FrameError("truncated frame header", frame_type="UNKNOWN", offset=offset + len(data))
    payload_length = int.from_bytes(data[offset : offset + 3], "big")
    total = 9 + payload_length
    if offset + total > len(data):
        raise FrameError(
            "truncated frame payload",
            frame_type=_frame_type_name(data[offset + 3]),
            offset=len(data),
        )
    frame = decode_frame(
        data[offset : offset + total],
        max_frame_size=max_frame_size,
        base_offset=offset,
    )
    return frame, offset + total


def clone_without_raw(frame: Frame) -> Frame:
    return replace(frame, raw_bytes=None)
