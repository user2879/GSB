"""Deterministic tunnel framing, fragmentation, and strict decoding."""

from __future__ import annotations

import hashlib
import math
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .config import MAX_MTU, MIN_MTU, SUPPORTED_CIPHERS
from .errors import ErrorCode, TunnelError

MAGIC = b"TVPN"
VERSION = 1
HEADER_STRUCT = struct.Struct(">4sBBBB32sQIIIIHH16s")
HEADER_SIZE = HEADER_STRUCT.size  # 84
DIGEST_SIZE = 16

FLAG_FRAGMENTED = 0x01
FLAG_LAST_FRAGMENT = 0x02
_ALLOWED_FLAGS = FLAG_FRAGMENTED | FLAG_LAST_FRAGMENT

CIPHER_IDS = {
    "NONE": 0,
    "AES-128-GCM": 1,
    "AES-256-GCM": 2,
    "CHACHA20-POLY1305": 3,
}
ID_CIPHERS = {value: key for key, value in CIPHER_IDS.items()}


@dataclass(frozen=True)
class EncodedMessage:
    tunnel_id: str
    sequence: int
    cipher: str
    payload: bytes
    payload_digest: str
    frames: tuple[bytes, ...]
    fragmented: bool
    wire_bytes: int

    @property
    def frame_digests(self) -> tuple[str, ...]:
        return tuple(decode_frame(frame).digest_hex for frame in self.frames)


@dataclass(frozen=True)
class DecodedFrame:
    tunnel_id: str
    sequence: int
    cipher: str
    flags: int
    payload: bytes
    payload_length: int
    original_length: int
    fragment_offset: int
    fragment_size: int
    fragment_index: int
    fragment_count: int
    digest: bytes

    @property
    def digest_hex(self) -> str:
        return self.digest.hex()

    @property
    def fragmented(self) -> bool:
        return bool(self.flags & FLAG_FRAGMENTED)

    @property
    def last_fragment(self) -> bool:
        return bool(self.flags & FLAG_LAST_FRAGMENT)


@dataclass(frozen=True)
class ReassembledMessage:
    tunnel_id: str
    sequence: int
    cipher: str
    payload: bytes
    payload_digest: str
    frame_digests: tuple[str, ...]
    duplicates: int
    fragment_count: int


@dataclass
class _ReassemblyState:
    tunnel_id: str
    sequence: int
    cipher: str
    original_length: int
    fragment_size: int
    fragment_count: int
    created_ns: int
    fragments: dict[int, bytes] = field(default_factory=dict)
    digests: dict[int, str] = field(default_factory=dict)
    duplicates: int = 0


def _tunnel_id_bytes(tunnel_id: str) -> bytes:
    if not isinstance(tunnel_id, str) or not tunnel_id:
        raise TunnelError(ErrorCode.E_ENCAP, "隧道标识不能为空", offset=8)
    try:
        encoded = tunnel_id.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TunnelError(
            ErrorCode.E_ENCAP, "隧道标识不是合法 UTF-8", tunnel_id=tunnel_id, offset=8
        ) from exc
    if len(encoded) > 32:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "隧道标识 UTF-8 编码超过 32 字节",
            tunnel_id=tunnel_id,
            offset=8,
        )
    if b"\x00" in encoded:
        raise TunnelError(
            ErrorCode.E_ENCAP, "隧道标识不得包含 NUL", tunnel_id=tunnel_id, offset=8
        )
    return encoded.ljust(32, b"\x00")


def _cipher_id(cipher: str) -> int:
    if cipher not in SUPPORTED_CIPHERS or cipher not in CIPHER_IDS:
        raise TunnelError(
            ErrorCode.E_ENCAP, f"不支持的加密套件标识: {cipher}", offset=6
        )
    return CIPHER_IDS[cipher]


def payload_digest(payload: bytes) -> bytes:
    """Stable logical payload digest used for end-to-end comparison."""

    return hashlib.sha256(b"PAYLOAD\x00" + payload).digest()[:DIGEST_SIZE]


def payload_digest_hex(payload: bytes) -> str:
    return payload_digest(payload).hex()


def _frame_digest(header_without_digest: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(b"FRAME\x00" + header_without_digest + payload).digest()[
        :DIGEST_SIZE
    ]


def _build_frame(
    *,
    tunnel_id: str,
    sequence: int,
    cipher: str,
    payload: bytes,
    original_length: int,
    fragment_offset: int,
    fragment_size: int,
    fragment_index: int,
    fragment_count: int,
    last: bool,
    fragmented: bool,
) -> bytes:
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "序号必须是非负整数",
            tunnel_id=tunnel_id,
            offset=40,
        )
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TunnelError(
            ErrorCode.E_ENCAP, "负载必须是字节串", tunnel_id=tunnel_id, offset=48
        )
    body = bytes(payload)
    flags = 0
    if fragmented:
        flags |= FLAG_FRAGMENTED
    if last:
        flags |= FLAG_LAST_FRAGMENT
    tunnel_raw = _tunnel_id_bytes(tunnel_id)
    cipher_value = _cipher_id(cipher)
    header_tail = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        flags,
        cipher_value,
        0,
        tunnel_raw,
        sequence,
        len(body),
        original_length,
        fragment_offset,
        fragment_size,
        fragment_index,
        fragment_count,
        b"\x00" * DIGEST_SIZE,
    )[: HEADER_SIZE - DIGEST_SIZE]
    digest = _frame_digest(header_tail, body)
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        flags,
        cipher_value,
        0,
        tunnel_raw,
        sequence,
        len(body),
        original_length,
        fragment_offset,
        fragment_size,
        fragment_index,
        fragment_count,
        digest,
    )
    return header + body


def encode_message(
    *,
    tunnel_id: str,
    sequence: int,
    cipher: str,
    mtu: int,
    payload: bytes,
    force_fragment: bool = False,
) -> EncodedMessage:
    """Encode one logical message into one or more MTU-bounded wire frames.

    No mutable state is touched, so encoding is deterministic and thread-safe.
    ``force_fragment`` is for equivalence tests: when possible it forces at
    least two fragments even if the payload would fit directly.
    """

    if isinstance(mtu, bool) or not isinstance(mtu, int) or not (MIN_MTU <= mtu <= MAX_MTU):
        raise TunnelError(ErrorCode.E_ENCAP, "MTU 非法", tunnel_id=tunnel_id, offset=0)
    if mtu <= HEADER_SIZE:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "MTU 必须大于封装头长度",
            tunnel_id=tunnel_id,
            offset=0,
        )
    body = bytes(payload)
    chunk_size = mtu - HEADER_SIZE
    should_fragment = len(body) > chunk_size
    if force_fragment and len(body) > 1:
        should_fragment = True
        chunk_size = max(1, min(chunk_size, math.ceil(len(body) / 2)))

    if not should_fragment:
        frame = _build_frame(
            tunnel_id=tunnel_id,
            sequence=sequence,
            cipher=cipher,
            payload=body,
            original_length=len(body),
            fragment_offset=0,
            fragment_size=len(body),
            fragment_index=0,
            fragment_count=1,
            last=False,
            fragmented=False,
        )
        frames = (frame,)
    else:
        offsets = list(range(0, len(body), chunk_size))
        fragment_count = len(offsets)
        result: list[bytes] = []
        for index, offset in enumerate(offsets):
            fragment = body[offset : offset + chunk_size]
            result.append(
                _build_frame(
                    tunnel_id=tunnel_id,
                    sequence=sequence,
                    cipher=cipher,
                    payload=fragment,
                    original_length=len(body),
                    fragment_offset=offset,
                    fragment_size=chunk_size,
                    fragment_index=index,
                    fragment_count=fragment_count,
                    last=index == fragment_count - 1,
                    fragmented=True,
                )
            )
        frames = tuple(result)
    return EncodedMessage(
        tunnel_id=tunnel_id,
        sequence=sequence,
        cipher=cipher,
        payload=body,
        payload_digest=payload_digest_hex(body),
        frames=frames,
        fragmented=len(frames) > 1,
        wire_bytes=sum(len(frame) for frame in frames),
    )


def decode_frame(frame: bytes, *, expected_mtu: int | None = None) -> DecodedFrame:
    raw = bytes(frame)
    if len(raw) < HEADER_SIZE:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "封装头被截断",
            offset=len(raw),
            details={"header_size": HEADER_SIZE, "actual_size": len(raw)},
        )
    (
        magic,
        version,
        flags,
        cipher_id,
        reserved,
        tunnel_raw,
        sequence,
        payload_length,
        original_length,
        fragment_offset,
        fragment_size,
        fragment_index,
        fragment_count,
        digest,
    ) = HEADER_STRUCT.unpack_from(raw, 0)

    tunnel_id: str | None = None
    tunnel_id_error: TunnelError | None = None
    try:
        tunnel_id = _decode_tunnel_id(tunnel_raw)
    except TunnelError as exc:
        tunnel_id_error = exc
    if magic != MAGIC:
        mismatch = next(
            (index for index, (a, b) in enumerate(zip(magic, MAGIC)) if a != b), 0
        )
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "封装魔数不匹配",
            tunnel_id=tunnel_id,
            offset=mismatch,
        )
    if version != VERSION:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            f"不支持的封装版本: {version}",
            tunnel_id=tunnel_id,
            offset=4,
        )
    if flags & ~_ALLOWED_FLAGS:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "封装标志含未定义位",
            tunnel_id=tunnel_id,
            offset=5,
            details={"flags": flags},
        )
    cipher = ID_CIPHERS.get(cipher_id)
    if cipher is None:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            f"未知加密套件编号: {cipher_id}",
            tunnel_id=tunnel_id,
            offset=6,
        )
    if reserved != 0:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "保留字段必须为 0",
            tunnel_id=tunnel_id,
            offset=7,
        )
    if tunnel_id_error is not None:
        raise tunnel_id_error
    assert tunnel_id is not None
    if payload_length != len(raw) - HEADER_SIZE:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "负载长度字段与实际长度不一致",
            tunnel_id=tunnel_id,
            offset=48,
            details={
                "declared": payload_length,
                "actual": len(raw) - HEADER_SIZE,
            },
        )
    if expected_mtu is not None and len(raw) > expected_mtu:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "帧长度超过隧道路由 MTU",
            tunnel_id=tunnel_id,
            offset=0,
            details={"frame_length": len(raw), "mtu": expected_mtu},
        )
    fragmented = bool(flags & FLAG_FRAGMENTED)
    last_fragment = bool(flags & FLAG_LAST_FRAGMENT)
    _validate_fragment_fields(
        tunnel_id=tunnel_id,
        fragmented=fragmented,
        last_fragment=last_fragment,
        payload_length=payload_length,
        original_length=original_length,
        fragment_offset=fragment_offset,
        fragment_size=fragment_size,
        fragment_index=fragment_index,
        fragment_count=fragment_count,
    )

    expected_digest = _frame_digest(raw[: HEADER_SIZE - DIGEST_SIZE], raw[HEADER_SIZE:])
    if digest != expected_digest:
        first = next(
            (
                index
                for index, (actual, wanted) in enumerate(zip(digest, expected_digest))
                if actual != wanted
            ),
            0,
        )
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "封装摘要不匹配",
            tunnel_id=tunnel_id,
            offset=HEADER_SIZE - DIGEST_SIZE + first,
        )
    return DecodedFrame(
        tunnel_id=tunnel_id,
        sequence=sequence,
        cipher=cipher,
        flags=flags,
        payload=raw[HEADER_SIZE:],
        payload_length=payload_length,
        original_length=original_length,
        fragment_offset=fragment_offset,
        fragment_size=fragment_size,
        fragment_index=fragment_index,
        fragment_count=fragment_count,
        digest=digest,
    )


def _decode_tunnel_id(raw: bytes) -> str:
    if not raw or raw[0] == 0:
        raise TunnelError(ErrorCode.E_ENCAP, "隧道标识为空", offset=8)
    zero_index = raw.find(b"\x00")
    if zero_index < 0:
        encoded = raw
    else:
        encoded = raw[:zero_index]
        if any(raw[zero_index:]):
            raise TunnelError(
                ErrorCode.E_ENCAP, "隧道标识填充区必须全为 NUL", offset=8
            )
    try:
        return encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "隧道标识不是合法 UTF-8",
            offset=8 + exc.start,
        ) from exc


def _validate_fragment_fields(
    *,
    tunnel_id: str,
    fragmented: bool,
    last_fragment: bool,
    payload_length: int,
    original_length: int,
    fragment_offset: int,
    fragment_size: int,
    fragment_index: int,
    fragment_count: int,
) -> None:
    if not fragmented:
        if last_fragment:
            raise TunnelError(
                ErrorCode.E_ENCAP,
                "未分片帧不得设置最后分片标志",
                tunnel_id=tunnel_id,
                offset=5,
            )
        if (
            fragment_offset != 0
            or fragment_index != 0
            or fragment_count != 1
            or fragment_size != payload_length
            or original_length != payload_length
        ):
            raise TunnelError(
                ErrorCode.E_ENCAP,
                "未分片帧的分片元数据不自洽",
                tunnel_id=tunnel_id,
                offset=52,
            )
        return
    if fragment_count < 1 or fragment_index >= fragment_count:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "分片序号或总数非法",
            tunnel_id=tunnel_id,
            offset=64,
        )
    if fragment_size < 1:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "分片大小必须为正",
            tunnel_id=tunnel_id,
            offset=60,
        )
    expected_offset = fragment_index * fragment_size
    if fragment_offset != expected_offset:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "分片偏移与分片序号不一致",
            tunnel_id=tunnel_id,
            offset=56,
            details={"expected": expected_offset, "actual": fragment_offset},
        )
    if original_length < 1 or fragment_offset + payload_length > original_length:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "分片范围越界",
            tunnel_id=tunnel_id,
            offset=52,
        )
    is_last_index = fragment_index == fragment_count - 1
    if last_fragment != is_last_index:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "最后分片标志与分片序号不一致",
            tunnel_id=tunnel_id,
            offset=5,
        )
    expected_length = min(fragment_size, original_length - fragment_offset)
    if payload_length != expected_length:
        raise TunnelError(
            ErrorCode.E_ENCAP,
            "分片负载长度与声明范围不一致",
            tunnel_id=tunnel_id,
            offset=48,
            details={"expected": expected_length, "actual": payload_length},
        )


class SequenceAllocator:
    """Thread-safe monotonically increasing per-tunnel sequence allocation."""

    def __init__(self, watermarks: dict[str, int] | None = None) -> None:
        self._lock = threading.Lock()
        self._next: dict[str, int] = {}
        if watermarks:
            for tunnel_id, watermark in watermarks.items():
                if isinstance(watermark, bool) or not isinstance(watermark, int):
                    raise TunnelError(
                        ErrorCode.E_STATE,
                        "序号水位必须为非负整数",
                        tunnel_id=tunnel_id,
                    )
                if watermark < 0:
                    raise TunnelError(
                        ErrorCode.E_STATE,
                        "序号水位必须为非负整数",
                        tunnel_id=tunnel_id,
                    )
                self._next[tunnel_id] = watermark + 1

    def allocate(self, tunnel_id: str, *, start: int = 0) -> int:
        with self._lock:
            if tunnel_id not in self._next:
                self._next[tunnel_id] = start
            sequence = self._next[tunnel_id]
            self._next[tunnel_id] = sequence + 1
            return sequence

    def observe(self, tunnel_id: str, sequence: int) -> None:
        with self._lock:
            current = self._next.get(tunnel_id, 0)
            self._next[tunnel_id] = max(current, sequence + 1)

    def watermark(self, tunnel_id: str) -> int | None:
        with self._lock:
            value = self._next.get(tunnel_id)
            return None if value is None else value - 1

    def watermarks(self) -> dict[str, int]:
        with self._lock:
            return {key: value - 1 for key, value in sorted(self._next.items())}


class FrameDecoder:
    """Strict per-tunnel receive sequence enforcement.

    Equal sequence numbers are allowed only among fragments of one unfinished
    logical message.  The complete message must be marked complete before the
    same sequence can be rejected.
    """

    def __init__(self, watermarks: dict[str, int] | None = None) -> None:
        self._lock = threading.Lock()
        self._last: dict[str, int] = dict(watermarks or {})
        self._last_fragmented: dict[str, bool] = {}
        self._completed: set[tuple[str, int]] = set()

    def decode(self, frame: bytes, *, expected_mtu: int | None = None) -> DecodedFrame:
        decoded = decode_frame(frame, expected_mtu=expected_mtu)
        with self._lock:
            key = (decoded.tunnel_id, decoded.sequence)
            last = self._last.get(decoded.tunnel_id)
            if last is not None:
                if decoded.sequence < last:
                    raise TunnelError(
                        ErrorCode.E_ENCAP,
                        "隧道序号未单调递增",
                        tunnel_id=decoded.tunnel_id,
                        offset=40,
                        details={"last": last, "received": decoded.sequence},
                    )
                if decoded.sequence == last:
                    if (
                        not decoded.fragmented
                        or not self._last_fragmented.get(decoded.tunnel_id, False)
                        or key in self._completed
                    ):
                        raise TunnelError(
                            ErrorCode.E_ENCAP,
                            "重复的隧道序号",
                            tunnel_id=decoded.tunnel_id,
                            offset=40,
                            details={"sequence": decoded.sequence},
                        )
            else:
                self._last[decoded.tunnel_id] = decoded.sequence
            self._last_fragmented[decoded.tunnel_id] = decoded.fragmented
        return decoded

    def mark_complete(self, tunnel_id: str, sequence: int) -> None:
        with self._lock:
            self._completed.add((tunnel_id, sequence))

    def watermark(self, tunnel_id: str) -> int | None:
        with self._lock:
            return self._last.get(tunnel_id)

    def watermarks(self) -> dict[str, int]:
        with self._lock:
            return {key: value for key, value in sorted(self._last.items())}


class Reassembler:
    """Fragment reassembly with explicit duplicate, gap, and timeout rules.

    * Out-of-order fragments are accepted.
    * Byte-identical duplicate fragments are ignored and counted.
    * Conflicting duplicate or inconsistent metadata is E_ENCAP.
    * Missing fragments become E_FRAG on timeout or forced finalization.
    """

    def __init__(
        self,
        *,
        timeout_ms: int,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.timeout_ms = timeout_ms
        self.clock_ns = clock_ns
        self._lock = threading.Lock()
        self._states: dict[tuple[str, int], _ReassemblyState] = {}

    def add(self, frame: DecodedFrame) -> ReassembledMessage | None:
        if not frame.fragmented:
            return ReassembledMessage(
                tunnel_id=frame.tunnel_id,
                sequence=frame.sequence,
                cipher=frame.cipher,
                payload=frame.payload,
                payload_digest=payload_digest_hex(frame.payload),
                frame_digests=(frame.digest_hex,),
                duplicates=0,
                fragment_count=1,
            )

        key = (frame.tunnel_id, frame.sequence)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = _ReassemblyState(
                    tunnel_id=frame.tunnel_id,
                    sequence=frame.sequence,
                    cipher=frame.cipher,
                    original_length=frame.original_length,
                    fragment_size=frame.fragment_size,
                    fragment_count=frame.fragment_count,
                    created_ns=self.clock_ns(),
                )
                self._states[key] = state
            else:
                if (
                    state.cipher != frame.cipher
                    or state.original_length != frame.original_length
                    or state.fragment_size != frame.fragment_size
                    or state.fragment_count != frame.fragment_count
                ):
                    raise TunnelError(
                        ErrorCode.E_ENCAP,
                        "同一序号分片的元数据不一致",
                        tunnel_id=frame.tunnel_id,
                        offset=52,
                        details={"sequence": frame.sequence},
                    )
            prior = state.fragments.get(frame.fragment_offset)
            if prior is not None:
                if prior == frame.payload:
                    state.duplicates += 1
                else:
                    first_diff = next(
                        (
                            index
                            for index, (a, b) in enumerate(zip(prior, frame.payload))
                            if a != b
                        ),
                        min(len(prior), len(frame.payload)),
                    )
                    raise TunnelError(
                        ErrorCode.E_ENCAP,
                        "同一分片偏移出现冲突内容",
                        tunnel_id=frame.tunnel_id,
                        offset=HEADER_SIZE + first_diff,
                        details={
                            "sequence": frame.sequence,
                            "fragment_offset": frame.fragment_offset,
                        },
                    )
            else:
                state.fragments[frame.fragment_offset] = frame.payload
                state.digests[frame.fragment_offset] = frame.digest_hex

            if len(state.fragments) != state.fragment_count:
                return None
            expected_offsets = [
                index * state.fragment_size for index in range(state.fragment_count)
            ]
            if any(offset not in state.fragments for offset in expected_offsets):
                return None
            assembled = b"".join(state.fragments[offset] for offset in expected_offsets)
            if len(assembled) != state.original_length:
                raise TunnelError(
                    ErrorCode.E_ENCAP,
                    "重组长度与原始长度不一致",
                    tunnel_id=frame.tunnel_id,
                    offset=52,
                    details={
                        "expected": state.original_length,
                        "actual": len(assembled),
                    },
                )
            digests = tuple(state.digests[offset] for offset in expected_offsets)
            duplicates = state.duplicates
            del self._states[key]
            return ReassembledMessage(
                tunnel_id=frame.tunnel_id,
                sequence=frame.sequence,
                cipher=state.cipher,
                payload=assembled,
                payload_digest=payload_digest_hex(assembled),
                frame_digests=digests,
                duplicates=duplicates,
                fragment_count=state.fragment_count,
            )

    def missing_offsets(self, tunnel_id: str, sequence: int) -> list[int]:
        with self._lock:
            state = self._states.get((tunnel_id, sequence))
            if state is None:
                return []
            expected = [
                index * state.fragment_size for index in range(state.fragment_count)
            ]
            return [offset for offset in expected if offset not in state.fragments]

    def sweep(self, *, force: bool = False) -> list[TunnelError]:
        now = self.clock_ns()
        errors: list[TunnelError] = []
        expired: list[tuple[str, int]] = []
        with self._lock:
            for key, state in self._states.items():
                age_ms = (now - state.created_ns) // 1_000_000
                if force or age_ms >= self.timeout_ms:
                    expected = [
                        index * state.fragment_size
                        for index in range(state.fragment_count)
                    ]
                    missing = [
                        offset
                        for offset in expected
                        if offset not in state.fragments
                    ]
                    errors.append(
                        TunnelError(
                            ErrorCode.E_FRAG,
                            "分片重组超时",
                            tunnel_id=state.tunnel_id,
                            details={
                                "sequence": state.sequence,
                                "missing_offsets": missing,
                                "received_offsets": sorted(state.fragments),
                            },
                        )
                    )
                    expired.append(key)
            for key in expired:
                del self._states[key]
        return errors

    def pending(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "tunnel_id": state.tunnel_id,
                    "sequence": state.sequence,
                    "received_offsets": sorted(state.fragments),
                    "fragment_count": state.fragment_count,
                }
                for state in self._states.values()
            ]


def assemble_frames(
    frames: Iterable[bytes],
    *,
    decoder: FrameDecoder,
    reassembler: Reassembler,
    mtu_by_tunnel: dict[str, int] | None = None,
) -> list[ReassembledMessage]:
    """Convenience helper that preserves input order and marks completed groups."""

    messages: list[ReassembledMessage] = []
    for frame in frames:
        expected_mtu = (
            None
            if mtu_by_tunnel is None
            else mtu_by_tunnel.get(decoded_tunnel_id(frame))
        )
        decoded = decoder.decode(
            frame,
            expected_mtu=expected_mtu,
        )
        message = reassembler.add(decoded)
        if message is not None:
            decoder.mark_complete(message.tunnel_id, message.sequence)
            messages.append(message)
    return messages


def decoded_tunnel_id(frame: bytes) -> str:
    if len(frame) < 40:
        raise TunnelError(ErrorCode.E_ENCAP, "封装头被截断", offset=len(frame))
    return _decode_tunnel_id(frame[8:40])
