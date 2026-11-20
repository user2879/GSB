"""环形通信的显式二进制消息协议。

帧结构（大端）：
  魔数 MAGIC(4B) | 协议版本(1B) | 消息类型(1B) | 阶段(1B) |
  训练步 step(uint32) | 源 rank(uint16) | 目标 rank(uint16) |
  分片序号 chunk(uint16) | 轮次 round(uint16) | 块序号 block(uint16) |
  负载 dtype(1B) | 负载元素数(uint32) | CRC32(4B) | 负载原始字节

所有字段均为显式定长字段；消息乱序到达时由块序号 (chunk, round) 去重与
归位，协议违规统一以 E_PROTO 报告。
"""
from __future__ import annotations

import socket
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .errors import E_PROTO, GradCompError

MAGIC = b"GCMP"
PROTOCOL_VERSION = 1
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
HEADER_SIZE = 30

# 消息类型
MSG_HANDSHAKE = 1
MSG_DATA = 2
MSG_PHASE = 3
MSG_FINALIZE = 4

# 阶段标记
PHASE_HANDSHAKE = 0
PHASE_REDUCE_SCATTER = 1
PHASE_ALL_GATHER = 2
PHASE_BARRIER = 3
PHASE_FINALIZE = 4

_PHASE_NAMES = {
    PHASE_HANDSHAKE: "handshake",
    PHASE_REDUCE_SCATTER: "reduce_scatter",
    PHASE_ALL_GATHER: "all_gather",
    PHASE_BARRIER: "barrier",
    PHASE_FINALIZE: "finalize",
}

_DTYPE_CODES = {
    np.dtype(np.int8): 1,
    np.dtype(np.int16): 2,
    np.dtype(np.int32): 3,
    np.dtype(np.int64): 4,
    np.dtype(np.float64): 9,
}
_CODE_DTYPE = {code: dtype for dtype, code in _DTYPE_CODES.items()}

_HEADER = struct.Struct(">4sBBBHIHHHHBII")


def phase_name(phase: int) -> str:
    return _PHASE_NAMES.get(phase, f"unknown({phase})")


@dataclass(frozen=True)
class Message:
    msg_type: int
    phase: int
    step: int
    src: int
    dst: int
    chunk: int
    round_index: int
    block: int
    payload: np.ndarray

    def key(self) -> tuple[int, int, int]:
        """乱序归位/去重键：(分片序号, 轮次, 块序号)。"""
        return (self.chunk, self.round_index, self.block)


def _proto_fail(message: str, details: dict | None = None):
    raise GradCompError(E_PROTO, message, details=details)


def encode(msg: Message) -> bytes:
    payload = np.ascontiguousarray(msg.payload)
    dtype = payload.dtype
    if dtype not in _DTYPE_CODES:
        _proto_fail(f"不支持的负载 dtype: {dtype}")
    raw = payload.tobytes()
    if len(raw) > MAX_PAYLOAD_BYTES:
        _proto_fail("负载超过协议上限 64MiB", {"bytes": len(raw)})
    header = _HEADER.pack(
        MAGIC, PROTOCOL_VERSION, msg.msg_type, msg.phase,
        msg.step & 0xFFFFFFFF, msg.src & 0xFFFF, msg.dst & 0xFFFF,
        msg.chunk & 0xFFFF, msg.round_index & 0xFFFF, msg.block & 0xFFFF,
        _DTYPE_CODES[dtype], payload.size, zlib.crc32(raw),
    )
    return header + raw


def decode(frame: bytes) -> Message:
    if len(frame) < HEADER_SIZE:
        _proto_fail("帧短于固定头长度", {"length": len(frame)})
    (magic, version, msg_type, phase, step, src, dst, chunk,
     round_index, block, dtype_code, n_elem, crc) = _HEADER.unpack(
         frame[:HEADER_SIZE])
    if magic != MAGIC:
        _proto_fail("魔数不匹配", {"magic": magic.hex()})
    if version != PROTOCOL_VERSION:
        _proto_fail(f"协议版本不匹配: 收到 {version}，期望 {PROTOCOL_VERSION}")
    if msg_type not in (MSG_HANDSHAKE, MSG_DATA, MSG_PHASE, MSG_FINALIZE):
        _proto_fail(f"未知消息类型 {msg_type}")
    if phase not in _PHASE_NAMES:
        _proto_fail(f"未知阶段标记 {phase}")
    if dtype_code not in _CODE_DTYPE:
        _proto_fail(f"未知负载 dtype 编码 {dtype_code}")
    raw = frame[HEADER_SIZE:]
    if zlib.crc32(raw) != crc:
        _proto_fail("负载 CRC32 校验失败")
    dtype = _CODE_DTYPE[dtype_code]
    expected = n_elem * dtype.itemsize
    if len(raw) != expected:
        _proto_fail("负载长度与元素数/dtype 不一致",
                    {"actual": len(raw), "expected": expected})
    payload = np.frombuffer(raw, dtype=dtype, count=n_elem).copy()
    return Message(msg_type, phase, step, src, dst, chunk, round_index,
                   block, payload)


def send_message(sock: socket.socket, msg: Message) -> None:
    frame = encode(msg)
    sock.sendall(struct.pack(">I", len(frame)) + frame)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        part = sock.recv(remaining)
        if not part:
            _proto_fail("套接字在消息接收中途关闭")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Message:
    size_raw = _recv_exactly(sock, 4)
    (size,) = struct.unpack(">I", size_raw)
    if size == 0 or size > MAX_PAYLOAD_BYTES + HEADER_SIZE:
        _proto_fail("非法帧长度", {"size": size})
    return decode(_recv_exactly(sock, size))
