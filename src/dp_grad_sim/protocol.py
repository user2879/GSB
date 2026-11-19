from __future__ import annotations

import struct
from dataclasses import dataclass

from .errors import SimError


FRAME_MAGIC = b"DPF1"
FRAME_VERSION = 1
FRAME_HEADER_STRUCT = struct.Struct(">4sBHHHHII")
FRAME_HEADER_SIZE = FRAME_HEADER_STRUCT.size


@dataclass(frozen=True)
class Frame:
    origin_worker: int
    src_holder: int
    dst_worker: int
    route_round: int
    window: int
    packet: bytes

    @property
    def wire_size(self) -> int:
        return FRAME_HEADER_SIZE + len(self.packet)


def build_frame(
    *,
    origin_worker: int,
    src_holder: int,
    dst_worker: int,
    route_round: int,
    window: int,
    packet: bytes,
) -> bytes:
    if len(packet) > 0xFFFFFFFF:
        raise SimError("E_COMM_FRAME_INVALID", "packet is too large for the transport frame")
    return FRAME_HEADER_STRUCT.pack(
        FRAME_MAGIC,
        FRAME_VERSION,
        origin_worker,
        src_holder,
        dst_worker,
        route_round,
        window,
        len(packet),
    ) + packet


def parse_frame(frame: bytes) -> Frame:
    if len(frame) < FRAME_HEADER_SIZE:
        raise SimError("E_COMM_FRAME_INVALID", "truncated transport frame header")
    (
        magic,
        version,
        origin_worker,
        src_holder,
        dst_worker,
        route_round,
        window,
        packet_len,
    ) = FRAME_HEADER_STRUCT.unpack_from(frame, 0)
    if magic != FRAME_MAGIC:
        raise SimError("E_COMM_FRAME_INVALID", "invalid transport frame magic")
    if version != FRAME_VERSION:
        raise SimError("E_COMM_FRAME_INVALID", f"unsupported transport frame version {version}")
    if len(frame) != FRAME_HEADER_SIZE + packet_len:
        raise SimError("E_COMM_FRAME_INVALID", "transport frame length does not match header")
    return Frame(
        origin_worker=origin_worker,
        src_holder=src_holder,
        dst_worker=dst_worker,
        route_round=route_round,
        window=window,
        packet=frame[FRAME_HEADER_SIZE:],
    )

