"""自定义紧凑二进制线格式抓包样本的编解码。

帧布局（大端，单帧总长度 = 36 字节头 + 负载 + 4 字节 CRC32）::

    偏移  长度  字段
    0     4     魔数 b"NDF1"
    4     8     时间戳（微秒，uint64）
    12    4     源 IPv4 地址
    16    4     目的 IPv4 地址
    20    2     源端口（uint16）
    22    2     目的端口（uint16）
    24    1     协议（6=TCP，17=UDP）
    25    1     标志位（SYN/ACK/FIN/RST/RETX）
    26    4     32 位序号（uint32）
    30    2     负载长度（uint16）
    32    4     流标识 flow_tag（五元组规范化哈希，校验序号空间归属）
    36    N     负载
    36+N  4     CRC32（覆盖头 + 负载）

解码在首个出错帧处停止并抛出 ``E_WIRE``，上下文给出该帧的文件偏移。
"""

from __future__ import annotations

import binascii
import ipaddress
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .constants import (
    FLAG_ACK,
    FLAG_FIN,
    FLAG_RESERVED_MASK,
    FLAG_RETX,
    FLAG_RST,
    FLAG_SYN,
    HEADER_SIZE,
    MAX_PAYLOAD,
    TRAILER_SIZE,
    WIRE_MAGIC,
)
from .errors import WireFormatError

_FLOW_TAG_MASK = 0xFFFFFFFF
_PROTO_TCP = 6
_PROTO_UDP = 17
_SUPPORTED_PROTOCOLS = (_PROTO_TCP, _PROTO_UDP)

# 合法标志位组合（保留位必须为 0）：
# SYN 与 FIN/RST 互斥；RST 与其他控制位互斥；RETX 只能与 ACK 组合。
_FLAG_CONTROL = FLAG_SYN | FLAG_FIN | FLAG_RST


def compute_flow_tag(src_ip: str, dst_ip: str, src_port: int,
                     dst_port: int, protocol: int) -> int:
    """依据规范化五元组计算 32 位流标识。

    规范化键对方向不敏感（地址/端口排序后哈希），因此同一条双向连接
    两个方向得到相同 flow_tag，用于校验序号空间归属是否一致。
    """
    endpoints_a = (src_ip, src_port)
    endpoints_b = (dst_ip, dst_port)
    ordered = sorted((endpoints_a, endpoints_b))
    material = f"{protocol}|{ordered[0][0]}:{ordered[0][1]}|{ordered[1][0]}:{ordered[1][1]}"
    return binascii.crc32(material.encode("ascii")) & _FLOW_TAG_MASK


def flow_key(src_ip: str, dst_ip: str, src_port: int, dst_port: int,
             protocol: int) -> str:
    """方向不敏感的流键，用于把帧归属到同一条流。"""
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    low, high = sorted((a, b))
    proto_name = "tcp" if protocol == _PROTO_TCP else "udp"
    return f"{proto_name}:{low[0]}:{low[1]}-{high[0]}:{high[1]}"


def direction_key(src_ip: str, dst_ip: str, src_port: int,
                  dst_port: int) -> str:
    """方向键：同一流的两个方向分别做序号分析。"""
    return f"{src_ip}:{src_port}->{dst_ip}:{dst_port}"


def validate_flags(flags: int) -> None:
    if flags & FLAG_RESERVED_MASK:
        raise ValueError("保留标志位非 0")
    if flags & FLAG_RST:
        if flags & (FLAG_SYN | FLAG_FIN | FLAG_ACK | FLAG_RETX):
            raise ValueError("RST 不得与 SYN/FIN/ACK/RETX 组合")
        return
    if flags & FLAG_SYN and flags & FLAG_FIN:
        raise ValueError("SYN 与 FIN 互斥")
    if flags & FLAG_RETX:
        if flags & (FLAG_SYN | FLAG_FIN):
            raise ValueError("RETX 不得与 SYN/FIN 组合")
        if not (flags & FLAG_ACK):
            raise ValueError("RETX 必须与 ACK 组合")
    if not (flags & _FLAG_CONTROL | FLAG_ACK | FLAG_RETX):
        raise ValueError("标志位全 0，无任何控制/数据语义")


@dataclass(frozen=True)
class Frame:
    """解码后的单帧观测。"""

    offset: int
    timestamp_us: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    flags: int
    seq: int
    payload_len: int
    flow_tag: int
    payload: bytes
    index: int = 0  # 文件内帧序号，从 0 开始

    @property
    def flow_key(self) -> str:
        return flow_key(self.src_ip, self.dst_ip, self.src_port,
                        self.dst_port, self.protocol)

    @property
    def direction_key(self) -> str:
        return direction_key(self.src_ip, self.dst_ip, self.src_port,
                             self.dst_port)

    @property
    def is_retransmit(self) -> bool:
        return bool(self.flags & FLAG_RETX)

    def to_dict(self) -> dict:
        return {
            "offset": self.offset,
            "index": self.index,
            "timestamp_us": self.timestamp_us,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "flags": self.flags,
            "seq": self.seq,
            "payload_len": self.payload_len,
            "flow_tag": self.flow_tag,
        }


@dataclass
class DecodedCapture:
    """整份抓包样本的解码结果。"""

    frames: List[Frame] = field(default_factory=list)
    flows: Dict[str, List[Frame]] = field(default_factory=dict)

    def ordered_sequence(self, flow: Optional[str] = None) -> List[Frame]:
        """按流重组为有序观测序列（确定性排序：序号、时间戳、文件偏移）。"""
        if flow is None:
            ordered: List[Frame] = []
            for frames in self.flows.values():
                ordered.extend(frames)
        else:
            ordered = list(self.flows.get(flow, []))
        return sorted(ordered, key=lambda f: (f.flow_key, f.direction_key,
                                              f.seq, f.timestamp_us, f.offset))

    def coverage(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """返回每流每方向的序号覆盖区间 [min_seq, max_seq]。"""
        result: Dict[str, Dict[str, Tuple[int, int]]] = {}
        for frame in self.frames:
            per_dir = result.setdefault(frame.flow_key, {})
            prev = per_dir.get(frame.direction_key)
            if prev is None:
                per_dir[frame.direction_key] = (frame.seq, frame.seq)
            else:
                per_dir[frame.direction_key] = (min(prev[0], frame.seq),
                                                max(prev[1], frame.seq))
        return result


def encode_frame(*, timestamp_us: int, src_ip: str, dst_ip: str,
                 src_port: int, dst_port: int, protocol: int, flags: int,
                 seq: int, payload: bytes = b"",
                 flow_tag: Optional[int] = None) -> bytes:
    """编码单帧；flow_tag 缺省时按规范化五元组计算。"""
    try:
        src_addr = ipaddress.IPv4Address(src_ip)
        dst_addr = ipaddress.IPv4Address(dst_ip)
    except ipaddress.AddressValueError as exc:
        raise WireFormatError(f"非法 IPv4 地址: {exc}",
                              {"field": "src_ip/dst_ip"}) from exc
    if protocol not in _SUPPORTED_PROTOCOLS:
        raise WireFormatError(f"不支持的协议号 {protocol}",
                              {"field": "protocol", "value": protocol})
    if not 0 <= src_port <= 0xFFFF or not 0 <= dst_port <= 0xFFFF:
        raise WireFormatError("端口超出 uint16 范围", {"field": "ports"})
    try:
        validate_flags(flags)
    except ValueError as exc:
        raise WireFormatError(f"标志位非法组合: {exc}",
                              {"field": "flags", "value": flags}) from exc
    if not 0 <= timestamp_us <= 0xFFFFFFFFFFFFFFFF:
        raise WireFormatError("时间戳超出 uint64 范围", {"field": "timestamp_us"})
    if not 0 <= seq <= 0xFFFFFFFF:
        raise WireFormatError("序号超出 uint32 范围", {"field": "seq"})
    if len(payload) > MAX_PAYLOAD:
        raise WireFormatError(
            f"负载长度 {len(payload)} 超过上限 {MAX_PAYLOAD}",
            {"field": "payload_len", "value": len(payload)})
    tag = flow_tag if flow_tag is not None else compute_flow_tag(
        src_ip, dst_ip, src_port, dst_port, protocol)
    if not 0 <= tag <= 0xFFFFFFFF:
        raise WireFormatError("flow_tag 超出 uint32 范围", {"field": "flow_tag"})

    header = struct.pack(
        "<4sQIIHHBBIHI",
        WIRE_MAGIC,
        timestamp_us,
        int(src_addr),
        int(dst_addr),
        src_port,
        dst_port,
        protocol,
        flags,
        seq,
        len(payload),
        tag,
    )
    body = header + payload
    crc = binascii.crc32(body) & 0xFFFFFFFF
    return body + struct.pack("<I", crc)


def _fail(offset: int, message: str, **context) -> None:
    ctx = {"frame_offset": offset}
    ctx.update(context)
    raise WireFormatError(message, ctx)


def decode_buffer(data: bytes) -> DecodedCapture:
    """解码整份抓包缓冲区，首个错误即以 E_WIRE 中止。"""
    result = DecodedCapture()
    offset = 0
    index = 0
    total = len(data)

    while offset < total:
        frame_start = offset
        remaining = total - offset
        if remaining < HEADER_SIZE:
            _fail(frame_start,
                  f"帧截断：剩余 {remaining} 字节不足 {HEADER_SIZE} 字节帧头",
                  reason="truncated_header", have=remaining)
        header = data[offset:offset + HEADER_SIZE]
        magic = header[0:4]
        if magic != WIRE_MAGIC:
            _fail(frame_start, f"魔数不匹配: 期望 {WIRE_MAGIC!r}，实际 {magic!r}",
                  reason="bad_magic")
        (timestamp_us, src_ip_int, dst_ip_int, src_port, dst_port,
         protocol, flags, seq, payload_len, flow_tag) = struct.unpack(
            "<QIIHHBBIHI", header[4:HEADER_SIZE])

        if payload_len > MAX_PAYLOAD:
            _fail(frame_start,
                  f"长度字段越界：payload_len={payload_len} 超过 {MAX_PAYLOAD}",
                  reason="payload_length_out_of_range", value=payload_len)
        frame_size = HEADER_SIZE + payload_len + TRAILER_SIZE
        if remaining < frame_size:
            _fail(frame_start,
                  f"帧截断：声明 {frame_size} 字节，实际只剩 {remaining} 字节",
                  reason="truncated_frame", declared=frame_size, have=remaining)

        payload = data[offset + HEADER_SIZE:offset + HEADER_SIZE + payload_len]
        trailer = data[offset + HEADER_SIZE + payload_len:offset + frame_size]
        (crc_actual,) = struct.unpack("<I", trailer)
        crc_expect = binascii.crc32(
            data[offset:offset + HEADER_SIZE + payload_len]) & 0xFFFFFFFF
        if crc_actual != crc_expect:
            _fail(frame_start, "CRC32 校验失败", reason="crc_mismatch",
                  expected=crc_expect, actual=crc_actual)

        if protocol not in _SUPPORTED_PROTOCOLS:
            _fail(frame_start, f"不支持的协议号 {protocol}",
                  reason="bad_protocol", value=protocol)
        try:
            validate_flags(flags)
        except ValueError as exc:
            _fail(frame_start, f"标志位非法组合: {exc}",
                  reason="bad_flags", value=flags)

        src_ip = str(ipaddress.IPv4Address(src_ip_int))
        dst_ip = str(ipaddress.IPv4Address(dst_ip_int))
        expect_tag = compute_flow_tag(src_ip, dst_ip, src_port, dst_port,
                                      protocol)
        if flow_tag != expect_tag:
            _fail(frame_start,
                  "序号空间标记与流标识不一致：flow_tag 与五元组规范化哈希不符",
                  reason="flow_tag_mismatch", expected=expect_tag,
                  actual=flow_tag)

        frame = Frame(
            offset=frame_start,
            timestamp_us=timestamp_us,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            flags=flags,
            seq=seq,
            payload_len=payload_len,
            flow_tag=flow_tag,
            payload=payload,
            index=index,
        )
        result.frames.append(frame)
        result.flows.setdefault(frame.flow_key, []).append(frame)
        offset += frame_size
        index += 1

    return result


def decode_file(path: str | Path) -> DecodedCapture:
    return decode_buffer(Path(path).read_bytes())


def encode_capture(frames: Iterable[dict]) -> bytes:
    """批量编码，供样本生成与回环重放使用。"""
    return b"".join(encode_frame(**item) for item in frames)
