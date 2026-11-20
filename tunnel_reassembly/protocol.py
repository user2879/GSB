"""数据报片头编解码（大端、定长、CRC32 校验）。

片头布局（46 字节）：
  magic        4B  固定 b'TURP'
  version      2B  片头版本，当前 1
  channel_id  16B  ASCII，不足 NUL 填充，超出拒绝
  group_seq    4B  无符号整数
  frag_offset  4B
  frag_length  4B  仅负载字节数
  total_frags  4B
  send_ms      4B
  crc32        4B  覆盖 magic..send_ms 的全部片头字节
其后紧跟 frag_length 字节负载。
"""

from __future__ import annotations

import struct
import zlib

from .errors import BusinessError, E_PROTOCOL

MAGIC = b"TURP"
HEADER_VERSION = 1
CHANNEL_ID_BYTES = 16
HEADER_SIZE = 4 + 2 + CHANNEL_ID_BYTES + 4 * 6
_HEADER_WITHOUT_CRC = struct.Struct(">4sH16sIIIII")


class PacketCodec:
    """无状态片头编解码器。"""

    def encode(
        self,
        channel_id: str,
        group_seq: int,
        frag_offset: int,
        frag_length: int,
        total_frags: int,
        send_ms: int,
        payload: bytes,
    ) -> bytes:
        if len(payload) != frag_length:
            raise BusinessError(
                E_PROTOCOL, f"片长度 {frag_length} 与负载长度 {len(payload)} 不自洽"
            )
        try:
            channel_bytes = channel_id.encode("ascii")
        except UnicodeEncodeError:
            raise BusinessError(E_PROTOCOL, f"通道标识必须为 ASCII: {channel_id}")
        if len(channel_bytes) > CHANNEL_ID_BYTES:
            raise BusinessError(E_PROTOCOL, f"通道标识超过 {CHANNEL_ID_BYTES} 字节")
        for name, value in (
            ("group_seq", group_seq),
            ("frag_offset", frag_offset),
            ("frag_length", frag_length),
            ("total_frags", total_frags),
            ("send_ms", send_ms),
        ):
            if not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF:
                raise BusinessError(E_PROTOCOL, f"{name} 超出 uint32 范围: {value!r}")

        prefix = _HEADER_WITHOUT_CRC.pack(
            MAGIC,
            HEADER_VERSION,
            channel_bytes.ljust(CHANNEL_ID_BYTES, b"\x00"),
            group_seq,
            frag_offset,
            frag_length,
            total_frags,
            send_ms,
        )
        crc = zlib.crc32(prefix) & 0xFFFFFFFF
        return prefix + struct.pack(">I", crc) + payload

    def decode(self, datagram: bytes, payload_limit: int) -> dict:
        """解码并执行全部片头自洽校验，返回字段字典（含 payload）。"""
        if len(datagram) < HEADER_SIZE:
            raise BusinessError(
                E_PROTOCOL, f"数据报短于片头 {HEADER_SIZE} 字节，实际 {len(datagram)}"
            )
        prefix = datagram[: HEADER_SIZE - 4]
        (crc_field,) = struct.unpack(">I", datagram[HEADER_SIZE - 4 : HEADER_SIZE])
        if zlib.crc32(prefix) & 0xFFFFFFFF != crc_field:
            raise BusinessError(E_PROTOCOL, "片头 CRC32 校验失败")

        magic, version, channel_raw, group_seq, frag_offset, frag_length, total_frags, send_ms = (
            _HEADER_WITHOUT_CRC.unpack(prefix)
        )
        if magic != MAGIC:
            raise BusinessError(E_PROTOCOL, f"magic 不匹配: {magic!r}")
        if version != HEADER_VERSION:
            raise BusinessError(E_PROTOCOL, f"不支持的片头版本: {version}")
        if total_frags <= 0:
            raise BusinessError(E_PROTOCOL, f"总片数必须为正整数，实际 {total_frags}")
        if frag_length <= 0:
            raise BusinessError(E_PROTOCOL, f"片长度必须为正整数，实际 {frag_length}")
        if frag_length > payload_limit:
            raise BusinessError(
                E_PROTOCOL, f"片长度 {frag_length} 超过通道负载上限 {payload_limit}"
            )

        payload = datagram[HEADER_SIZE:]
        if len(payload) != frag_length:
            raise BusinessError(
                E_PROTOCOL,
                f"片长度与数据报实际负载不自洽：声明 {frag_length} 实际 {len(payload)}",
            )
        if frag_offset + frag_length > 0xFFFFFFFF:
            raise BusinessError(E_PROTOCOL, "片偏移加片长度超过 uint32 总长度上限")

        channel_id = channel_raw.rstrip(b"\x00").decode("ascii")
        return {
            "channel_id": channel_id,
            "group_seq": group_seq,
            "frag_offset": frag_offset,
            "frag_length": frag_length,
            "total_frags": total_frags,
            "send_ms": send_ms,
            "payload": payload,
        }
