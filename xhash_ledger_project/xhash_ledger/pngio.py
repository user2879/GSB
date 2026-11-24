"""最小化 PNG 编解码（仅依赖标准库 zlib/struct）。

写入端只生成 8 位真彩（color type 2）、filter type 0 的 PNG；
读取端支持灰度/真彩/带 alpha 的 8 位图像以及全部五种行过滤器，
并在校验层暴露 PNG 签名与 IHDR 容器头字段以便复核。
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngError(Exception):
    pass


@dataclass
class PngImage:
    width: int
    height: int
    bit_depth: int
    color_type: int
    # shape = (height, width, channels)，dtype=uint8
    pixels: "object"

    @property
    def channels(self) -> int:
        return self.pixels.shape[2]


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def write_png(path: str, pixels) -> None:
    """把 HxWx3 uint8 数组写成确定性 PNG（每行前置 filter=0）。"""
    import numpy as np

    arr = np.ascontiguousarray(pixels, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise PngError("write_png 仅支持 HxWx3 的 8 位真彩图像")
    height, width, _ = arr.shape
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = bytearray()
    for row in arr:
        raw.append(0)
        raw.extend(row.tobytes())
    with open(path, "wb") as fh:
        fh.write(PNG_SIGNATURE)
        fh.write(_chunk(b"IHDR", ihdr))
        fh.write(_chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
        fh.write(_chunk(b"IEND", b""))


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png(path: str) -> PngImage:
    """读取 PNG 并做容器头校验，返回 :class:`PngImage`。"""
    import numpy as np

    with open(path, "rb") as fh:
        blob = fh.read()
    if not blob.startswith(PNG_SIGNATURE):
        raise PngError("PNG 签名（容器头）非法")
    pos = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while pos < len(blob):
        if pos + 8 > len(blob):
            raise PngError("PNG 数据块长度字段被截断")
        (length,) = struct.unpack(">I", blob[pos : pos + 4])
        tag = blob[pos + 4 : pos + 8]
        start = pos + 8
        end = start + length
        if end + 4 > len(blob):
            raise PngError(f"PNG 数据块 {tag!r} 被截断")
        payload = blob[start:end]
        crc_stored = struct.unpack(">I", blob[end : end + 4])[0]
        if zlib.crc32(tag + payload) & 0xFFFFFFFF != crc_stored:
            raise PngError(f"PNG 数据块 {tag!r} CRC 校验失败")
        if tag == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if comp != 0 or filt != 0 or interlace != 0:
                raise PngError("仅支持压缩/滤波/隔行均为标准 0 的 PNG")
        elif tag == b"IDAT":
            idat.extend(payload)
        elif tag == b"IEND":
            break
        pos = end + 4

    if width is None:
        raise PngError("缺少 IHDR 容器头")
    if bit_depth != 8:
        raise PngError(f"仅支持 8 位位深，实际 {bit_depth}")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise PngError(f"不支持的 color type {color_type}")

    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise PngError("解压后扫描线长度与 IHDR 分辨率不一致")

    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    offset = 0
    for y in range(height):
        filter_type = raw[offset]
        line = np.frombuffer(raw[offset + 1 : offset + 1 + stride], dtype=np.uint8).copy()
        offset += stride + 1
        if filter_type == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif filter_type == 2:
            line = (line + prev) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((int(left) + int(prev[i])) // 2)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                c = prev[i - channels] if i >= channels else 0
                line[i] = (line[i] + _paeth(a, int(prev[i]), c)) & 0xFF
        elif filter_type != 0:
            raise PngError(f"未知行过滤器 {filter_type}")
        out[y] = line
        prev = line
    pixels = out.reshape(height, width, channels)
    return PngImage(width=width, height=height, bit_depth=bit_depth, color_type=color_type, pixels=pixels)
