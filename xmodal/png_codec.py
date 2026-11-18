"""最小 PNG 编解码器（纯标准库实现，确定性）。

支持 8 位灰度(0)、真彩色(2)、带 alpha 真彩色(6)，解码支持全部五种
PNG 行过滤器；编码统一使用过滤器 0（None）+ zlib 压缩，保证字节确定。
"""

import struct
import zlib

from .errors import ErrorCode, PipelineError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

COLOR_GRAY = 0
COLOR_RGB = 2
COLOR_RGBA = 6
_CHANNELS = {COLOR_GRAY: 1, COLOR_RGB: 3, COLOR_RGBA: 4}


def _paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def encode_png(array):
    """uint8 HxW 或 HxWxC 数组 -> PNG 字节串。"""
    import numpy as np

    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "PNG 编码仅支持 uint8 数组")
    if arr.ndim == 2:
        color_type, channels = COLOR_GRAY, 1
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        channels = arr.shape[2]
        color_type = COLOR_RGB if channels == 3 else COLOR_RGBA
    else:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "不支持的图像形状 %r" % (arr.shape,))
    height, width = arr.shape[:2]
    flat = arr.reshape(height, width * channels)
    raw = bytearray()
    for row in flat:
        raw.append(0)
        raw.extend(row.tolist())
    compressed = zlib.compress(bytes(raw), level=6)

    def chunk(tag, data):
        out = struct.pack(">I", len(data)) + tag + data
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return out + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) \
        + chunk(b"IEND", b"")


def _iter_chunks(blob):
    if not blob.startswith(PNG_SIGNATURE):
        raise PipelineError(ErrorCode.SAMPLE_INVALID, "PNG 容器头非法")
    pos = len(PNG_SIGNATURE)
    seen_ihdr = False
    idat = bytearray()
    while pos + 8 <= len(blob):
        (length,) = struct.unpack(">I", blob[pos:pos + 4])
        tag = blob[pos + 4:pos + 8]
        end = pos + 8 + length + 4
        if end > len(blob):
            raise PipelineError(ErrorCode.SAMPLE_INVALID,
                                "PNG 数据块长度越界: " + tag.decode("latin1"))
        data = blob[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", blob[pos + 8 + length:end])
        if (zlib.crc32(tag + data) & 0xFFFFFFFF) != crc:
            raise PipelineError(ErrorCode.SAMPLE_INVALID,
                                "PNG 数据块 CRC 校验失败: "
                                + tag.decode("latin1"))
        if tag == b"IHDR":
            seen_ihdr = True
            yield tag, data
        elif tag == b"IDAT":
            idat.extend(data)
        elif tag == b"IEND":
            break
        else:
            yield tag, data
        pos = end
    if not seen_ihdr:
        raise PipelineError(ErrorCode.SAMPLE_INVALID, "PNG 缺少 IHDR 块")
    if not idat:
        raise PipelineError(ErrorCode.SAMPLE_INVALID, "PNG 缺少 IDAT 块")
    yield b"IDAT", bytes(idat)


def decode_png(blob, expected_size=None):
    """PNG 字节串 -> (uint8 HxWxC 数组, (width, height), color_type)。"""
    import numpy as np

    width = height = bit_depth = color_type = None
    idat = None
    for tag, data in _iter_chunks(blob):
        if tag == b"IHDR":
            (width, height, bit_depth, color_type, comp, flt, interlace) = \
                struct.unpack(">IIBBBBB", data)
            if comp != 0 or flt != 0 or interlace != 0:
                raise PipelineError(ErrorCode.SAMPLE_INVALID,
                                    "PNG 使用了不支持的压缩/隔行参数")
        elif tag == b"IDAT":
            idat = data
    if bit_depth != 8 or color_type not in _CHANNELS:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "PNG 位深或颜色类型不受支持")
    if expected_size is not None and (width, height) != tuple(expected_size):
        raise PipelineError(
            ErrorCode.SAMPLE_INVALID,
            "PNG 尺寸 %r 与期望 %r 不一致"
            % ((width, height), tuple(expected_size)))
    try:
        raw = zlib.decompress(idat)
    except zlib.error as exc:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "PNG 图像数据解压失败: " + str(exc))
    channels = _CHANNELS[color_type]
    stride = width * channels
    expected_len = height * (stride + 1)
    if len(raw) < expected_len:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "PNG 图像数据被截断")
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    pos = 0
    for r in range(height):
        filter_type = raw[pos]
        pos += 1
        if filter_type > 4:
            raise PipelineError(ErrorCode.SAMPLE_INVALID,
                                "PNG 行过滤器类型非法: %d" % filter_type)
        line = np.frombuffer(raw, dtype=np.uint8, count=stride,
                             offset=pos).copy()
        pos += stride
        if filter_type == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif filter_type == 2:
            line = (line + prev) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((int(left) + int(prev[i])) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                c = prev[i - channels] if i >= channels else 0
                line[i] = (line[i] + _paeth(a, int(prev[i]), c)) & 0xFF
        out[r] = line
        prev = line
    shape = (height, width) if channels == 1 else (height, width, channels)
    return out.reshape(shape), (width, height), color_type
