"""三种块编码：原始（RAW）、字典（DICT）、游程（RLE）。

确定性约定（保证重复编解码逐字节一致）：

* 整数统一为有符号 64 位小端，字符串统一为 UTF-8，长度前缀使用 4 字节小端；
* 字典编码的字典项顺序由显式规则确定：整数按数值升序，字符串按 UTF-8
  字节序升序（``bytes`` 比较），不依赖 dict/map 的遍历顺序；
* 游程编码的重复次数与字典项下标均为 4 字节小端；
* 压缩使用 zlib 固定压缩级别，因此同一载荷的压缩结果稳定；
* 每个块的完整编码载荷（含小魔数与类型标记）再计算 SHA-256 摘要。
"""
from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass

from .errors import EncodingError, PayloadError

RAW = "RAW"
DICT = "DICT"
RLE = "RLE"
ENCODINGS = (RAW, DICT, RLE)

INT = "int64"
STR = "string"
TYPES = (INT, STR)

_MAGIC = b"CBLK"
_ZLIB_LEVEL = 6


def _is_int(values: list) -> bool:
    return all(isinstance(v, int) and not isinstance(v, bool) for v in values)


def infer_type(values: list) -> str:
    if _is_int(values):
        return INT
    if all(isinstance(v, str) for v in values):
        return STR
    raise EncodingError("块内值必须全部为 int64 或全部为 string",
                        {"sample_types": sorted({type(v).__name__
                                                 for v in values})})


# ---------------------------------------------------------------------------
# 标量 / 序列的长度前缀读写
# ---------------------------------------------------------------------------
def _put_int(buf: bytearray, value: int) -> None:
    if not (-(1 << 63) <= value < (1 << 63)):
        raise EncodingError(f"整数 {value} 超出 int64 范围")
    buf.extend(struct.pack("<q", value))


def _get_int(buf: bytes, pos: int) -> tuple[int, int]:
    if pos + 8 > len(buf):
        raise PayloadError("整数载荷被截断", {"offset": pos,
                                             "remaining": len(buf) - pos})
    return struct.unpack_from("<q", buf, pos)[0], pos + 8


def _put_blob(buf: bytearray, blob: bytes) -> None:
    buf.extend(struct.pack("<I", len(blob)))
    buf.extend(blob)


def _get_blob(buf: bytes, pos: int) -> tuple[bytes, int]:
    if pos + 4 > len(buf):
        raise PayloadError("长度前缀被截断", {"offset": pos})
    (length,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    if pos + length > len(buf):
        raise PayloadError("变长载荷被截断",
                           {"offset": pos, "declared": length,
                            "remaining": len(buf) - pos})
    return buf[pos:pos + length], pos + length


def _put_count(buf: bytearray, value: int) -> None:
    if not (0 <= value < (1 << 32)):
        raise EncodingError(f"计数 {value} 超出 uint32 范围")
    buf.extend(struct.pack("<I", value))


def _get_count(buf: bytes, pos: int) -> tuple[int, int]:
    if pos + 4 > len(buf):
        raise PayloadError("计数前缀被截断", {"offset": pos})
    return struct.unpack_from("<I", buf, pos)[0], pos + 4


def _scalar_bytes(value, col_type: str) -> bytes:
    if col_type == INT:
        return struct.pack("<q", value)
    return value.encode("utf-8")


def _scalar_from_bytes(blob: bytes, col_type: str):
    if col_type == INT:
        if len(blob) != 8:
            raise PayloadError("int64 标量长度必须为 8", {"got": len(blob)})
        return struct.unpack("<q", blob)[0]
    return blob.decode("utf-8")


# ---------------------------------------------------------------------------
# 三种编码
# ---------------------------------------------------------------------------
def _encode_raw(values: list, col_type: str) -> bytes:
    buf = bytearray()
    buf.extend(b"R")
    _put_count(buf, len(values))
    for value in values:
        _put_blob(buf, _scalar_bytes(value, col_type))
    return bytes(buf)


def _decode_raw(buf: bytes, pos: int, col_type: str) -> tuple[list, int]:
    count, pos = _get_count(buf, pos)
    values = []
    for _ in range(count):
        blob, pos = _get_blob(buf, pos)
        values.append(_scalar_from_bytes(blob, col_type))
    return values, pos


def _sorted_distinct(values: list, col_type: str) -> list:
    if col_type == INT:
        return sorted(set(values))
    return sorted(set(values), key=lambda item: item.encode("utf-8"))


def _encode_dict(values: list, col_type: str) -> bytes:
    dictionary = _sorted_distinct(values, col_type)
    index = {value: slot for slot, value in enumerate(dictionary)}
    buf = bytearray()
    buf.extend(b"D")
    _put_count(buf, len(dictionary))
    for value in dictionary:  # 显式排序后的字典顺序
        _put_blob(buf, _scalar_bytes(value, col_type))
    _put_count(buf, len(values))
    for value in values:
        _put_count(buf, index[value])
    return bytes(buf)


def _decode_dict(buf: bytes, pos: int, col_type: str) -> tuple[list, int]:
    dict_size, pos = _get_count(buf, pos)
    dictionary = []
    for _ in range(dict_size):
        blob, pos = _get_blob(buf, pos)
        dictionary.append(_scalar_from_bytes(blob, col_type))
    count, pos = _get_count(buf, pos)
    values = []
    for _ in range(count):
        slot, pos = _get_count(buf, pos)
        if slot >= len(dictionary):
            raise PayloadError("字典下标越界",
                               {"slot": slot, "dict_size": len(dictionary)})
        values.append(dictionary[slot])
    return values, pos


def _runs(values: list) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))
    return runs


def _encode_rle(values: list, col_type: str) -> bytes:
    run_list = _runs(values)
    buf = bytearray()
    buf.extend(b"X")
    _put_count(buf, len(values))
    _put_count(buf, len(run_list))
    for value, length in run_list:
        _put_blob(buf, _scalar_bytes(value, col_type))
        _put_count(buf, length)
    return bytes(buf)


def _decode_rle(buf: bytes, pos: int, col_type: str) -> tuple[list, int]:
    total, pos = _get_count(buf, pos)
    run_count, pos = _get_count(buf, pos)
    values: list = []
    for _ in range(run_count):
        blob, pos = _get_blob(buf, pos)
        length, pos = _get_count(buf, pos)
        value = _scalar_from_bytes(blob, col_type)
        values.extend([value] * length)
    if len(values) != total:
        raise PayloadError("RLE 行数与块头声明不一致",
                           {"declared": total, "decoded": len(values)})
    return values, pos


_DECODERS = {
    b"R": _decode_raw,
    b"D": _decode_dict,
    b"X": _decode_rle,
}


@dataclass(frozen=True)
class EncodedBlock:
    """编码结果：原始/压缩后字节数、摘要与可落盘的完整载荷。"""

    encoding: str
    col_type: str
    row_count: int
    pre_bytes: int
    post_bytes: int
    compressed: bool
    digest: str
    payload: bytes  # 含 _MAGIC 与类型/压缩标记的完整块载荷


def encode_block(values: list, encoding: str, *, compressed: bool) -> EncodedBlock:
    if encoding not in ENCODINGS:
        raise EncodingError(f"未知编码方式 {encoding!r}",
                            {"allowed": list(ENCODINGS)})
    col_type = infer_type(values)
    if encoding == RAW:
        body = _encode_raw(values, col_type)
    elif encoding == DICT:
        body = _encode_dict(values, col_type)
    else:
        body = _encode_rle(values, col_type)

    pre_bytes = len(body)
    if compressed:
        stored = zlib.compress(body, _ZLIB_LEVEL)
    else:
        stored = body
    payload = bytearray()
    payload.extend(_MAGIC)
    payload.append(TYPES.index(col_type))
    payload.append(ENCODINGS.index(encoding))
    payload.append(1 if compressed else 0)
    _put_blob(payload, stored)
    payload = bytes(payload)
    return EncodedBlock(
        encoding=encoding,
        col_type=col_type,
        row_count=len(values),
        pre_bytes=pre_bytes,
        post_bytes=len(payload),
        compressed=compressed,
        digest=hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )


def decode_block(payload: bytes, *, expect_encoding: str | None = None,
                 expect_type: str | None = None,
                 expect_digest: str | None = None,
                 truncate: int | None = None) -> list:
    """逐块解码；可注入截断并校验摘要与编码声明。"""
    if truncate is not None:
        payload = payload[:truncate]
    if expect_digest is not None:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expect_digest:
            raise PayloadError("块摘要不匹配",
                               {"expected": expect_digest, "actual": actual})
    if len(payload) < 6 or payload[:4] != _MAGIC:
        raise PayloadError("块魔数损坏或载荷被截断",
                           {"size": len(payload)})
    type_index = payload[4]
    encoding_index = payload[5]
    compressed_flag = payload[6]
    if type_index >= len(TYPES) or encoding_index >= len(ENCODINGS):
        raise EncodingError("块头类型/编码标记非法",
                            {"type_index": type_index,
                             "encoding_index": encoding_index})
    col_type = TYPES[type_index]
    encoding = ENCODINGS[encoding_index]
    if expect_encoding is not None and encoding != expect_encoding:
        raise EncodingError("块头编码与载荷编码不匹配",
                            {"declared": expect_encoding, "actual": encoding})
    if expect_type is not None and col_type != expect_type:
        raise EncodingError("块头列类型与载荷类型不匹配",
                            {"declared": expect_type, "actual": col_type})
    stored, _ = _get_blob(payload, 7)
    try:
        body = zlib.decompress(stored) if compressed_flag else stored
    except zlib.error as exc:
        raise PayloadError(f"zlib 解压失败：{exc}") from exc
    marker = body[:1]
    decoder = _DECODERS.get(marker)
    if decoder is None:
        raise EncodingError("载荷编码标记无法识别", {"marker": marker.hex()})
    payload_encoding = {b"R": RAW, b"D": DICT, b"X": RLE}[marker]
    if payload_encoding != encoding:
        raise EncodingError("块头编码与载荷内嵌编码不匹配",
                            {"header": encoding, "payload": payload_encoding})
    values, end = decoder(body, 1, col_type)
    if end != len(body):
        raise PayloadError("块存在未消费的尾部字节",
                           {"consumed": end, "size": len(body)})
    return values


def block_summary(values: list) -> str:
    """结果行集合的稳定摘要（规范化 JSON 后 SHA-256）。"""
    return hashlib.sha256(canonical_values(values)).hexdigest()


def canonical_values(values: list) -> bytes:
    """把结果行集合规范化为确定性字节序列。"""
    col_type = infer_type(values)
    parts = [col_type.encode("ascii"), b"|",
             str(len(values)).encode("ascii"), b"\n"]
    for value in values:
        parts.append(_scalar_bytes(value, col_type))
        parts.append(b"\n")
    return b"".join(parts)
