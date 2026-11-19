"""确定性编码与内容哈希。

约定（满足“输出 UTF-8、数值固定字节序”）：
- 整数：固定小端 8 字节有符号；
- 浮点：固定小端 IEEE-754 double；
- 文本：UTF-8，带 4 字节小端长度前缀；
- 列表/元组：长度前缀 + 逐元素编码，元素顺序显式保留；
- 字典：按键的 UTF-8 字节序排序后编码，结果不依赖 dict 遍历顺序；
- 内容哈希统一使用 SHA-256（hex）。
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any, Iterable

ENDIAN = "<"
TEXT_ENCODING = "utf-8"


def encode_int(value: int) -> bytes:
    return struct.pack(ENDIAN + "q", int(value))


def encode_float(value: float) -> bytes:
    return struct.pack(ENDIAN + "d", float(value))


def encode_bool(value: bool) -> bytes:
    return b"\x01" if value else b"\x00"


def encode_text(value: str) -> bytes:
    raw = value.encode(TEXT_ENCODING)
    return struct.pack(ENDIAN + "I", len(raw)) + raw


def encode_bytes(value: bytes) -> bytes:
    return struct.pack(ENDIAN + "I", len(value)) + value


def canonical_encoding(value: Any) -> bytes:
    """把嵌套结构编码为确定性字节串。"""
    out = bytearray()
    _emit(value, out)
    return bytes(out)


def _emit(value: Any, out: bytearray) -> None:
    if value is None:
        out.append(ord("N"))
    elif isinstance(value, bool):
        out.append(ord("b"))
        out.extend(encode_bool(value))
    elif isinstance(value, int):
        out.append(ord("i"))
        out.extend(encode_int(value))
    elif isinstance(value, float):
        out.append(ord("f"))
        out.extend(encode_float(value))
    elif isinstance(value, str):
        out.append(ord("s"))
        out.extend(encode_text(value))
    elif isinstance(value, (bytes, bytearray)):
        out.append(ord("y"))
        out.extend(encode_bytes(bytes(value)))
    elif isinstance(value, dict):
        out.append(ord("{"))
        items = sorted(
            ((canonical_encoding(k), v) for k, v in value.items()),
            key=lambda kv: kv[0],
        )
        out.extend(encode_int(len(items)))
        for key_bytes, val in items:
            out.extend(encode_bytes(key_bytes))
            _emit(val, out)
    elif isinstance(value, (list, tuple)):
        tag = ord("[") if isinstance(value, list) else ord("(")
        out.append(tag)
        out.extend(encode_int(len(value)))
        for item in value:
            _emit(item, out)
    else:
        raise TypeError(f"不可确定性编码的类型: {type(value)!r}")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_encoding(value)).hexdigest()


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def windows_path(path: str | Path) -> str:
    """把任意本地路径表示为 Windows 风格（反斜杠、盘符大写）。

    运行环境可能是 POSIX，但报告与清单中的路径统一以 Windows 风格输出，
    保证跨平台复现实验时路径文本一致。POSIX 绝对路径映射到 ``C:\`` 下。
    """
    text = str(path).replace("/", "\\")
    if "\\" in text and ":" in text:
        return text
    if text.startswith("\\"):
        return "C:" + text
    if text.startswith("\\"):
        return "C:" + text
    return "C:\\" + text.lstrip("\\")


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=TEXT_ENCODING, newline="\n")


def read_text_file(path: Path) -> str:
    return path.read_text(encoding=TEXT_ENCODING)


def stable_join(parts: Iterable[Any]) -> str:
    return "|".join(str(p) for p in parts)
