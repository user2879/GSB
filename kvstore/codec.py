"""键值记录与索引条目的持久化编解码及合法性校验。

键规则：非空字符串，段由 ``[A-Za-z0-9_-]+`` 组成，段之间用 ``/`` 分隔，
总长不超过 :data:`MAX_KEY_LEN`。前缀必须是“段对齐”的前缀（允许空前缀表示
整个键空间根）。所有校验失败均抛出 :class:`RecordError`（业务码 E_RECORD），
并在 ``details`` 中给出首个出错记录的键与字段。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .errors import RecordError

FORMAT_VERSION = 1
MAX_KEY_LEN = 512
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")

RECORD_FIELDS = ("key", "value", "expire_ms", "write_seq", "format_version")
INDEX_FIELDS = ("prefix", "index_value", "entry_version", "ref_key")


def validate_key(key: Any) -> str:
    if not isinstance(key, str):
        raise RecordError(
            "键必须为字符串",
            details={"key": _safe(key), "field": "key", "reason": "not_a_string"},
        )
    if not key or len(key) > MAX_KEY_LEN:
        raise RecordError(
            "键长度非法",
            details={"key": _safe(key), "field": "key", "reason": "bad_length"},
        )
    segments = key.split("/")
    for segment in segments:
        if not _SEGMENT_RE.match(segment):
            raise RecordError(
                "键包含非法段",
                details={
                    "key": key,
                    "field": "key",
                    "reason": "bad_segment",
                    "segment": _safe(segment),
                },
            )
    return key


def validate_prefix(prefix: Any) -> str:
    if prefix is None:
        prefix = ""
    if not isinstance(prefix, str):
        raise RecordError(
            "前缀必须为字符串",
            details={"key": _safe(prefix), "field": "prefix", "reason": "not_a_string"},
        )
    if prefix == "":
        return prefix
    if len(prefix) > MAX_KEY_LEN or prefix.startswith("/") or prefix.endswith("/"):
        raise RecordError(
            "前缀格式非法",
            details={"key": prefix, "field": "prefix", "reason": "bad_shape"},
        )
    for segment in prefix.split("/"):
        if not _SEGMENT_RE.match(segment):
            raise RecordError(
                "前缀包含非法段",
                details={"key": prefix, "field": "prefix", "reason": "bad_segment"},
            )
    return prefix


def prefixes_of(key: str) -> list[str]:
    """返回键的全部段对齐前缀路径，包含键自身（最深层）。"""
    validate_key(key)
    segments = key.split("/")
    result: list[str] = []
    for depth in range(1, len(segments) + 1):
        result.append("/".join(segments[:depth]))
    return result


def validate_record(record: Any) -> dict[str, Any]:
    """校验一条键值记录，返回规范化后的记录；首错即整体拒绝。"""
    if not isinstance(record, dict):
        raise RecordError(
            "记录必须为对象",
            details={"key": None, "field": "record", "reason": "not_an_object"},
        )
    key = validate_key(record.get("key"))

    value = record.get("value")
    if not isinstance(value, str):
        raise RecordError(
            "值必须为字符串",
            details={"key": key, "field": "value", "reason": "not_a_string"},
        )

    expire_ms = record.get("expire_ms")
    if not isinstance(expire_ms, int) or isinstance(expire_ms, bool) or expire_ms < 0:
        raise RecordError(
            "毫秒级过期时间戳必须为非负整数",
            details={"key": key, "field": "expire_ms", "reason": "bad_expire_ms"},
        )

    write_seq = record.get("write_seq")
    if not isinstance(write_seq, int) or isinstance(write_seq, bool) or write_seq < 0:
        raise RecordError(
            "写入序号必须为非负整数",
            details={"key": key, "field": "write_seq", "reason": "bad_write_seq"},
        )

    format_version = record.get("format_version", FORMAT_VERSION)
    if format_version != FORMAT_VERSION:
        raise RecordError(
            "不支持的记录格式版本",
            details={
                "key": key,
                "field": "format_version",
                "reason": "unsupported_version",
                "actual": format_version,
                "expected": FORMAT_VERSION,
            },
        )

    return {
        "key": key,
        "value": value,
        "expire_ms": expire_ms,
        "write_seq": write_seq,
        "format_version": FORMAT_VERSION,
    }


def validate_index_entry(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise RecordError(
            "索引条目必须为对象",
            details={"key": None, "field": "index_entry", "reason": "not_an_object"},
        )
    prefix = validate_prefix(entry.get("prefix"))
    ref_key = validate_key(entry.get("ref_key"))
    index_value = entry.get("index_value")
    if not isinstance(index_value, str):
        raise RecordError(
            "索引项值必须为字符串",
            details={"key": ref_key, "field": "index_value", "reason": "not_a_string"},
        )
    entry_version = entry.get("entry_version")
    if not isinstance(entry_version, int) or isinstance(entry_version, bool) or entry_version < 0:
        raise RecordError(
            "条目版本号必须为非负整数",
            details={"key": ref_key, "field": "entry_version", "reason": "bad_version"},
        )
    return {
        "prefix": prefix,
        "index_value": index_value,
        "entry_version": entry_version,
        "ref_key": ref_key,
    }


def check_write_seq_monotonic(previous_seq: int | None, write_seq: int, key: str) -> None:
    """同一键上写入序号必须严格单调递增。"""
    if previous_seq is not None and write_seq <= previous_seq:
        raise RecordError(
            "同一键的写入序号必须单调递增",
            details={
                "key": key,
                "field": "write_seq",
                "reason": "non_monotonic_seq",
                "previous": previous_seq,
                "actual": write_seq,
            },
        )


def _safe(value: Any) -> Any:
    if isinstance(value, str) and len(value) <= 128:
        return value
    try:
        return repr(value)[:128]
    except Exception:  # pragma: no cover
        return "<unprintable>"


def canonical_json(value: Any) -> bytes:
    """确定性 JSON 编码：键排序、无多余空白，供摘要与逐字节比对使用。"""
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def digest_records(records: Iterable[dict[str, Any]]) -> str:
    import hashlib

    hasher = hashlib.sha256()
    for record in records:
        hasher.update(canonical_json(record))
        hasher.update(b"\n")
    return hasher.hexdigest()
