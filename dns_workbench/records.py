"""缓存记录 v1/v2/v3 与上游应答 u1/u2 的编解码、强校验与归一。

版本字段：
- 缓存记录 ``version`` 取值 v1/v2/v3；
- 上游应答 ``format`` 取值 u1/u2。
未知版本抛 E_VER；必需字段缺失、TTL 越界等抛 E_FORMAT 并列出字段；
v3 来源序列号倒退抛 E_VER。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

from .errors import FormatError, VersionError

CACHE_VERSIONS = ("v1", "v2", "v3")
UPSTREAM_FORMATS = ("u1", "u2")
MAX_TTL = 7 * 24 * 3600

# v1 基础字段；v2/v3 扩展字段在视图降级时被稳定剥离
V1_FIELDS = ("version", "name", "qtype", "value", "ttl")
V2_ADDED_FIELDS = ("rcode", "aa", "source")
V3_ADDED_FIELDS = ("ecs_prefix", "hits", "seq")
STRIP_TO_V1 = tuple(sorted(V2_ADDED_FIELDS + V3_ADDED_FIELDS))

# v3 视图补齐使用的显式默认值
V3_DEFAULTS = {"ecs_prefix": None, "hits": 0, "seq": 0}

SOURCE_AUTHORITATIVE = "authoritative"
SOURCE_UPSTREAM = "upstream"
SOURCE_TYPES = (SOURCE_AUTHORITATIVE, SOURCE_UPSTREAM)


def canonical_digest(payload: dict[str, Any]) -> str:
    """对归一内容计算稳定 SHA-256 摘要（与来源格式版本无关）。"""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    """三版本归一后的内部模型。"""

    name: str
    qtype: int
    value: str
    ttl: int
    source_version: str
    content_digest: str
    rcode: int = 0
    aa: bool = False
    source: str = SOURCE_AUTHORITATIVE
    ecs_prefix: str | None = None
    hits: int = 0
    seq: int = 0

    def key(self) -> tuple[str, int]:
        return (self.name, self.qtype)

    def normalized_payload(self, remaining_ttl: int | None = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "qtype": self.qtype,
            "value": self.value,
            "ttl": self.ttl if remaining_ttl is None else remaining_ttl,
            "rcode": self.rcode,
            "aa": self.aa,
            "source": self.source,
            "ecs_prefix": self.ecs_prefix,
            "hits": self.hits,
            "seq": self.seq,
        }


@dataclass
class ViewResult:
    """视图导出结果，附稳定的剥离/补齐字段集合。"""

    record: dict[str, Any]
    stripped_fields: tuple[str, ...] = ()
    filled_fields: tuple[str, ...] = ()


def _require_fields(payload: dict[str, Any], required: Iterable[str]) -> None:
    missing = [name for name in required if name not in payload]
    if missing:
        raise FormatError("必需字段缺失", fields=missing, context={"version": payload.get("version")})


def _validate_ttl(payload: dict[str, Any]) -> int:
    ttl = payload.get("ttl")
    if not isinstance(ttl, int) or isinstance(ttl, bool):
        raise FormatError("TTL 必须为整数", fields=["ttl"], context={"ttl": ttl})
    if ttl < 0:
        raise FormatError("TTL 不得为负", fields=["ttl"], context={"ttl": ttl})
    if ttl > MAX_TTL:
        raise FormatError("TTL 超过上限", fields=["ttl"], context={"ttl": ttl, "max_ttl": MAX_TTL})
    return ttl


def _base_validation(payload: dict[str, Any]) -> tuple[str, int, str, int]:
    version = payload.get("version")
    if version not in CACHE_VERSIONS:
        raise VersionError(
            "未知缓存记录版本",
            fields=["version"],
            context={"version": version, "known": list(CACHE_VERSIONS)},
        )
    _require_fields(payload, V1_FIELDS)
    if not isinstance(payload["name"], str) or not payload["name"]:
        raise FormatError("查询名非法", fields=["name"])
    if not isinstance(payload["qtype"], int) or isinstance(payload["qtype"], bool) or payload["qtype"] <= 0:
        raise FormatError("查询类型非法", fields=["qtype"], context={"qtype": payload["qtype"]})
    if not isinstance(payload["value"], str):
        raise FormatError("记录值摘要必须为字符串", fields=["value"])
    ttl = _validate_ttl(payload)
    return version, payload["qtype"], payload["name"], ttl


def load_cache_record(payload: dict[str, Any], *, last_seq_by_key: dict[tuple[str, int], int] | None = None) -> CacheEntry:
    """加载单条缓存记录并归一；v3 校验来源序列号单调性。"""
    if not isinstance(payload, dict):
        raise FormatError("缓存记录必须为 JSON 对象")
    version, _qtype, _name, ttl = _base_validation(payload)
    entry = CacheEntry(
        name=payload["name"],
        qtype=payload["qtype"],
        value=payload["value"],
        ttl=ttl,
        source_version=version,
        content_digest="",
    )
    if version in ("v2", "v3"):
        _require_fields(payload, V2_ADDED_FIELDS)
        if not isinstance(payload["rcode"], int) or not 0 <= payload["rcode"] <= 15:
            raise FormatError("应答码必须为 0..15 的整数", fields=["rcode"])
        if not isinstance(payload["aa"], bool):
            raise FormatError("权威标志必须为布尔值", fields=["aa"])
        if payload["source"] not in SOURCE_TYPES:
            raise FormatError("来源类型标识非法", fields=["source"], context={"source": payload["source"]})
        entry.rcode = payload["rcode"]
        entry.aa = payload["aa"]
        entry.source = payload["source"]
    if version == "v3":
        _require_fields(payload, V3_ADDED_FIELDS)
        ecs = payload["ecs_prefix"]
        if ecs is not None and not isinstance(ecs, str):
            raise FormatError("EDNS 客户端子网前缀必须为字符串或 null", fields=["ecs_prefix"])
        if not isinstance(payload["hits"], int) or isinstance(payload["hits"], bool) or payload["hits"] < 0:
            raise FormatError("命中计数必须为非负整数", fields=["hits"])
        if not isinstance(payload["seq"], int) or isinstance(payload["seq"], bool) or payload["seq"] < 0:
            raise FormatError("来源序列号必须为非负整数", fields=["seq"])
        entry.ecs_prefix = ecs
        entry.hits = payload["hits"]
        entry.seq = payload["seq"]
        last_seq = (last_seq_by_key or {}).get(entry.key())
        if last_seq is not None and entry.seq < last_seq:
            raise VersionError(
                "来源序列号倒退",
                fields=["seq"],
                context={"key": list(entry.key()), "seq": entry.seq, "last_seq": last_seq},
            )
    entry.content_digest = canonical_digest(entry.normalized_payload())
    return entry


def load_cache_records(payloads: Iterable[dict[str, Any]]) -> list[CacheEntry]:
    last_seq: dict[tuple[str, int], int] = {}
    entries: list[CacheEntry] = []
    for payload in payloads:
        entry = load_cache_record(payload, last_seq_by_key=last_seq)
        previous = last_seq.get(entry.key())
        if previous is None or entry.seq > previous:
            last_seq[entry.key()] = entry.seq
        entries.append(entry)
    return entries


def _entry_view_base(entry: CacheEntry, remaining_ttl: int | None) -> dict[str, Any]:
    ttl = entry.ttl if remaining_ttl is None else remaining_ttl
    return {
        "name": entry.name,
        "qtype": entry.qtype,
        "value": entry.value,
        "ttl": ttl,
    }


def export_view(entry: CacheEntry, target_version: str, *, remaining_ttl: int | None = None) -> ViewResult:
    """导出指定版本视图。

    - 向 v1 降级：稳定列出被剥离的 v2/v3 字段；
    - 向 v3 补齐：按 V3_DEFAULTS 显式填充并列出补齐字段；
    - 任何视图下 name/qtype/剩余 TTL 逐字节一致。
    """
    if target_version not in CACHE_VERSIONS:
        raise VersionError("未知目标视图版本", fields=["version"], context={"version": target_version})
    base = _entry_view_base(entry, remaining_ttl)
    base["version"] = target_version
    ordered = {"version": target_version}
    ordered.update(base)
    if target_version == "v1":
        stripped = list(STRIP_TO_V1) if entry.source_version != "v1" else []
        return ViewResult(record=dict(ordered), stripped_fields=tuple(stripped))
    ordered.update({
        "rcode": entry.rcode,
        "aa": entry.aa,
        "source": entry.source,
    })
    filled: list[str] = []
    if target_version == "v2":
        if entry.source_version == "v1":
            filled = ["aa", "rcode", "source"]
        return ViewResult(record=dict(ordered), filled_fields=tuple(sorted(filled)))
    # v3 扩展字段：高版本原值保留，低版本按显式默认值补齐
    filled = []
    if entry.source_version == "v1":
        filled = sorted(set(V2_ADDED_FIELDS) | set(V3_ADDED_FIELDS))
    elif entry.source_version == "v2":
        filled = list(V3_ADDED_FIELDS)
    ordered["ecs_prefix"] = entry.ecs_prefix if entry.ecs_prefix is not None else V3_DEFAULTS["ecs_prefix"]
    ordered["hits"] = entry.hits
    ordered["seq"] = entry.seq
    return ViewResult(record=dict(ordered), filled_fields=tuple(sorted(filled)))


@dataclass
class UpstreamAnswer:
    """u1/u2 归一后的上游应答模型。"""

    format_version: str
    txn_id: int
    question: dict[str, Any]
    answers: list[dict[str, Any]]
    edns: bool = False
    dnssec_ok: bool = False
    content_digest: str = ""


U1_FIELDS = ("format", "txn_id", "question", "answers")
U2_ADDED_FIELDS = ("edns", "dnssec_ok")


def load_upstream_answer(payload: dict[str, Any]) -> UpstreamAnswer:
    if not isinstance(payload, dict):
        raise FormatError("上游应答必须为 JSON 对象")
    fmt = payload.get("format")
    if fmt not in UPSTREAM_FORMATS:
        raise VersionError(
            "未知上游应答格式",
            fields=["format"],
            context={"format": fmt, "known": list(UPSTREAM_FORMATS)},
        )
    _require_fields(payload, U1_FIELDS)
    txn_id = payload["txn_id"]
    if not isinstance(txn_id, int) or not 0 <= txn_id <= 0xFFFF:
        raise FormatError("事务标识必须为 0..65535 的整数", fields=["txn_id"])
    question = payload["question"]
    if not isinstance(question, dict) or not {"name", "qtype"}.issubset(question):
        raise FormatError("问题回显必须含 name/qtype", fields=["question"])
    if not isinstance(question.get("name"), str) or not isinstance(question.get("qtype"), int):
        raise FormatError("问题回显字段类型非法", fields=["question"])
    answers = payload["answers"]
    if not isinstance(answers, list):
        raise FormatError("应答记录必须为数组", fields=["answers"])
    for index, answer in enumerate(answers):
        if not isinstance(answer, dict) or not {"name", "qtype", "value", "ttl"}.issubset(answer):
            raise FormatError("应答记录字段不完整", fields=[f"answers[{index}]"])
    normalized = UpstreamAnswer(
        format_version=fmt,
        txn_id=txn_id,
        question=dict(question),
        answers=[dict(answer) for answer in answers],
    )
    if fmt == "u2":
        _require_fields(payload, U2_ADDED_FIELDS)
        if not isinstance(payload["edns"], bool) or not isinstance(payload["dnssec_ok"], bool):
            raise FormatError("EDNS/DO 标志必须为布尔值", fields=list(U2_ADDED_FIELDS))
        normalized.edns = payload["edns"]
        normalized.dnssec_ok = payload["dnssec_ok"]
    normalized.content_digest = canonical_digest(
        {
            "txn_id": normalized.txn_id,
            "question": normalized.question,
            "answers": normalized.answers,
            "edns": normalized.edns,
            "dnssec_ok": normalized.dnssec_ok,
        }
    )
    return normalized


def dump_upstream_answer(answer: UpstreamAnswer) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": answer.format_version,
        "txn_id": answer.txn_id,
        "question": dict(answer.question),
        "answers": [dict(item) for item in answer.answers],
    }
    if answer.format_version == "u2":
        payload["edns"] = answer.edns
        payload["dnssec_ok"] = answer.dnssec_ok
    return payload


def entry_as_dict(entry: CacheEntry) -> dict[str, Any]:
    data = asdict(entry)
    return data
