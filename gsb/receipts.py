"""h1/h2/h3 三种参数格式版本回执的编解码与归一。

线格式（确定性强校验）：
    行1: GSB-RECEIPT <v1|v2|v3>
    行2: LEN <payload 字节数>
    其后: <payload 字节>（canonical JSON，键排序、无空白）

解码校验：长度字段自洽、必需字段齐全、摘要绑定关系。
归一：旧格式缺失字段从派发上下文补齐，并记录补齐依据。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .errors import e_format

FORMAT_VERSIONS = ("v1", "v2", "v3")
FORMAT_RANK = {"v1": 1, "v2": 2, "v3": 3}
ROUTES = ("stable", "gray")

REQUIRED_FIELDS = {
    "v1": ("batch_id", "route", "output_digest"),
    "v2": (
        "batch_id",
        "route",
        "output_digest",
        "param_digest",
        "quant_profile",
        "elapsed_ms",
    ),
    "v3": (
        "batch_id",
        "route",
        "output_digest",
        "param_digest",
        "quant_profile",
        "elapsed_ms",
        "sample_ids",
        "param_version",
    ),
}

HEADER = "GSB-RECEIPT"


def compute_output_digest(
    batch_id: str, route: str, param_digest: str, sample_ids
) -> str:
    joined = ",".join(sample_ids)
    payload = f"gsb|{batch_id}|{route}|{param_digest}|{joined}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_frame(version: str, data: dict) -> str:
    if version not in FORMAT_VERSIONS:
        raise ValueError(f"未知格式版本: {version}")
    payload = _canonical_json(data)
    return f"{HEADER} {version}\nLEN {len(payload)}\n".join(
        ["", payload.decode("utf-8")]
    )


def decode_frame(frame: str) -> tuple[str, dict]:
    """解码并强校验线格式，返回 (格式版本, 字段字典)。"""
    if not isinstance(frame, str):
        raise e_format("回执帧必须为字符串", batch_id="<unknown>", field="frame")
    head, sep, payload = frame.partition("\n")
    if not sep:
        raise e_format("回执帧缺少头部分隔", batch_id="<unknown>", field="frame")
    parts = head.split(" ")
    if len(parts) != 2 or parts[0] != HEADER:
        raise e_format("回执帧头部非法", batch_id="<unknown>", field="frame")
    version = parts[1]
    if version not in FORMAT_VERSIONS:
        raise e_format(
            "未知格式版本", batch_id="<unknown>", field="format_version"
        )
    len_line, sep, payload = payload.partition("\n")
    if not sep or not len_line.startswith("LEN "):
        raise e_format("缺少 LEN 长度行", batch_id="<unknown>", field="length")
    try:
        declared_len = int(len_line[4:])
    except ValueError:
        raise e_format(
            "LEN 长度字段不是整数", batch_id="<unknown>", field="length"
        ) from None
    actual_len = len(payload.encode("utf-8"))
    if declared_len != actual_len:
        raise e_format(
            f"长度字段不自洽: 声明 {declared_len} 实际 {actual_len}",
            batch_id="<unknown>",
            field="length",
        )
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        raise e_format(
            "回执负载不是合法 JSON", batch_id="<unknown>", field="payload"
        ) from None
    if not isinstance(data, dict):
        raise e_format("回执负载必须为对象", batch_id="<unknown>", field="payload")
    batch_id = data.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise e_format(
            "缺少必需字段 batch_id", batch_id="<unknown>", field="batch_id"
        )
    for name in REQUIRED_FIELDS[version]:
        if name not in data:
            raise e_format(
                f"缺少必需字段 {name}", batch_id=batch_id, field=name
            )
    if data["route"] not in ROUTES:
        raise e_format("route 取值非法", batch_id=batch_id, field="route")
    if not isinstance(data["output_digest"], str) or len(data["output_digest"]) != 64:
        raise e_format(
            "output_digest 必须为 64 位十六进制摘要",
            batch_id=batch_id,
            field="output_digest",
        )
    if version in ("v2", "v3"):
        if not isinstance(data["param_digest"], str) or not data["param_digest"]:
            raise e_format(
                "param_digest 必须为非空字符串", batch_id=batch_id, field="param_digest"
            )
        if not isinstance(data["quant_profile"], str) or not data["quant_profile"]:
            raise e_format(
                "quant_profile 必须为非空字符串",
                batch_id=batch_id,
                field="quant_profile",
            )
        elapsed = data["elapsed_ms"]
        if not isinstance(elapsed, int) or isinstance(elapsed, bool) or elapsed < 0:
            raise e_format(
                "elapsed_ms 必须为非负整数", batch_id=batch_id, field="elapsed_ms"
            )
    if version == "v3":
        sample_ids = data["sample_ids"]
        if not isinstance(sample_ids, list) or not all(
            isinstance(s, str) and s for s in sample_ids
        ):
            raise e_format(
                "sample_ids 必须为非空字符串数组",
                batch_id=batch_id,
                field="sample_ids",
            )
        pv = data["param_version"]
        if not isinstance(pv, int) or isinstance(pv, bool) or pv <= 0:
            raise e_format(
                "param_version 必须为正整数", batch_id=batch_id, field="param_version"
            )
    return version, data


@dataclass(frozen=True)
class NormContext:
    """归一上下文：来自派发时刻冻结的信息与后端声明。"""

    sample_ids: tuple[str, ...]
    param_digest: str
    param_version: int
    quant_profile: str


@dataclass
class NormalizedReceipt:
    batch_id: str
    route: str
    output_digest: str
    param_digest: str
    quant_profile: str
    elapsed_ms: int
    sample_ids: tuple[str, ...]
    param_version: int
    source_format: str
    filled_fields: list[str] = field(default_factory=list)
    fill_basis: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "route": self.route,
            "output_digest": self.output_digest,
            "param_digest": self.param_digest,
            "quant_profile": self.quant_profile,
            "elapsed_ms": self.elapsed_ms,
            "sample_ids": list(self.sample_ids),
            "param_version": self.param_version,
            "source_format": self.source_format,
            "filled_fields": list(self.filled_fields),
            "fill_basis": dict(self.filled_basis_sorted()),
        }

    def filled_basis_sorted(self):
        for key in sorted(self.fill_basis):
            yield key, self.fill_basis[key]

    @staticmethod
    def from_dict(data: dict) -> "NormalizedReceipt":
        return NormalizedReceipt(
            batch_id=data["batch_id"],
            route=data["route"],
            output_digest=data["output_digest"],
            param_digest=data["param_digest"],
            quant_profile=data["quant_profile"],
            elapsed_ms=data["elapsed_ms"],
            sample_ids=tuple(data["sample_ids"]),
            param_version=data["param_version"],
            source_format=data["source_format"],
            filled_fields=list(data.get("filled_fields", [])),
            fill_basis=dict(data.get("fill_basis", {})),
        )


def normalize(version: str, data: dict, ctx: NormContext) -> NormalizedReceipt:
    """把任一格式版本的已解码回执归一到 v3 全集字段。"""
    batch_id = data["batch_id"]
    filled: list[str] = []
    basis: dict[str, str] = {}

    def take(name: str, default, basis_text: str):
        if name in data:
            return data[name]
        filled.append(name)
        basis[name] = basis_text
        return default

    param_digest = take("param_digest", ctx.param_digest, "dispatch:param_digest")
    quant_profile = take("quant_profile", ctx.quant_profile, "backend_decl:quant_profile")
    elapsed_ms = take("elapsed_ms", 0, "default:elapsed_ms=0")
    sample_ids = take("sample_ids", list(ctx.sample_ids), "dispatch:sample_ids")
    param_version = take("param_version", ctx.param_version, "dispatch:param_version")

    sample_ids = tuple(sorted(sample_ids))
    expected = compute_output_digest(batch_id, data["route"], param_digest, sample_ids)
    if data["output_digest"] != expected:
        raise e_format(
            "摘要绑定校验失败: output_digest 与字段内容不匹配",
            batch_id=batch_id,
            field="output_digest",
        )
    return NormalizedReceipt(
        batch_id=batch_id,
        route=data["route"],
        output_digest=data["output_digest"],
        param_digest=param_digest,
        quant_profile=quant_profile,
        elapsed_ms=elapsed_ms,
        sample_ids=sample_ids,
        param_version=param_version,
        source_format=version,
        filled_fields=filled,
        fill_basis=basis,
    )


def decode_and_normalize(frame: str, ctx: NormContext) -> NormalizedReceipt:
    version, data = decode_frame(frame)
    return normalize(version, data, ctx)
