"""r1 / r2 / r3 历史回执格式的编解码与归一。

线路信封（长度前缀显式自洽）::

    magic=b"RLR1" | format_version:1 字节(1/2/3) | payload_len:4 字节大端
    | payload: 规范化 JSON 字节（长度恰为 payload_len，禁止尾随字节）

字段集合：
  r1  batch_id, worker_id, destination, output_digest
  r2  r1 + param_digest, quant, elapsed_ms
  r3  r2 + request_seqs(升序整数集合), param_version
"""

from .digest import canonical_json
from .errors import CodecError

MAGIC = b"RLR1"
SUPPORTED_FORMATS = ("r1", "r2", "r3")
_FORMAT_RANK = {"r1": 1, "r2": 2, "r3": 3}

_BASE_FIELDS = ("batch_id", "worker_id", "destination", "output_digest")
_R2_FIELDS = ("param_digest", "quant", "elapsed_ms")
_R3_FIELDS = ("request_seqs", "param_version")


def format_rank(fmt: str) -> int:
    return _FORMAT_RANK[fmt]


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_str(obj, key, fmt):
    if not isinstance(obj.get(key), str) or not obj[key]:
        raise CodecError(
            f"{fmt} 字段 {key} 必须是非空字符串", field=key, format=fmt
        )


def _check_digest(obj, key, fmt):
    value = obj.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise CodecError(
            f"{fmt} 摘要字段 {key} 必须是 64 位十六进制字符串",
            field=key, format=fmt,
        )
    int(value, 16)


def encode_receipt(fmt: str, fields: dict) -> bytes:
    """按目标格式裁剪字段并编码为长度前缀信封。"""
    if fmt not in SUPPORTED_FORMATS:
        raise CodecError(f"未知回执格式版本 {fmt}", field="format")
    payload_obj = {}
    for key in _BASE_FIELDS:
        payload_obj[key] = fields[key]
    if _FORMAT_RANK[fmt] >= 2:
        for key in _R2_FIELDS:
            payload_obj[key] = fields[key]
    if _FORMAT_RANK[fmt] >= 3:
        payload_obj["request_seqs"] = sorted(fields["request_seqs"])
        payload_obj["param_version"] = fields["param_version"]
    payload = canonical_json(payload_obj)
    header = MAGIC + bytes((_FORMAT_RANK[fmt],)) + len(payload).to_bytes(4, "big")
    return header + payload


def decode_receipt(data: bytes):
    """解码并做长度自洽与字段类型校验，返回 (格式, 字段字典, 原始字节长度)。"""
    if not isinstance(data, (bytes, bytearray)):
        raise CodecError("回执必须是字节串", field="envelope")
    if len(data) < 9:
        raise CodecError(
            f"回执长度 {len(data)} 小于信封头 9 字节，长度字段不自洽",
            field="payload_len",
        )
    if bytes(data[:4]) != MAGIC:
        raise CodecError("回执魔数不匹配", field="magic")
    rank = data[4]
    if rank not in (1, 2, 3):
        raise CodecError(f"不支持的回执格式序号 {rank}", field="format")
    fmt = f"r{rank}"
    declared_len = int.from_bytes(data[5:9], "big")
    if declared_len != len(data) - 9:
        raise CodecError(
            f"长度字段声明 {declared_len} 与实际载荷 {len(data) - 9} 不一致",
            field="payload_len", declared=declared_len,
            actual=len(data) - 9,
        )
    import json
    try:
        obj = json.loads(data[9:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodecError(f"载荷不是合法 JSON: {exc}", field="payload", format=fmt)
    if not isinstance(obj, dict):
        raise CodecError("载荷必须是 JSON 对象", field="payload", format=fmt)
    _validate_fields(fmt, obj)
    return fmt, obj, len(data)


def _validate_fields(fmt: str, obj: dict) -> None:
    for key in ("batch_id", "worker_id", "destination"):
        _check_str(obj, key, fmt)
    _check_digest(obj, "output_digest", fmt)
    if _FORMAT_RANK[fmt] >= 2:
        _check_digest(obj, "param_digest", fmt)
        _check_str(obj, "quant", fmt)
        if not _is_int(obj.get("elapsed_ms")) or obj["elapsed_ms"] < 0:
            raise CodecError(
                f"{fmt} 字段 elapsed_ms 必须是非负整数",
                field="elapsed_ms", format=fmt,
            )
    if _FORMAT_RANK[fmt] >= 3:
        seqs = obj.get("request_seqs")
        if not isinstance(seqs, list) or not seqs:
            raise CodecError(
                f"{fmt} 字段 request_seqs 必须是非空整数数组",
                field="request_seqs", format=fmt,
            )
        if not all(_is_int(v) for v in seqs):
            raise CodecError(
                f"{fmt} 字段 request_seqs 元素必须是整数",
                field="request_seqs", format=fmt,
            )
        if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
            raise CodecError(
                f"{fmt} 字段 request_seqs 必须升序且无重复",
                field="request_seqs", format=fmt,
            )
        if not _is_int(obj.get("param_version")) or obj["param_version"] <= 0:
            raise CodecError(
                f"{fmt} 字段 param_version 必须是正整数",
                field="param_version", format=fmt,
            )


# 归一视图字段全集（顺序固定，避免任何遍历顺序依赖）。
NORMALIZED_FIELDS = (
    "source_format", "batch_id", "worker_id", "destination", "output_digest",
    "param_digest", "quant", "elapsed_ms", "request_seqs", "param_version",
    "filled_fields", "raw_len",
)

_FILL_BASIS = {
    "param_digest": "context:worker-parameter-summary",
    "quant": "context:worker-declaration",
    "elapsed_ms": "default:0:not-carried-by-legacy-format",
    "request_seqs": "context:batch-plan-request-sequences",
    "param_version": "context:rule-param-version",
}


def normalize_receipt(fmt: str, obj: dict, context: dict, raw_len: int) -> dict:
    """把历史格式归一为内部视图，携带来源格式、补齐字段与补齐依据。

    context 必含 batch_id, worker_id, destination, quant, param_version,
    request_seqs, param_digest；仅在历史格式未携带该字段时补齐。
    """
    view = {
        "source_format": fmt,
        "batch_id": obj["batch_id"],
        "worker_id": obj["worker_id"],
        "destination": obj["destination"],
        "output_digest": obj["output_digest"],
        "param_digest": None,
        "quant": None,
        "elapsed_ms": None,
        "request_seqs": None,
        "param_version": None,
        "filled_fields": [],
        "raw_len": raw_len,
    }
    rank = _FORMAT_RANK[fmt]
    if rank >= 2:
        view["param_digest"] = obj["param_digest"]
        view["quant"] = obj["quant"]
        view["elapsed_ms"] = obj["elapsed_ms"]
    else:
        view["param_digest"] = context["param_digest"]
        view["quant"] = context["quant"]
        view["elapsed_ms"] = 0
    if rank >= 3:
        view["request_seqs"] = list(obj["request_seqs"])
        view["param_version"] = obj["param_version"]
    else:
        view["request_seqs"] = sorted(context["request_seqs"])
        view["param_version"] = context["param_version"]
    for field_name in NORMALIZED_FIELDS[5:10]:
        carried = (rank >= 2 and field_name in _R2_FIELDS) or (
            rank >= 3 and field_name in _R3_FIELDS
        )
        if not carried:
            view["filled_fields"].append(
                {"field": field_name, "basis": _FILL_BASIS[field_name]}
            )
    view["filled_fields"].sort(key=lambda item: item["field"])
    return {key: view[key] for key in NORMALIZED_FIELDS}

