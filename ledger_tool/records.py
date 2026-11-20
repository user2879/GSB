"""r1/r2/r3 归属记录编解码与强校验.

r1: 请求标识、批次槽位、输出摘要
r2: + 到达序号、后端版本指纹、整数耗时毫秒
r3: + 组内名次、被覆盖的重复投递引用、参与批次的指纹快照
非法记录整体拒绝, 以 E_FORMAT 指出首个出错字段.
"""

from .canon import is_hex_digest
from .errors import E_FORMAT, LedgerError

CURRENT_FORMAT = "r3"

FORMAT_FIELDS = {
    "r1": ("request_id", "batch_slot", "output_digest"),
    "r2": ("request_id", "batch_slot", "output_digest",
           "arrival_seq", "backend_fingerprint", "latency_ms"),
    "r3": ("request_id", "batch_slot", "output_digest",
           "arrival_seq", "backend_fingerprint", "latency_ms",
           "group_rank", "superseded_refs", "fingerprint_snapshot"),
}

# 校验顺序固定, 保证"首个出错字段"可复现.
FIELD_ORDER = (
    "request_id", "batch_slot", "output_digest",
    "arrival_seq", "backend_fingerprint", "latency_ms",
    "group_rank", "superseded_refs", "fingerprint_snapshot",
)

SNAPSHOT_KEYS = ("backend_fingerprint", "param_digest", "input_digest", "batch_digest")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_field(field, value, supported_slots):
    if field == "request_id":
        if not isinstance(value, str) or not value:
            return "request_id must be a non-empty string"
    elif field == "batch_slot":
        if not _is_int(value) or value not in supported_slots:
            return f"batch_slot {value!r} not in supported set {sorted(supported_slots)}"
    elif field in ("output_digest", "backend_fingerprint"):
        if not is_hex_digest(value):
            return f"{field} must be a 64-char hex digest"
    elif field in ("arrival_seq", "latency_ms", "group_rank"):
        if not _is_int(value) or value < 0:
            return f"{field} must be a non-negative integer"
    elif field == "superseded_refs":
        if not isinstance(value, list) or any(not _is_int(x) or x < 0 for x in value):
            return "superseded_refs must be a list of non-negative integers"
    elif field == "fingerprint_snapshot":
        if not isinstance(value, dict):
            return "fingerprint_snapshot must be an object"
        for key in SNAPSHOT_KEYS:
            if not is_hex_digest(value.get(key)):
                return f"fingerprint_snapshot.{key} must be a 64-char hex digest"
    return None


def decode(payload, supported_slots):
    """解码并强校验一条记录, 返回仅含声明字段的字典."""
    if not isinstance(payload, dict):
        raise LedgerError(E_FORMAT, "record payload must be an object",
                          field="<payload>", stage="decode")
    fmt = payload.get("fmt")
    if fmt not in FORMAT_FIELDS:
        raise LedgerError(E_FORMAT, f"unknown format version {fmt!r}",
                          record_id=payload.get("request_id"),
                          field="fmt", stage="decode")
    expected = set(FORMAT_FIELDS[fmt]) | {"fmt"}
    extra = sorted(set(payload) - expected)
    if extra:
        raise LedgerError(E_FORMAT, f"unexpected field {extra[0]!r} for {fmt}",
                          record_id=payload.get("request_id"),
                          field=extra[0], stage="decode")
    for field in FORMAT_FIELDS[fmt]:
        if field not in payload:
            raise LedgerError(E_FORMAT, f"missing field {field!r} for {fmt}",
                              record_id=payload.get("request_id"),
                              field=field, stage="decode")
    for field in FIELD_ORDER:
        if field in FORMAT_FIELDS[fmt]:
            err = _validate_field(field, payload[field], supported_slots)
            if err:
                raise LedgerError(E_FORMAT, err,
                                  record_id=payload.get("request_id"),
                                  field=field, stage="decode")
    return {"fmt": fmt, **{f: payload[f] for f in FORMAT_FIELDS[fmt]}}


def encode(fmt, view):
    """从归一视图编码出指定格式的记录字典."""
    if fmt not in FORMAT_FIELDS:
        raise ValueError(f"unknown format {fmt!r}")
    return {"fmt": fmt, **{f: view[f] for f in FORMAT_FIELDS[fmt]}}
