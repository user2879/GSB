"""r1/r2/r3 回执的编解码与归一（需求 2）。

格式演进：
  r1: batch_id, request_id, output_digest
  r2: r1 + param_version, elapsed_ms, worker_id
  r3: r2 + rank_in_batch, dup_ref

归一结果携带来源格式版本、被补齐字段集合与每个补齐字段的依据；
解码阶段对耗时非负、批内名次不重复且不越界做 E_FIELD 校验，
对重复投递引用做 E_REF 校验。
"""

from dataclasses import dataclass, field

from .errors import FieldError, RefError

R1_FIELDS = ("batch_id", "request_id", "output_digest")
R2_FIELDS = R1_FIELDS + ("param_version", "elapsed_ms", "worker_id")
R3_FIELDS = R2_FIELDS + ("rank_in_batch", "dup_ref")
FORMAT_FIELDS = {1: R1_FIELDS, 2: R2_FIELDS, 3: R3_FIELDS}


@dataclass
class NormalizedReceipt:
    source_format: int
    batch_id: str
    request_id: str
    output_digest: str
    param_version: int
    elapsed_ms: int
    worker_id: str
    rank_in_batch: int
    dup_ref: object
    filled_fields: list = field(default_factory=list)
    fill_basis: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "source_format": self.source_format,
            "batch_id": self.batch_id,
            "request_id": self.request_id,
            "output_digest": self.output_digest,
            "param_version": self.param_version,
            "elapsed_ms": self.elapsed_ms,
            "worker_id": self.worker_id,
            "rank_in_batch": self.rank_in_batch,
            "dup_ref": self.dup_ref,
            "filled_fields": list(self.filled_fields),
            "fill_basis": dict(self.fill_basis),
        }


def encode_receipt(normalized, fmt):
    """把归一结果编码为指定历史格式（高版本 -> 低版本丢弃新增字段）。"""
    if fmt not in FORMAT_FIELDS:
        raise FieldError(f"未知回执格式版本 {fmt}", field="fmt")
    full = {
        "batch_id": normalized.batch_id,
        "request_id": normalized.request_id,
        "output_digest": normalized.output_digest,
        "param_version": normalized.param_version,
        "elapsed_ms": normalized.elapsed_ms,
        "worker_id": normalized.worker_id,
        "rank_in_batch": normalized.rank_in_batch,
        "dup_ref": normalized.dup_ref,
    }
    return {key: full[key] for key in FORMAT_FIELDS[fmt]}


def normalize_receipt(raw, context):
    """解码并归一单条历史回执。

    ``context`` 必须提供：
      batch_size            批大小（名次越界校验）
      known_request_ids     所有存在过的请求标识集合（E_REF 校验）
      worker_id             批次归属工作进程（补齐依据）
      param_version         批次参数版本（补齐依据）
      rank_of(request_id)   批内稳定名次函数（补齐依据）
      dup_ref_of(request_id) 重复投递引用函数（补齐依据，无则 None）
    """
    fmt = _detect_format(raw)
    location = context.get("location", "/receipt")

    batch_id = _require_str(raw, "batch_id", location)
    request_id = _require_str(raw, "request_id", location)
    output_digest = _require_str(raw, "output_digest", location)

    filled_fields = []
    fill_basis = {}

    param_version = raw.get("param_version")
    if fmt >= 2:
        _check_nonneg_int(param_version, "param_version", location)
    else:
        param_version = context["param_version"]
        filled_fields.append("param_version")
        fill_basis["param_version"] = "context:batch_param_version"

    elapsed_ms = raw.get("elapsed_ms")
    if fmt >= 2:
        if not _is_nonneg_int(elapsed_ms):
            raise FieldError("elapsed_ms 必须是非负整数", location=location, field="elapsed_ms")
    else:
        elapsed_ms = 0
        filled_fields.append("elapsed_ms")
        fill_basis["elapsed_ms"] = "default:r1_has_no_timing"

    worker_id = raw.get("worker_id")
    if fmt >= 2:
        if not isinstance(worker_id, str) or not worker_id:
            raise FieldError("worker_id 必须是非空字符串", location=location, field="worker_id")
    else:
        worker_id = context["worker_id"]
        filled_fields.append("worker_id")
        fill_basis["worker_id"] = "context:batch_worker"

    rank_in_batch = raw.get("rank_in_batch")
    dup_ref = raw.get("dup_ref")
    if fmt >= 3:
        if not _is_nonneg_int(rank_in_batch):
            raise FieldError(
                "rank_in_batch 必须是非负整数", location=location, field="rank_in_batch"
            )
        if rank_in_batch >= context["batch_size"]:
            raise FieldError(
                f"rank_in_batch={rank_in_batch} 越界，批大小为 {context['batch_size']}",
                location=location,
                field="rank_in_batch",
            )
        if dup_ref is not None and dup_ref not in context["known_request_ids"]:
            raise RefError(
                f"dup_ref {dup_ref!r} 指向不存在的请求", location=location, field="dup_ref"
            )
    else:
        rank_in_batch = context["rank_of"](request_id)
        filled_fields.append("rank_in_batch")
        fill_basis["rank_in_batch"] = "derive:stable_arrival_order"
        dup_ref = context["dup_ref_of"](request_id)
        filled_fields.append("dup_ref")
        fill_basis["dup_ref"] = "derive:supersession_map"

    return NormalizedReceipt(
        source_format=fmt,
        batch_id=batch_id,
        request_id=request_id,
        output_digest=output_digest,
        param_version=param_version,
        elapsed_ms=elapsed_ms,
        worker_id=worker_id,
        rank_in_batch=rank_in_batch,
        dup_ref=dup_ref,
        filled_fields=filled_fields,
        fill_basis=fill_basis,
    )


def normalize_batch(raw_receipts, context):
    """解码同批多条回执，并校验批内名次不重复（E_FIELD）。"""
    normalized = [normalize_receipt(raw, context) for raw in raw_receipts]
    ranks = [item.rank_in_batch for item in normalized if item.source_format >= 3]
    if len(ranks) != len(set(ranks)):
        raise FieldError(
            "批内名次 rank_in_batch 出现重复",
            location=context.get("location", "/receipt"),
            field="rank_in_batch",
        )
    return normalized


def _detect_format(raw):
    if not isinstance(raw, dict):
        raise FieldError("回执必须是对象", field="<receipt>")
    for required in R1_FIELDS:
        if required not in raw:
            raise FieldError(f"r1 必填字段 {required} 缺失", field=required)
    has_r2 = all(key in raw for key in ("param_version", "elapsed_ms", "worker_id"))
    has_r3 = "rank_in_batch" in raw or "dup_ref" in raw
    if has_r3:
        if not has_r2:
            raise FieldError("r3 回执必须同时包含 r2 字段", field="rank_in_batch")
        return 3
    if has_r2:
        return 2
    return 1


def _is_nonneg_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _check_nonneg_int(value, field_name, location):
    if not _is_nonneg_int(value):
        raise FieldError(f"{field_name} 必须是非负整数", location=location, field=field_name)


def _require_str(raw, key, location):
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise FieldError(f"{key} 必须是非空字符串", location=location, field=key)
    return value
