"""三类来源记录的多版本编解码与归一。"""

from .errors import WorkbenchError
from .models import METRICS, NormalizedRecord

# 各类型低版本记录的声明式默认值
DEFAULT_BLOCK_SIZE = 4096
DEFAULT_ROUNDING = "floor"


def _fail(stype, sid, field, message):
    raise WorkbenchError("E_FIELD", message,
                         {"source_type": stype, "source_id": sid, "field": field})


def _check_field_count(rec):
    """长度字段自洽校验：field_count 必须等于 fields 实际键数。"""
    fields = rec["fields"]
    declared = rec.get("field_count")
    if declared is None:
        _fail(rec["source_type"], rec["source_id"], "field_count",
              "缺少长度字段 field_count")
    if declared != len(fields):
        _fail(rec["source_type"], rec["source_id"], "field_count",
              f"长度字段不自洽：声明 {declared}，实际 {len(fields)}")


def _decode_capacity(rec):
    f = rec["fields"]
    version = rec["format_version"]
    defaulted = []
    cap = f["capacity_limit_bytes"]
    if version == 1:
        block_size = DEFAULT_BLOCK_SIZE
        block_count = -(-cap // block_size)  # 向上取整
        defaulted = ["block_size_bytes", "block_count"]
    else:
        block_size = f["block_size_bytes"]
        block_count = f["block_count"]
        if block_size * block_count < cap:
            _fail("capacity_claim", rec["source_id"], "block_count",
                  f"块数口径不自洽：{block_size}*{block_count} < {cap}")
    return ({"capacity_limit_bytes": cap}, defaulted, 0,
            {"block_size_bytes": block_size, "block_count": block_count})


def _decode_rate(rec):
    f = rec["fields"]
    version = rec["format_version"]
    defaulted = []
    ops = f["ops_per_second"]
    if version == 1:
        burst = ops
        scale = 1
        defaulted = ["burst_bucket_capacity", "scale_factor"]
    else:
        burst = f["burst_bucket_capacity"]
        scale = f["scale_factor"]
    metrics = {
        "ops_per_second": ops * scale,
        "burst_bucket_capacity": burst * scale,
    }
    # 整数缩放因子即记录粒度，量化误差上界取缩放因子本身
    return (metrics, defaulted, scale,
            {"raw_ops_per_second": ops, "raw_burst_bucket_capacity": burst,
             "scale_factor": scale})


def _decode_derived(rec, base_lookup):
    f = rec["fields"]
    version = rec["format_version"]
    defaulted = []
    metric = f["metric"]
    if metric not in METRICS:
        _fail("derived_claim", rec["source_id"], "metric",
              f"派生指标 {metric!r} 不在受支持集合 {list(METRICS)}")
    base_id = f["base_source_id"]
    base = base_lookup(base_id)
    if base is None:
        _fail("derived_claim", rec["source_id"], "base_source_id",
              f"派生来源引用了不存在的基准来源 {base_id}")
    if metric not in base.metrics:
        _fail("derived_claim", rec["source_id"], "base_source_id",
              f"基准来源 {base_id} 不提供指标 {metric}")
    if version == 1:
        divisor = 1
        rounding = DEFAULT_ROUNDING
        defaulted = ["divisor", "rounding"]
    else:
        divisor = f["divisor"]
        rounding = f["rounding"]
        if rounding not in ("floor", "ceil", "trunc"):
            _fail("derived_claim", rec["source_id"], "rounding",
                  f"取整方向 {rounding!r} 不受支持")
    numerator = f["coefficient"] * base.metrics[metric]
    if rounding == "floor":
        value = numerator // divisor
    elif rounding == "ceil":
        value = -(-numerator // divisor)
    else:  # trunc：向零截断
        value = (numerator // divisor) if numerator >= 0 else -((-numerator) // divisor)
    value += f["offset"]
    if value < 0:
        _fail("derived_claim", rec["source_id"], "offset",
              f"派生结果越界为负数：{value}")
    # 量化误差合成：|系数|/除数 放大基准误差 + 除法取整误差
    err = (abs(f["coefficient"]) * base.quantization_error) // divisor
    if divisor > 1:
        err += 1
    return ({metric: value}, defaulted, err,
            {"base_source_id": base_id, "coefficient": f["coefficient"],
             "offset": f["offset"], "divisor": divisor, "rounding": rounding})


def normalize_source(rec, base_lookup):
    """逐条解码并归一；低版本记录补齐默认值并保留来源版本号。"""
    _check_field_count(rec)
    stype = rec["source_type"]
    if stype == "capacity_claim":
        metrics, defaulted, err, extra = _decode_capacity(rec)
    elif stype == "rate_claim":
        metrics, defaulted, err, extra = _decode_rate(rec)
    else:
        metrics, defaulted, err, extra = _decode_derived(rec, base_lookup)
    return NormalizedRecord(
        source_id=rec["source_id"],
        source_type=stype,
        format_version=rec["format_version"],
        metrics=metrics,
        defaulted_fields=defaulted,
        quantization_error=err,
        extra=extra,
    )


def normalize_volume_sources(volume_id, raw_sources):
    """归一一个卷的全部来源；派生来源的基准按 source_id 在同卷内解析。"""
    normalized = {}
    order = sorted(raw_sources, key=lambda r: r["source_id"])

    def base_lookup(base_id):
        return normalized.get(base_id)

    # 先归一非派生来源，再归一派生来源（派生可引用派生时按 source_id 序迭代收敛）
    deferred = []
    for rec in order:
        if rec["source_type"] == "derived_claim":
            deferred.append(rec)
        else:
            n = normalize_source(rec, base_lookup)
            normalized[n.source_id] = n
    pending = list(deferred)
    while pending:
        progressed = False
        for rec in list(pending):
            base_id = rec["fields"].get("base_source_id")
            if base_id in normalized:
                n = normalize_source(rec, base_lookup)
                normalized[n.source_id] = n
                pending.remove(rec)
                progressed = True
        if not progressed:
            rec = pending[0]
            _fail("derived_claim", rec["source_id"], "base_source_id",
                  "派生来源基准无法解析（缺失或循环引用）")
    return [normalized[sid] for sid in sorted(normalized)]
