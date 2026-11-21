"""多来源配额记录声明的加载与强校验（E_FIELD）。"""

import json

from .errors import WorkbenchError
from .models import SOURCE_TYPES

SUPPORTED_VERSIONS = {
    "capacity_claim": (1, 2),
    "rate_claim": (1, 2),
    "derived_claim": (1, 2),
}

# 每类来源每版本的必需字段
REQUIRED_FIELDS = {
    ("capacity_claim", 1): ("capacity_limit_bytes",),
    ("capacity_claim", 2): ("capacity_limit_bytes", "block_size_bytes", "block_count"),
    ("rate_claim", 1): ("ops_per_second",),
    ("rate_claim", 2): ("ops_per_second", "burst_bucket_capacity", "scale_factor"),
    ("derived_claim", 1): ("metric", "base_source_id", "coefficient", "offset"),
    ("derived_claim", 2): ("metric", "base_source_id", "coefficient", "offset",
                           "divisor", "rounding"),
}

# 非负整数字段 / 正整数字段 / 任意整数字段
NON_NEG_FIELDS = {"capacity_limit_bytes", "ops_per_second", "burst_bucket_capacity",
                  "block_count"}
POSITIVE_FIELDS = {"block_size_bytes", "scale_factor", "divisor"}
INT_FIELDS = NON_NEG_FIELDS | POSITIVE_FIELDS | {"coefficient", "offset"}


def _fail(source_type, source_id, field, message):
    raise WorkbenchError(
        "E_FIELD", message,
        {"source_type": source_type, "source_id": source_id, "field": field})


def _check_int(stype, sid, name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(stype, sid, name, f"字段 {name} 必须为整数，实际为 {type(value).__name__}")
    if name in NON_NEG_FIELDS and value < 0:
        _fail(stype, sid, name, f"字段 {name} 必须为非负整数，实际为 {value}")
    if name in POSITIVE_FIELDS and value <= 0:
        _fail(stype, sid, name, f"字段 {name} 必须为正整数，实际为 {value}")


def validate_source(rec):
    """逐条强校验来源记录，违规以 E_FIELD 整条拒绝。"""
    stype = rec.get("source_type")
    sid = rec.get("source_id")
    if stype not in SOURCE_TYPES:
        _fail(stype, sid, "source_type",
              f"来源类型 {stype!r} 不在受支持集合 {sorted(SOURCE_TYPES)}")
    version = rec.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int):
        _fail(stype, sid, "format_version", "来源格式版本必须为整数")
    if version not in SUPPORTED_VERSIONS[stype]:
        _fail(stype, sid, "format_version",
              f"来源类型 {stype} 的格式版本 {version} 不受支持，"
              f"受支持版本为 {list(SUPPORTED_VERSIONS[stype])}")
    fields = rec.get("fields")
    if not isinstance(fields, dict):
        _fail(stype, sid, "fields", "记录缺少 fields 对象")
    for name in REQUIRED_FIELDS[(stype, version)]:
        if name not in fields:
            _fail(stype, sid, name,
                  f"版本 {version} 的 {stype} 缺少必需字段 {name}")
    for name, value in fields.items():
        if name in INT_FIELDS:
            _check_int(stype, sid, name, value)
    return True


def load_declaration(path):
    """加载卷与来源引用关系声明，返回 {"volumes": {vid: [sid]}, "sources": {sid: rec}}。"""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    volumes_raw = data.get("volumes")
    sources_raw = data.get("sources")
    if not isinstance(volumes_raw, list) or not isinstance(sources_raw, list):
        raise WorkbenchError("E_FIELD", "声明文件须包含 volumes 与 sources 数组",
                             {"source_type": None, "source_id": None,
                              "field": "volumes/sources"})
    sources = {}
    for rec in sources_raw:
        validate_source(rec)
        sid = rec["source_id"]
        if sid in sources:
            _fail(rec["source_type"], sid, "source_id", f"来源标识 {sid} 重复")
        sources[sid] = rec
    volumes = {}
    for vol in volumes_raw:
        vid = vol.get("volume_id")
        if not isinstance(vid, str) or not vid:
            _fail(None, None, "volume_id", "卷标识必须为非空字符串")
        if vid in volumes:
            _fail(None, None, "volume_id", f"卷标识 {vid} 不唯一")
        refs = vol.get("sources")
        if not isinstance(refs, list) or not refs:
            _fail(None, None, "sources", f"卷 {vid} 的来源引用列表为空或缺失")
        for sid in refs:
            if sid not in sources:
                _fail(None, None, "sources",
                      f"卷 {vid} 引用了不存在的来源 {sid}")
        volumes[vid] = list(refs)
    return {"volumes": volumes, "sources": sources}
