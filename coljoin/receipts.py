"""执行回执 q1/q2/q3 三版格式的编解码与跨版本归一。

磁盘格式为 JSON Lines（每行一条回执，行号从 1 开始计）：
- q1：shard_id + 聚合摘要；
- q2：q1 + plan_version + elapsed_ms；
- q3：q2 + executor_id + cost_vector + clock_start_ms/clock_end_ms。

归一化对低版本缺失字段按显式默认规则补齐，并记录来源版本、
被补齐字段集合与补齐依据；任何格式非法统一抛 E_FORMAT 并定位行号/字段。
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .canonical import canonical_text
from .errors import FormatError

KNOWN_VERSIONS = ("q1", "q2", "q3")
REQUIRED_FIELDS = {
    "q1": ("version", "shard_id", "aggregate"),
    "q2": ("version", "shard_id", "aggregate", "plan_version", "elapsed_ms"),
    "q3": (
        "version",
        "shard_id",
        "aggregate",
        "plan_version",
        "elapsed_ms",
        "executor_id",
        "cost_vector",
        "clock_start_ms",
        "clock_end_ms",
    ),
}

DEFAULTS = {
    "executor_id": {"default": "executor-unknown", "basis": "q3 字段缺失：按未知执行器占位补齐"},
    "cost_vector": {"default": None, "basis": "q3 字段缺失：由计划代价向量按计划标识补齐"},
    "clock_start_ms": {"default": 0, "basis": "q3 字段缺失：单调时钟起点不可恢复，按 0 补齐"},
    "clock_end_ms": {"default": 0, "basis": "q3 字段缺失：单调时钟终点不可恢复，按 0 补齐"},
    "plan_version": {"default": 0, "basis": "q1 字段缺失：计划版本未知，按 0 补齐"},
    "elapsed_ms": {"default": 0, "basis": "q1/q2 语义缺失时按零耗时补齐"},
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def encode_receipt(version: str, payload: Dict[str, Any]) -> str:
    if version not in KNOWN_VERSIONS:
        raise FormatError(f"未知回执版本: {version}", field="version")
    record = dict(payload)
    record["version"] = version
    allowed = set(REQUIRED_FIELDS[version])
    unknown = set(record) - allowed - {"projection_digest"}
    if unknown:
        raise FormatError(
            f"版本 {version} 出现非法字段 {sorted(unknown)[0]}",
            field=sorted(unknown)[0],
        )
    return canonical_text(record)


def _read_lines(path: str) -> List[Tuple[int, str]]:
    if not os.path.exists(path):
        raise FormatError(f"回执文件不存在: {path}", field="file")
    with open(path, "r", encoding="utf-8") as handle:
        return [
            (index, line.strip())
            for index, line in enumerate(handle, start=1)
            if line.strip()
        ]


def _validate_aggregate(record: Dict[str, Any], line_no: int):
    aggregate = record.get("aggregate")
    if not isinstance(aggregate, dict) or not aggregate:
        raise FormatError(
            "聚合摘要必须为非空对象", line=line_no, field="aggregate"
        )
    for group_key, metrics in aggregate.items():
        if not isinstance(group_key, str):
            raise FormatError("分组键必须为字符串", line=line_no, field="aggregate")
        if not isinstance(metrics, dict) or not metrics:
            raise FormatError(
                "每个分组的聚合指标必须为非空对象",
                line=line_no,
                field=f"aggregate.{group_key}",
            )
        for metric_name, value in metrics.items():
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise FormatError(
                    "聚合指标值必须为数值或 null",
                    line=line_no,
                    field=f"aggregate.{group_key}.{metric_name}",
                )


def _validate_record(
    record: Dict[str, Any],
    line_no: int,
    expected_cost_vector_len: Optional[int],
) -> str:
    version = record.get("version")
    if version not in KNOWN_VERSIONS:
        raise FormatError(
            f"未知回执版本: {version}", line=line_no, field="version"
        )
    for field_name in REQUIRED_FIELDS[version]:
        if field_name not in record:
            raise FormatError(
                f"版本 {version} 缺少必需字段 {field_name}",
                line=line_no,
                field=field_name,
            )
    shard_id = record.get("shard_id")
    if not isinstance(shard_id, str) or not shard_id:
        raise FormatError(
            "shard_id 必须为非空字符串", line=line_no, field="shard_id"
        )
    _validate_aggregate(record, line_no)
    if version in ("q2", "q3"):
        if not _is_int(record["plan_version"]) or record["plan_version"] < 0:
            raise FormatError(
                "plan_version 必须为非负整数",
                line=line_no,
                field="plan_version",
            )
        if not _is_int(record["elapsed_ms"]) or record["elapsed_ms"] < 0:
            raise FormatError(
                "elapsed_ms 必须为非负整数", line=line_no, field="elapsed_ms"
            )
    if version == "q3":
        if not isinstance(record["executor_id"], str) or not record["executor_id"]:
            raise FormatError(
                "executor_id 必须为非空字符串", line=line_no, field="executor_id"
            )
        vector = record["cost_vector"]
        if not isinstance(vector, dict):
            raise FormatError(
                "cost_vector 必须为对象", line=line_no, field="cost_vector"
            )
        if expected_cost_vector_len is not None and len(vector) != expected_cost_vector_len:
            raise FormatError(
                f"cost_vector 长度 {len(vector)} 与计划代价向量长度 "
                f"{expected_cost_vector_len} 不符",
                line=line_no,
                field="cost_vector",
            )
        for clock_field in ("clock_start_ms", "clock_end_ms"):
            if not _is_int(record[clock_field]) or record[clock_field] < 0:
                raise FormatError(
                    f"{clock_field} 必须为非负整数",
                    line=line_no,
                    field=clock_field,
                )
        if record["clock_end_ms"] < record["clock_start_ms"]:
            raise FormatError(
                "clock_end_ms 不得早于 clock_start_ms",
                line=line_no,
                field="clock_end_ms",
            )
    return version


def normalize_record(
    record: Dict[str, Any],
    line_no: int,
    expected_cost_vector: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    expected_len = len(expected_cost_vector) if expected_cost_vector else None
    version = _validate_record(record, line_no, expected_len)
    normalized = dict(record)
    filled: List[Dict[str, str]] = []

    def fill(field_name: str, value: Any):
        if field_name not in normalized or normalized[field_name] is None:
            normalized[field_name] = value
            filled.append({"field": field_name, "basis": DEFAULTS[field_name]["basis"]})

    fill("plan_version", DEFAULTS["plan_version"]["default"])
    fill("elapsed_ms", DEFAULTS["elapsed_ms"]["default"])
    fill("executor_id", DEFAULTS["executor_id"]["default"])
    if expected_cost_vector is not None:
        fill("cost_vector", dict(expected_cost_vector))
    else:
        fill("cost_vector", {})
    fill("clock_start_ms", DEFAULTS["clock_start_ms"]["default"])
    fill("clock_end_ms", DEFAULTS["clock_end_ms"]["default"])
    normalized["_source_format"] = version
    normalized["_filled_fields"] = filled
    normalized["_normalized_version"] = "q3-normalized"
    return normalized


def load_receipts(
    path: str, expected_cost_vector: Optional[Dict[str, int]] = None
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line_no, line in _read_lines(path):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FormatError(
                f"回执 JSON 解析失败: {exc.msg}", line=line_no, field="json"
            ) from exc
        if not isinstance(record, dict):
            raise FormatError("回执必须为 JSON 对象", line=line_no, field="record")
        records.append(normalize_record(record, line_no, expected_cost_vector))
    return records


def write_receipts(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_text(record) + "\n")
