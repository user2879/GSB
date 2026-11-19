"""自有 JSON 行格式解析与序列化。

必需字段按 schema_version 分档校验：
  v1: tick/peer_id/op_type/payload/base_snapshot_version/schema_version
  v2: v1 + local_seq/payload_encoding
  v3: v2 + time_base
低版本缺失的新增字段由迁移表补齐。
"""

from __future__ import annotations

import json

from ..errors import E_SCHEMA_FIELD, PipelineError
from ..model import Event

REQUIRED_BY_VERSION = {
    1: ("tick", "peer_id", "op_type", "payload", "base_snapshot_version", "schema_version"),
    2: ("tick", "peer_id", "op_type", "payload", "base_snapshot_version", "schema_version",
        "local_seq", "payload_encoding"),
    3: ("tick", "peer_id", "op_type", "payload", "base_snapshot_version", "schema_version",
        "local_seq", "payload_encoding", "time_base"),
}


def parse_line(line: str, line_no: int = 0) -> Event:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise PipelineError(E_SCHEMA_FIELD, f"第 {line_no} 行不是合法 JSON: {exc}", field="line")
    if not isinstance(data, dict):
        raise PipelineError(E_SCHEMA_FIELD, f"第 {line_no} 行不是 JSON 对象", field="line")
    version = data.get("schema_version")
    if not isinstance(version, int):
        raise PipelineError(E_SCHEMA_FIELD, "缺少或非法 schema_version", field="schema_version")
    required = REQUIRED_BY_VERSION.get(version)
    if required is None:
        # 版本超出迁移表覆盖范围时，仅做必需字段的最小校验，版本合法性由迁移层裁决。
        required = REQUIRED_BY_VERSION[min(max(REQUIRED_BY_VERSION), 3)] if version > 3 else ()
        if version > 3:
            required = REQUIRED_BY_VERSION[3]
    for name in required:
        if name not in data:
            raise PipelineError(E_SCHEMA_FIELD, f"schema v{version} 缺少必需字段 {name}", field=name)
    if "peer_id" in data and not isinstance(data["peer_id"], str):
        raise PipelineError(E_SCHEMA_FIELD, "peer_id 必须为字符串", field="peer_id")
    return Event(
        tick=int(data["tick"]),
        peer_id=str(data["peer_id"]),
        local_seq=data.get("local_seq"),
        op_type=str(data["op_type"]),
        payload=dict(data["payload"]),
        base_snapshot_version=int(data["base_snapshot_version"]),
        schema_version=version,
        payload_encoding=data.get("payload_encoding"),
        time_base=data.get("time_base"),
        source_format="jsonl",
    )


def dump_line(event: Event, schema_version: int = 3) -> str:
    data = event.identity_dict()
    if schema_version < 3:
        data.pop("time_base", None)
    if schema_version < 2:
        data.pop("local_seq", None)
        data.pop("payload_encoding", None)
    data["schema_version"] = schema_version
    return json.dumps(data, sort_keys=True, ensure_ascii=False)
