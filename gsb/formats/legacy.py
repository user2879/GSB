"""旧版 JSON（schema v1，字段被裁剪）解析与规范化。

旧版特征：
  - peer_id 为数字型字段 "peer"，规范化为字符串 "peer-<n>"
  - 时间戳为毫秒 "ts_ms"，按 TICK_MS 换算为 tick
  - 缺少 local_seq / payload_encoding / time_base，由管线按流序赋序号、迁移表补齐
"""

from __future__ import annotations

import json

from ..errors import E_SCHEMA_FIELD, PipelineError
from ..model import TICK_MS, Event

REQUIRED_LEGACY_FIELDS = ("peer", "ts_ms", "op", "data", "base")


def parse_line(line: str, line_no: int = 0) -> Event:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise PipelineError(E_SCHEMA_FIELD, f"第 {line_no} 行不是合法 JSON: {exc}", field="line")
    if not isinstance(data, dict):
        raise PipelineError(E_SCHEMA_FIELD, f"第 {line_no} 行不是 JSON 对象", field="line")
    for name in REQUIRED_LEGACY_FIELDS:
        if name not in data:
            raise PipelineError(E_SCHEMA_FIELD, f"旧版帧缺少字段 {name}", field=name)

    peer_num = data["peer"]
    if not isinstance(peer_num, int) or isinstance(peer_num, bool):
        raise PipelineError(E_SCHEMA_FIELD, "旧版 peer 必须为数字", field="peer")

    ts_ms = data["ts_ms"]
    if not isinstance(ts_ms, int) or ts_ms < 0:
        raise PipelineError(E_SCHEMA_FIELD, "旧版 ts_ms 必须为非负整数毫秒", field="ts_ms")
    if ts_ms % TICK_MS != 0:
        raise PipelineError(E_SCHEMA_FIELD, "ts_ms 不能整除 tick 基准", field="ts_ms")

    return Event(
        tick=ts_ms // TICK_MS,
        peer_id=f"peer-{peer_num}",
        local_seq=None,
        op_type=str(data["op"]),
        payload=dict(data["data"]),
        base_snapshot_version=int(data["base"]),
        schema_version=int(data.get("schema_version", 1)),
        payload_encoding=None,  # 旧版未声明，交由迁移表补齐
        time_base=None,
        source_format="legacy",
    )


def dump_line(event: Event) -> str:
    peer_num = int(event.peer_id.split("-", 1)[1])
    data = {
        "schema_version": 1,
        "peer": peer_num,
        "ts_ms": event.tick * TICK_MS,
        "op": event.op_type,
        "data": event.payload,
        "base": event.base_snapshot_version,
    }
    return json.dumps(data, sort_keys=True, ensure_ascii=False)
