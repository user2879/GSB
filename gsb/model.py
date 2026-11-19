"""统一内部事件模型与确定性哈希。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any

CURRENT_SCHEMA_VERSION = 3
MIN_SCHEMA_VERSION = 1
TICK_MS = 1000  # 旧版毫秒时间戳到 tick 的换算基准

OP_TYPES = ("move", "spawn", "destroy", "set_attr")
OP_TYPE_TO_ID = {name: idx for idx, name in enumerate(OP_TYPES)}
ID_TO_OP_TYPE = {idx: name for name, idx in OP_TYPE_TO_ID.items()}

PAYLOAD_ENCODING_CURRENT = "utf8-json-v1"
TIME_BASE_CURRENT = "tick"


def canonical_json(obj: Any) -> str:
    """确定性 JSON 序列化：键排序、紧凑分隔、固定编码。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


@dataclass
class Event:
    """统一内部事件模型（当前 schema_version = 3）。

    三种承载（JSON 行 / 二进制帧 / 旧版 JSON）解析后都必须落到该结构。
    """

    tick: int
    peer_id: str
    local_seq: int
    op_type: str
    payload: dict
    base_snapshot_version: int
    schema_version: int = CURRENT_SCHEMA_VERSION
    payload_encoding: str = PAYLOAD_ENCODING_CURRENT
    time_base: str = TIME_BASE_CURRENT
    source_format: str = ""  # jsonl / binary / legacy（解析期标注，不参与等价比对）
    event_id: str = ""

    def identity_dict(self) -> dict:
        """参与内容哈希与跨格式等价比对的字段集合（不含来源标注）。"""
        return {
            "tick": self.tick,
            "peer_id": self.peer_id,
            "local_seq": self.local_seq,
            "op_type": self.op_type,
            "payload": self.payload,
            "base_snapshot_version": self.base_snapshot_version,
            "schema_version": self.schema_version,
            "payload_encoding": self.payload_encoding,
            "time_base": self.time_base,
        }

    def compute_event_id(self) -> str:
        return content_hash(self.identity_dict())

    def finalize(self) -> "Event":
        if self.op_type not in OP_TYPES:
            raise ValueError(f"unknown op_type: {self.op_type}")
        self.event_id = self.compute_event_id()
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Event":
        return Event(**data)


@dataclass
class AdjudicationRecord:
    """裁决账本条目：一次字段级覆盖。"""

    tick: int
    entity_id: str
    field: str
    winner_event_id: str
    winner_peer: str
    winner_format: str
    loser_event_id: str
    loser_peer: str
    loser_format: str
    rule: str = "field-lww-by-commit-order"
    final_value: Any = None


@dataclass
class Snapshot:
    version: int
    parent_version: int
    tick: int
    state: dict  # entity_id -> {"kind": str, "fields": {name: value}}
    state_hash: str = ""

    def compute_hash(self) -> str:
        return content_hash({"version": self.version, "state": self.state})

    def finalize(self) -> "Snapshot":
        self.state_hash = self.compute_hash()
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Snapshot":
        return Snapshot(**data)
