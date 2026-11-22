"""候选流 JSONL 与配置加载层（要求 1）。

逐行解析、逐条校验；非法记录写入失败清单并继续，不中断整体加载。
校验项：请求 ID/通道名非空、(请求,通道) 内物品 ID 唯一、原始得分有限、
特征维度一致、通道内名次为正整数且无重复、分片序号为非负整数。
"""
from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .config import FusionConfig
from .errors import E_RECORD_INVALID, Failure


@dataclass
class CandidateRecord:
    request_id: str
    channel: str
    item_id: str
    raw_score: float
    features: List[float]
    channel_rank: int
    shard: int


@dataclass
class LoadedStream:
    records: List[CandidateRecord]
    failures: List[Failure] = field(default_factory=list)
    requests: Dict[str, Dict[str, List[CandidateRecord]]] = \
        field(default_factory=dict)
    request_items: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def request_ids(self) -> List[str]:
        return sorted(self.requests)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def load_stream(path: str, cfg: FusionConfig) -> LoadedStream:
    failures: List[Failure] = []
    records: List[CandidateRecord] = []
    seen_item: Dict[Tuple[str, str], set] = {}
    seen_rank: Dict[Tuple[str, str], set] = {}

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as exc:
                failures.append(Failure(
                    E_RECORD_INVALID, f"{path}:{lineno}",
                    f"JSON 解析失败: {exc}"))
                continue
            if not isinstance(obj, dict):
                failures.append(Failure(
                    E_RECORD_INVALID, f"{path}:{lineno}", "记录必须是对象"))
                continue

            loc = f"{path}:{lineno}"

            def bad(reason: str) -> None:
                failures.append(Failure(E_RECORD_INVALID, loc, reason,
                                        {"request_id": obj.get("request_id"),
                                         "item_id": obj.get("item_id")}))

            req = obj.get("request_id")
            chan = obj.get("channel")
            item = obj.get("item_id")
            if not isinstance(req, str) or not req:
                bad("request_id 必须为非空字符串")
                continue
            if not isinstance(chan, str) or not chan:
                bad("channel 必须为非空字符串")
                continue
            if chan not in cfg.channels:
                bad(f"channel {chan} 未在配置中声明")
                continue
            if not isinstance(item, str) or not item:
                bad("item_id 必须为非空字符串")
                continue

            score = obj.get("raw_score")
            if not isinstance(score, (int, float)) or isinstance(score, bool) \
                    or not math.isfinite(float(score)):
                bad("raw_score 必须为有限数")
                continue

            feats = obj.get("features")
            if (not isinstance(feats, list) or len(feats) != cfg.feature_dim
                    or not all(isinstance(v, (int, float))
                               and not isinstance(v, bool)
                               and math.isfinite(float(v)) for v in feats)):
                bad(f"features 必须是长度 {cfg.feature_dim} 的有限数向量")
                continue

            rank = obj.get("channel_rank")
            if not _is_int(rank) or rank <= 0:
                bad("channel_rank 必须为正整数")
                continue

            shard = obj.get("shard")
            if not _is_int(shard) or shard < 0:
                bad("shard 必须为非负整数")
                continue

            key = (req, chan)
            if item in seen_item.setdefault(key, set()):
                bad(f"物品 ID {item} 在请求 {req} 通道 {chan} 内重复")
                continue
            if rank in seen_rank.setdefault(key, set()):
                bad(f"通道内名次 {rank} 在请求 {req} 通道 {chan} 内重复")
                continue
            seen_item[key].add(item)
            seen_rank[key].add(rank)

            records.append(CandidateRecord(
                request_id=req, channel=chan, item_id=item,
                raw_score=float(score), features=[float(v) for v in feats],
                channel_rank=int(rank), shard=int(shard)))

    requests: Dict[str, Dict[str, List[CandidateRecord]]] = {}
    request_items: Dict[str, List[str]] = {}
    for rec in records:
        requests.setdefault(rec.request_id, {}) \
            .setdefault(rec.channel, []).append(rec)
        request_items.setdefault(rec.request_id, [])
        if rec.item_id not in request_items[rec.request_id]:
            request_items[rec.request_id].append(rec.item_id)

    for req in request_items:
        request_items[req] = sorted(request_items[req])
    for req in requests:
        for chan in requests[req]:
            requests[req][chan].sort(key=lambda r: r.channel_rank)

    return LoadedStream(records, failures, requests, request_items)
