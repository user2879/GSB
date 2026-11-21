"""显式整数热度公式。

heat = heat_weight_recency * (anchor_ms - last_access_ms)
       + heat_weight_frequency * recency_rank
其中:
- anchor_ms 为同批对象最大 last_access_ms（与时间单位无关的相对年龄）
- recency_rank 为按 last_access_ms 降序、对象键升序得到的稳定 0 基名次
全部为整数运算，无浮点；分值越小越热。
"""
from __future__ import annotations

from dataclasses import dataclass

from .manifest import Manifest


@dataclass(frozen=True)
class HeatScore:
    key: str
    heat: int
    age_ms: int
    recency_rank: int


def compute_heat(manifest: Manifest) -> dict[str, HeatScore]:
    objects = manifest.objects
    if not objects:
        return {}
    anchor = max(obj.last_access_ms for obj in objects)
    ordered = sorted(objects, key=lambda o: (-o.last_access_ms, o.key))
    scores: dict[str, HeatScore] = {}
    for rank, obj in enumerate(ordered):
        age = anchor - obj.last_access_ms
        heat = (manifest.rules.heat_weight_recency * age
                + manifest.rules.heat_weight_frequency * rank)
        scores[obj.key] = HeatScore(
            key=obj.key, heat=heat, age_ms=age, recency_rank=rank)
    return scores


def eviction_order(manifest: Manifest, scores: dict[str, HeatScore],
                   keys: list[str]) -> list[str]:
    """淘汰顺序：热度升序（冷的先迁出），平分按对象键升序稳定打破。"""
    return sorted(keys, key=lambda k: (scores[k].heat, k))


def promotion_order(manifest: Manifest, scores: dict[str, HeatScore],
                    keys: list[str]) -> list[str]:
    """晋升顺序：热度降序（热的先迁入），平分按对象键升序稳定打破。"""
    return sorted(keys, key=lambda k: (-scores[k].heat, k))
