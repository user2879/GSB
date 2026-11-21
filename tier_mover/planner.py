"""基于热度公式与层容量的确定性迁移计划推导。

规则（两层：hot/cold）：
1. 热层占用超过 hot_watermark_permille 时，按热度升序（冷者先迁）迁出，
   直到占用回落到水位线；
2. 冷层占用低于 cold_reclaim_permille 时，按热度降序（热者先迁）晋升回热层，
   直到达到回收线或没有候选；
3. 全程模拟容量，任何时刻不允许超过任一层容量上限；
4. 同一输入重复推导得到逐字节一致的计划（顺序由显式整数键决定）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .errors import CapacityError
from .heat import HeatScore, compute_heat, eviction_order, promotion_order
from .manifest import Manifest, ObjectSpec


@dataclass(frozen=True)
class PlanStep:
    seq: int
    key: str
    direction: str  # evict: hot->cold, promote: cold->hot
    source_tier: str
    target_tier: str
    size_bytes: int
    heat: int
    recency_rank: int
    ranking: int  # 1 基排序位次，按本方向候选顺序


@dataclass(frozen=True)
class Plan:
    spec_version: int
    steps: tuple[PlanStep, ...]
    retained_keys: tuple[str, ...]
    evict_keys: tuple[str, ...]
    promote_keys: tuple[str, ...]
    scores: dict[str, HeatScore]
    simulated_occupancy: dict[str, int]
    plan_digest: str

    def to_json(self) -> str:
        payload = {
            "spec_version": self.spec_version,
            "steps": [step.__dict__ for step in self.steps],
            "retained_keys": list(self.retained_keys),
            "evict_keys": list(self.evict_keys),
            "promote_keys": list(self.promote_keys),
            "simulated_occupancy": self.simulated_occupancy,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))


def derive_plan(manifest: Manifest,
                placement: dict[str, str] | None = None) -> Plan:
    """推导迁移计划。

    placement: 对象键 -> 当前层 id；缺省表示全部初始放置于热层。
    推导只做容量模拟，不触碰磁盘；容量不足按 on_capacity 规则改写或报错。
    """
    hot = manifest.hot_tier
    cold = manifest.cold_tier
    obj_by_key = {obj.key: obj for obj in manifest.objects}
    if placement is None:
        placement = {obj.key: hot.tier_id for obj in manifest.objects}
    _check_placement(manifest, placement)

    occupancy = {hot.tier_id: 0, cold.tier_id: 0}
    for key, tier_id in placement.items():
        occupancy[tier_id] += obj_by_key[key].size_bytes

    scores = compute_heat(manifest)
    rules = manifest.rules
    hot_limit = hot.capacity_bytes * rules.hot_watermark_permille // 1000
    cold_target = cold.capacity_bytes * rules.cold_reclaim_permille // 1000

    steps: list[PlanStep] = []
    evict_keys: list[str] = []
    promote_keys: list[str] = []

    # ---- 阶段 1：热层水位迁出 ----
    hot_keys = [k for k, t in placement.items() if t == hot.tier_id]
    for ranking, key in enumerate(eviction_order(manifest, scores, hot_keys), 1):
        if occupancy[hot.tier_id] <= hot_limit:
            break
        obj = obj_by_key[key]
        if not _fits(cold, occupancy[cold.tier_id], obj.size_bytes):
            if rules.on_capacity == "reject":
                raise CapacityError(
                    f"冷层 {cold.tier_id} 余量不足以接纳迁出对象 {key}",
                    tier_id=cold.tier_id, rejected_keys=[key],
                    free_bytes=cold.capacity_bytes - occupancy[cold.tier_id],
                    required_bytes=obj.size_bytes,
                    field="capacity_bytes")
            continue  # rewrite：跳过该对象，继续尝试更冷的候选
        _apply(occupancy, hot.tier_id, cold.tier_id, obj.size_bytes)
        evict_keys.append(key)
        steps.append(_step(len(steps), key, "evict", hot, cold, obj,
                           scores, ranking))

    # ---- 阶段 2：冷层回收晋升 ----
    cold_keys = [k for k, t in placement.items() if t == cold.tier_id]
    # 已在本计划中迁出到冷层的对象也是晋升候选
    cold_current = set(cold_keys) | set(evict_keys)
    for ranking, key in enumerate(promotion_order(
            manifest, scores, sorted(cold_current)), 1):
        if occupancy[cold.tier_id] <= cold_target:
            break
        obj = obj_by_key[key]
        if not _fits(hot, occupancy[hot.tier_id], obj.size_bytes):
            if rules.on_capacity == "reject":
                raise CapacityError(
                    f"热层 {hot.tier_id} 余量不足以接纳晋升对象 {key}",
                    tier_id=hot.tier_id, rejected_keys=[key],
                    free_bytes=hot.capacity_bytes - occupancy[hot.tier_id],
                    required_bytes=obj.size_bytes,
                    field="capacity_bytes")
            continue
        _apply(occupancy, cold.tier_id, hot.tier_id, obj.size_bytes)
        promote_keys.append(key)
        steps.append(_step(len(steps), key, "promote", cold, hot, obj,
                           scores, ranking))

    moved = set(evict_keys) | set(promote_keys)
    retained = sorted(k for k in placement if k not in moved)

    # 执行后容量不变量校验（防御性）
    for tier in manifest.tiers:
        if occupancy[tier.tier_id] > tier.capacity_bytes:  # pragma: no cover
            raise CapacityError(
                f"计划推导后层 {tier.tier_id} 超出容量",
                tier_id=tier.tier_id, rejected_keys=[],
                free_bytes=0,
                required_bytes=occupancy[tier.tier_id] - tier.capacity_bytes,
                field="capacity_bytes")

    payload = {
        "spec_version": manifest.spec_version,
        "placement": dict(sorted(placement.items())),
        "steps": [step.__dict__ for step in steps],
        "retained_keys": retained,
        "simulated_occupancy": dict(sorted(occupancy.items())),
        "spec_digest": manifest.spec_digest,
    }
    import hashlib
    plan_digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()

    return Plan(
        spec_version=manifest.spec_version,
        steps=tuple(steps),
        retained_keys=tuple(retained),
        evict_keys=tuple(evict_keys),
        promote_keys=tuple(promote_keys),
        scores=scores,
        simulated_occupancy=dict(sorted(occupancy.items())),
        plan_digest=plan_digest,
    )


def split_batches(plan: Plan, batch_limit: int) -> list[tuple[PlanStep, ...]]:
    """按迁移批次上限切分计划，保持计划顺序。"""
    if batch_limit <= 0:
        raise ValueError("batch_limit 必须为正整数")
    return [tuple(plan.steps[i:i + batch_limit])
            for i in range(0, len(plan.steps), batch_limit)]


def _step(seq_index: int, key: str, direction: str, source, target,
          obj: ObjectSpec, scores: dict[str, HeatScore], ranking: int) -> PlanStep:
    return PlanStep(
        seq=seq_index + 1, key=key, direction=direction,
        source_tier=source.tier_id, target_tier=target.tier_id,
        size_bytes=obj.size_bytes, heat=scores[key].heat,
        recency_rank=scores[key].recency_rank, ranking=ranking)


def _apply(occupancy: dict[str, int], source: str, target: str,
           size_bytes: int) -> None:
    occupancy[source] -= size_bytes
    occupancy[target] += size_bytes


def _fits(tier, used: int, size_bytes: int) -> bool:
    return used + size_bytes <= tier.capacity_bytes


def _check_placement(manifest: Manifest, placement: dict[str, str]) -> None:
    known = {tier.tier_id for tier in manifest.tiers}
    keys = {obj.key for obj in manifest.objects}
    if set(placement) != keys:
        missing = sorted(keys - set(placement))
        extra = sorted(set(placement) - keys)
        raise CapacityError(
            "对象归属与清单不匹配",
            tier_id="", rejected_keys=missing or extra,
            free_bytes=0, required_bytes=0,
            details={"missing": missing, "extra": extra})
    unknown = sorted({t for t in placement.values() if t not in known})
    if unknown:
        raise CapacityError(
            f"归属引用了未知层: {unknown}", tier_id=str(unknown[0]),
            rejected_keys=[], free_bytes=0, required_bytes=0)
