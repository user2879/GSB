"""Deterministic migration plan derivation.

The planner is a pure function of the manifest plus the current ownership map.
It never touches the file system or a database, so the same inputs always
derive a byte-identical plan.  All ordering goes through the total-order
helpers in :mod:`tier_mover.heat`; nothing relies on dict/hash iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .heat import heat_value, hottest_first
from .models import Manifest, Obj, Tier


@dataclass(frozen=True)
class PlanAction:
    seq: int
    key: str
    action: str  # place | migrate | postpone
    direction: str  # to_hot | to_cold | none
    source_tier: Optional[str]
    target_tier: Optional[str]
    reason: str
    size_bytes: int
    digest: str
    heat: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "key": self.key,
            "action": self.action,
            "direction": self.direction,
            "source_tier": self.source_tier,
            "target_tier": self.target_tier,
            "reason": self.reason,
            "size_bytes": self.size_bytes,
            "digest": self.digest,
            "heat": self.heat,
        }


def _action(
    seq: int,
    obj: Obj,
    action: str,
    direction: str,
    source: Optional[Tier],
    target: Optional[Tier],
    reason: str,
) -> PlanAction:
    return PlanAction(
        seq=seq,
        key=obj.key,
        action=action,
        direction=direction,
        source_tier=source.tier_id if source else None,
        target_tier=target.tier_id if target else None,
        reason=reason,
        size_bytes=obj.size_bytes,
        digest=obj.digest,
        heat=heat_value(obj),
    )


def derive_initial_plan(manifest: Manifest) -> List[PlanAction]:
    """First-session placement: hottest objects fill hot, the rest go cold."""
    hot = manifest.hot()
    cold = manifest.cold()
    total_bytes = sum(obj.size_bytes for obj in manifest.objects)
    if total_bytes > hot.capacity_bytes + cold.capacity_bytes:
        from .errors import CapacityError

        raise CapacityError(
            "total object bytes exceed combined tier capacity",
            {
                "total_bytes": total_bytes,
                "hot_capacity": hot.capacity_bytes,
                "cold_capacity": cold.capacity_bytes,
            },
        )
    # The hot tier must remain a viable tier.  An injection that shrinks it
    # below the smallest object (or to zero) cannot be satisfied and is a hard
    # capacity violation rather than silent overflow into cold.
    smallest = min(obj.size_bytes for obj in manifest.objects)
    if hot.capacity_bytes < smallest:
        from .errors import CapacityError

        raise CapacityError(
            "hot tier capacity cannot accept any object",
            {
                "tier_id": hot.tier_id,
                "hot_capacity": hot.capacity_bytes,
                "smallest_object_bytes": smallest,
            },
        )

    actions: List[PlanAction] = []
    hot_used = 0
    cold_used = 0
    seq = 0
    for obj in hottest_first(manifest.objects):
        seq += 1
        if hot_used + obj.size_bytes <= hot.capacity_bytes:
            hot_used += obj.size_bytes
            actions.append(
                _action(seq, obj, "place", "to_hot", None, hot, "initial hot-first placement")
            )
        elif cold_used + obj.size_bytes <= cold.capacity_bytes:
            cold_used += obj.size_bytes
            actions.append(
                _action(seq, obj, "place", "to_cold", None, cold, "hot tier full; overflow to cold")
            )
        else:
            from .errors import CapacityError

            raise CapacityError(
                "no tier can accept object during initial placement",
                {
                    "key": obj.key,
                    "size_bytes": obj.size_bytes,
                    "hot_available": hot.capacity_bytes - hot_used,
                    "cold_available": cold.capacity_bytes - cold_used,
                },
            )
    return actions


def derive_cycle_plan(
    manifest: Manifest, ownership: Mapping[str, str]
) -> Dict[str, Any]:
    """Derive demotions, promotions and retained sets for one cycle."""
    hot = manifest.hot()
    cold = manifest.cold()
    rule = manifest.rule

    obj_by_key = {obj.key: obj for obj in manifest.objects}
    hot_objs = [obj_by_key[key] for key, tier_id in ownership.items() if tier_id == hot.tier_id]
    cold_objs = [obj_by_key[key] for key, tier_id in ownership.items() if tier_id == cold.tier_id]

    hot_used = sum(obj.size_bytes for obj in hot_objs)
    cold_used = sum(obj.size_bytes for obj in cold_objs)
    hot_initial = hot_used
    cold_initial = cold_used

    # Demotion candidates: coldest first, tier evict_priority breaks ties.
    demotion_order = sorted(
        hot_objs,
        key=lambda o: (hot.evict_priority, heat_value(o), o.key, o.digest),
    )
    # Promotion candidates: hottest first.
    promotion_order = hottest_first(cold_objs)

    moves: List[PlanAction] = []
    postponed: List[PlanAction] = []
    seq = 0
    budget = rule.batch_size

    # Phase 1: demote until hot usage is at or below the hot watermark.
    for obj in demotion_order:
        if hot_used <= rule.hot_watermark_bytes or budget == 0:
            break
        if cold_used + obj.size_bytes > cold.capacity_bytes:
            seq += 1
            postponed.append(
                _action(
                    seq,
                    obj,
                    "postpone",
                    "none",
                    hot,
                    cold,
                    "cold tier lacks free capacity; demotion deferred",
                )
            )
            continue
        seq += 1
        budget -= 1
        hot_used -= obj.size_bytes
        cold_used += obj.size_bytes
        moves.append(
            _action(
                seq,
                obj,
                "migrate",
                "to_cold",
                hot,
                cold,
                "hot usage above hot watermark; demote coldest object",
            )
        )

    # Phase 2: reclaim cold by promoting the hottest objects while hot stays
    # within both its watermark and hard capacity.
    for obj in promotion_order:
        if cold_used <= rule.cold_reclaim_bytes or budget == 0:
            break
        if hot_used + obj.size_bytes > min(rule.hot_watermark_bytes, hot.capacity_bytes):
            seq += 1
            postponed.append(
                _action(
                    seq,
                    obj,
                    "postpone",
                    "none",
                    cold,
                    hot,
                    "hot tier would cross watermark/capacity; promotion deferred",
                )
            )
            continue
        seq += 1
        budget -= 1
        cold_used -= obj.size_bytes
        hot_used += obj.size_bytes
        moves.append(
            _action(
                seq,
                obj,
                "migrate",
                "to_hot",
                cold,
                hot,
                "cold usage above reclaim threshold; promote hottest object",
            )
        )

    moved_keys = {action.key for action in moves}
    retained_hot = sorted(
        key for key, tier_id in ownership.items() if tier_id == hot.tier_id and key not in moved_keys
    )
    retained_cold = sorted(
        key for key, tier_id in ownership.items() if tier_id == cold.tier_id and key not in moved_keys
    )

    return {
        "policy_version": manifest.policy_version,
        "tier_param_version": manifest.tier_param_version,
        "phase": "cycle",
        "hot_tier": hot.tier_id,
        "cold_tier": cold.tier_id,
        "initial_usage": {hot.tier_id: hot_initial, cold.tier_id: cold_initial},
        "projected_usage": {hot.tier_id: hot_used, cold.tier_id: cold_used},
        "actions": [action.to_dict() for action in moves],
        "postponed": [action.to_dict() for action in postponed],
        "retained_hot": retained_hot,
        "retained_cold": retained_cold,
    }


def plan_to_dict(manifest: Manifest, actions: List[PlanAction], phase: str) -> Dict[str, Any]:
    hot = manifest.hot()
    cold = manifest.cold()
    hot_used = sum(
        action.size_bytes for action in actions if action.target_tier == hot.tier_id
    )
    cold_used = sum(
        action.size_bytes for action in actions if action.target_tier == cold.tier_id
    )
    return {
        "policy_version": manifest.policy_version,
        "tier_param_version": manifest.tier_param_version,
        "phase": phase,
        "hot_tier": hot.tier_id,
        "cold_tier": cold.tier_id,
        "projected_usage": {hot.tier_id: hot_used, cold.tier_id: cold_used},
        "actions": [action.to_dict() for action in actions],
        "postponed": [],
        "retained_hot": sorted(
            action.key for action in actions if action.target_tier == hot.tier_id
        ),
        "retained_cold": sorted(
            action.key for action in actions if action.target_tier == cold.tier_id
        ),
    }
