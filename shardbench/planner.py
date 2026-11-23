"""Compile normalized predicates into deterministic route plans."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import algorithms
from .canon import digest
from .config_loader import ALGO_MOD, ALGO_RANGE, ALGO_RING
from .predicates import extract_constraints, normalize, shard_key_tuples


@dataclass
class RoutePlan:
    unit_id: str
    predicate: dict
    predicate_digest: str
    targets: list  # list of {"db": int, "table": int}
    reasons: list  # aligned human-readable derivation per hit
    rule_version: int
    ring_version: int
    plan_digest: str = field(default="")

    def summary_view(self):
        return {
            "unit_id": self.unit_id,
            "predicate_digest": self.predicate_digest,
            "targets": self.targets,
            "reasons": self.reasons,
            "rule_version": self.rule_version,
            "ring_version": self.ring_version,
        }


def compile_plan(config, unit_id, predicate, rings=None) -> RoutePlan:
    unit = config.units[unit_id]
    norm = normalize(predicate)
    constraints = extract_constraints(norm)
    covered, key_tuples, intervals = shard_key_tuples(unit, constraints)

    if unit.algo_type == ALGO_MOD:
        hits = (
            algorithms.route_mod(unit, key_tuples)
            if covered
            else algorithms.broadcast_hits(
                unit, f"shard key {list(unit.shard_keys)} not fully constrained"
            )
        )
    elif unit.algo_type == ALGO_RANGE:
        if covered:
            hits = algorithms.route_range(unit, key_tuples, {})
        elif unit.shard_keys[0] in intervals:
            hits = algorithms.route_range(unit, [], intervals)
        else:
            hits = algorithms.broadcast_hits(
                unit, f"shard key {list(unit.shard_keys)} not constrained"
            )
    else:  # consistent_hash
        ring = rings[unit_id] if rings else algorithms.HashRing(unit, config.ring_version)
        hits = (
            algorithms.route_consistent_hash(unit, ring, key_tuples)
            if covered
            else algorithms.broadcast_hits(
                unit, f"shard key {list(unit.shard_keys)} not fully constrained"
            )
        )

    grouped = {}
    for h in hits:
        grouped.setdefault((h.db, h.table), []).append(h.reason)
    targets = [{"db": db, "table": table} for db, table in sorted(grouped)]
    reasons = [" && ".join(grouped[(t["db"], t["table"])]) for t in targets]
    plan = RoutePlan(
        unit_id=unit_id,
        predicate=norm,
        predicate_digest=digest(norm),
        targets=targets,
        reasons=reasons,
        rule_version=config.config_version,
        ring_version=config.ring_version,
    )
    plan.plan_digest = digest(plan.summary_view())
    return plan
