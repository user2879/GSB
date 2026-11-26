from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import assert_consistent, canonical_bytes
from .engine import Engine, replay_bundle
from .errors import DivergenceError
from .report import build_conclusion
from .spec import Bundle, NodeSpec
from .statemachine import ACTIVE_STATES


@dataclass(frozen=True)
class EvolutionChange:
    seq: int
    new_quota_version: int | None = None
    removed_node_ids: tuple[str, ...] = ()
    changed_seed_node_ids: tuple[str, ...] = ()
    added_node_ids: tuple[str, ...] = ()
    reason: str = "declared_evolution"


def _lease_order_key(conclusion: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    return {
        lease["lease_id"]: (lease["node_id"], int(lease["last_event_seq"]), lease["lease_id"])
        for lease in conclusion["leases"]
    }


def find_impact(conclusion: dict[str, Any], change: EvolutionChange) -> dict[str, Any]:
    order_keys = _lease_order_key(conclusion)
    affected_reasons: dict[str, list[str]] = {}
    affected_basis: dict[str, str] = {}

    def affect(lease_id: str, reason: str, basis: str) -> None:
        affected_reasons.setdefault(lease_id, [])
        if reason not in affected_reasons[lease_id]:
            affected_reasons[lease_id].append(reason)
        affected_basis.setdefault(lease_id, basis)

    for lease in conclusion["leases"]:
        if lease["state"] not in ACTIVE_STATES:
            continue
        lease_id = lease["lease_id"]
        if change.new_quota_version is not None and int(lease["quota_version"]) < change.new_quota_version:
            affect(
                lease_id,
                "quota_version",
                f"lease quota_version {lease['quota_version']} < evolved {change.new_quota_version}",
            )
        if lease["node_id"] in change.removed_node_ids:
            affect(lease_id, "node_removed", f"node {lease['node_id']} removed from declaration")
        if lease["node_id"] in change.changed_seed_node_ids:
            affect(lease_id, "node_seed_changed", f"node {lease['node_id']} seed/content derivation changed")

    affected_lease_ids = sorted(affected_reasons, key=lambda lease_id: order_keys[lease_id])
    affected_set = set(affected_lease_ids)
    affected_holds = [
        hold
        for hold in conclusion["active_holds"]
        if hold["lease_id"] in affected_set
    ]
    affected_shard_rows = [
        {
            "artifact_key": hold["artifact_key"],
            "shard_index": int(hold["shard_index"]),
            "node_id": hold["node_id"],
            "lease_id": hold["lease_id"],
            "event_seq": int(hold["opened_seq"]),
        }
        for hold in affected_holds
    ]
    affected_shards = sorted(
        affected_shard_rows,
        key=lambda row: (row["node_id"], row["event_seq"], row["artifact_key"], row["shard_index"]),
    )
    affected_node_ids = sorted(
        {lease["node_id"] for lease in conclusion["leases"] if lease["lease_id"] in affected_set}
        | set(change.removed_node_ids)
        | set(change.changed_seed_node_ids)
        | set(change.added_node_ids)
    )
    event_refs = sorted(
        {int(lease["last_event_seq"]) for lease in conclusion["leases"] if lease["lease_id"] in affected_set}
        | {int(hold["opened_seq"]) for hold in affected_holds}
    )

    steps: list[dict[str, Any]] = []
    order = 0

    def add_step(from_ref: str, to_ref: str, relation: str, event_seq: int | None, basis: str) -> None:
        nonlocal order
        order += 1
        steps.append(
            {
                "order": order,
                "from_ref": from_ref,
                "to_ref": to_ref,
                "relation": relation,
                "event_seq": event_seq,
                "basis": basis,
            }
        )

    if change.new_quota_version is not None:
        add_step(
            f"quota_version:{conclusion['quota_version']}",
            f"quota_version:{change.new_quota_version}",
            "quota_generation_reference",
            change.seq,
            "declared quota version replacement",
        )
    for node_id in sorted(set(change.removed_node_ids) | set(change.changed_seed_node_ids)):
        relation = "node_removal" if node_id in change.removed_node_ids else "node_seed_change"
        add_step(f"node:{node_id}", "node_set:evolved", relation, change.seq, change.reason)

    lease_by_id = {lease["lease_id"]: lease for lease in conclusion["leases"]}
    for lease_id in affected_lease_ids:
        lease = lease_by_id[lease_id]
        event_seq = int(lease["last_event_seq"])
        for reason in affected_reasons[lease_id]:
            if reason == "quota_version":
                add_step(
                    f"quota_version:{lease['quota_version']}",
                    f"lease:{lease_id}",
                    "quota_to_lease_generation",
                    event_seq,
                    affected_basis[lease_id],
                )
            else:
                add_step(
                    f"node:{lease['node_id']}",
                    f"lease:{lease_id}",
                    "node_to_lease_reference",
                    event_seq,
                    affected_basis[lease_id],
                )
        add_step(
            f"lease:{lease_id}",
            f"generation:{lease_id}#{lease['generation']}",
            "lease_to_generation_chain",
            event_seq,
            "current recognized generation is the invalidation base",
        )
        for hold in sorted(
            [hold for hold in affected_holds if hold["lease_id"] == lease_id],
            key=lambda row: int(row["shard_index"]),
        ):
            add_step(
                f"generation:{lease_id}#{lease['generation']}",
                f"shard:{hold['artifact_key']}#{hold['shard_index']}",
                "generation_to_hold_to_shard",
                int(hold["opened_seq"]),
                f"hold {hold['hold_id']} occupies node slot {hold['slot_id']}",
            )
            add_step(
                f"event:{hold['opened_seq']}",
                f"hold:{hold['hold_id']}",
                "event_reference",
                int(hold["opened_seq"]),
                "grant event opened the hold",
            )

    unaffected = sorted(set(lease_by_id) - affected_set)
    return {
        "evolution_seq": change.seq,
        "reason": change.reason,
        "affected_node_ids": affected_node_ids,
        "affected_lease_ids": affected_lease_ids,
        "affected_shards": affected_shards,
        "affected_event_seqs": event_refs,
        "unaffected_lease_ids": unaffected,
        "lease_reasons": {lease_id: affected_reasons[lease_id] for lease_id in affected_lease_ids},
        "steps": steps,
    }


def hydrate_engine(bundle: Bundle, conclusion: dict[str, Any]) -> Engine:
    engine = Engine(bundle)
    engine.quota_version = int(conclusion["quota_version"])
    engine.leases = {lease["lease_id"]: dict(lease) for lease in conclusion["leases"]}
    engine.transitions = [dict(row) for row in conclusion["transitions"]]
    engine.holds = {hold["hold_id"]: dict(hold) for hold in conclusion["active_holds"]}
    engine.decisions = [dict(row) for row in conclusion["decisions"]]
    engine.reclaims = [dict(row) for row in conclusion["reclaims"]]
    for lease_id, lease in engine.leases.items():
        engine.chains[lease_id] = engine._chain(lease_id)
        engine.chains[lease_id].generation = int(lease["generation"])
    return engine


def apply_evolution(engine: Engine, change: EvolutionChange, impact: dict[str, Any]) -> dict[str, Any]:
    seq = change.seq
    transitions: list[dict[str, Any]] = []
    leases: list[dict[str, Any]] = []
    holds: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    reclaims: list[dict[str, Any]] = []
    lease_order = impact["affected_lease_ids"]
    for ordinal, lease_id in enumerate(lease_order):
        lease = engine.leases[lease_id]
        node_id = lease["node_id"]
        before = int(lease["generation"])
        after = before + 1
        reasons = impact["lease_reasons"][lease_id]
        if "node_removed" in reasons:
            reason_kind = "node_removal"
            why = f"node_removed:{node_id}"
            hold_status = "reclaimed_node"
            basis = "node_removed_reclaimed"
        elif "node_seed_changed" in reasons:
            reason_kind = "node_seed_change"
            why = f"node_seed_changed:{node_id}"
            hold_status = "reclaimed_seed"
            basis = "node_seed_changed_reclaimed"
        else:
            reason_kind = "quota_invalidation"
            why = f"quota_version_superseded:{lease['quota_version']}->{change.new_quota_version}"
            hold_status = "reclaimed_quota"
            basis = "quota_invalidation_reclaimed"
        transition = {
            "event_seq": seq,
            "ordinal": ordinal,
            "lease_id": lease_id,
            "node_id": node_id,
            "trigger": "invalidate",
            "reason": why,
            "state_before": lease["state"],
            "state_after": "expired",
            "generation_before": before,
            "generation_after": after,
            "carried_label": None,
            "quota_version": int(lease["quota_version"]),
        }
        chain = engine._chain(lease_id)
        chain.append(**{k: transition[k] for k in (
            "event_seq", "ordinal", "node_id", "trigger", "reason", "state_before",
            "state_after", "generation_before", "generation_after", "carried_label", "quota_version"
        )})
        engine.transitions.append(transition)
        transitions.append(transition)
        lease["state"] = "expired"
        lease["generation"] = after
        lease["last_event_seq"] = seq
        leases.append(dict(lease))
        closed = engine._close_holds(lease_id=lease_id, seq=seq, status=hold_status, reason=why)
        holds.extend(closed)
        reclaim = {
            "reclaim_id": f"evolution-reclaim:{seq}:{ordinal}:{lease_id}",
            "event_seq": seq,
            "ordinal": ordinal,
            "lease_id": lease_id,
            "node_id": node_id,
            "reason_kind": reason_kind,
            "generation_before": before,
            "generation_after": after,
            "hold_ids": [row["hold_id"] for row in closed],
        }
        engine.reclaims.append(reclaim)
        reclaims.append(reclaim)
        decision = {
            "decision_id": f"evolution-ownership:{seq}:{ordinal}:{lease_id}",
            "event_seq": seq,
            "ordinal": ordinal,
            "artifact_key": lease["artifact_key"],
            "owner_node_id": None,
            "basis": basis,
            "generation": after,
            "quota_version": int(lease["quota_version"]),
            "decision": {
                "owner_node_id": None,
                "previous_owner_node_id": node_id,
                "lease_id": lease_id,
                "evolution": change.reason,
            },
        }
        engine.decisions.append(decision)
        decisions.append(decision)
    if change.new_quota_version is not None:
        engine.quota_version = int(change.new_quota_version)
    return {
        "transitions": transitions,
        "leases": leases,
        "holds": holds,
        "decisions": decisions,
        "reclaims": reclaims,
    }


def _node_dicts(nodes: list[NodeSpec] | tuple[NodeSpec, ...]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": node.node_id,
            "host": node.host,
            "port": node.port,
            "capacity_slots": node.capacity_slots,
            "max_leases": node.max_leases,
            "seed": node.seed,
        }
        for node in sorted(nodes, key=lambda item: item.node_id)
    ]


def evolve_and_verify(
    bundle: Bundle,
    *,
    cutoff_seq: int | None,
    change: EvolutionChange,
    evolved_nodes: list[NodeSpec] | tuple[NodeSpec, ...] | None = None,
) -> dict[str, Any]:
    base_engine = replay_bundle(bundle, through_seq=cutoff_seq)
    base = build_conclusion(
        spec_digest=bundle.spec_digest(),
        quota_version=base_engine.quota_version,
        nodes=_node_dicts(bundle.nodes),
        leases=list(base_engine.leases.values()),
        transitions=base_engine.transitions,
        holds=list(base_engine.holds.values()),
        decisions=base_engine.decisions,
        reclaims=base_engine.reclaims,
        cutoff_seq=cutoff_seq,
    )
    impact = find_impact(base, change)
    unaffected_before = {
        lease_id: canonical_bytes(next(lease for lease in base["leases"] if lease["lease_id"] == lease_id))
        for lease_id in impact["unaffected_lease_ids"]
    }

    incremental_engine = hydrate_engine(bundle, base)
    apply_evolution(incremental_engine, change, impact)
    nodes_for_report = _node_dicts(evolved_nodes if evolved_nodes is not None else bundle.nodes)
    incremental = build_conclusion(
        spec_digest=bundle.spec_digest(),
        quota_version=incremental_engine.quota_version,
        nodes=nodes_for_report,
        leases=list(incremental_engine.leases.values()),
        transitions=incremental_engine.transitions,
        holds=list(incremental_engine.holds.values()),
        decisions=incremental_engine.decisions,
        reclaims=incremental_engine.reclaims,
        cutoff_seq=change.seq,
    )

    full_engine = replay_bundle(bundle, through_seq=cutoff_seq)
    full_impact = find_impact(
        build_conclusion(
            spec_digest=bundle.spec_digest(),
            quota_version=full_engine.quota_version,
            nodes=_node_dicts(bundle.nodes),
            leases=list(full_engine.leases.values()),
            transitions=full_engine.transitions,
            holds=list(full_engine.holds.values()),
            decisions=full_engine.decisions,
            reclaims=full_engine.reclaims,
            cutoff_seq=cutoff_seq,
        ),
        change,
    )
    apply_evolution(full_engine, change, full_impact)
    full = build_conclusion(
        spec_digest=bundle.spec_digest(),
        quota_version=full_engine.quota_version,
        nodes=nodes_for_report,
        leases=list(full_engine.leases.values()),
        transitions=full_engine.transitions,
        holds=list(full_engine.holds.values()),
        decisions=full_engine.decisions,
        reclaims=full_engine.reclaims,
        cutoff_seq=change.seq,
    )
    assert_consistent(full, incremental, DivergenceError)

    unaffected_after = {
        lease_id: canonical_bytes(next(lease for lease in incremental["leases"] if lease["lease_id"] == lease_id))
        for lease_id in impact["unaffected_lease_ids"]
    }
    untouched_ok = unaffected_before == unaffected_after
    if not untouched_ok:
        raise DivergenceError("unaffected lease canonical bytes changed", field="leases")

    return {
        "ok": True,
        "impact": impact,
        "conclusion": incremental,
        "digest": incremental["content_digest"],
        "untouched_ok": untouched_ok,
        "incremental_matches_full_rebuild": True,
    }
