from __future__ import annotations

from typing import Any

from .canonical import digest_json
from .spec import EventStream, Spec, stable_topological_order


def dependent_graph(spec: Spec) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {shard.shard_id: [] for shard in spec.shards}
    for shard in spec.shards:
        for dependency in shard.dependencies:
            graph[dependency].append(shard.shard_id)
    for values in graph.values():
        values.sort()
    return graph


def transitive_dependents(spec: Spec, root_ids: set[str] | list[str]) -> set[str]:
    graph = dependent_graph(spec)
    invalid = set(root_ids)
    stack = sorted(set(root_ids))
    while stack:
        current = stack.pop()
        for dependent in graph.get(current, []):
            if dependent not in invalid:
                invalid.add(dependent)
                stack.append(dependent)
    return invalid


def invalidation_plan(
    spec: Spec,
    direct_shard_ids: set[str] | list[str],
    *,
    reason: str = "rollback_to_historical_quota_version",
    node_id: str | None = None,
    quota_version: int | None = None,
    target_version: int | None = None,
) -> dict[str, Any]:
    direct = set(direct_shard_ids)
    invalid = transitive_dependents(spec, direct)
    order = stable_topological_order(spec.shards)
    position = {shard_id: index for index, shard_id in enumerate(order)}
    steps: list[dict[str, Any]] = []
    for shard_id in order:
        if shard_id not in invalid:
            continue
        if shard_id in direct:
            steps.append({
                "shard_id": shard_id,
                "kind": "direct",
                "basis": reason,
                "node_id": node_id,
                "quota_version": quota_version,
                "target_version": target_version,
            })
            continue
        shard = spec.shard_map[shard_id]
        invalid_deps = [dep for dep in shard.dependencies if dep in invalid]
        steps.append({
            "shard_id": shard_id,
            "kind": "indirect_dependency",
            "basis": "upstream_dependency_invalid",
            "invalid_dependencies": invalid_deps,
            "stable_order_index": position[shard_id],
        })
    ordered = sorted(invalid, key=lambda shard_id: (position[shard_id], shard_id))
    return {
        "invalid_shards": ordered,
        "count": len(ordered),
        "steps": steps,
        "order_rule": "stable Kahn topological order with lexicographic shard_id tie-break",
    }


def event_hash(event: Any) -> str:
    if hasattr(event, "seq"):
        return digest_json({"seq": event.seq, "type": event.type, **event.data})
    return digest_json(event)


def compare_event_streams(old: EventStream | None, new: EventStream) -> dict[str, Any]:
    old_events = list(old.events) if old else []
    new_events = list(new.events)
    first_changed: int | None = None
    for index in range(max(len(old_events), len(new_events))):
        old_event = old_events[index] if index < len(old_events) else None
        new_event = new_events[index] if index < len(new_events) else None
        if old_event is None or new_event is None or event_hash(old_event) != event_hash(new_event):
            first_changed = index + 1
            break
    unchanged_prefix = (first_changed - 1) if first_changed is not None else len(new_events)
    changed_events = new_events[unchanged_prefix:] if first_changed is not None else []
    target_shards: set[str] = set()
    target_nodes: set[str] = set()
    dirty_water: set[str] = set()
    for event in changed_events:
        data = event.data
        if "shard_id" in data:
            target_shards.add(data["shard_id"])
        if "node_id" in data:
            target_nodes.add(data["node_id"])
            dirty_water.add(f"node:{data['node_id']}")
        if event.type in {"dispatch", "cancel", "rollback", "quota_upgrade"}:
            dirty_water.add("global")
        if event.type == "dispatch":
            dirty_water.add(f"node:{data['node_id']}")
    affected_shards = transitive_dependents(_spec_placeholder, target_shards) if False else set(target_shards)
    return {
        "first_changed_seq": first_changed,
        "unchanged_prefix_seq": unchanged_prefix,
        "changed_event_seqs": [event.seq for event in changed_events],
        "target_shards": sorted(target_shards),
        "target_nodes": sorted(target_nodes),
        "dirty_water_scopes": sorted(dirty_water),
        "needs_suffix_replay": first_changed is not None,
    }


def event_edit_impact(spec: Spec, old: EventStream | None, new: EventStream) -> dict[str, Any]:
    base = compare_event_streams(old, new)
    old_by_seq = {event.seq: event for event in old.events} if old is not None else {}
    if old is None:
        content_changed_events = list(new.events)
    else:
        content_changed_events = [
            event for event in new.events
            if event.seq not in old_by_seq or event_hash(old_by_seq[event.seq]) != event_hash(event)
        ]
    dispatch_node = {
        event.data["shard_id"]: event.data["node_id"]
        for event in new.events
        if event.type == "dispatch"
    }
    state_targets: set[str] = set()
    aggregate_targets: set[str] = set()
    quota_nodes: set[str] = set()
    rollback_plans: list[dict[str, Any]] = []
    for event in content_changed_events:
        data = event.data
        if "shard_id" in data:
            state_targets.add(data["shard_id"])
        if event.type == "dispatch":
            aggregate_targets.add(data["shard_id"])
        elif event.type in {"cancel", "node_offline"}:
            # Cancellation/offline changes ownership/scheduling state, not the
            # deterministic bytes of an already completed unrelated aggregate.
            pass
        elif event.type == "priority":
            # Priority only changes admission order; aggregation inputs are unchanged.
            pass
        elif event.type == "quota_upgrade":
            node_id = data.get("node_id")
            if node_id:
                quota_nodes.add(node_id)
                aggregate_targets.update(sid for sid, nid in dispatch_node.items() if nid == node_id)
            else:
                quota_nodes.update(spec.node_map)
                aggregate_targets.update(dispatch_node)
        elif event.type == "rollback":
            node_id = data.get("node_id")
            scoped = set(dispatch_node) if node_id is None else {sid for sid, nid in dispatch_node.items() if nid == node_id}
            aggregate_targets.update(scoped)
            direct = scoped or set(data.get("shard_id", "").split()) if data.get("shard_id") else scoped
            plan = invalidation_plan(
                spec,
                direct,
                node_id=node_id,
                target_version=data.get("target_version"),
            )
            rollback_plans.append({"seq": event.seq, **plan})
    affected = transitive_dependents(spec, state_targets)
    order = stable_topological_order(spec.shards)
    position = {shard_id: index for index, shard_id in enumerate(order)}
    base["affected_shards"] = sorted(affected, key=lambda sid: (position[sid], sid))
    base["state_affected_shards"] = sorted(affected, key=lambda sid: (position[sid], sid))
    base["content_changed_seqs"] = [event.seq for event in content_changed_events]
    base["aggregate_recompute_shards"] = sorted(aggregate_targets, key=lambda sid: (position.get(sid, 10**9), sid))
    if content_changed_events and all(event.type == "priority" for event in content_changed_events):
        base["aggregate_recompute_shards"] = []
        base["aggregate_recompute_reason"] = "priority changes admission order only; deterministic aggregate inputs are unchanged"
    base["quota_nodes"] = sorted(quota_nodes)
    base["rollback_plans"] = rollback_plans
    base["aggregate_cache_hint"] = "reuse ledger aggregate/sample bytes unless shard dispatch/node/quota input changed"
    base["water_recompute_scopes"] = sorted(set(base["dirty_water_scopes"]) | {f"node:{node_id}" for node_id in quota_nodes})
    base["decision_replay_from_seq"] = base["first_changed_seq"]
    return base


def spec_edit_impact(old_spec: Spec | None, new_spec: Spec) -> dict[str, Any]:
    if old_spec is None or old_spec.digest() == new_spec.digest():
        return {"spec_changed": False, "changed_shards": [], "replace_snapshot": False}
    old_shards = old_spec.shard_map if old_spec else {}
    old_nodes = old_spec.node_map if old_spec else {}
    changed_shards: set[str] = set()
    changed_nodes: set[str] = set()
    for shard_id, shard in new_spec.shard_map.items():
        if shard_id not in old_shards or digest_json(old_shards[shard_id]) != digest_json(shard):
            changed_shards.add(shard_id)
    for node_id, node in new_spec.node_map.items():
        if node_id not in old_nodes or digest_json(old_nodes[node_id]) != digest_json(node):
            changed_nodes.add(node_id)
            # Node seed/port/caliber can affect every shard assigned there;
            # assignment is event-derived, so conservatively mark graph roots.
    affected = transitive_dependents(new_spec, changed_shards)
    return {
        "spec_changed": True,
        "replace_snapshot": True,
        "changed_shards": sorted(changed_shards),
        "changed_nodes": sorted(changed_nodes),
        "affected_shards": sorted(affected),
        "reason": "spec_digest_mismatch_requires_whole_snapshot_replacement",
    }


_spec_placeholder = None
