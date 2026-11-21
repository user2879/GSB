"""Compile manifest versions into immutable per-key distribution artifacts.

Compilation is key-scoped and deterministic: an unchanged item produces a
byte-identical artifact digest across versions.  Invalidation walks the
derived-from graph (direct and transitive dependents) together with node
reference changes, and emits a stable topological order with propagation
evidence for every step.
"""
from __future__ import annotations

from typing import Any

from .fixedpoint import Grid, derive_qvalue, error_bound, quantize
from .models import CompiledItem, ConfigItem, ManifestVersion
from .util import canonical_json, content_digest, stable_topological


def _item_fingerprint(item: ConfigItem, version: ManifestVersion) -> dict[str, Any]:
    """Change signature: key value, target set and derived coefficients."""
    return {
        "key": item.key,
        "value": item.value,
        "targets": sorted(version.targets),
        "derived_from": dict(sorted(item.derived_from.items())),
        "strategy": version.strategy.to_dict(),
    }


def compile_version(version: ManifestVersion, grid: Grid) -> dict[str, CompiledItem]:
    edges = {key: set(item.derived_from) for key, item in version.items.items()}
    order = stable_topological(set(version.items), edges)
    compiled: dict[str, CompiledItem] = {}
    for key in order:
        item = version.items[key]
        qcoeff = {dep: quantize(coeff, grid) for dep, coeff in item.derived_from.items()}
        if item.is_derived():
            qinputs = {dep: compiled[dep].qvalue for dep in item.derived_from}
            qvalue = derive_qvalue(qcoeff, qinputs, grid)
            bound = error_bound(qcoeff, qinputs, grid)
        else:
            qvalue = quantize(item.value, grid)
            bound = None
        artifact_body = {
            "key": item.key,
            "qvalue": qvalue,
            "is_derived": item.is_derived(),
            "derived_from_q": dict(sorted(qcoeff.items())),
            "targets": sorted(version.targets),
            "strategy": version.strategy.to_dict(),
            "error_bound": bound,
        }
        compiled[key] = CompiledItem(
            key=key,
            qvalue=qvalue,
            is_derived=item.is_derived(),
            derived_from_q=qcoeff,
            targets=tuple(sorted(version.targets)),
            strategy=version.strategy.to_dict(),
            error_bound=bound,
            artifact_digest=content_digest(artifact_body),
        )
    return compiled


def _dependents_graph(version: ManifestVersion) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {key: set() for key in version.items}
    for key, item in version.items.items():
        for dep in item.derived_from:
            graph[dep].add(key)
    return graph


def diff_invalidation(previous: ManifestVersion, current: ManifestVersion,
                      grid: Grid) -> dict[str, Any]:
    """Compute changed keys and propagate invalidation through dependencies.

    Returns a report containing fingerprints, the directly changed set, the
    full invalidation set in stable topological order, per-step propagation
    evidence and a separate node-reference flag.
    """
    prev_fingerprints = {key: canonical_json(_item_fingerprint(item, previous))
                         for key, item in previous.items.items()}
    cur_fingerprints = {key: canonical_json(_item_fingerprint(item, current))
                        for key, item in current.items.items()}
    added = sorted(set(current.items) - set(previous.items))
    removed = sorted(set(previous.items) - set(current.items))
    direct_changed = sorted(
        key for key in current.items
        if key in prev_fingerprints and prev_fingerprints[key] != cur_fingerprints[key]
    )
    node_targets_changed = sorted(previous.targets) != sorted(current.targets)
    strategy_changed = previous.strategy.to_dict() != current.strategy.to_dict()

    graph = _dependents_graph(current)
    affected: set[str] = set(added) | set(direct_changed)
    steps: list[dict[str, Any]] = []
    for key in added:
        steps.append({"key": key, "reason": "new_item", "propagated_from": None})
    for key in direct_changed:
        changed_dimensions = _changed_dimensions(previous.items[key], current.items[key],
                                                 previous, current)
        steps.append({"key": key, "reason": "fingerprint_changed",
                      "propagated_from": None, "dimensions": changed_dimensions})

    frontier = set(affected)
    while frontier:
        next_frontier: set[str] = set()
        for changed_key in sorted(frontier):
            for dependent in sorted(graph.get(changed_key, ())):
                if dependent not in affected:
                    affected.add(dependent)
                    next_frontier.add(dependent)
                    steps.append({
                        "key": dependent,
                        "reason": "derived_dependency_propagation",
                        "propagated_from": changed_key,
                    })
        frontier = next_frontier

    edges = {key: set(item.derived_from) for key, item in current.items.items()}
    ordered = stable_topological(set(current.items), edges)
    invalidation_order = [key for key in ordered if key in affected]
    prev_compiled = compile_version(previous, grid)
    cur_compiled = compile_version(current, grid)
    unchanged = sorted(key for key in current.items
                       if key in prev_compiled
                       and prev_compiled[key].artifact_digest == cur_compiled[key].artifact_digest
                       and key not in affected)
    return {
        "previous_version": previous.version,
        "current_version": current.version,
        "added": added,
        "removed": removed,
        "direct_changed": direct_changed,
        "node_targets_changed": node_targets_changed,
        "strategy_changed": strategy_changed,
        "invalidation_set": invalidation_order,
        "propagation_steps": steps,
        "byte_identical_unchanged": unchanged,
        "artifact_digests": {
            "previous": {key: value.artifact_digest for key, value in prev_compiled.items()},
            "current": {key: value.artifact_digest for key, value in cur_compiled.items()},
        },
    }


def _changed_dimensions(old: ConfigItem, new: ConfigItem,
                        previous: ManifestVersion, current: ManifestVersion) -> list[str]:
    dimensions: list[str] = []
    if old.value != new.value:
        dimensions.append("value")
    if old.derived_from != new.derived_from:
        dimensions.append("derived_coefficients")
    if sorted(previous.targets) != sorted(current.targets):
        dimensions.append("target_nodes")
    if previous.strategy.to_dict() != current.strategy.to_dict():
        dimensions.append("strategy")
    return dimensions or ["fingerprint"]


def build_dispatch_plan(compiled: dict[str, CompiledItem],
                        version: ManifestVersion) -> dict[str, Any]:
    """Per-node shard plan carrying only the keys targeted at that node."""
    plan: dict[str, Any] = {
        "version": version.version,
        "strategy": version.strategy.to_dict(),
        "nodes": {},
    }
    for node_id in version.targets:
        plan["nodes"][node_id] = {
            "shard_count": version.strategy.shard_count,
            "keys": sorted(compiled),
            "items": {key: compiled[key].to_dict() for key in sorted(compiled)},
        }
    plan["plan_digest"] = content_digest({k: v for k, v in plan.items() if k != "plan_digest"})
    return plan
