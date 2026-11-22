"""Cross-version diff measurement and derivation impact analysis.

All outputs are built from sorted keys and lexicographic topological order,
so results never depend on dict or hash iteration order.  Only genuinely
changed items become impact roots; an unchanged item whose entire ancestor
closure is unchanged is never emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Manifest

ADDED = "added"
REMOVED = "removed"
VALUE_CHANGED = "value_changed"
TARGETS_CHANGED = "targets_changed"
COEFFICIENT_CHANGED = "coefficient_changed"


@dataclass
class ItemDiff:
    key: str
    change_types: list[str]
    baseline_value: float | None
    candidate_value: float | None
    baseline_targets: list[str]
    candidate_targets: list[str]
    coefficient_changes: list[dict[str, float | None]]
    baseline_parents: list[str]
    candidate_parents: list[str]


@dataclass
class ImpactChain:
    root_key: str
    change_types: list[str]
    downstream_order: list[str]
    chain: list[str]
    affected_nodes: list[str]


@dataclass
class DiffResult:
    baseline_version: int
    candidate_version: int
    item_diffs: dict[str, ItemDiff]
    changed_keys: list[str]
    impact_chains: list[ImpactChain]
    topological_order: list[str]
    affected_node_union: list[str]
    unchanged_keys: list[str]

    def as_dict(self) -> dict:
        return {
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "changed_keys": self.changed_keys,
            "unchanged_keys": self.unchanged_keys,
            "item_diffs": {
                key: {
                    "key": diff.key,
                    "change_types": diff.change_types,
                    "baseline_value": diff.baseline_value,
                    "candidate_value": diff.candidate_value,
                    "baseline_targets": diff.baseline_targets,
                    "candidate_targets": diff.candidate_targets,
                    "baseline_parents": diff.baseline_parents,
                    "candidate_parents": diff.candidate_parents,
                    "coefficient_changes": diff.coefficient_changes,
                }
                for key, diff in sorted(self.item_diffs.items())
            },
            "impact_chains": [
                {
                    "root_key": chain.root_key,
                    "change_types": chain.change_types,
                    "downstream_order": chain.downstream_order,
                    "chain": chain.chain,
                    "affected_nodes": chain.affected_nodes,
                }
                for chain in self.impact_chains
            ],
            "affected_node_union": self.affected_node_union,
            "topological_order": self.topological_order,
        }


def diff_manifests(baseline: Manifest, candidate: Manifest) -> DiffResult:
    base_keys = set(baseline.items)
    cand_keys = set(candidate.items)
    shared = sorted(base_keys & cand_keys)
    item_diffs: dict[str, ItemDiff] = {}
    changed_keys: list[str] = []

    for key in sorted(cand_keys - base_keys):
        item = candidate.items[key]
        item_diffs[key] = ItemDiff(
            key=key, change_types=[ADDED], baseline_value=None,
            candidate_value=item.value, baseline_targets=[],
            candidate_targets=list(item.targets), coefficient_changes=[],
            baseline_parents=[], candidate_parents=_parents(candidate, key),
        )
        changed_keys.append(key)

    for key in sorted(base_keys - cand_keys):
        item = baseline.items[key]
        item_diffs[key] = ItemDiff(
            key=key, change_types=[REMOVED], baseline_value=item.value,
            candidate_value=None, baseline_targets=list(item.targets),
            candidate_targets=[], coefficient_changes=[],
            baseline_parents=_parents(baseline, key), candidate_parents=[],
        )
        changed_keys.append(key)

    for key in shared:
        b_item = baseline.items[key]
        c_item = candidate.items[key]
        change_types: list[str] = []
        if b_item.value != c_item.value:
            change_types.append(VALUE_CHANGED)
        if b_item.targets != c_item.targets:
            change_types.append(TARGETS_CHANGED)
        coefficient_changes = _edge_changes(baseline, candidate, key)
        if coefficient_changes:
            change_types.append(COEFFICIENT_CHANGED)
        if change_types:
            item_diffs[key] = ItemDiff(
                key=key, change_types=change_types,
                baseline_value=b_item.value, candidate_value=c_item.value,
                baseline_targets=list(b_item.targets),
                candidate_targets=list(c_item.targets),
                coefficient_changes=coefficient_changes,
                baseline_parents=_parents(baseline, key),
                candidate_parents=_parents(candidate, key),
            )
            changed_keys.append(key)

    changed_keys.sort()
    unchanged_keys = sorted(key for key in shared if key not in item_diffs)
    impact_chains = _build_impact_chains(candidate, item_diffs, changed_keys)
    affected_union = sorted({
        node
        for chain in impact_chains
        for node in chain.affected_nodes
    })
    return DiffResult(
        baseline_version=baseline.version,
        candidate_version=candidate.version,
        item_diffs=item_diffs,
        changed_keys=changed_keys,
        impact_chains=impact_chains,
        topological_order=list(candidate.order),
        affected_node_union=affected_union,
        unchanged_keys=unchanged_keys,
    )


def _parents(manifest: Manifest, key: str) -> list[str]:
    return sorted(edge.parent for edge in manifest.items[key].derived_from)


def _edge_changes(baseline: Manifest, candidate: Manifest,
                  key: str) -> list[dict[str, float | None]]:
    b_edges = {edge.parent: edge.coefficient
               for edge in baseline.items[key].derived_from}
    c_edges = {edge.parent: edge.coefficient
               for edge in candidate.items[key].derived_from}
    changes: list[dict[str, float | None]] = []
    for parent in sorted(set(b_edges) | set(c_edges)):
        b_coeff = b_edges.get(parent)
        c_coeff = c_edges.get(parent)
        if b_coeff != c_coeff:
            changes.append({"parent": parent,
                            "baseline_coefficient": b_coeff,
                            "candidate_coefficient": c_coeff})
    return changes


def _downstream(manifest: Manifest, root: str) -> list[str]:
    """Reachable descendants of *root* in candidate DAG, topo sorted."""

    children: dict[str, list[str]] = {key: [] for key in manifest.items}
    for key, item in manifest.items.items():
        for edge in item.derived_from:
            children.setdefault(edge.parent, []).append(key)
    seen: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        for child in sorted(children.get(current, [])):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return [key for key in manifest.order if key in seen]


def _build_impact_chains(manifest: Manifest,
                         item_diffs: dict[str, ItemDiff],
                         changed_keys: list[str]) -> list[ImpactChain]:
    chains: list[ImpactChain] = []
    for root in changed_keys:
        if REMOVED in item_diffs[root].change_types:
            # Removed items have no candidate downstream; record the removal
            # itself with the baseline-side nodes stored on the ItemDiff.
            baseline_nodes = list(item_diffs[root].baseline_targets)
            chains.append(ImpactChain(
                root_key=root,
                change_types=list(item_diffs[root].change_types),
                downstream_order=[],
                chain=[root],
                affected_nodes=sorted(baseline_nodes),
            ))
            continue
        downstream = _downstream(manifest, root)
        affected: set[str] = set(manifest.items[root].targets)
        for key in downstream:
            affected.update(manifest.items[key].targets)
        chains.append(ImpactChain(
            root_key=root,
            change_types=list(item_diffs[root].change_types),
            downstream_order=downstream,
            chain=[root, *downstream],
            affected_nodes=sorted(affected),
        ))
    chains.sort(key=lambda chain: chain.root_key)
    return chains
