from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .spec import RevisionSpec, Spec, topological_shards


@dataclass(frozen=True)
class InvalidationStep:
    shard_id: str
    invalid: bool
    reason: str
    source: str
    basis: dict[str, Any]


@dataclass(frozen=True)
class InvalidationPlan:
    revision: int
    invalid_shard_ids: tuple[str, ...]
    stable_topological_order: tuple[str, ...]
    steps: tuple[InvalidationStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "invalid_shard_ids": list(self.invalid_shard_ids),
            "stable_topological_order": list(self.stable_topological_order),
            "steps": [step.__dict__ | {"basis": step.basis} for step in self.steps],
        }


def build_invalidation_plan(spec: Spec, revision: RevisionSpec | int) -> InvalidationPlan:
    if isinstance(revision, int):
        revision = next(rev for rev in spec.revisions if rev.revision == revision)

    by_id = spec.shard_by_id
    topo = [shard.shard_id for shard in topological_shards(spec)]
    descendants: dict[str, set[str]] = {shard_id: set() for shard_id in by_id}
    for shard in spec.shards:
        for dep in shard.dependencies:
            descendants[dep].add(shard.shard_id)

    direct: dict[str, tuple[str, dict[str, Any]]] = {}

    def mark_direct(shard_id: str, reason: str, basis: dict[str, Any]) -> None:
        if shard_id not in direct:
            direct[shard_id] = (reason, basis)

    global_config_changed = bool(revision.config) and not revision.changed_shard_ids and not revision.changed_node_ids
    if global_config_changed:
        for shard in spec.shards:
            mark_direct(
                shard.shard_id,
                "global_revision_config_changed",
                {"revision": revision.revision, "config_keys": sorted(revision.config)},
            )

    for shard_id in revision.changed_shard_ids:
        mark_direct(
            shard_id,
            "explicit_shard_reference_in_revision",
            {"revision": revision.revision, "field": "changed_shard_ids/changes"},
        )

    for node_id in revision.changed_node_ids:
        referencing = [
            shard.shard_id
            for shard in sorted(spec.shards, key=lambda s: s.shard_id)
            if node_id in shard.node_refs
        ]
        if not referencing:
            # A changed node can affect shards last attributed to it. The plan is
            # still stable; runtime/verifier augments this with ledger node refs.
            referencing = []
        for shard_id in referencing:
            mark_direct(
                shard_id,
                "node_reference_changed",
                {"node_id": node_id, "reference_field": "node_refs"},
            )

    invalid_reasons: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for shard_id, (reason, basis) in direct.items():
        invalid_reasons[shard_id] = (reason, f"revision:{revision.revision}", basis)

    # Propagate from direct failures through reverse dependency edges. Process in
    # stable topological order so the recorded path/basis is reproducible.
    changed = True
    while changed:
        changed = False
        for shard_id in topo:
            shard = by_id[shard_id]
            invalid_deps = [dep for dep in shard.dependencies if dep in invalid_reasons]
            if shard_id not in invalid_reasons and invalid_deps:
                dep = sorted(invalid_deps)[0]
                prior_reason, prior_source, prior_basis = invalid_reasons[dep]
                invalid_reasons[shard_id] = (
                    "upstream_invalidated",
                    f"dependency:{dep}",
                    {
                        "dependency": dep,
                        "dependency_reason": prior_reason,
                        "dependency_source": prior_source,
                        "edge": f"{dep}->{shard_id}",
                    },
                )
                changed = True

    invalid_set = set(invalid_reasons)
    invalid_order = tuple(shard_id for shard_id in topo if shard_id in invalid_set)
    steps: list[InvalidationStep] = []
    for shard_id in topo:
        if shard_id in invalid_reasons:
            reason, source, basis = invalid_reasons[shard_id]
            steps.append(InvalidationStep(shard_id, True, reason, source, basis))
        else:
            deps = list(by_id[shard_id].dependencies)
            steps.append(
                InvalidationStep(
                    shard_id,
                    False,
                    "no_direct_change_and_no_invalid_upstream",
                    "dependency_graph",
                    {"dependencies": sorted(deps)},
                )
            )
    return InvalidationPlan(
        revision=revision.revision,
        invalid_shard_ids=invalid_order,
        stable_topological_order=tuple(topo),
        steps=tuple(steps),
    )


def ledger_node_referenced_plan(
    spec: Spec, base_plan: InvalidationPlan, shard_nodes: dict[str, str], changed_node_ids: set[str]
) -> InvalidationPlan:
    """Augment a declaration plan with node references recovered from the ledger."""

    extra = [
        shard_id
        for shard_id, node_id in sorted(shard_nodes.items())
        if node_id in changed_node_ids and shard_id not in set(base_plan.invalid_shard_ids)
    ]
    if not extra:
        return base_plan
    # Recompute through the same stable constructor by adding synthetic revision refs.
    rev = next(rev for rev in spec.revisions if rev.revision == base_plan.revision)
    raw = dict(rev.raw)
    raw_shards = list(raw.get("changed_shard_ids", [])) + extra
    raw["changed_shard_ids"] = tuple(dict.fromkeys(raw_shards))
    augmented = RevisionSpec(
        revision=rev.revision,
        parent_revision=rev.parent_revision,
        config=rev.config,
        changed_shard_ids=tuple(raw_shards),
        changed_node_ids=rev.changed_node_ids,
        raw=raw,
    )
    return build_invalidation_plan(spec, augmented)
