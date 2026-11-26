"""Invalidation propagation after metadata-version / node-set evolution.

A version bump or node-set change propagates along node -> object key ->
parent/merge references -> lease references. Only the affected subset is
recomputed; untouched objects keep byte-identical digests, adoption and lease
ownership. The incremental result is cross-checked against a from-scratch full
rebuild field by field.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import MergeCandidate, NodeSpec, ObjectManifest


@dataclass
class PropagationStep:
    order: int
    relation: str
    source: str
    affected: str
    basis: str

    def as_dict(self) -> dict:
        return {
            "order": self.order,
            "relation": self.relation,
            "source": self.source,
            "affected": self.affected,
            "basis": self.basis,
        }


@dataclass
class InvalidationResult:
    affected_keys: list[str]
    affected_shards: list[tuple[str, int]]
    steps: list[PropagationStep] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "affected_keys": self.affected_keys,
            "affected_shards": [list(item) for item in self.affected_shards],
            "steps": [step.as_dict() for step in self.steps],
        }


def propagate_invalidation(
    *,
    old_nodes: dict[str, NodeSpec],
    new_nodes: dict[str, NodeSpec],
    manifests: dict[str, ObjectManifest],
    candidates: list[MergeCandidate],
    bumped_versions: dict[str, int] | None = None,
) -> InvalidationResult:
    bumped_versions = bumped_versions or {}
    steps: list[PropagationStep] = []
    order = 0
    affected_nodes: set[str] = set()

    # Step 1: nodes whose meta version changed or whose set changed.
    for node_id in sorted(set(old_nodes) | set(new_nodes)):
        old = old_nodes.get(node_id)
        new = new_nodes.get(node_id)
        if old is None or new is None:
            affected_nodes.add(node_id)
            order += 1
            steps.append(PropagationStep(
                order, "node-set", node_id, node_id,
                "node added or removed",
            ))
        elif old.meta_version != new.meta_version or node_id in bumped_versions:
            affected_nodes.add(node_id)
            order += 1
            steps.append(PropagationStep(
                order, "meta-version", node_id, node_id,
                f"version {old.meta_version} -> {new.meta_version}",
            ))

    # Step 2: node -> object key / shard via declared placements.
    affected_shards: set[tuple[str, int]] = set()
    affected_keys: set[str] = set()
    for key in sorted(manifests):
        manifest = manifests[key]
        for placement in manifest.placements:
            if placement.node_id in affected_nodes:
                affected_shards.add((key, placement.shard_index))
                affected_keys.add(key)
                order += 1
                steps.append(PropagationStep(
                    order, "node->object", placement.node_id,
                    f"{key}#{placement.shard_index}",
                    "placement references affected node",
                ))

    # Step 3: merge / parent reference chain. A shard invalidates its parents
    # and children along parent_merge_id within the same object.
    by_id = {candidate.merge_id: candidate for candidate in candidates}
    queue = list(affected_shards)
    seen_pairs: set[tuple[str, int]] = set(affected_shards)
    while queue:
        key, shard_index = queue.pop(0)
        related = [c for c in candidates
                   if c.object_key == key and c.shard_index == shard_index]
        for candidate in related:
            neighbors: list[str] = []
            if candidate.parent_merge_id and candidate.parent_merge_id in by_id:
                neighbors.append(candidate.parent_merge_id)
            for other in candidates:
                if other.parent_merge_id == candidate.merge_id:
                    neighbors.append(other.merge_id)
            for merge_id in neighbors:
                target = by_id[merge_id]
                pair = (target.object_key, target.shard_index)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    queue.append(pair)
                    affected_keys.add(target.object_key)
                    order += 1
                    steps.append(PropagationStep(
                        order, "merge-ref", candidate.merge_id,
                        f"{target.object_key}#{target.shard_index}",
                        f"parent/child merge reference {merge_id}",
                    ))

    # Step 4: lease references (any lease tied to an affected shard).
    for candidate in candidates:
        pair = (candidate.object_key, candidate.shard_index)
        if pair in seen_pairs and candidate.lease_id:
            order += 1
            steps.append(PropagationStep(
                order, "lease-ref", candidate.merge_id, candidate.lease_id,
                "lease attached to invalidated shard",
            ))

    ordered_shards = sorted(seen_pairs)
    return InvalidationResult(
        affected_keys=sorted(affected_keys),
        affected_shards=ordered_shards,
        steps=steps,
    )


def filter_unaffected(
    candidates: list[MergeCandidate],
    invalidation: InvalidationResult,
) -> list[MergeCandidate]:
    invalid_set = set(invalidation.affected_shards)
    return [candidate for candidate in candidates
            if (candidate.object_key, candidate.shard_index) not in invalid_set]


def compare_field_by_field(left: list[MergeCandidate],
                           right: list[MergeCandidate]) -> list[dict]:
    """Return an empty list when two candidate sets are field-identical."""
    def keyfn(candidate: MergeCandidate):
        return (candidate.object_key, candidate.shard_index, candidate.merge_seq,
                candidate.merge_id)

    left_map = {keyfn(c): c for c in left}
    right_map = {keyfn(c): c for c in right}
    mismatches: list[dict] = []
    for identity in sorted(set(left_map) | set(right_map)):
        if identity not in left_map or identity not in right_map:
            mismatches.append({"identity": list(identity), "field": "presence"})
            continue
        a, b = left_map[identity], right_map[identity]
        for field_name in ("shard_digest", "content_check", "meta_version",
                           "source_format", "node_id", "shard_bytes"):
            if getattr(a, field_name) != getattr(b, field_name):
                mismatches.append({
                    "identity": list(identity),
                    "field": field_name,
                    "left": getattr(a, field_name),
                    "right": getattr(b, field_name),
                })
    return mismatches


__all__ = [
    "propagate_invalidation",
    "filter_unaffected",
    "compare_field_by_field",
    "InvalidationResult",
    "PropagationStep",
]
