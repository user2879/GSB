"""Requirement 7: dependency-chain invalidation and subset recomputation.

Propagation chain: layer parameter version -> batch -> loss summary -> checkpoint.

A spec change (layer scale, dimension or inter-layer reference) first marks the
changed layers, then expands to their derived (transitive child) layers. Batches
whose scope intersects that closure are invalidated; their loss summaries and
any checkpoints frozen on those batches/param versions are invalidated in
stable shard-id then batch-id order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .compute import ComputeResult, compute_batch
from .digest import canonical
from .errors import DivergeError
from .spec import LayerSpec, ShardSpec, Spec


@dataclass
class PropagationStep:
    order: int
    shard_id: str
    kind: str            # layer | batch | loss_summary | checkpoint
    object_id: str
    basis: str

    def to_dict(self) -> dict:
        return {"order": self.order, "shard_id": self.shard_id,
                "kind": self.kind, "object_id": self.object_id, "basis": self.basis}


@dataclass
class InvalidationReport:
    changed_layers: dict[str, list[str]]
    affected_layers: dict[str, list[str]]
    invalid_batches: dict[str, list[str]]
    invalid_checkpoints: list[str]
    steps: list[PropagationStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "changed_layers": {k: list(v) for k, v in sorted(self.changed_layers.items())},
            "affected_layers": {k: list(v) for k, v in sorted(self.affected_layers.items())},
            "invalid_batches": {k: list(v) for k, v in sorted(self.invalid_batches.items())},
            "invalid_checkpoints": list(self.invalid_checkpoints),
            "steps": [step.to_dict() for step in self.steps],
        }


def _derived_closure(layers: tuple[LayerSpec, ...], roots: set[str]) -> set[str]:
    children: dict[str, list[str]] = {}
    for layer in layers:
        if layer.parent:
            children.setdefault(layer.parent, []).append(layer.layer_id)
    closure = set(roots)
    stack = list(roots)
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            if child not in closure:
                closure.add(child)
                stack.append(child)
    return closure


def diff_layer_specs(old: Spec, new: Spec) -> dict[str, list[str]]:
    """Return shard_id -> changed layer_id (scale, dimension or parent)."""
    changed: dict[str, list[str]] = {}
    old_by_shard = {shard.shard_id: {layer.layer_id: layer for layer in shard.layers}
                    for shard in old.shards}
    for new_shard in new.shards:
        old_layers = old_by_shard.get(new_shard.shard_id, {})
        ids: list[str] = []
        for layer in new_shard.layers:
            previous = old_layers.get(layer.layer_id)
            if previous is None:
                ids.append(layer.layer_id)
                continue
            if (previous.scale != layer.scale or previous.dim_in != layer.dim_in
                    or previous.dim_out != layer.dim_out
                    or previous.parent != layer.parent):
                ids.append(layer.layer_id)
        if ids:
            changed[new_shard.shard_id] = sorted(ids)
    return changed


def plan_invalidation(old: Spec, new: Spec,
                      batch_scopes: dict[str, dict[str, tuple[str, ...]]],
                      batch_param_versions: dict[str, dict[str, int]]
                      ) -> InvalidationReport:
    """``batch_scopes[shard][batch]`` = scope layer tuple used at dispatch."""
    changed = diff_layer_specs(old, new)
    affected: dict[str, list[str]] = {}
    invalid_batches: dict[str, list[str]] = {}
    steps: list[PropagationStep] = []
    order = 0
    new_shards = {shard.shard_id: shard for shard in new.shards}
    for shard_id in sorted(changed):
        roots = set(changed[shard_id])
        closure = _derived_closure(new_shards[shard_id].layers, roots)
        affected[shard_id] = sorted(closure)
        for layer_id in sorted(roots):
            order += 1
            steps.append(PropagationStep(order, shard_id, "layer", layer_id,
                                         "spec changed: scale/dimension/parent"))
        for layer_id in sorted(closure - roots):
            order += 1
            steps.append(PropagationStep(order, shard_id, "layer", layer_id,
                                         "derived from changed layer (transitive)"))
        scopes = batch_scopes.get(shard_id, {})
        hit_batches: list[str] = []
        for batch_id in sorted(scopes):
            scope = set(scopes[batch_id]) or {layer.layer_id for layer in
                                              new_shards[shard_id].layers}
            if scope & closure:
                hit_batches.append(batch_id)
                order += 1
                steps.append(PropagationStep(
                    order, shard_id, "batch", batch_id,
                    f"scope intersects affected layers {sorted(scope & closure)}"))
                order += 1
                steps.append(PropagationStep(
                    order, shard_id, "loss_summary", batch_id,
                    "loss summary depends on invalidated batch parameters"))
        if hit_batches:
            invalid_batches[shard_id] = hit_batches
    invalid_cps: list[str] = []
    for cp in sorted(new.checkpoints, key=lambda c: (c.shard_id, c.generation)):
        shard_batches = invalid_batches.get(cp.shard_id, [])
        for batch_id in shard_batches:
            pv = batch_param_versions.get(cp.shard_id, {}).get(batch_id)
            # checkpoint freezes a param version; match the invalidated batch pv.
            if pv == cp.param_version:
                order += 1
                steps.append(PropagationStep(
                    order, cp.shard_id, "checkpoint", cp.checkpoint_id,
                    f"frozen param_version {pv} depends on invalidated batch {batch_id}"))
                invalid_cps.append(cp.checkpoint_id)
                break
    return InvalidationReport(changed, affected, invalid_batches,
                              sorted(set(invalid_cps)), steps)


def recompute_subset(new: Spec, report: InvalidationReport,
                     batch_inputs: dict[str, dict[str, dict]]
                     ) -> dict[str, dict[str, ComputeResult]]:
    """Recompute only invalidated batches; untouched results stay byte-identical."""
    shards = {shard.shard_id: shard for shard in new.shards}
    rebuilt: dict[str, dict[str, ComputeResult]] = {}
    for shard_id in sorted(report.invalid_batches):
        rebuilt[shard_id] = {}
        shard = shards[shard_id]
        for batch_id in report.invalid_batches[shard_id]:
            entry = batch_inputs[shard_id][batch_id]
            rebuilt[shard_id][batch_id] = compute_batch(
                shard, batch_id, entry["param_version"],
                tuple(entry["sample_ids"]), tuple(entry.get("scope_layers", ())))
    return rebuilt


def verify_against_full_rebuild(new: Spec,
                                subset_results: dict[str, dict[str, ComputeResult]],
                                all_inputs: dict[str, dict[str, dict]]
                                ) -> None:
    """Field-by-field check versus a from-blank full rebuild; E_DIVERGE on diff."""
    shards = {shard.shard_id: shard for shard in new.shards}
    for shard_id in sorted(all_inputs):
        shard = shards[shard_id]
        for batch_id in sorted(all_inputs[shard_id]):
            entry = all_inputs[shard_id][batch_id]
            full = compute_batch(shard, batch_id, entry["param_version"],
                                 tuple(entry["sample_ids"]),
                                 tuple(entry.get("scope_layers", ())))
            subset = subset_results.get(shard_id, {}).get(batch_id)
            if subset is None:
                continue  # untouched batch: baseline preserved, nothing to compare
            if canonical(full.to_dict()) != canonical(subset.to_dict()):
                raise DivergeError(
                    "subset recompute differs from blank full rebuild",
                    f"shard:{shard_id}", batch_id,
                    detail={"full": full.to_dict(), "subset": subset.to_dict()})


def verify_untouched_byte_stable(before: dict[str, ComputeResult],
                                 after: dict[str, ComputeResult],
                                 invalid_batch_ids: set[str]) -> None:
    """Ensure non-invalidated loss records are byte-for-byte unchanged."""
    for batch_id, result in before.items():
        if batch_id in invalid_batch_ids:
            continue
        if canonical(result.to_dict()) != canonical(after[batch_id].to_dict()):
            raise DivergeError("untouched batch summary changed",
                               "invalidation", batch_id)
