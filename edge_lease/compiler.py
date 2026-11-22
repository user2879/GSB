"""Compile versioned manifests into byte-stable dispatch plans.

The compiler is purely deterministic: given the same manifest version, content
seed and scale it always emits identical compilation artifacts (verified by a
canonical per-path ``artifact_digest``).  Incremental evolution recompiles only
the paths whose shard layout, target set or derived coefficients changed, and
propagates invalidation through the dependency graph in a stable topological
order with an explicit reason for every step.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .content import (
    canonical_bytes,
    content_hash,
    fnv1a_32,
    shard_hash,
    shard_hit_digest,
    shard_intervals,
)
from .fixedpoint import (
    derived_digest,
    derived_error_bound,
    leaf_digest,
    leaf_error_bound,
    quant_version,
    quantize_scalar,
)
from .manifest import topological_order
from .models import (
    CompiledPath,
    PathEntry,
    PlanItem,
    VersionManifest,
)


def artifact_digest(payload: Dict[str, Any]) -> str:
    """Stable 64-bit text digest of a canonical compilation payload."""
    return f"art{fnv1a_32(canonical_bytes(payload)):08x}"


class Compiler:
    def __init__(self, content_seed: int, scale: int) -> None:
        self.content_seed = int(content_seed)
        self.scale = int(scale)
        self.quant = quant_version(scale)

    def compile_version(
        self, manifest: VersionManifest
    ) -> Tuple[Dict[str, CompiledPath], List[PlanItem]]:
        entry_map = manifest.entry_map()
        order = topological_order(entry_map)
        compiled: Dict[str, CompiledPath] = {}
        for name in order:
            compiled[name] = self._compile_path(manifest.version, entry_map[name], compiled)
        items = self._build_items(manifest, compiled)
        return compiled, items

    def _compile_path(
        self,
        version: int,
        entry: PathEntry,
        prior: Dict[str, CompiledPath],
    ) -> CompiledPath:
        weight_q, weight_single = quantize_scalar(entry.weight, self.scale)
        chash = content_hash(self.content_seed, entry.path, entry.length)
        coeff_rows: List[Tuple[str, int, int]] = []
        derived_from: List[str] = []
        if entry.derived is None:
            digest, saturated = leaf_digest(chash, weight_q)
            error = leaf_error_bound(weight_single)
        else:
            terms: List[Tuple[int, int]] = []
            error_terms: List[Tuple[int, int, int, int]] = []
            for source, coefficient in entry.derived.sources:
                coeff_q, coeff_single = quantize_scalar(coefficient, self.scale)
                upstream = prior[source]
                coeff_rows.append((source, coeff_q, coeff_single))
                derived_from.append(source)
                terms.append((coeff_q, upstream.digest))
                error_terms.append(
                    (coeff_q, coeff_single, upstream.digest, upstream.error["bound"])
                )
            digest, saturated, _raw = derived_digest(terms, self.scale)
            error = derived_error_bound(error_terms, self.scale)
        payload = {
            "path": entry.path,
            "strategy": entry.strategy,
            "priority": entry.priority,
            "targets": list(entry.targets),
            "shards": [dict(s) for s in entry.shards],
            "weight_q": weight_q,
            "weight_single": weight_single,
            "coeff_q": [list(row) for row in coeff_rows],
            "content_hash": chash,
            "digest": digest,
            "error": error,
            "saturated": saturated,
        }
        return CompiledPath(
            path=entry.path,
            version=version,
            strategy=entry.strategy,
            priority=entry.priority,
            targets=entry.targets,
            shards=entry.shards,
            weight_q=weight_q,
            weight_single=weight_single,
            coeff_q=tuple(coeff_rows),
            derived_from=tuple(derived_from),
            content_hash=chash,
            digest=digest,
            error=dict(error),
            saturated=saturated,
            artifact_digest=artifact_digest(payload),
        )

    def _build_items(
        self, manifest: VersionManifest, compiled: Dict[str, CompiledPath]
    ) -> List[PlanItem]:
        order = topological_order(manifest.entry_map())
        items: List[PlanItem] = []
        for name in order:
            entry = manifest.entry_map()[name]
            cp = compiled[name]
            intervals = shard_intervals(entry.shards, entry.length)
            sh_hashes = tuple(
                shard_hash(self.content_seed, name, start, end, entry.length)
                for start, end in intervals
            )
            sh_digest = shard_hit_digest(sh_hashes)
            for node_id in entry.targets:
                items.append(
                    PlanItem(
                        item_id=f"{manifest.version}:{name}:{node_id}",
                        version=manifest.version,
                        path=name,
                        node_id=node_id,
                        strategy=entry.strategy,
                        priority=entry.priority,
                        expected_digest=cp.digest,
                        shard_hashes=sh_hashes,
                        shard_hit_digest=sh_digest,
                        expected_error=dict(cp.error),
                        quant_version=self.quant,
                        compiled_artifact_digest=cp.artifact_digest,
                    )
                )
        items.sort(key=lambda item: (item.priority, item.path, item.node_id))
        return items

    # ------------------------------------------------------------------
    # Incremental evolution / invalidation propagation
    # ------------------------------------------------------------------
    def diff_paths(
        self, old: VersionManifest, new: VersionManifest
    ) -> Dict[str, str]:
        """Direct per-path change reason; missing paths count as changed."""
        old_map, new_map = old.entry_map(), new.entry_map()
        reasons: Dict[str, str] = {}
        for name in sorted(set(old_map) | set(new_map)):
            if name not in old_map:
                reasons[name] = "path_added"
                continue
            if name not in new_map:
                reasons[name] = "path_removed"
                continue
            before, after = old_map[name], new_map[name]
            if [dict(s) for s in before.shards] != [dict(s) for s in after.shards]:
                reasons[name] = "shards_changed"
            elif list(before.targets) != list(after.targets):
                reasons[name] = "targets_changed"
            elif self._coeff_signature(before) != self._coeff_signature(after):
                reasons[name] = "derived_coeff_changed"
            elif before.weight != after.weight or before.strategy != after.strategy:
                reasons[name] = "weight_or_strategy_changed"
        return reasons

    @staticmethod
    def _coeff_signature(entry: PathEntry) -> List[List[Any]]:
        if entry.derived is None:
            return []
        return [[p, str(c)] for p, c in entry.derived.sources]

    def propagate_invalidation(
        self,
        old: VersionManifest,
        new: VersionManifest,
        direct: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], List[Dict[str, str]]]:
        """Return ``(invalidated_in_topo_order, propagation_steps)``.

        Propagation follows derived edges (a changed source invalidates its
        dependents) and node references (target set changes also invalidate the
        path's own plan items).  Every step records its parent and reason.
        """
        direct = self.diff_paths(old, new) if direct is None else direct
        new_map = new.entry_map()
        reasons = dict(direct)
        steps: List[Dict[str, str]] = [
            {"path": name, "parent": "<root>", "reason": reason}
            for name, reason in sorted(direct.items())
        ]
        dependents: Dict[str, List[str]] = {name: [] for name in new_map}
        for name, entry in new_map.items():
            if entry.derived is not None:
                for source, _coef in entry.derived.sources:
                    if source in dependents:
                        dependents[source].append(name)
        queue = sorted(name for name in direct if name in new_map)
        seen: set = set()
        while queue:
            parent = queue.pop(0)
            for child in sorted(dependents.get(parent, [])):
                if child not in reasons:
                    reasons[child] = "downstream_invalidated"
                # Even when the child has its own direct change reason, the
                # propagation edge is still part of the auditable evidence.
                steps.append(
                    {
                        "path": child,
                        "parent": parent,
                        "reason": "downstream_invalidated",
                    }
                )
                if child not in seen and child not in queue:
                    seen.add(child)
                    queue.append(child)
                    queue.sort()
        invalidated = [
            name for name in topological_order(new_map) if name in reasons
        ]
        return invalidated, steps

    def incremental_compile(
        self,
        old_manifest: VersionManifest,
        new_manifest: VersionManifest,
        old_compiled: Dict[str, CompiledPath],
    ) -> Tuple[Dict[str, CompiledPath], List[PlanItem], List[str], List[Dict[str, str]]]:
        """Recompile only invalidated paths; keep unchanged artifacts byte-stable."""
        invalidated, steps = self.propagate_invalidation(old_manifest, new_manifest)
        new_compiled, _all_items = self.compile_version(new_manifest)
        old_map = old_manifest.entry_map()
        new_map = new_manifest.entry_map()
        stable_check: List[str] = []
        merged: Dict[str, CompiledPath] = {}
        for name in topological_order(new_map):
            if (
                name in old_compiled
                and name in old_map
                and name not in invalidated
                and old_compiled[name].artifact_digest == new_compiled[name].artifact_digest
            ):
                merged[name] = old_compiled[name]
                stable_check.append(name)
            else:
                merged[name] = new_compiled[name]
        items = self._build_items(new_manifest, merged)
        return merged, items, stable_check, steps
