"""两版清单差异度量、派生链影响范围与定点不可区分判定（需求 2、3）。"""
from __future__ import annotations

from dataclasses import dataclass

from .canon import canonical_digest, canonical_bytes, downstream_of
from .fixedpoint import (
    QuantizedManifest,
    difference_error_bound,
    error_bounds,
    quantize_manifest,
    real_values,
)
from .manifest import Manifest, key_set_delta

VALUE_CHANGED = "value_changed"
TARGET_CHANGED = "target_changed"
COEFF_CHANGED = "coeff_changed"
ADDED = "added"
REMOVED = "removed"
UNCHANGED = "unchanged"


@dataclass(frozen=True)
class ItemDiff:
    key: str
    change_types: tuple[str, ...]
    baseline_value: float | None
    candidate_value: float | None
    baseline_targets: tuple[str, ...]
    candidate_targets: tuple[str, ...]
    baseline_q: int | None
    candidate_q: int | None
    abs_diff: float | None
    error_bound: float | None
    indistinguishable: bool
    indistinguishable_reason: str | None


@dataclass(frozen=True)
class ImpactChain:
    trigger: str
    trigger_changes: tuple[str, ...]
    downstream: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class DiffReport:
    baseline_version: int
    candidate_version: int
    key_delta: dict[str, list[str]]
    item_diffs: dict[str, ItemDiff]
    impact_chains: dict[str, ImpactChain]
    precision: int
    report_digest: str

    def changed_keys(self) -> list[str]:
        return sorted(
            key
            for key, diff in self.item_diffs.items()
            if diff.change_types and UNCHANGED not in diff.change_types
        )

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "precision": self.precision,
            "key_delta": self.key_delta,
            "item_diffs": {
                key: {
                    "key": diff.key,
                    "change_types": list(diff.change_types),
                    "baseline_value": diff.baseline_value,
                    "candidate_value": diff.candidate_value,
                    "baseline_targets": list(diff.baseline_targets),
                    "candidate_targets": list(diff.candidate_targets),
                    "baseline_q": diff.baseline_q,
                    "candidate_q": diff.candidate_q,
                    "abs_diff": diff.abs_diff,
                    "error_bound": _json_safe_float(diff.error_bound),
                    "indistinguishable": diff.indistinguishable,
                    "indistinguishable_reason": diff.indistinguishable_reason,
                }
                for key, diff in sorted(self.item_diffs.items())
            },
            "impact_chains": {
                key: {
                    "trigger": chain.trigger,
                    "trigger_changes": list(chain.trigger_changes),
                    "downstream": list(chain.downstream),
                }
                for key, chain in sorted(self.impact_chains.items())
            },
        }


def _json_safe_float(value: float | None) -> str | float | None:
    if value is None:
        return None
    if value == float("inf"):
        return "inf"
    return value


def _classify_shared(key: str, base: Manifest, cand: Manifest) -> tuple[str, ...]:
    changes: list[str] = []
    base_item = base.items[key]
    cand_item = cand.items[key]
    if base_item.derived != cand_item.derived:
        changes.append(COEFF_CHANGED if cand_item.derived else VALUE_CHANGED)
    if base_item.derived:
        base_coeff = dict(sorted(base_item.coeffs.items()))
        cand_coeff = dict(sorted(cand_item.coeffs.items()))
        if base_coeff != cand_coeff or base_item.const != cand_item.const:
            changes.append(COEFF_CHANGED)
    elif base_item.value != cand_item.value:
        changes.append(VALUE_CHANGED)
    if tuple(base_item.targets) != tuple(cand_item.targets):
        changes.append(TARGET_CHANGED)
    return tuple(changes) if changes else (UNCHANGED,)


def build_diff_report(baseline: Manifest, candidate: Manifest, precision: int) -> DiffReport:
    from .fixedpoint import make_spec

    spec = make_spec(precision)
    base_quant: QuantizedManifest = quantize_manifest(baseline, spec)
    cand_quant = quantize_manifest(candidate, spec)
    base_real = real_values(baseline)
    cand_real = real_values(candidate)
    base_bounds = error_bounds(baseline, spec, base_quant)
    cand_bounds = error_bounds(candidate, spec, cand_quant)
    delta = key_set_delta(baseline, candidate)

    item_diffs: dict[str, ItemDiff] = {}
    trigger_changes: dict[str, tuple[str, ...]] = {}
    for key in delta["added"]:
        item = candidate.items[key]
        item_diffs[key] = ItemDiff(
            key=key,
            change_types=(ADDED,),
            baseline_value=None,
            candidate_value=cand_real[key],
            baseline_targets=(),
            candidate_targets=item.targets,
            baseline_q=None,
            candidate_q=cand_quant.item_q[key],
            abs_diff=None,
            error_bound=None,
            indistinguishable=False,
            indistinguishable_reason=None,
        )
        trigger_changes[key] = (ADDED,)
    for key in delta["removed"]:
        item = baseline.items[key]
        item_diffs[key] = ItemDiff(
            key=key,
            change_types=(REMOVED,),
            baseline_value=base_real[key],
            candidate_value=None,
            baseline_targets=item.targets,
            candidate_targets=(),
            baseline_q=base_quant.item_q[key],
            candidate_q=None,
            abs_diff=None,
            error_bound=None,
            indistinguishable=False,
            indistinguishable_reason=None,
        )
        trigger_changes[key] = (REMOVED,)
    for key in delta["shared"]:
        change_types = _classify_shared(key, baseline, candidate)
        base_item = baseline.items[key]
        cand_item = candidate.items[key]
        abs_diff = abs(cand_real[key] - base_real[key])
        if change_types == (UNCHANGED,):
            bound = None
            indistinguishable = False
            reason = None
        else:
            bound = difference_error_bound(cand_item, baseline, candidate, base_bounds, cand_bounds)
            indistinguishable = bound == float("inf") or abs_diff < bound
            if bound == float("inf"):
                reason = "定点饱和导致误差界无界，差异不可区分"
            else:
                reason = (
                    f"|Δ|={abs_diff:.6g} 小于定点误差界 {bound:.6g}"
                    f"（基线 E={base_bounds[key]:.6g}，候选 E={cand_bounds[key]:.6g}）"
                )
        item_diffs[key] = ItemDiff(
            key=key,
            change_types=change_types,
            baseline_value=base_real[key],
            candidate_value=cand_real[key],
            baseline_targets=base_item.targets,
            candidate_targets=cand_item.targets,
            baseline_q=base_quant.item_q[key],
            candidate_q=cand_quant.item_q[key],
            abs_diff=abs_diff if change_types != (UNCHANGED,) else 0.0,
            error_bound=bound,
            indistinguishable=indistinguishable,
            indistinguishable_reason=reason if indistinguishable else None,
        )
        if change_types != (UNCHANGED,):
            trigger_changes[key] = change_types

    impact_chains = _build_impact_chains(baseline, candidate, trigger_changes)
    report = DiffReport(
        baseline_version=baseline.version,
        candidate_version=candidate.version,
        key_delta=delta,
        item_diffs=item_diffs,
        impact_chains=impact_chains,
        precision=precision,
        report_digest="",
    )
    digest = canonical_digest(report.to_dict())
    return DiffReport(
        baseline_version=report.baseline_version,
        candidate_version=report.candidate_version,
        key_delta=report.key_delta,
        item_diffs=report.item_diffs,
        impact_chains=report.impact_chains,
        precision=precision,
        report_digest=digest,
    )


def _build_impact_chains(
    baseline: Manifest,
    candidate: Manifest,
    trigger_changes: dict[str, tuple[str, ...]],
) -> dict[str, ImpactChain]:
    cand_edges = candidate.edges()
    base_edges = baseline.edges()
    chains: dict[str, ImpactChain] = {}
    for trigger in sorted(trigger_changes):
        if trigger in candidate.items:
            depths = downstream_of(trigger, cand_edges)
            graph_version = "candidate"
        else:
            depths = downstream_of(trigger, base_edges)
            graph_version = "baseline"
        downstream = tuple(
            sorted(
                (
                    {"item": key, "depth": depth, "graph": graph_version}
                    for key, depth in depths.items()
                ),
                key=lambda entry: (entry["depth"], entry["item"]),
            )
        )
        chains[trigger] = ImpactChain(trigger, trigger_changes[trigger], downstream)
    return chains
