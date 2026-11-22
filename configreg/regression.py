"""逐项差异判定、回归预算与派生链级归因（需求 4、5）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canon import canonical_bytes, canonical_digest
from .collect import Observation, SessionParams
from .diff import (
    COEFF_CHANGED,
    DiffReport,
    ImpactChain,
)
from .fixedpoint import real_values
from .manifest import Manifest

DIRECT_VALUE_THRESHOLD = 0.5
COEFF_AMPLIFICATION_THRESHOLD = 1.5

CAUSE_DIRECT = "direct_value_diff"
CAUSE_AMPLIFIED = "derived_coefficient_amplification"
CAUSE_DISCRETE = "node_execution_dispersion"


@dataclass(frozen=True)
class Budget:
    max_elapsed_delta_ms: int
    max_mismatch_nodes: int


@dataclass(frozen=True)
class NodeVerdict:
    node_id: str
    baseline_present: bool
    candidate_present: bool
    summary_equal: bool
    baseline_elapsed_ms: int | None
    candidate_elapsed_ms: int | None
    elapsed_delta_ms: int | None
    elapsed_breach: bool
    mismatch_fields: tuple[str, ...]


@dataclass(frozen=True)
class Attribution:
    item: str
    trigger: str
    cause: str
    cause_rule: str
    dominant_layer: int
    path: tuple[str, ...]
    layer_contributions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ItemVerdict:
    item: str
    status: str
    impacted: bool
    node_verdicts: tuple[NodeVerdict, ...]
    mismatch_node_count: int
    max_elapsed_delta_ms: int | None
    elapsed_budget_remaining_ms: int | None
    mismatch_budget_remaining: int | None
    indistinguishable: bool
    attribution: Attribution | None


@dataclass(frozen=True)
class RegressionReport:
    baseline_version: int
    candidate_version: int
    budget: Budget
    item_verdicts: dict[str, ItemVerdict]
    regression_items: tuple[str, ...]
    report_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "budget": {
                "max_elapsed_delta_ms": self.budget.max_elapsed_delta_ms,
                "max_mismatch_nodes": self.budget.max_mismatch_nodes,
            },
            "regression_items": list(self.regression_items),
            "item_verdicts": {
                key: _verdict_to_dict(verdict) for key, verdict in sorted(self.item_verdicts.items())
            },
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _verdict_to_dict(verdict: ItemVerdict) -> dict[str, Any]:
    return {
        "item": verdict.item,
        "status": verdict.status,
        "impacted": verdict.impacted,
        "indistinguishable": verdict.indistinguishable,
        "mismatch_node_count": verdict.mismatch_node_count,
        "max_elapsed_delta_ms": verdict.max_elapsed_delta_ms,
        "elapsed_budget_remaining_ms": verdict.elapsed_budget_remaining_ms,
        "mismatch_budget_remaining": verdict.mismatch_budget_remaining,
        "node_verdicts": [
            {
                "node_id": node.node_id,
                "baseline_present": node.baseline_present,
                "candidate_present": node.candidate_present,
                "summary_equal": node.summary_equal,
                "baseline_elapsed_ms": node.baseline_elapsed_ms,
                "candidate_elapsed_ms": node.candidate_elapsed_ms,
                "elapsed_delta_ms": node.elapsed_delta_ms,
                "elapsed_breach": node.elapsed_breach,
                "mismatch_fields": list(node.mismatch_fields),
            }
            for node in verdict.node_verdicts
        ],
        "attribution": None
        if verdict.attribution is None
        else {
            "item": verdict.attribution.item,
            "trigger": verdict.attribution.trigger,
            "cause": verdict.attribution.cause,
            "cause_rule": verdict.attribution.cause_rule,
            "dominant_layer": verdict.attribution.dominant_layer,
            "path": list(verdict.attribution.path),
            "layer_contributions": list(verdict.attribution.layer_contributions),
        },
    }


def _compare_summary(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    fields = sorted(set(left) | set(right))
    mismatches = tuple(
        field_name
        for field_name in fields
        if field_name not in ("elapsed_ms_observed", "extra_sleep_ms")
        and left.get(field_name) != right.get(field_name)
    )
    return not mismatches, mismatches


def _node_verdict(
    node_id: str,
    base_obs: Observation | None,
    cand_obs: Observation | None,
    budget: Budget,
) -> NodeVerdict:
    mismatch_fields: list[str] = []
    summary_equal = False
    elapsed_delta: int | None = None
    elapsed_breach = False
    if base_obs is None:
        mismatch_fields.append("baseline_observation_missing")
    if cand_obs is None:
        mismatch_fields.append("candidate_observation_missing")
    if base_obs is not None and cand_obs is not None:
        summary_equal, summary_mismatch = _compare_summary(base_obs.summary, cand_obs.summary)
        mismatch_fields.extend(summary_mismatch)
        elapsed_delta = cand_obs.elapsed_ms - base_obs.elapsed_ms
        elapsed_breach = elapsed_delta > budget.max_elapsed_delta_ms
        if elapsed_breach:
            mismatch_fields.append("elapsed_ms")
    return NodeVerdict(
        node_id=node_id,
        baseline_present=base_obs is not None,
        candidate_present=cand_obs is not None,
        summary_equal=summary_equal,
        baseline_elapsed_ms=None if base_obs is None else base_obs.elapsed_ms,
        candidate_elapsed_ms=None if cand_obs is None else cand_obs.elapsed_ms,
        elapsed_delta_ms=elapsed_delta,
        elapsed_breach=elapsed_breach,
        mismatch_fields=tuple(mismatch_fields),
    )


def evaluate(
    baseline: Manifest,
    candidate: Manifest,
    diff_report: DiffReport,
    observations: dict[tuple[int, str, str], Observation],
    budget: Budget,
    params: SessionParams,
) -> RegressionReport:
    base_real = real_values(baseline)
    cand_real = real_values(candidate)
    verdicts: dict[str, ItemVerdict] = {}
    all_keys = sorted(set(baseline.items) | set(candidate.items))
    downstream_map: dict[str, set[str]] = {}
    for trigger, chain in diff_report.impact_chains.items():
        downstream_map.setdefault(trigger, set()).add(trigger)
        for entry in chain.downstream:
            downstream_map.setdefault(trigger, set()).add(str(entry["item"]))
    impacted_keys = set().union(*downstream_map.values()) if downstream_map else set()
    for key in all_keys:
        base_targets = set(baseline.items[key].targets) if key in baseline.items else set()
        cand_targets = set(candidate.items[key].targets) if key in candidate.items else set()
        node_ids = sorted(base_targets | cand_targets)
        node_verdicts: list[NodeVerdict] = []
        for node_id in node_ids:
            base_obs = observations.get((baseline.version, key, node_id))
            cand_obs = observations.get((candidate.version, key, node_id))
            node_verdicts.append(_node_verdict(node_id, base_obs, cand_obs, budget))
        mismatch_nodes = sum(
            1 for node in node_verdicts if node.mismatch_fields or not node.baseline_present or not node.candidate_present
        )
        positive_deltas = [node.elapsed_delta_ms for node in node_verdicts if node.elapsed_delta_ms is not None]
        max_delta = max(positive_deltas, default=None)
        elapsed_remaining = None if max_delta is None else budget.max_elapsed_delta_ms - max_delta
        mismatch_remaining = budget.max_mismatch_nodes - mismatch_nodes
        is_regression = any(node.mismatch_fields for node in node_verdicts) or mismatch_remaining < 0
        status = "E_REGRESSION" if is_regression else "ok"
        item_diff = diff_report.item_diffs.get(key)
        indistinguishable = bool(item_diff and item_diff.indistinguishable)
        attribution = None
        if is_regression:
            attribution = attribute(
                key,
                baseline,
                candidate,
                diff_report,
                node_verdicts,
                base_real,
                cand_real,
            )
        verdicts[key] = ItemVerdict(
            item=key,
            status=status,
            impacted=key in impacted_keys and (is_regression or key in diff_report.impact_chains),
            node_verdicts=tuple(node_verdicts),
            mismatch_node_count=mismatch_nodes,
            max_elapsed_delta_ms=max_delta,
            elapsed_budget_remaining_ms=elapsed_remaining,
            mismatch_budget_remaining=mismatch_remaining,
            indistinguishable=indistinguishable,
            attribution=attribution,
        )
    regression_items = tuple(sorted(key for key, verdict in verdicts.items() if verdict.status == "E_REGRESSION"))
    report = RegressionReport(
        baseline_version=baseline.version,
        candidate_version=candidate.version,
        budget=budget,
        item_verdicts=verdicts,
        regression_items=regression_items,
        report_digest="",
    )
    return RegressionReport(
        baseline_version=report.baseline_version,
        candidate_version=report.candidate_version,
        budget=report.budget,
        item_verdicts=report.item_verdicts,
        regression_items=report.regression_items,
        report_digest=canonical_digest(report.to_dict()),
    )


def _find_trigger(item: str, diff_report: DiffReport) -> tuple[str, ImpactChain]:
    if item in diff_report.impact_chains:
        chain = diff_report.impact_chains[item]
        return item, chain
    candidates: list[tuple[int, str, ImpactChain]] = []
    for trigger, chain in diff_report.impact_chains.items():
        for downstream in chain.downstream:
            if downstream["item"] == item:
                candidates.append((int(downstream["depth"]), trigger, chain))
    if not candidates:
        chain = ImpactChain(item, (), ())
        return item, chain
    _, trigger, chain = sorted(candidates, key=lambda row: (row[0], row[1]))[0]
    return trigger, chain


def _path_to_item(trigger: str, item: str, candidate: Manifest, baseline: Manifest) -> list[str]:
    if trigger == item:
        return [trigger]
    graph = candidate.edges() if item in candidate.items else baseline.edges()
    queue: list[tuple[str, list[str]]] = [(trigger, [trigger])]
    seen = {trigger}
    while queue:
        node, trail = queue.pop(0)
        for nxt in sorted(graph.get(node, ())):
            if nxt in seen:
                continue
            next_trail = trail + [nxt]
            if nxt == item:
                return next_trail
            seen.add(nxt)
            queue.append((nxt, next_trail))
    return [trigger, item]


def _edge_coefficient(child: str, parent: str, manifest: Manifest) -> float:
    item = manifest.items.get(child)
    if item is None or not item.derived:
        return 1.0
    return abs(item.coeffs.get(parent, 0.0))


def attribute(
    item: str,
    baseline: Manifest,
    candidate: Manifest,
    diff_report: DiffReport,
    node_verdicts: list[NodeVerdict],
    base_real: dict[str, float],
    cand_real: dict[str, float],
) -> Attribution:
    trigger, _ = _find_trigger(item, diff_report)
    path = _path_to_item(trigger, item, candidate, baseline)
    trigger_diff = diff_report.item_diffs.get(trigger)
    trigger_has_coeff_change = bool(
        trigger_diff and COEFF_CHANGED in trigger_diff.change_types
    )

    def real_or_zero(manifest: Manifest, values: dict[str, float], key: str) -> float:
        return values[key] if key in manifest.items else 0.0

    trigger_delta = abs(
        real_or_zero(candidate, cand_real, trigger) - real_or_zero(baseline, base_real, trigger)
    )
    layer_values: list[float] = [trigger_delta]
    compound_products: list[float] = []
    propagated = trigger_delta
    for index in range(1, len(path)):
        coefficient = _edge_coefficient(path[index], path[index - 1], candidate)
        propagated = propagated * coefficient
        compound_products.append(coefficient)
        layer_values.append(propagated)
    total = sum(layer_values)
    if total <= 0.0:
        contributions = [1.0 if index == 0 else 0.0 for index in range(len(path))]
    else:
        contributions = [value / total for value in layer_values]
    dominant_layer = min(
        range(len(contributions)),
        key=lambda index: (-contributions[index], index),
    )
    layer_contributions = tuple(
        {
            "layer": index,
            "item": path[index],
            "contribution_ratio": round(contributions[index], 9),
            "propagated_abs_delta": layer_values[index],
        }
        for index in range(len(path))
    )

    only_timing = all(
        field_name == "elapsed_ms"
        for node in node_verdicts
        for field_name in node.mismatch_fields
    ) and any(node.mismatch_fields for node in node_verdicts)
    max_compound = max(compound_products, default=1.0)
    if only_timing:
        cause = CAUSE_DISCRETE
        rule = (
            "阈值规则：所有节点摘要逐字段相等，仅整数毫秒耗时增量超预算，"
            "归因为节点执行离散"
        )
    elif trigger_has_coeff_change:
        cause = CAUSE_AMPLIFIED
        rule = (
            "阈值规则：触发项自身的派生系数/常量发生变更，系数变更沿派生链放大，"
            f"路径最大派生系数绝对值 {max_compound:.6g}，归因为派生系数放大"
        )
    elif dominant_layer >= 1 and max_compound >= COEFF_AMPLIFICATION_THRESHOLD:
        cause = CAUSE_AMPLIFIED
        rule = (
            f"阈值规则：主导层级 depth={dominant_layer}>=1 且路径最大派生系数绝对值 "
            f"{max_compound:.6g}>={COEFF_AMPLIFICATION_THRESHOLD}，归因为派生系数放大"
        )
    else:
        cause = CAUSE_DIRECT
        rule = (
            f"阈值规则：触发层贡献占比 {contributions[0]:.6g} "
            f"（直接取值差异判定阈值 {DIRECT_VALUE_THRESHOLD}）且不满足放大规则，"
            "归因为直接取值差异"
        )
    return Attribution(
        item=item,
        trigger=trigger,
        cause=cause,
        cause_rule=rule,
        dominant_layer=dominant_layer,
        path=tuple(path),
        layer_contributions=layer_contributions,
    )
