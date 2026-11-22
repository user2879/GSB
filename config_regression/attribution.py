"""Derivation-chain level attribution for E_REGRESSION findings.

Attribution is computed entirely from sorted inputs and integer permille
arithmetic, so the same logical input reproduces byte-identical output no
matter in which order node receipts arrived.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .differ import (
    ADDED,
    COEFFICIENT_CHANGED,
    REMOVED,
    TARGETS_CHANGED,
    VALUE_CHANGED,
    DiffResult,
)
from .errorbound import QuantizedVersion
from .compare import Finding

DIRECT_VALUE = "direct_value_change"
COEFFICIENT_AMPLIFICATION = "derived_coefficient_amplification"
NODE_DISPERSION = "node_execution_dispersion"
MIXED = "mixed"


@dataclass
class Attribution:
    node_id: str
    key: str
    root_key: str
    path: list[str]
    layer_contributions_permille: list[dict[str, int]]
    dominant_layer: str
    cause: str
    cause_rule: str
    shares_total_permille: int

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "key": self.key,
            "root_key": self.root_key,
            "path": self.path,
            "layer_contributions_permille":
                self.layer_contributions_permille,
            "dominant_layer": self.dominant_layer,
            "cause": self.cause,
            "cause_rule": self.cause_rule,
            "shares_total_permille": self.shares_total_permille,
        }


def _chain_path(diff: DiffResult, root: str, target: str) -> list[str]:
    for chain in diff.impact_chains:
        if chain.root_key == root and target in chain.chain:
            return list(chain.chain[:chain.chain.index(target) + 1])
    return [root, target]


def _root_for(diff: DiffResult, key: str) -> str | None:
    """Pick the deepest changed ancestor root reaching *key*.

    Added/removed items are changes in their own right and always attribute
    to themselves first.  Otherwise, among all changed roots whose impact
    chain contains *key*, choose the one whose chain is longest (deepest
    ancestor), with lexicographic tie-breaking for determinism.
    """

    own = diff.item_diffs.get(key)
    if own is not None and any(change in own.change_types
                               for change in (ADDED, REMOVED, VALUE_CHANGED,
                                              COEFFICIENT_CHANGED)):
        return key
    candidates = [(len(chain.chain), chain.root_key)
                  for chain in diff.impact_chains if key in chain.chain]
    if not candidates:
        return None
    candidates.sort(key=lambda pair: (-pair[0], pair[1]))
    return candidates[0][1]


def _integer_contributions(diff: DiffResult, q_base: QuantizedVersion,
                           q_cand: QuantizedVersion,
                           path: list[str]) -> list[tuple[str, int]]:
    contributions: list[tuple[str, int]] = []
    for layer_key in path:
        item_diff = diff.item_diffs.get(layer_key)
        if item_diff is not None and VALUE_CHANGED in item_diff.change_types:
            delta = abs(q_cand.values.get(layer_key, 0)
                        - q_base.values.get(layer_key, 0))
            contributions.append((layer_key, max(1, delta)))
            continue
        if item_diff is not None and COEFFICIENT_CHANGED in item_diff.change_types:
            raw = 0
            for parent_key in sorted({
                    edge[0] for edge in q_cand.edges
                    if edge[1] == layer_key} | {
                    edge[0] for edge in q_base.edges
                    if edge[1] == layer_key}):
                b_coeff = q_base.edges.get((parent_key, layer_key), 0)
                c_coeff = q_cand.edges.get((parent_key, layer_key), 0)
                raw += abs(c_coeff - b_coeff)
            contributions.append((layer_key, max(1, raw)))
            continue
        # Propagation-only layer: fixed minimum unit so the layer still has
        # an explicit, stable weight in the share breakdown.
        contributions.append((layer_key, 1))
    return contributions


def _normalize_permille(contribs: list[tuple[str, int]]) -> list[dict[str, int]]:
    total = sum(weight for _, weight in contribs) or 1
    rows = []
    allocated = 0
    for index, (layer_key, weight) in enumerate(contribs):
        if index == len(contribs) - 1:
            share = 1000 - allocated
        else:
            share = (weight * 1000) // total
            allocated += share
        rows.append({"layer": layer_key, "contribution_weight": weight,
                     "share_permille": share})
    return rows


def attribute_findings(findings: list[Finding], diff: DiffResult,
                       q_base: QuantizedVersion,
                       q_cand: QuantizedVersion,
                       session) -> list[Attribution]:
    attributions: list[Attribution] = []
    for finding in sorted(findings, key=lambda f: (f.node_id, f.key)):
        if finding.status != "E_REGRESSION":
            continue
        root = _root_for(diff, finding.key)
        if root is None:
            # Target-only change has no numeric root; attribute to the item.
            root = finding.key
        path = _chain_path(diff, root, finding.key)
        contribs = _integer_contributions(diff, q_base, q_cand, path)
        rows = _normalize_permille(contribs)
        dominant = max(
            enumerate(rows),
            key=lambda pair: (pair[1]["share_permille"], -pair[0]))[1]
        cause, rule = _classify(finding, diff, root, rows, session,
                                dominant)
        attributions.append(Attribution(
            node_id=finding.node_id, key=finding.key, root_key=root,
            path=path, layer_contributions_permille=rows,
            dominant_layer=dominant["layer"], cause=cause, cause_rule=rule,
            shares_total_permille=sum(r["share_permille"] for r in rows)))
    return attributions


def _classify(finding: Finding, diff: DiffResult, root: str,
              rows: list[dict], session, dominant: dict) -> tuple[str, str]:
    only_elapsed = (finding.categories == ["elapsed_out_of_tolerance"])
    root_change = diff.item_diffs.get(root)
    change_types = root_change.change_types if root_change else [
        TARGETS_CHANGED]
    dominant_share = dominant["share_permille"]
    if only_elapsed and dominant_share >= session.dispersion_threshold_permille:
        return (NODE_DISPERSION,
                f"elapsed-only regression and dominant layer share "
                f"{dominant_share}/1000 >= dispersion threshold "
                f"{session.dispersion_threshold_permille}/1000")
    if COEFFICIENT_CHANGED in change_types and dominant_share >= \
            session.coefficient_threshold_permille:
        return (COEFFICIENT_AMPLIFICATION,
                f"root {root} changes a derivation coefficient and dominant "
                f"layer share {dominant_share}/1000 >= coefficient threshold "
                f"{session.coefficient_threshold_permille}/1000")
    if VALUE_CHANGED in change_types and dominant_share >= \
            session.direct_threshold_permille:
        return (DIRECT_VALUE,
                f"root {root} changes its value directly and dominant layer "
                f"share {dominant_share}/1000 >= direct threshold "
                f"{session.direct_threshold_permille}/1000")
    return (MIXED,
            f"no single cause reached its permille threshold "
            f"(direct={session.direct_threshold_permille}, coefficient="
            f"{session.coefficient_threshold_permille}, dispersion="
            f"{session.dispersion_threshold_permille})")
