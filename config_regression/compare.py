"""Baseline vs candidate result comparison and regression classification.

Findings are emitted per (node, config item) in sorted order.  A node that
never returns a receipt is never treated as "no difference": it gets the
``no_receipt`` category and participates in the overall regression verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .errors import E_EXHAUST, E_REGRESSION

OK = "ok"
NO_RECEIPT = "no_receipt"
REGRESSION = E_REGRESSION


@dataclass
class Finding:
    node_id: str
    key: str
    status: str
    categories: list[str] = field(default_factory=list)
    baseline_summary: dict | None = None
    candidate_summary: dict | None = None
    summary_deltas: dict | None = None
    first_diff_group: str | None = None
    tolerance: dict | None = None
    baseline_elapsed_ms: int | None = None
    candidate_elapsed_ms: int | None = None
    elapsed_delta_ms: int | None = None
    baseline_received: bool = False
    candidate_received: bool = False
    detail: str | None = None

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "key": self.key,
            "status": self.status,
            "categories": self.categories,
            "baseline_received": self.baseline_received,
            "candidate_received": self.candidate_received,
            "baseline_summary": self.baseline_summary,
            "candidate_summary": self.candidate_summary,
            "summary_deltas": self.summary_deltas,
            "first_diff_group": self.first_diff_group,
            "tolerance": self.tolerance,
            "baseline_elapsed_ms": self.baseline_elapsed_ms,
            "candidate_elapsed_ms": self.candidate_elapsed_ms,
            "elapsed_delta_ms": self.elapsed_delta_ms,
            "detail": self.detail,
        }


def _index_results(job_results: list) -> dict[tuple[str, int], dict]:
    indexed: dict[tuple[str, int], dict] = {}
    for result in job_results:
        if result.status == "ok" and result.response is not None:
            indexed[(result.node_id, result.manifest_version)] = result.response
    return indexed


def _exhaust_nodes(job_results: list) -> dict[str, str]:
    out: dict[str, str] = {}
    for result in job_results:
        if result.status == E_EXHAUST:
            out[(result.node_id, result.manifest_version)] = result.error or ""
    return out


def compare_results(baseline, candidate, session,
                    base_jobs: list, cand_jobs: list) -> list[Finding]:
    base_responses = _index_results(base_jobs)
    cand_responses = _index_results(cand_jobs)
    base_exhaust = _exhaust_nodes(base_jobs)
    cand_exhaust = _exhaust_nodes(cand_jobs)

    pairs: set[tuple[str, str]] = set()
    for key, item in baseline.items.items():
        for node_id in item.targets:
            pairs.add((node_id, key))
    for key, item in candidate.items.items():
        for node_id in item.targets:
            pairs.add((node_id, key))

    findings: list[Finding] = []
    for node_id, key in sorted(pairs):
        in_base = key in baseline.items and node_id in baseline.items[key].targets
        in_cand = key in candidate.items and node_id in candidate.items[key].targets
        base_resp = base_responses.get((node_id, baseline.version))
        cand_resp = cand_responses.get((node_id, candidate.version))
        base_item = _extract_item(base_resp, key) if base_resp else None
        cand_item = _extract_item(cand_resp, key) if cand_resp else None
        finding = Finding(node_id=node_id, key=key, status=OK)

        exhaust_note = _exhaust_note(node_id, baseline.version,
                                     candidate.version, base_exhaust,
                                     cand_exhaust, in_base, in_cand)
        if (in_base and base_item is None) or (in_cand and cand_item is None):
            finding.status = NO_RECEIPT
            finding.categories = ["missing_receipt"]
            finding.baseline_received = in_base and base_item is not None
            finding.candidate_received = in_cand and cand_item is not None
            finding.detail = exhaust_note or "node did not return a receipt"
            findings.append(finding)
            continue

        finding.baseline_received = in_base
        finding.candidate_received = in_cand
        if in_base != in_cand:
            finding.status = REGRESSION
            finding.categories.append("distribution_target_changed")
            finding.detail = ("item removed from node" if in_base
                              else "item added to node")

        if in_base and in_cand:
            finding.baseline_summary = base_item["summary"]
            finding.candidate_summary = cand_item["summary"]
            finding.baseline_elapsed_ms = int(base_item["elapsed_ms"])
            finding.candidate_elapsed_ms = int(cand_item["elapsed_ms"])
            finding.elapsed_delta_ms = abs(
                finding.candidate_elapsed_ms - finding.baseline_elapsed_ms)
            finding.tolerance = {
                "value_abs": session.value_tolerance,
                "elapsed_ms": session.elapsed_tolerance_ms,
            }
            first_group, deltas, value_regress = _compare_summaries(
                base_item["summary"], cand_item["summary"],
                session.value_tolerance)
            finding.first_diff_group = first_group
            finding.summary_deltas = deltas
            elapsed_regress = (finding.elapsed_delta_ms
                               > session.elapsed_tolerance_ms)
            if value_regress:
                finding.status = REGRESSION
                finding.categories.append("summary_out_of_tolerance")
            if elapsed_regress:
                finding.status = REGRESSION
                finding.categories.append("elapsed_out_of_tolerance")
        findings.append(finding)
    return findings


def _exhaust_note(node_id: str, base_version: int, cand_version: int,
                  base_exhaust: dict, cand_exhaust: dict,
                  in_base: bool, in_cand: bool) -> str | None:
    notes = []
    if in_base and (node_id, base_version) in base_exhaust:
        notes.append(f"baseline: {base_exhaust[(node_id, base_version)]}")
    if in_cand and (node_id, cand_version) in cand_exhaust:
        notes.append(f"candidate: {cand_exhaust[(node_id, cand_version)]}")
    return "; ".join(notes) if notes else None


def _extract_item(response: dict, key: str) -> dict | None:
    for item in response.get("results", []):
        if item["key"] == key:
            return item
    return None


def _compare_summaries(base: dict, cand: dict,
                       tolerance: float) -> tuple[str | None, dict, bool]:
    first_group: str | None = None
    deltas: dict[str, dict[str, float]] = {}
    out_of_tolerance = False
    for group in sorted(set(base) | set(cand)):
        b_record = base.get(group, {})
        c_record = cand.get(group, {})
        group_deltas: dict[str, float] = {}
        for metric in sorted(set(b_record) | set(c_record)):
            b_val = b_record.get(metric)
            c_val = c_record.get(metric)
            if b_val is None or c_val is None:
                delta = math.inf
            else:
                delta = abs(c_val - b_val)
            group_deltas[metric] = None if delta == math.inf else delta
            if delta > tolerance:
                out_of_tolerance = True
                if first_group is None:
                    first_group = group
        deltas[group] = group_deltas
    return first_group, deltas, out_of_tolerance
