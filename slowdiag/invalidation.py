from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json, sha256_canonical
from .config import WorkbenchConfig
from .constants import METRICS
from .storage import Ledger, SourceDataset


@dataclass(frozen=True)
class PlanDecision:
    plan_version: int
    previous_plan_version: int | None
    changed_sources: tuple[str, ...]
    added_sources: tuple[str, ...]
    removed_sources: tuple[str, ...]
    policy_changed: bool
    initial_build: bool

    @property
    def version_changed(self) -> bool:
        return self.previous_plan_version != self.plan_version


def source_fingerprints(config: WorkbenchConfig) -> dict[str, dict[str, Any]]:
    return {
        source.id: source.config_fingerprint()
        for source in config.sorted_sources()
    }


def decide_plan(ledger: Ledger, config: WorkbenchConfig) -> PlanDecision:
    previous = ledger.get_plan_state()
    current_fingerprints = source_fingerprints(config)
    policy_sha = config.fingerprint_sha256()
    if previous is None:
        return PlanDecision(
            plan_version=1,
            previous_plan_version=None,
            changed_sources=(),
            added_sources=tuple(sorted(current_fingerprints)),
            removed_sources=(),
            policy_changed=False,
            initial_build=True,
        )

    old_fingerprints = previous["source_fingerprints"]
    old_ids = set(old_fingerprints)
    new_ids = set(current_fingerprints)
    added = tuple(sorted(new_ids - old_ids))
    removed = tuple(sorted(old_ids - new_ids))
    changed = tuple(
        sorted(
            source_id
            for source_id in old_ids & new_ids
            if old_fingerprints[source_id] != current_fingerprints[source_id]
        )
    )
    policy_changed = previous["policy_sha256"] != policy_sha
    needs_bump = bool(added or removed or changed or policy_changed)
    plan_version = previous["plan_version"] + (1 if needs_bump else 0)
    return PlanDecision(
        plan_version=plan_version,
        previous_plan_version=previous["plan_version"],
        changed_sources=changed,
        added_sources=added,
        removed_sources=removed,
        policy_changed=policy_changed,
        initial_build=False,
    )


def commit_plan(ledger: Ledger, decision: PlanDecision, config: WorkbenchConfig) -> None:
    ledger.set_plan_state(
        decision.plan_version,
        source_fingerprints(config),
        config.fingerprint_sha256(),
    )


def _query_map(view: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not view:
        return {}
    return {item["query_id"]: item for item in view.get("queries", [])}


def _evidence_map(query: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for metric in METRICS:
        item = query.get("metrics", {}).get(metric, {})
        adopted = item.get("adopted")
        if adopted:
            result[(metric, adopted["source_id"])] = adopted
        for comparison in item.get("comparisons", []):
            result[(metric, comparison["source_id"])] = comparison
    return result


def _query_referenced_sources(query: dict[str, Any]) -> set[str]:
    return {source_id for _, source_id in _evidence_map(query)}


def _migration_pair_source_ids(pair: dict[str, Any]) -> set[str]:
    return {pair["from_source"], pair["to_source"]}


def _affected_queries(
    config: WorkbenchConfig,
    decision: PlanDecision,
    old_view: dict[str, Any] | None,
    new_merged: list[dict[str, Any]],
    migration_document: dict[str, Any],
) -> tuple[list[str], list[str], set[str]]:
    all_new_queries = sorted(item["query_id"] for item in new_merged)
    if decision.initial_build:
        return [], all_new_queries, set()
    changed_reference_sources = set(decision.changed_sources)
    changed_reference_sources.update(decision.added_sources)
    changed_reference_sources.update(decision.removed_sources)
    if decision.policy_changed:
        old_queries = set(_query_map(old_view))
        new_queries = set(all_new_queries)
        return sorted(old_queries | new_queries), [], set(all_new_queries)
    if not changed_reference_sources:
        return [], all_new_queries, set()

    affected: set[str] = set()
    old_map = _query_map(old_view)
    for query_id, query in old_map.items():
        if _query_referenced_sources(query) & changed_reference_sources:
            affected.add(query_id)
    for query in new_merged:
        if _query_referenced_sources(query) & changed_reference_sources:
            affected.add(query["query_id"])
    for pair in migration_document["pairs"]:
        pair_sources = _migration_pair_source_ids(pair) & changed_reference_sources
        if not pair_sources:
            continue
        for result in pair["results"]:
            for side in ("source_a", "source_b"):
                source_id = result[side]["source_id"]
                if source_id not in pair_sources:
                    continue
                if any(
                    result[side]["metrics"][metric]["evidence_count"] > 0
                    for metric in METRICS
                ):
                    affected.add(result["query_id"])
    unaffected = sorted(set(all_new_queries) - affected)
    return sorted(affected), unaffected, set(all_new_queries)


def _changed_evidence(
    old_view: dict[str, Any] | None,
    new_merged: list[dict[str, Any]],
    affected_queries: set[str],
    changed_sources: set[str],
) -> list[dict[str, Any]]:
    old_map = _query_map(old_view)
    changes: list[dict[str, Any]] = []
    for query in sorted(new_merged, key=lambda item: item["query_id"]):
        query_id = query["query_id"]
        if query_id not in affected_queries:
            continue
        old_evidence = _evidence_map(old_map.get(query_id, {}))
        new_evidence = _evidence_map(query)
        for metric in METRICS:
            keys = sorted(
                {
                    source_id
                    for key_metric, source_id in old_evidence
                    if key_metric == metric
                }
                | {
                    source_id
                    for key_metric, source_id in new_evidence
                    if key_metric == metric
                }
            )
            for source_id in keys:
                before = old_evidence.get((metric, source_id))
                after = new_evidence.get((metric, source_id))
                before_signature = None if before is None else {
                    "value": before.get("value"),
                    "format_version": before.get("format_version"),
                    "sample_period_ms": before.get("sample_period_ms"),
                    "elapsed_semantics": before.get("elapsed_semantics"),
                    "scan_semantics": before.get("scan_semantics"),
                }
                after_signature = None if after is None else {
                    "value": after.get("value"),
                    "format_version": after.get("format_version"),
                    "sample_period_ms": after.get("sample_period_ms"),
                    "elapsed_semantics": after.get("elapsed_semantics"),
                    "scan_semantics": after.get("scan_semantics"),
                }
                if before_signature == after_signature and source_id not in changed_sources:
                    continue
                reason = "source_parameter_changed" if source_id in changed_sources else "evidence_changed"
                changes.append(
                    {
                        "query_id": query_id,
                        "metric": metric,
                        "source_id": source_id,
                        "reason": reason,
                        "before": before_signature,
                        "after": after_signature,
                    }
                )
    changes.sort(key=lambda item: (item["query_id"], item["metric"], item["source_id"]))
    return changes


def _stale_scope(
    new_merged: list[dict[str, Any]],
    migration_document: dict[str, Any],
    affected_queries: set[str],
    changed_pair_sources: set[str],
    policy_changed: bool,
) -> dict[str, Any]:
    migration_scope: list[dict[str, str]] = []
    for pair in migration_document["pairs"]:
        pair_changed_sources = _migration_pair_source_ids(pair) & changed_pair_sources
        for result in pair["results"]:
            source_evidence_changed = False
            for side in ("source_a", "source_b"):
                source_id = result[side]["source_id"]
                if source_id not in pair_changed_sources:
                    continue
                if any(
                    result[side]["metrics"][metric]["evidence_count"] > 0
                    for metric in METRICS
                ):
                    source_evidence_changed = True
            if policy_changed or source_evidence_changed:
                migration_scope.append(
                    {"pair_id": pair["pair_id"], "query_id": result["query_id"]}
                )
    migration_scope.sort(key=lambda item: (item["pair_id"], item["query_id"]))
    return {
        "merged_queries": sorted(affected_queries),
        "migration_results": migration_scope,
        "metrics": list(METRICS),
        "scope_rule": "query -> metric -> evidence references and migration source pairs",
    }


def _attribution_mentions_changed_evidence(
    result: dict[str, Any], changed_sources: set[str]
) -> bool:
    for conflict in result.get("conflicts", []):
        for side in ("evidence_a", "evidence_b"):
            evidence = conflict.get(side, {})
            if (
                evidence.get("source_id") in changed_sources
                and evidence.get("evidence_count", 0) > 0
            ):
                return True
    return False


def derive_invalidation(
    config: WorkbenchConfig,
    decision: PlanDecision,
    old_view: dict[str, Any] | None,
    new_merged: list[dict[str, Any]],
    migration_document: dict[str, Any],
    attribution_document: dict[str, Any],
    datasets: list[SourceDataset],
) -> dict[str, Any]:
    affected_list, unaffected_list, _ = _affected_queries(
        config, decision, old_view, new_merged, migration_document
    )
    affected = set(affected_list)
    changed_pair_sources = set(decision.changed_sources)
    changed_pair_sources.update(decision.added_sources)
    changed_pair_sources.update(decision.removed_sources)
    if decision.policy_changed:
        changed_pair_sources.update(dataset.config.id for dataset in datasets)

    stale_scope = _stale_scope(
        new_merged,
        migration_document,
        affected,
        changed_pair_sources,
        decision.policy_changed,
    )
    stale_scope["attribution_results"] = [
        {"pair_id": result["pair_id"], "query_id": result["query_id"]}
        for result in attribution_document["results"]
        if decision.policy_changed
        or _attribution_mentions_changed_evidence(result, changed_pair_sources)
    ]

    old_map = _query_map(old_view)
    new_map = _query_map({"queries": new_merged})
    byte_checks: list[dict[str, Any]] = []
    all_identical = True
    for query_id in unaffected_list:
        before = old_map.get(query_id)
        after = new_map.get(query_id)
        before_sha = None if before is None else sha256_canonical(before)
        after_sha = None if after is None else sha256_canonical(after)
        identical = True if old_view is None else before_sha == after_sha
        all_identical = all_identical and identical
        byte_checks.append(
            {
                "query_id": query_id,
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "byte_identical": identical,
            }
        )

    changed_evidence = _changed_evidence(
        old_view,
        new_merged,
        affected,
        set(decision.changed_sources) | set(decision.added_sources) | set(decision.removed_sources),
    )
    reasons: list[str] = []
    if decision.initial_build:
        reasons.append("initial_build")
    if decision.policy_changed:
        reasons.append("policy_changed")
    for source_id in decision.changed_sources:
        reasons.append(f"source_parameters_changed:{source_id}")
    for source_id in decision.added_sources:
        reasons.append(f"source_added:{source_id}")
    for source_id in decision.removed_sources:
        reasons.append(f"source_removed:{source_id}")
    if not reasons:
        reasons.append("no_plan_change")
    return {
        "plan_version": decision.plan_version,
        "previous_plan_version": decision.previous_plan_version,
        "version_changed": decision.version_changed,
        "changed_sources": list(decision.changed_sources),
        "added_sources": list(decision.added_sources),
        "removed_sources": list(decision.removed_sources),
        "policy_changed": decision.policy_changed,
        "reasons": reasons,
        "replacement_mode": "whole_view",
        "affected_queries": affected_list,
        "unaffected_queries": unaffected_list,
        "stale_query_count": len(affected_list),
        "stale_scope": stale_scope,
        "changed_evidence": changed_evidence,
        "unaffected_byte_identical": all_identical,
        "unaffected_checks": byte_checks,
    }
