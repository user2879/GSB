"""Invalidation propagation and minimally recomputed incremental runs."""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .canonical import dumps
from .changes import ChangeSet
from .engine import (
    ExpandedSource,
    Pack,
    PackPlan,
    PipelineState,
    assemble_result,
    compute_pack,
    deterministic_view,
    expand_source,
    make_pack_plans,
    run_full,
)
from .errors import E_DIVERGE, fail
from .graph import DependencyGraph, build_graph
from .scheduler import ScheduleResult, schedule_packs
from .source import SourceData, load_source
from .spec import RuleSpec, Spec


@dataclass(frozen=True)
class Trigger:
    id: str
    kind: str
    source_id: str
    bucket_seeds: tuple[str, ...]
    source_wide: bool
    affected_rows: tuple[int, ...]
    field: str
    detail: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source_id": self.source_id,
            "bucket_seeds": list(self.bucket_seeds),
            "source_wide": self.source_wide,
            "affected_rows": list(self.affected_rows),
            "field": self.field,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class IncrementalOutcome:
    state: PipelineState
    report: Mapping[str, Any]


def _updated_spec(spec: Spec, changes: ChangeSet, baseline_graph: DependencyGraph) -> Spec:
    if changes.delete_rules:
        baseline_graph.assert_deletable(spec, changes.delete_rules)
    updated_rules: list[RuleSpec] = []
    for rule in spec.rules:
        if rule.id in changes.delete_rules:
            continue
        if rule.id in changes.rule_updates:
            updated_rules.append(replace(rule, boundaries=changes.rule_updates[rule.id]))
        else:
            updated_rules.append(rule)
    return spec.replace(rules=tuple(sorted(updated_rules, key=lambda item: item.id)))


def _trace_for_trigger(
    graph: DependencyGraph,
    trigger: Trigger,
    deleted: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    queue: list[tuple[int, str, tuple[str, ...]]] = []
    for seed in trigger.bucket_seeds:
        heapq.heappush(queue, (0, seed, (seed,)))
    visited: dict[str, int] = {}
    while queue:
        depth, node, path = heapq.heappop(queue)
        if node in visited and visited[node] <= depth:
            continue
        visited[node] = depth
        rows.append(
            {
                "trigger": trigger.id,
                "trigger_kind": trigger.kind,
                "bucket_column": node,
                "depth": depth,
                "direct": depth == 0,
                "path": list(path),
                "deleted": node in deleted,
            }
        )
        for child in graph.children.get(node, ()):
            if child not in visited:
                heapq.heappush(queue, (depth + 1, child, path + (child,)))
    return rows


def _plan_signature(plans: Iterable[PackPlan]) -> tuple[tuple[str, int, int, int], ...]:
    return tuple(
        (plan.pack_id, plan.row_start, plan.row_end, plan.width_used)
        for plan in sorted(plans, key=lambda item: item.pack_id)
    )


def _baseline_plan_map(state: PipelineState) -> dict[str, PackPlan]:
    return {
        pack.pack_id: PackPlan(
            pack_id=pack.pack_id,
            source_id=pack.source_id,
            index=pack.index,
            row_start=pack.row_start,
            row_end=pack.row_end,
            width_used=pack.width_used,
        )
        for pack in state.packs
    }


def _packs_covering_rows(packs: Sequence[Pack], rows: Iterable[int]) -> set[str]:
    row_set = set(rows)
    selected: set[str] = set()
    for pack in packs:
        if any(pack.row_start <= row < pack.row_end for row in row_set):
            selected.add(pack.pack_id)
    return selected


def _compare_results(
    incremental: Mapping[str, Any],
    full: Mapping[str, Any],
) -> None:
    inc_view = deterministic_view(incremental)
    full_view = deterministic_view(full)
    if dumps(inc_view) == dumps(full_view):
        return
    inc_packs = {pack["pack_id"]: pack for pack in incremental.get("packs", [])}
    full_packs = {pack["pack_id"]: pack for pack in full.get("packs", [])}
    for pack_id in sorted(set(inc_packs) | set(full_packs)):
        if inc_packs.get(pack_id) != full_packs.get(pack_id):
            inc_pack = inc_packs.get(pack_id, {})
            full_pack = full_packs.get(pack_id, {})
            inc_counts = inc_pack.get("bucket_counts", {})
            full_counts = full_pack.get("bucket_counts", {})
            bucket_ids = sorted(set(inc_counts) | set(full_counts))
            first_bucket = None
            for bucket_id in bucket_ids:
                if inc_counts.get(bucket_id) != full_counts.get(bucket_id) or inc_pack.get(
                    "null_counts", {}
                ).get(bucket_id) != full_pack.get("null_counts", {}).get(bucket_id):
                    first_bucket = bucket_id
                    break
            fail(
                E_DIVERGE,
                f"$.incremental.packs[{pack_id}]",
                "incremental pack result differs from full rebuild",
                first_differing_pack_id=pack_id,
                first_differing_bucket_id=first_bucket,
                incremental=inc_pack,
                full=full_pack,
            )
    inc_sources = incremental.get("sources", {})
    full_sources = full.get("sources", {})
    for source_id in sorted(set(inc_sources) | set(full_sources)):
        inc_source = inc_sources.get(source_id, {})
        full_source = full_sources.get(source_id, {})
        inc_buckets = inc_source.get("bucket_columns", {})
        full_buckets = full_source.get("bucket_columns", {})
        for bucket_id in sorted(set(inc_buckets) | set(full_buckets)):
            if inc_buckets.get(bucket_id) != full_buckets.get(bucket_id):
                first_pack = next(
                    (pack["pack_id"] for pack in incremental.get("packs", []) if pack["source_id"] == source_id),
                    "",
                )
                fail(
                    E_DIVERGE,
                    f"$.incremental.sources[{source_id}].bucket_columns[{bucket_id}]",
                    "incremental bucket expansion differs from full rebuild",
                    first_differing_pack_id=first_pack,
                    first_differing_bucket_id=bucket_id,
                    incremental=inc_buckets.get(bucket_id),
                    full=full_buckets.get(bucket_id),
                )
        if inc_source != full_source:
            first_pack = next(
                (pack["pack_id"] for pack in incremental.get("packs", []) if pack["source_id"] == source_id),
                "",
            )
            fail(
                E_DIVERGE,
                f"$.incremental.sources[{source_id}]",
                "incremental source expansion differs from full rebuild",
                first_differing_pack_id=first_pack,
                first_differing_bucket_id=None,
            )
    fail(
        E_DIVERGE,
        "$.incremental",
        "incremental result differs from full rebuild",
        first_differing_pack_id=next(iter(sorted(inc_packs)), ""),
        first_differing_bucket_id=None,
    )


def run_incremental(
    spec: Spec,
    changes: ChangeSet,
    *,
    baseline: PipelineState | None = None,
    on_pack_complete: Any = None,
) -> IncrementalOutcome:
    """Apply one change set, recompute only invalid buckets/packs, verify vs full."""

    if baseline is None:
        baseline = run_full(spec)
    if baseline.spec.digest != spec.digest:
        fail(
            E_DIVERGE,
            "$.baseline.spec_digest",
            "incremental baseline was built for a different specification",
            expected=spec.digest,
            actual=baseline.spec.digest,
        )
    new_spec = _updated_spec(spec, changes, baseline.graph)
    new_graph = build_graph(new_spec)

    deleted = set(changes.delete_rules)
    triggers: list[Trigger] = []
    source_wide_sources: set[str] = set()
    affected_rows_by_source: dict[str, set[int]] = {}
    for change in changes.values:
        seeds = baseline.graph.columns_triggering(change.source, change.column)
        source = spec.source(change.source)
        assert source is not None
        is_pk = change.column == source.primary_key
        if seeds or is_pk:
            affected_rows_by_source.setdefault(change.source, set()).add(change.row)
        triggers.append(
            Trigger(
                id=f"value:{change.source}:{change.row}:{change.column}",
                kind="value",
                source_id=change.source,
                bucket_seeds=tuple(sorted(seeds)),
                source_wide=False,
                affected_rows=(change.row,),
                field=f"$.changes.values[{change.index}]",
                detail={"value": change.value},
            )
        )
    for change in changes.nulls:
        seeds = baseline.graph.columns_triggering(change.source, change.column)
        source = spec.source(change.source)
        assert source is not None
        is_pk = change.column == source.primary_key
        if seeds or is_pk:
            affected_rows_by_source.setdefault(change.source, set()).add(change.row)
        triggers.append(
            Trigger(
                id=f"null:{change.source}:{change.row}:{change.column}",
                kind="null",
                source_id=change.source,
                bucket_seeds=tuple(sorted(seeds)),
                source_wide=False,
                affected_rows=(change.row,),
                field=f"$.changes.nulls[{change.index}]",
                detail={"is_null": change.is_null},
            )
        )
    for rule_id in sorted(changes.rule_updates):
        rule = spec.rule(rule_id)
        assert rule is not None
        source_wide_sources.add(rule.source)
        triggers.append(
            Trigger(
                id=f"boundaries:{rule_id}",
                kind="boundaries",
                source_id=rule.source,
                bucket_seeds=(rule_id,),
                source_wide=True,
                affected_rows=tuple(range(spec.source(rule.source).rows)),  # type: ignore[union-attr]
                field=f"$.changes.rules[id={rule_id}].boundaries",
                detail={"boundaries": list(changes.rule_updates[rule_id])},
            )
        )
    for rule_id in changes.delete_rules:
        rule = spec.rule(rule_id)
        assert rule is not None
        source_wide_sources.add(rule.source)
        triggers.append(
            Trigger(
                id=f"delete:{rule_id}",
                kind="delete",
                source_id=rule.source,
                bucket_seeds=(rule_id,),
                source_wide=True,
                affected_rows=tuple(range(spec.source(rule.source).rows)),  # type: ignore[union-attr]
                field=f"$.changes.delete_rules[id={rule_id}]",
                detail={"deleted": rule_id},
            )
        )
    triggers.sort(key=lambda item: item.id)

    seed_set: set[str] = set()
    for trigger in triggers:
        seed_set.update(trigger.bucket_seeds)
    invalidated_all = set(baseline.graph.descendants(seed_set))
    invalidated_existing = invalidated_all - deleted
    trace: list[dict[str, Any]] = []
    for trigger in triggers:
        trace.extend(_trace_for_trigger(baseline.graph, trigger, deleted))
    trace.sort(
        key=lambda item: (
            item["trigger"],
            item["depth"],
            item["bucket_column"],
            tuple(item["path"]),
        )
    )

    changed_source_ids = set(changes.changed_source_ids())
    rule_changed_sources = {
        spec.rule(rule_id).source
        for rule_id in set(changes.rule_updates) | set(changes.delete_rules)
        if spec.rule(rule_id) is not None
    }
    source_data: dict[str, SourceData] = dict(baseline.source_data)
    expansions: dict[str, ExpandedSource] = {}
    counters: dict[str, list[str]] = {"recomputed": [], "reused": []}
    value_overrides = changes.value_overrides()
    null_overrides = changes.null_overrides()
    for source in new_spec.sources:
        source_is_untouched = source.id not in changed_source_ids and source.id not in rule_changed_sources
        if source_is_untouched:
            expansions[source.id] = baseline.expansions[source.id]
            counters["reused"].extend(sorted(baseline.expansions[source.id].buckets))
            continue
        data = load_source(
            new_spec,
            source,
            value_overrides=value_overrides.get(source.id, {}),
            null_overrides=null_overrides.get(source.id, {}),
        )
        source_data[source.id] = data
        invalidated_for_source = {
            bucket_id
            for bucket_id in invalidated_existing
            if (new_spec.rule(bucket_id) is not None and new_spec.rule(bucket_id).source == source.id)
        }
        expansions[source.id] = expand_source(
            new_spec,
            new_graph,
            data,
            baseline=baseline.expansions[source.id],
            invalidated=invalidated_for_source,
            counters=counters,
        )

    baseline_plans = _baseline_plan_map(baseline)
    baseline_packs = {pack.pack_id: pack for pack in baseline.packs}
    all_new_plans: list[PackPlan] = []
    affected_plans: list[PackPlan] = []
    reused_pack_ids: list[str] = []
    invalidated_pack_ids: set[str] = set()
    for source in new_spec.sources:
        new_plans = make_pack_plans(expansions[source.id], new_spec.packing)
        all_new_plans.extend(new_plans)
        old_plans = [
            baseline_plans[pack.pack_id]
            for pack in baseline.packs
            if pack.source_id == source.id
        ]
        same_layout = _plan_signature(old_plans) == _plan_signature(new_plans)
        source_wide = source.id in source_wide_sources
        source_invalidated_packs = {
            pack.pack_id for pack in baseline.packs if pack.source_id == source.id
        } if source_wide else set()
        rows = affected_rows_by_source.get(source.id, set())
        if rows:
            source_invalidated_packs.update(
                _packs_covering_rows(
                    [pack for pack in baseline.packs if pack.source_id == source.id], rows
                )
            )
        invalidated_pack_ids.update(source_invalidated_packs)
        if not same_layout:
            affected_plans.extend(new_plans)
            continue
        for plan in new_plans:
            if plan.pack_id in source_invalidated_packs:
                affected_plans.append(plan)
            else:
                reused_pack_ids.append(plan.pack_id)

    affected_plans.sort(key=lambda item: item.pack_id)
    if affected_plans:
        def compute(plan: PackPlan) -> Pack:
            return compute_pack(
                plan,
                expansions[plan.source_id],
                new_spec.accounting.unit_cost_micros,
            )

        schedule, computed = schedule_packs(
            affected_plans,
            new_spec.packing,
            compute,
            on_result=on_pack_complete,
        )
    else:
        computed = {}
        schedule = ScheduleResult((), 0, new_spec.packing.global_width_budget)

    final_by_id: dict[str, Pack] = {
        pack_id: baseline_packs[pack_id] for pack_id in reused_pack_ids
    }
    for plan in all_new_plans:
        if plan.pack_id in computed:
            final_by_id[plan.pack_id] = computed[plan.pack_id]
        elif plan.pack_id not in final_by_id:
            fail(
                E_DIVERGE,
                f"$.packs[{plan.pack_id}]",
                "incremental packing did not produce or reuse a required pack",
                first_differing_pack_id=plan.pack_id,
                first_differing_bucket_id=None,
            )
    packs = tuple(final_by_id[pack_id] for pack_id in sorted(final_by_id))

    preserved_buckets: list[dict[str, Any]] = []
    for source_id in sorted(expansions):
        expanded = expansions[source_id]
        old_expanded = baseline.expansions.get(source_id)
        if old_expanded is None:
            continue
        for bucket_id in sorted(expanded.buckets):
            if bucket_id in invalidated_all:
                continue
            old_bucket = old_expanded.buckets.get(bucket_id)
            if old_bucket is None:
                continue
            same = dumps(old_bucket.to_dict()) == dumps(expanded.buckets[bucket_id].to_dict())
            preserved_buckets.append(
                {
                    "bucket_column": bucket_id,
                    "baseline_version": old_bucket.version,
                    "incremental_version": expanded.buckets[bucket_id].version,
                    "same": same,
                }
            )
            if not same:
                fail(
                    E_DIVERGE,
                    f"$.incremental.sources[{source_id}].bucket_columns[{bucket_id}]",
                    "an unaffected bucket column changed",
                    first_differing_pack_id="",
                    first_differing_bucket_id=bucket_id,
                )
    preserved_packs: list[dict[str, Any]] = []
    for pack_id in sorted(reused_pack_ids):
        old_pack = baseline_packs[pack_id]
        new_pack = final_by_id[pack_id]
        same = dumps(old_pack.to_dict()) == dumps(new_pack.to_dict())
        preserved_packs.append(
            {
                "pack_id": pack_id,
                "baseline_digest": old_pack.digest,
                "incremental_digest": new_pack.digest,
                "same": same,
            }
        )
        if not same:
            fail(
                E_DIVERGE,
                f"$.incremental.packs[{pack_id}]",
                "an unaffected pack changed",
                first_differing_pack_id=pack_id,
                first_differing_bucket_id=None,
            )
    preserved_buckets.sort(key=lambda item: item["bucket_column"])
    preserved_packs.sort(key=lambda item: item["pack_id"])

    recompute = {
        "mode": "incremental",
        "triggers": [trigger.to_dict() for trigger in triggers],
        "recomputed_bucket_columns": sorted(set(counters["recomputed"])),
        "reused_bucket_columns": sorted(set(counters["reused"])),
        "recomputed_packs": sorted(computed),
        "reused_packs": sorted(reused_pack_ids),
        "trace": trace,
    }
    result = assemble_result(
        new_spec,
        new_graph,
        source_data,
        expansions,
        packs,
        schedule,
        recompute,
    )
    full_state = run_full(
        new_spec,
        value_overrides=value_overrides,
        null_overrides=null_overrides,
    )
    _compare_results(result, full_state.result)
    report = {
        "baseline_content_digest": baseline.result["content_digest"],
        "changed_spec_digest": new_spec.digest,
        "invalidation": {
            "triggers": [trigger.to_dict() for trigger in triggers],
            "bucket_columns": sorted(invalidated_all),
            "deleted_bucket_columns": sorted(deleted),
            "packs": sorted(invalidated_pack_ids),
            "trace": trace,
        },
        "equivalence": {
            "equal": True,
            "incremental_content_digest": result["content_digest"],
            "full_content_digest": full_state.result["content_digest"],
            "first_difference": None,
            "fields_compared": [
                "graph",
                "sources",
                "packs",
                "merged",
                "direct",
                "derived_statistics",
                "cost_ledger",
                "policy",
            ],
        },
        "preserved": {
            "bucket_columns": preserved_buckets,
            "packs": preserved_packs,
        },
        "recomputed": {
            "bucket_columns": recompute["recomputed_bucket_columns"],
            "packs": recompute["recomputed_packs"],
            "reused_bucket_columns": recompute["reused_bucket_columns"],
            "reused_packs": recompute["reused_packs"],
        },
    }
    return IncrementalOutcome(
        state=PipelineState(
            spec=new_spec,
            graph=new_graph,
            source_data=source_data,
            expansions=expansions,
            packs=packs,
            schedule=schedule,
            result=result,
        ),
        report=report,
    )
