"""Deterministic two-route aggregation execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .canonical import canonical_bytes, canonical_json, digest_value
from .errors import EPage, ERoute
from .fixedpoint import (
    FIXED_POINT_VERSION,
    GroupAccumulator,
    MetricAccumulator,
    result_digest,
    result_view,
)
from .models import PageCatalog, PageData
from .page_format import decode_page
from .planner import (
    RESCAN,
    RESIDENT,
    RouteDecision,
    SlotPlan,
    decide_route,
    plan_slots,
)


@dataclass
class PageOutcome:
    access_seq: int
    page_id: int
    outcome: str
    cache_hit: bool
    physical_scan: bool
    slot: int | None
    canonical_hit: bool
    canonical_slot: int
    evicted_page_id: int | None
    eviction_seq: int | None
    slot_occupancy_after: tuple[int | None, ...]
    occupancy_kind: str

    def payload(self) -> dict[str, Any]:
        return {
            "access_seq": self.access_seq,
            "page_id": self.page_id,
            "outcome": self.outcome,
            "cache_hit": self.cache_hit,
            "physical_scan": self.physical_scan,
            "slot": self.slot,
            "canonical_hit": self.canonical_hit,
            "canonical_slot": self.canonical_slot,
            "evicted_page_id": self.evicted_page_id,
            "eviction_seq": self.eviction_seq,
            "slot_occupancy_after": list(self.slot_occupancy_after),
            "occupancy_kind": self.occupancy_kind,
        }


@dataclass
class RescanEvent:
    event_seq: int
    access_seq: int
    page_id: int
    reason: str
    canonical_hit: bool
    canonical_slot: int

    def payload(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "access_seq": self.access_seq,
            "page_id": self.page_id,
            "reason": self.reason,
            "canonical_hit": self.canonical_hit,
            "canonical_slot": self.canonical_slot,
        }


@dataclass
class BatchRecord:
    batch_index: int
    cursor_start: int
    cursor_end: int
    page_outcomes: list[dict[str, Any]]
    eviction_events: list[dict[str, Any]]
    rescan_events: list[dict[str, Any]]
    group_state: list[dict[str, Any]]
    group_state_digest: str
    batch_digest: str

    def digest_payload(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "cursor_start": self.cursor_start,
            "cursor_end": self.cursor_end,
            "page_outcomes": self.page_outcomes,
            "eviction_events": self.eviction_events,
            "rescan_events": self.rescan_events,
            "group_state_digest": self.group_state_digest,
        }


@dataclass
class Execution:
    catalog: PageCatalog
    budget: int
    route: RouteDecision
    plan: SlotPlan
    batches: list[BatchRecord]
    page_outcomes: list[dict[str, Any]]
    eviction_events: list[dict[str, Any]]
    rescan_events: list[dict[str, Any]]
    final_group_state: list[dict[str, Any]]
    result: dict[str, Any]
    result_digest: str
    progress_cursor: int
    cache_hits: int
    physical_scans: int
    final_resident_pages: list[int]
    route_consistency: dict[str, Any] = field(default_factory=dict)

    @property
    def eviction_digest(self) -> str:
        return digest_value(self.eviction_events)

    @property
    def rescan_digest(self) -> str:
        return digest_value(self.rescan_events)

    @property
    def batch_digest(self) -> str:
        return digest_value([batch.batch_digest for batch in self.batches])

    def result_bytes(self) -> bytes:
        return canonical_bytes(self.result)

    def summary_payload(self) -> dict[str, Any]:
        return {
            "route_decision": self.route.payload(),
            "budget": self.budget,
            "scan_access_count": self.catalog.access_count,
            "distinct_page_count": len(self.catalog.pages),
            "progress_cursor": self.progress_cursor,
            "batch_count": len(self.batches),
            "cache_hits": self.cache_hits,
            "physical_scans": self.physical_scans,
            "hit_count": self.cache_hits,
            "eviction_count": len(self.eviction_events),
            "rescan_event_count": len(self.rescan_events),
            "eviction_digest": self.eviction_digest,
            "rescan_digest": self.rescan_digest,
            "batch_digest": self.batch_digest,
            "result_digest": self.result_digest,
            "result_bytes_sha256": __import__("hashlib").sha256(self.result_bytes()).hexdigest(),
            "fixed_point_version": FIXED_POINT_VERSION,
            "final_resident_pages": self.final_resident_pages,
            "route_consistency": self.route_consistency,
        }


def _runtime_page(catalog: PageCatalog, page_id: int) -> PageData:
    page = catalog.pages[page_id]
    decoded, actual_digest = decode_page(
        page.path,
        expected_page_id=page_id,
        expected_rows=catalog.rows_per_page,
        expected_columns=catalog.columns,
    )
    if actual_digest != page.content_sha256:
        raise EPage(
            "page content changed while executing",
            page_id=page_id,
            field="content_sha256",
            details={"expected": page.content_sha256, "actual": actual_digest},
        )
    return decoded


def _group_accumulators(catalog: PageCatalog) -> dict[int, GroupAccumulator]:
    return {}


def _ensure_group(
    groups: dict[int, GroupAccumulator],
    catalog: PageCatalog,
    group_key: int,
) -> GroupAccumulator:
    accumulator = groups.get(group_key)
    if accumulator is None:
        accumulator = GroupAccumulator(group_key=group_key)
        for column in catalog.metric_columns:
            accumulator.metrics[column.name] = MetricAccumulator(
                column=column.name,
                column_type=column.column_type,
            )
        groups[group_key] = accumulator
    return accumulator


def _add_page_to_groups(
    groups: dict[int, GroupAccumulator],
    catalog: PageCatalog,
    page: PageData,
    access_seq: int,
) -> None:
    group_values = page.columns[catalog.group_key]
    for raw_key in np.unique(group_values):
        group_key = int(raw_key)
        indices = np.flatnonzero(group_values == raw_key)
        accumulator = _ensure_group(groups, catalog, group_key)
        metric_values: dict[str, list[int | float | None]] = {}
        for column in catalog.metric_columns:
            values = page.columns[column.name][indices]
            nulls = page.null_maps[column.name][indices]
            converted: list[int | float | None] = []
            for value, is_null in zip(values.tolist(), nulls.tolist(), strict=True):
                if is_null:
                    converted.append(None)
                elif column.column_type == "int":
                    converted.append(int(value))
                else:
                    converted.append(float(value))
            metric_values[column.name] = converted
        accumulator.add_page(
            access_seq=access_seq,
            page_id=page.page_id,
            page_rows=page.rows,
            group_rows=len(indices),
            column_values=metric_values,
        )


def _snapshot_groups(groups: dict[int, GroupAccumulator]) -> list[dict[str, Any]]:
    return [groups[key].payload() for key in sorted(groups)]


def _rescan_reason(plan_access: Any) -> str:
    if plan_access.hit:
        return "RESCAN_ROUTE_NO_RESIDENCY"
    if plan_access.evicted_page_id is not None:
        return "RESCAN_ROUTE_OVER_BUDGET_EVICTION"
    return "RESCAN_ROUTE_INITIAL_NO_SLOT"


def build_execution(
    catalog: PageCatalog,
    budget: int,
    *,
    force_route: str | None = None,
    batch_size: int | None = None,
) -> Execution:
    effective_batch_size = catalog.batch_size if batch_size is None else batch_size
    if not isinstance(effective_batch_size, int) or isinstance(effective_batch_size, bool) or effective_batch_size <= 0:
        raise EPage("batch_size must be a positive integer", field="batch_size")

    route = decide_route(catalog, budget, force_route=force_route)
    plan = plan_slots(catalog.scan_sequence, budget)
    groups = _group_accumulators(catalog)
    cache: list[PageData | None] = [None] * budget
    page_outcomes: list[PageOutcome] = []
    eviction_events = [event.payload() for event in plan.evictions]
    rescan_events: list[RescanEvent] = []
    batches: list[BatchRecord] = []

    for plan_access in plan.accesses:
        page_id = plan_access.page_id
        if route.selected_route == RESIDENT:
            if plan_access.hit:
                page = cache[plan_access.slot]
                if page is None or page.page_id != page_id:
                    raise ERoute(
                        "resident cache hit conflicts with the canonical slot plan",
                        page_id=page_id,
                        event_seq=plan_access.access_seq,
                        field="cache_hit",
                    )
                outcome = PageOutcome(
                    access_seq=plan_access.access_seq,
                    page_id=page_id,
                    outcome="HIT",
                    cache_hit=True,
                    physical_scan=False,
                    slot=plan_access.slot,
                    canonical_hit=True,
                    canonical_slot=plan_access.slot,
                    evicted_page_id=None,
                    eviction_seq=None,
                    slot_occupancy_after=tuple(
                        item.page_id if item is not None else None for item in cache
                    ),
                    occupancy_kind="actual",
                )
            else:
                page = _runtime_page(catalog, page_id)
                if plan_access.evicted_page_id is not None:
                    victim = cache[plan_access.slot]
                    if victim is None or victim.page_id != plan_access.evicted_page_id:
                        raise ERoute(
                            "resident eviction victim conflicts with the canonical slot plan",
                            page_id=plan_access.evicted_page_id,
                            event_seq=plan_access.eviction_seq,
                            field="evicted_page_id",
                            details={
                                "expected_victim": plan_access.evicted_page_id,
                                "actual_victim": None if victim is None else victim.page_id,
                            },
                        )
                cache[plan_access.slot] = page
                outcome = PageOutcome(
                    access_seq=plan_access.access_seq,
                    page_id=page_id,
                    outcome="CACHE_MISS_SCAN",
                    cache_hit=False,
                    physical_scan=True,
                    slot=plan_access.slot,
                    canonical_hit=False,
                    canonical_slot=plan_access.slot,
                    evicted_page_id=plan_access.evicted_page_id,
                    eviction_seq=plan_access.eviction_seq,
                    slot_occupancy_after=tuple(
                        item.page_id if item is not None else None for item in cache
                    ),
                    occupancy_kind="actual",
                )
        elif route.selected_route == RESCAN:
            page = _runtime_page(catalog, page_id)
            reason = _rescan_reason(plan_access)
            rescan_events.append(
                RescanEvent(
                    event_seq=plan_access.access_seq,
                    access_seq=plan_access.access_seq,
                    page_id=page_id,
                    reason=reason,
                    canonical_hit=plan_access.hit,
                    canonical_slot=plan_access.slot,
                )
            )
            outcome = PageOutcome(
                access_seq=plan_access.access_seq,
                page_id=page_id,
                outcome="RESCAN",
                cache_hit=False,
                physical_scan=True,
                slot=None,
                canonical_hit=plan_access.hit,
                canonical_slot=plan_access.slot,
                evicted_page_id=plan_access.evicted_page_id,
                eviction_seq=plan_access.eviction_seq,
                slot_occupancy_after=plan_access.occupancy_after,
                occupancy_kind="shadow_plan",
            )
        else:
            raise ERoute("unknown execution route", field="route", details={"route": route.selected_route})

        _add_page_to_groups(groups, catalog, page, plan_access.access_seq)
        page_outcomes.append(outcome)
        if len(page_outcomes) % effective_batch_size == 0 or len(page_outcomes) == len(plan.accesses):
            batch_index = len(batches) + 1
            cursor_end = len(page_outcomes)
            cursor_start = (batch_index - 1) * effective_batch_size
            outcome_slice = [item.payload() for item in page_outcomes[cursor_start:cursor_end]]
            eviction_slice = [
                event
                for event in eviction_events
                if cursor_start < event["access_seq"] <= cursor_end
            ]
            rescan_slice = [
                event.payload()
                for event in rescan_events
                if cursor_start < event.access_seq <= cursor_end
            ]
            group_state = _snapshot_groups(groups)
            group_state_digest = digest_value(group_state)
            batch_payload = {
                "batch_index": batch_index,
                "cursor_start": cursor_start,
                "cursor_end": cursor_end,
                "page_outcomes": outcome_slice,
                "eviction_events": eviction_slice,
                "rescan_events": rescan_slice,
                "group_state_digest": group_state_digest,
            }
            batches.append(
                BatchRecord(
                    batch_index=batch_index,
                    cursor_start=cursor_start,
                    cursor_end=cursor_end,
                    page_outcomes=outcome_slice,
                    eviction_events=eviction_slice,
                    rescan_events=rescan_slice,
                    group_state=group_state,
                    group_state_digest=group_state_digest,
                    batch_digest=digest_value(batch_payload),
                )
            )

    if route.selected_route == RESIDENT:
        actual_final = tuple(item.page_id if item is not None else None for item in cache)
        if actual_final != plan.final_occupancy:
            first = next(
                (
                    index
                    for index, (actual, expected) in enumerate(zip(actual_final, plan.final_occupancy, strict=True))
                    if actual != expected
                ),
                0,
            )
            raise ERoute(
                "final resident slot occupancy conflicts with the canonical slot plan",
                page_id=next((item for item in actual_final if item is not None), None),
                event_seq=len(plan.accesses),
                field=f"final_slot[{first}]",
                details={"actual": list(actual_final), "expected": list(plan.final_occupancy)},
            )
        final_resident_pages = [item.page_id for item in cache if item is not None]
        consistency = {
            "canonical_plan_matches_actual_cache": True,
            "actual_cache_hits": plan.hit_count,
            "actual_evictions": len(eviction_events),
            "rescan_events_expected": 0,
            "rescan_events_actual": 0,
        }
        cache_hits = plan.hit_count
        physical_scans = plan.miss_count
    else:
        final_resident_pages = []
        expected_rescan_accesses = [access.access_seq for access in plan.accesses]
        actual_rescan_accesses = [event.access_seq for event in rescan_events]
        if expected_rescan_accesses != actual_rescan_accesses:
            mismatch_index = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(actual_rescan_accesses, expected_rescan_accesses, strict=False)
                    )
                    if actual != expected
                ),
                0,
            )
            page_id = rescan_events[mismatch_index].page_id if rescan_events else None
            raise ERoute(
                "rescan route does not cover the scan sequence exactly once per access",
                page_id=page_id,
                event_seq=mismatch_index + 1,
                field="rescan_access_seq",
            )
        eviction_incoming = {event["incoming_page_id"] for event in eviction_events}
        accessed_pages = {access.page_id for access in plan.accesses}
        if not eviction_incoming.issubset(accessed_pages):
            bad = sorted(eviction_incoming - accessed_pages)[0]
            raise ERoute(
                "eviction incoming page is absent from the scan sequence",
                page_id=bad,
                field="eviction_events.incoming_page_id",
            )
        consistency = {
            "shadow_plan_matches_eviction_events": True,
            "rescan_events_expected": len(plan.accesses),
            "rescan_events_actual": len(rescan_events),
            "actual_cache_hits": 0,
            "shadow_canonical_hits": plan.hit_count,
            "resident_set_is_empty": True,
        }
        cache_hits = 0
        physical_scans = len(plan.accesses)

    # Convert the complete event stream once for summaries and ledger writes.
    all_outcomes = [outcome.payload() for outcome in page_outcomes]
    all_rescans = [event.payload() for event in rescan_events]

    final_groups = _snapshot_groups(groups)
    view = result_view(groups, [column.payload() for column in catalog.columns])
    return Execution(
        catalog=catalog,
        budget=budget,
        route=route,
        plan=plan,
        batches=batches,
        page_outcomes=all_outcomes,
        eviction_events=eviction_events,
        rescan_events=all_rescans,
        final_group_state=final_groups,
        result=view,
        result_digest=result_digest(view),
        progress_cursor=len(plan.accesses),
        cache_hits=cache_hits,
        physical_scans=physical_scans,
        final_resident_pages=sorted(final_resident_pages),
        route_consistency=consistency,
    )


def assert_route_equivalence(
    resident: Execution,
    rescan: Execution,
    *,
    first_page_lookup: callable | None = None,
) -> dict[str, Any]:
    """Compare route-independent bytes and eviction bytes deterministically."""
    result_a = resident.result_bytes()
    result_b = rescan.result_bytes()
    if result_a != result_b:
        raise ERoute(
            "resident and rescan routes produced different aggregate result bytes",
            page_id=first_page_lookup() if first_page_lookup else None,
            field="aggregate_result",
            details={
                "resident_result_digest": resident.result_digest,
                "rescan_result_digest": rescan.result_digest,
            },
        )
    evictions_a = canonical_json(resident.eviction_events)
    evictions_b = canonical_json(rescan.eviction_events)
    if evictions_a != evictions_b:
        raise ERoute(
            "resident and rescan routes produced different eviction event bytes",
            page_id=first_page_lookup() if first_page_lookup else None,
            field="eviction_events",
        )
    return {
        "result_bytes_equal": True,
        "eviction_bytes_equal": True,
        "result_digest": resident.result_digest,
        "result_bytes_sha256": __import__("hashlib").sha256(result_a).hexdigest(),
        "eviction_digest": resident.eviction_digest,
        "estimated_group_count": len(resident.result.get("groups", [])),
        "resident_cache_hits": resident.cache_hits,
        "rescan_cache_hits": rescan.cache_hits,
    }
