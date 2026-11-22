"""Deterministic slot planning, LRU replacement, and route selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .errors import EBudget
from .models import PageCatalog

RESIDENT = "RESIDENT"
RESCAN = "RESCAN"
VALID_ROUTES = {RESIDENT, RESCAN}


@dataclass(frozen=True)
class RouteDecision:
    selected_route: str
    reason: str
    pressure_ppm: int
    overflow_pages: int
    overflow_ratio_ppm: int
    threshold_ppm: int
    boundary_margin_ppm: int
    boundary_low_ppm: int
    boundary_high_ppm: int
    boundary_affected: bool
    tie_rule: str
    forced: bool
    page_count: int
    access_count: int
    rows_per_page: int
    rows_reference: int
    budget: int

    def payload(self) -> dict[str, Any]:
        return {
            "selected_route": self.selected_route,
            "reason": self.reason,
            "key_scale_parameters": {
                "page_count": self.page_count,
                "access_count": self.access_count,
                "rows_per_page": self.rows_per_page,
                "rows_reference": self.rows_reference,
                "slot_budget": self.budget,
                "page_budget_ratio_ppm": (self.page_count * 1_000_000) // self.budget,
                "overflow_pages": self.overflow_pages,
                "overflow_ratio_ppm": self.overflow_ratio_ppm,
            },
            "decision": {
                "formula": "overflow_ratio_ppm * rows_per_page // rows_reference",
                "pressure_ppm": self.pressure_ppm,
                "threshold_ppm": self.threshold_ppm,
                "delta_ppm": self.pressure_ppm - self.threshold_ppm,
                "comparison": "RESCAN if pressure_ppm > threshold_ppm, else RESIDENT",
                "tie_rule": self.tie_rule,
            },
            "boundary": {
                "margin_ppm": self.boundary_margin_ppm,
                "low_ppm": self.boundary_low_ppm,
                "high_ppm": self.boundary_high_ppm,
                "affected": self.boundary_affected,
                "untriggered_boundary_value_ppm": self.threshold_ppm,
            },
            "forced": self.forced,
        }


@dataclass(frozen=True)
class EvictionEvent:
    event_seq: int
    access_seq: int
    slot: int
    victim_page_id: int
    incoming_page_id: int
    reason: str = "LRU_CAPACITY"

    def payload(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "access_seq": self.access_seq,
            "slot": self.slot,
            "victim_page_id": self.victim_page_id,
            "incoming_page_id": self.incoming_page_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PlanAccess:
    access_seq: int
    page_id: int
    hit: bool
    slot: int
    evicted_page_id: int | None
    eviction_seq: int | None
    occupancy_after: tuple[int | None, ...]

    def outcome_payload(self) -> dict[str, Any]:
        return {
            "access_seq": self.access_seq,
            "page_id": self.page_id,
            "canonical_hit": self.hit,
            "canonical_slot": self.slot,
            "evicted_page_id": self.evicted_page_id,
            "eviction_seq": self.eviction_seq,
            "slot_occupancy_after": list(self.occupancy_after),
        }


@dataclass(frozen=True)
class SlotPlan:
    budget: int
    accesses: tuple[PlanAccess, ...]
    evictions: tuple[EvictionEvent, ...]
    hit_count: int
    miss_count: int
    final_occupancy: tuple[int | None, ...]

    def eviction_payloads(self) -> list[dict[str, Any]]:
        return [event.payload() for event in self.evictions]


def decide_route(
    catalog: PageCatalog,
    budget: int,
    *,
    force_route: str | None = None,
    min_slots: int = 1,
) -> RouteDecision:
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < min_slots:
        raise EBudget(budget, min_slots)
    if force_route is not None:
        normalized = force_route.upper()
        if normalized not in VALID_ROUTES:
            raise ValueError(f"force_route must be one of {sorted(VALID_ROUTES)}")
    else:
        normalized = None

    page_count = len(catalog.pages)
    overflow_pages = max(0, page_count - budget)
    overflow_ratio_ppm = (overflow_pages * 1_000_000) // page_count
    pressure_ppm = (overflow_ratio_ppm * catalog.rows_per_page) // catalog.rows_reference
    threshold = catalog.overflow_ratio_threshold_ppm
    margin = catalog.boundary_margin_ppm
    delta = pressure_ppm - threshold
    boundary_affected = abs(delta) <= margin

    auto_route = RESCAN if pressure_ppm > threshold else RESIDENT
    if pressure_ppm == threshold:
        auto_reason = (
            "pressure_ppm equals threshold_ppm; stable tie rule selects RESIDENT"
        )
    elif pressure_ppm > threshold:
        auto_reason = "pressure_ppm exceeds threshold_ppm; RESCAN avoids slot replacement"
    else:
        auto_reason = "pressure_ppm is at or below threshold_ppm; RESIDENT keeps pages resident"

    if normalized is not None:
        selected = normalized
        forced = True
        reason = (
            f"route forced to {selected}; auto route would be {auto_route} "
            f"({auto_reason})"
        )
    else:
        selected = auto_route
        forced = False
        reason = auto_reason

    if boundary_affected:
        reason += (
            f"; pressure_ppm {pressure_ppm} is within boundary window "
            f"[{threshold - margin}, {threshold + margin}]"
        )

    return RouteDecision(
        selected_route=selected,
        reason=reason,
        pressure_ppm=pressure_ppm,
        overflow_pages=overflow_pages,
        overflow_ratio_ppm=overflow_ratio_ppm,
        threshold_ppm=threshold,
        boundary_margin_ppm=margin,
        boundary_low_ppm=threshold - margin,
        boundary_high_ppm=threshold + margin,
        boundary_affected=boundary_affected,
        tie_rule="ON_EQUAL_PRESSURE_SELECT_RESIDENT",
        forced=forced,
        page_count=page_count,
        access_count=len(catalog.scan_sequence),
        rows_per_page=catalog.rows_per_page,
        rows_reference=catalog.rows_reference,
        budget=budget,
    )


def plan_slots(scan_sequence: Sequence[int], budget: int, *, min_slots: int = 1) -> SlotPlan:
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < min_slots:
        raise EBudget(budget, min_slots)
    slots: list[int | None] = [None] * budget
    page_to_slot: dict[int, int] = {}
    last_used: dict[int, int] = {}
    accesses: list[PlanAccess] = []
    evictions: list[EvictionEvent] = []

    for index, page_id in enumerate(scan_sequence, start=1):
        evicted: int | None = None
        eviction_seq: int | None = None
        if page_id in page_to_slot:
            slot = page_to_slot[page_id]
            hit = True
        else:
            hit = False
            if None in slots:
                slot = slots.index(None)
            else:
                # LRU; page id is the stable tie breaker.
                victim_slot = min(
                    range(budget),
                    key=lambda candidate: (last_used[slots[candidate]], slots[candidate]),
                )
                slot = victim_slot
                evicted = slots[slot]
                assert evicted is not None
                del page_to_slot[evicted]
                del last_used[evicted]
                eviction_seq = len(evictions) + 1
                evictions.append(
                    EvictionEvent(
                        event_seq=eviction_seq,
                        access_seq=index,
                        slot=slot,
                        victim_page_id=evicted,
                        incoming_page_id=page_id,
                    )
                )
            slots[slot] = page_id
            page_to_slot[page_id] = slot
        last_used[page_id] = index
        accesses.append(
            PlanAccess(
                access_seq=index,
                page_id=page_id,
                hit=hit,
                slot=slot,
                evicted_page_id=evicted,
                eviction_seq=eviction_seq,
                occupancy_after=tuple(slots),
            )
        )

    hit_count = sum(1 for access in accesses if access.hit)
    return SlotPlan(
        budget=budget,
        accesses=tuple(accesses),
        evictions=tuple(evictions),
        hit_count=hit_count,
        miss_count=len(accesses) - hit_count,
        final_occupancy=tuple(slots),
    )
