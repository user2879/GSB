"""Deterministic incremental materialized-view engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import (
    canonical_bytes,
    group_sort_key,
    object_digest,
    sort_pages,
    typed_group_key,
)
from .errors import ColmvError
from .fixed import (
    FixedPointContract,
    saturate_int64,
    trunc_div_toward_zero,
)
from .model import Change, GroupRef, RuntimeConfig, ViewDefinition, stable_group_refs
from .storage import PageData, read_page_file


RESIDENT_THRESHOLD = Fraction(4, 1)
BOUNDARY_LOWER = Fraction(7, 2)
BOUNDARY_UPPER = Fraction(9, 2)
PAGE_SCAN_COST_UNITS = 8


def _fraction_payload(value: Fraction) -> dict[str, Any]:
    decimal = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": decimal,
    }


@dataclass(frozen=True)
class PathDecision:
    path: str
    reason: str
    metrics: dict[str, Any]
    boundary_affected: bool
    forced_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "reason": self.reason,
            "boundary_affected": self.boundary_affected,
            "forced_by": self.forced_by,
            "metrics": self.metrics,
        }


def select_path(
    *,
    change_count: int,
    affected_page_count: int,
    view_count: int,
    budget: int,
    path_mode: str = "auto",
    refresh_modes: Sequence[str] = (),
) -> PathDecision:
    if budget <= 0:
        raise ColmvError(
            "E_BUDGET",
            "slot budget must be a positive integer",
            field="budget",
            details={"budget": budget, "required_minimum": 1},
        )
    if change_count <= 0:
        raise ColmvError("E_CHANGE", "path decision requires at least one change", field="changes")
    if path_mode not in ("auto", "resident", "rescan"):
        raise ColmvError(
            "E_USAGE",
            f"unsupported path mode {path_mode!r}",
            field="path",
            details={"supported": ["auto", "resident", "rescan"]},
        )

    ratio = Fraction(change_count, budget)
    resident_work = change_count * max(1, view_count)
    rescan_work = affected_page_count * max(1, view_count) * PAGE_SCAN_COST_UNITS + change_count
    boundary_affected = BOUNDARY_LOWER <= ratio <= BOUNDARY_UPPER
    metric_base: dict[str, Any] = {
        "change_count": change_count,
        "affected_page_count": affected_page_count,
        "view_count": view_count,
        "budget": budget,
        "change_to_budget_ratio": _fraction_payload(ratio),
        "decision_threshold": _fraction_payload(RESIDENT_THRESHOLD),
        "boundary_band": {
            "lower_inclusive": _fraction_payload(BOUNDARY_LOWER),
            "upper_inclusive": _fraction_payload(BOUNDARY_UPPER),
        },
        "distance_to_threshold": _fraction_payload(abs(ratio - RESIDENT_THRESHOLD)),
        "boundary_affected": boundary_affected,
        "projected_work": {
            "resident": resident_work,
            "rescan": rescan_work,
        },
    }

    if path_mode != "auto":
        return PathDecision(
            path=path_mode,
            reason=f"forced by command line --path {path_mode}",
            metrics=metric_base,
            boundary_affected=boundary_affected,
            forced_by="command_line",
        )

    modes = set(refresh_modes)
    if len(modes) == 1 and next(iter(modes)) != "auto":
        mode = next(iter(modes))
        return PathDecision(
            path=mode,
            reason=f"forced by the common view refresh_mode {mode!r}",
            metrics=metric_base,
            boundary_affected=boundary_affected,
            forced_by="view_refresh_mode",
        )

    if not boundary_affected:
        if ratio < BOUNDARY_LOWER:
            path = "resident"
            reason = "change/budget ratio is below the boundary band; resident convergence wins"
        else:
            path = "rescan"
            reason = "change/budget ratio is above the boundary band; affected-page rescan wins"
        return PathDecision(
            path=path,
            reason=reason,
            metrics=metric_base,
            boundary_affected=False,
        )

    # Stable integer-only tie-break.  Resident wins exact ties.
    path = "resident" if resident_work <= rescan_work else "rescan"
    metric_base["tie_break"] = {
        "rule": "resident_work <= rescan_work; exact tie selects resident",
        "resident_work": resident_work,
        "rescan_work": rescan_work,
        "winner": path,
    }
    return PathDecision(
        path=path,
        reason=(
            "change/budget ratio is inside the explicit boundary band; "
            "applied deterministic integer work tie-break"
        ),
        metrics=metric_base,
        boundary_affected=True,
    )


@dataclass
class Contribution:
    row_id: int
    value_units: int | None
    has_value: bool
    float_value: bool
    seq: int
    order: int


@dataclass
class GroupAggregate:
    ref: GroupRef
    contributions: dict[int, Contribution] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.contributions)

    def ordered_contributions(self) -> list[Contribution]:
        return sorted(
            self.contributions.values(),
            key=lambda item: (item.seq, item.order, item.row_id),
        )

    def value_contributions(self) -> list[Contribution]:
        return [item for item in self.ordered_contributions() if item.has_value and item.value_units is not None]

    def sum_unbounded(self) -> int:
        total = 0
        for item in self.value_contributions():
            total += int(item.value_units or 0)
        return total

    def float_value_count(self) -> int:
        return sum(1 for item in self.value_contributions() if item.float_value)


@dataclass
class CacheEntry:
    ref: GroupRef
    slot_index: int
    last_touch: int


@dataclass
class CacheTouch:
    hit: bool
    admitted: bool
    denied: bool
    evicted: GroupRef | None
    slot_index: int | None


class SlotCache:
    """Global slot budget with integer-clock LRU and stable identity tie-breaks."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ColmvError(
                "E_BUDGET",
                "slot budget must be at least one",
                field="budget",
                details={"budget": capacity},
            )
        self.capacity = capacity
        self.slots: list[CacheEntry | None] = [None] * capacity
        self.clock = 0

    def entry_for(self, ref: GroupRef) -> CacheEntry | None:
        identity = ref.identity
        for entry in self.slots:
            if entry is not None and entry.ref.identity == identity:
                return entry
        return None

    def touch(self, ref: GroupRef, *, denied: set[tuple[str, str]]) -> CacheTouch:
        identity = ref.identity
        existing = self.entry_for(ref)
        if existing is not None:
            self.clock += 1
            existing.last_touch = self.clock
            return CacheTouch(True, False, False, None, existing.slot_index)
        if identity in denied:
            return CacheTouch(False, False, True, None, None)

        free_slot = next((index for index, entry in enumerate(self.slots) if entry is None), None)
        if free_slot is not None:
            self.clock += 1
            self.slots[free_slot] = CacheEntry(ref=ref, slot_index=free_slot, last_touch=self.clock)
            return CacheTouch(False, True, False, None, free_slot)

        occupied = [entry for entry in self.slots if entry is not None]
        victim = min(occupied, key=lambda entry: (entry.last_touch, entry.ref.view_id, entry.ref.key_sort))
        self.clock += 1
        self.slots[victim.slot_index] = CacheEntry(ref=ref, slot_index=victim.slot_index, last_touch=self.clock)
        denied.add(victim.ref.identity)
        return CacheTouch(False, True, False, victim.ref, victim.slot_index)

    def snapshot(self) -> list[GroupRef]:
        return sorted(
            (entry.ref for entry in self.slots if entry is not None),
            key=lambda ref: (ref.view_id, ref.key_sort),
        )


class MaterializationEngine:
    def __init__(
        self,
        config: RuntimeConfig,
        *,
        budget: int,
        path_mode: str = "auto",
        scheme_version: int = 1,
    ) -> None:
        if budget <= 0:
            raise ColmvError(
                "E_BUDGET",
                "integer slot budget must be greater than zero",
                field="budget",
                details={"budget": budget, "required_minimum": 1},
            )
        if path_mode not in ("auto", "resident", "rescan"):
            raise ColmvError(
                "E_USAGE",
                f"unsupported path mode {path_mode!r}",
                field="path",
            )
        self.config = config
        self.table = config.table
        self.views = config.views
        self.view_map = config.view_map
        self.budget = budget
        self.path_mode = path_mode
        self.scheme_version = scheme_version
        self.contract = FixedPointContract(config.scale_power)
        self.cache = SlotCache(budget)
        self.event_seq = 0
        self.batch_no = 0
        self.last_seq: int | None = None
        self.last_path: str | None = None
        self.switch_history: list[dict[str, Any]] = []
        self.batch_decisions: list[dict[str, Any]] = []
        self.total_hits = 0
        self.total_replacements = 0
        self.total_rescans = 0
        self.committed_changes = 0

        self.page_data: dict[str, PageData] = {}
        self.current_rows: dict[int, dict[str, Any]] = {}
        self.row_page: dict[int, str] = {}
        self.row_seq: dict[int, int] = {}
        self.row_order: dict[int, int] = {}
        self.rows_by_page: dict[str, set[int]] = {}
        self.seen_row_ids: set[int] = set()
        self.next_order = 0
        self.group_states: dict[tuple[str, str], GroupAggregate] = {}
        self.group_page_counts: dict[tuple[str, str], dict[str, int]] = {}

        self._load_base_and_initial_groups()

    # ------------------------------------------------------------------
    # Initial loading
    # ------------------------------------------------------------------
    def _load_base_and_initial_groups(self) -> None:
        for page_meta in self.table.pages:
            page_path = Path(page_meta.file)
            if not page_path.is_absolute():
                page_path = self.table.base_dir / page_path
            page = read_page_file(
                page_path,
                expected_page_id=page_meta.page_id,
                expected_start_row=page_meta.start_row,
                expected_end_row=page_meta.end_row,
            )
            self.page_data[page_meta.page_id] = page
            self.rows_by_page[page_meta.page_id] = set()
            for offset, row in enumerate(page.rows()):
                row_id = row[self.table.row_key]
                global_order = page_meta.start_row + offset
                self.current_rows[row_id] = dict(row)
                self.row_page[row_id] = page_meta.page_id
                self.row_seq[row_id] = 0
                self.row_order[row_id] = global_order
                self.rows_by_page[page_meta.page_id].add(row_id)
                self.seen_row_ids.add(row_id)
                for ref in self._row_groups(row):
                    self._add_contribution(
                        ref,
                        row_id=row_id,
                        row=row,
                        seq=0,
                        order=global_order,
                        page_id=page_meta.page_id,
                    )
        self.rows_by_page["delta"] = set()
        self.next_order = self.table.row_count

    # ------------------------------------------------------------------
    # Row, filter, and group semantics
    # ------------------------------------------------------------------
    def _filter_matches(self, row: Mapping[str, Any], view: ViewDefinition) -> bool:
        for filter_def in view.filters:
            value = row.get(filter_def.column)
            op = filter_def.op
            expected = filter_def.value
            if op == "is_null":
                if value is not None:
                    return False
                continue
            if op == "is_not_null":
                if value is None:
                    return False
                continue
            if op == "eq":
                if value != expected:
                    return False
                continue
            if op == "ne":
                if value == expected:
                    return False
                continue
            if op in ("gt", "gte", "lt", "lte"):
                if value is None or expected is None:
                    return False
                if op == "gt" and not value > expected:
                    return False
                if op == "gte" and not value >= expected:
                    return False
                if op == "lt" and not value < expected:
                    return False
                if op == "lte" and not value <= expected:
                    return False
                continue
            if op in ("in", "not_in"):
                contains = value in expected
                if op == "in" and not contains:
                    return False
                if op == "not_in" and contains:
                    return False
                continue
            return False
        return True

    def _row_groups(self, row: Mapping[str, Any]) -> list[GroupRef]:
        refs: list[GroupRef] = []
        for view in self.views:
            if not self._filter_matches(row, view):
                continue
            values = tuple(row.get(column) for column in view.group_by)
            if any(value is None for value in values):
                continue
            refs.append(GroupRef(view_id=view.view_id, values=values))
        return refs

    def _make_contribution(
        self,
        ref: GroupRef,
        *,
        row_id: int,
        row: Mapping[str, Any],
        seq: int,
        order: int,
    ) -> Contribution:
        view = self.view_map[ref.view_id]
        if view.value_column is None:
            return Contribution(row_id, None, False, False, seq, order)
        value = row.get(view.value_column)
        column_type = self.table.column_map[view.value_column].type
        quantized = self.contract.quantize(value, column=view.value_column)
        return Contribution(
            row_id=row_id,
            value_units=quantized,
            has_value=value is not None,
            float_value=value is not None and column_type == "float",
            seq=seq,
            order=order,
        )

    def _add_contribution(
        self,
        ref: GroupRef,
        *,
        row_id: int,
        row: Mapping[str, Any],
        seq: int,
        order: int,
        page_id: str,
    ) -> None:
        state = self.group_states.get(ref.identity)
        if state is None:
            state = GroupAggregate(ref=ref)
            self.group_states[ref.identity] = state
        state.contributions[row_id] = self._make_contribution(
            ref,
            row_id=row_id,
            row=row,
            seq=seq,
            order=order,
        )
        counts = self.group_page_counts.setdefault(ref.identity, {})
        counts[page_id] = counts.get(page_id, 0) + 1

    def _remove_contribution(self, ref: GroupRef, *, row_id: int, page_id: str) -> None:
        state = self.group_states.get(ref.identity)
        if state is not None:
            state.contributions.pop(row_id, None)
            if not state.contributions:
                self.group_states.pop(ref.identity, None)
        counts = self.group_page_counts.get(ref.identity)
        if counts is not None:
            counts[page_id] = counts.get(page_id, 0) - 1
            if counts[page_id] <= 0:
                counts.pop(page_id, None)
            if not counts:
                self.group_page_counts.pop(ref.identity, None)

    # ------------------------------------------------------------------
    # Change application
    # ------------------------------------------------------------------
    def _ensure_change_state(self, change: Change) -> None:
        if self.last_seq is not None and change.seq <= self.last_seq:
            raise ColmvError(
                "E_CHANGE",
                "change seq must be strictly increasing across engine batches",
                field="seq",
                details={"seq": change.seq, "previous_seq": self.last_seq},
            )
        if change.op == "insert":
            if change.row_id in self.seen_row_ids:
                raise ColmvError(
                    "E_CHANGE",
                    "insert row_id already exists in the base table or prior stream",
                    field="row_id",
                    details={"seq": change.seq, "row_id": change.row_id},
                )
        elif change.op in ("update", "delete"):
            if change.row_id not in self.current_rows:
                raise ColmvError(
                    "E_CHANGE",
                    f"{change.op} row_id is not present in the current table state",
                    field="row_id",
                    details={"seq": change.seq, "row_id": change.row_id},
                )
        else:
            raise ColmvError("E_CHANGE", f"unsupported operation {change.op!r}", field="op")

    def _apply_change(self, change: Change) -> str:
        self._ensure_change_state(change)
        page_id = "delta" if change.op == "insert" else self.row_page[change.row_id]
        if change.op == "insert":
            assert change.row is not None
            row = dict(change.row)
            self.current_rows[change.row_id] = row
            self.row_page[change.row_id] = "delta"
            self.row_seq[change.row_id] = change.seq
            self.row_order[change.row_id] = self.next_order
            self.next_order += 1
            self.seen_row_ids.add(change.row_id)
            self.rows_by_page.setdefault("delta", set()).add(change.row_id)
            for ref in self._row_groups(row):
                self._add_contribution(
                    ref,
                    row_id=change.row_id,
                    row=row,
                    seq=change.seq,
                    order=self.row_order[change.row_id],
                    page_id="delta",
                )
        elif change.op == "update":
            assert change.row is not None
            old_row = self.current_rows[change.row_id]
            if change.row.get(self.table.row_key) != change.row_id:
                raise ColmvError(
                    "E_CHANGE",
                    "updates may not change the row_key value",
                    field=self.table.row_key,
                    details={"seq": change.seq, "row_id": change.row_id},
                )
            for ref in self._row_groups(old_row):
                self._remove_contribution(ref, row_id=change.row_id, page_id=page_id)
            row = dict(change.row)
            self.current_rows[change.row_id] = row
            self.row_seq[change.row_id] = change.seq
            for ref in self._row_groups(row):
                self._add_contribution(
                    ref,
                    row_id=change.row_id,
                    row=row,
                    seq=change.seq,
                    order=self.row_order[change.row_id],
                    page_id=page_id,
                )
        else:  # delete
            old_row = self.current_rows[change.row_id]
            for ref in self._row_groups(old_row):
                self._remove_contribution(ref, row_id=change.row_id, page_id=page_id)
            self.current_rows.pop(change.row_id, None)
        self.last_seq = change.seq
        self.committed_changes += 1
        return page_id

    # ------------------------------------------------------------------
    # Preview and path decision
    # ------------------------------------------------------------------
    def preview_affected_pages(self, changes: Sequence[Change]) -> list[str]:
        pages: set[str] = set()
        for change in changes:
            if change.op == "insert":
                pages.add("delta")
            elif change.row_id in self.row_page:
                pages.add(self.row_page[change.row_id])
            elif change.row is not None:
                pages.add("delta")
        return sort_pages(pages)

    def preview_touched_groups(self, changes: Sequence[Change]) -> list[GroupRef]:
        refs: list[GroupRef] = []
        for change in changes:
            old_row = self.current_rows.get(change.row_id)
            if old_row is not None:
                refs.extend(self._row_groups(old_row))
            if change.row is not None and change.op in ("insert", "update"):
                refs.extend(self._row_groups(change.row))
        return stable_group_refs(refs)

    def decide_batch(self, changes: Sequence[Change]) -> PathDecision:
        if not changes:
            raise ColmvError("E_CHANGE", "batch has no changes", field="changes")
        pages = self.preview_affected_pages(changes)
        return select_path(
            change_count=len(changes),
            affected_page_count=len(pages),
            view_count=len(self.views),
            budget=self.budget,
            path_mode=self.path_mode,
            refresh_modes=[view.refresh_mode for view in self.views],
        )

    def _next_event_seq(self) -> int:
        self.event_seq += 1
        return self.event_seq

    def _replacement_event(
        self,
        *,
        batch_no: int,
        victim: GroupRef,
        incoming: GroupRef,
        slot_index: int,
    ) -> dict[str, Any]:
        self.total_replacements += 1
        return {
            "kind": "replacement",
            "event_seq": self._next_event_seq(),
            "batch_no": batch_no,
            "view_id": victim.view_id,
            "evicted_group_key": [typed_group_key([value])[0] for value in victim.values],
            "evicted_group_key_values": list(victim.values),
            "occupying_group_key": [typed_group_key([value])[0] for value in incoming.values],
            "occupying_group_key_values": list(incoming.values),
            "occupying_view_id": incoming.view_id,
            "slot_index": slot_index,
            "reason": "lru_full",
        }

    # ------------------------------------------------------------------
    # Rescan
    # ------------------------------------------------------------------
    def _page_scan_rows(self, page_id: str) -> list[tuple[int, dict[str, Any], int]]:
        if page_id == "delta":
            rows: list[tuple[int, dict[str, Any], int]] = []
            for row_id in sorted(self.rows_by_page.get("delta", set())):
                row = self.current_rows.get(row_id)
                if row is not None and self.row_page.get(row_id) == "delta":
                    rows.append((row_id, row, self.row_seq.get(row_id, 0)))
            return rows
        page_meta = self.table.page_by_id.get(page_id)
        if page_meta is None:
            return []
        page_path = Path(page_meta.file)
        if not page_path.is_absolute():
            page_path = self.table.base_dir / page_path
        page = read_page_file(
            page_path,
            expected_page_id=page_meta.page_id,
            expected_start_row=page_meta.start_row,
            expected_end_row=page_meta.end_row,
        )
        self.page_data[page_id] = page
        rows = []
        for offset, initial_row in enumerate(page.rows()):
            row_id = initial_row[self.table.row_key]
            row = self.current_rows.get(row_id)
            if row is None:
                continue
            rows.append((row_id, row, self.row_seq.get(row_id, 0)))
        return rows

    def _rebuild_group(
        self,
        ref: GroupRef,
        batch_group_pages: Mapping[tuple[str, str], set[str]],
    ) -> None:
        page_ids = set(self.group_page_counts.get(ref.identity, {}).keys())
        page_ids.update(batch_group_pages.get(ref.identity, set()))
        rebuilt = GroupAggregate(ref=ref)
        new_counts: dict[str, int] = {}
        for page_id in sort_pages(page_ids):
            for row_id, row, seq in self._page_scan_rows(page_id):
                if ref not in self._row_groups(row):
                    continue
                contribution = self._make_contribution(
                    ref,
                    row_id=row_id,
                    row=row,
                    seq=seq,
                    order=self.row_order[row_id],
                )
                rebuilt.contributions[row_id] = contribution
                new_counts[page_id] = new_counts.get(page_id, 0) + 1
        if rebuilt.contributions:
            self.group_states[ref.identity] = rebuilt
            self.group_page_counts[ref.identity] = new_counts
        else:
            self.group_states.pop(ref.identity, None)
            self.group_page_counts.pop(ref.identity, None)

    def _rescan_event(
        self,
        *,
        batch_no: int,
        ref: GroupRef,
        page_id: str,
        reason: str,
    ) -> dict[str, Any]:
        self.total_rescans += 1
        return {
            "kind": "rescan",
            "event_seq": self._next_event_seq(),
            "batch_no": batch_no,
            "view_id": ref.view_id,
            "group_key": [typed_group_key([value])[0] for value in ref.values],
            "group_key_values": list(ref.values),
            "page_id": page_id,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def process_batch(
        self,
        changes: Sequence[Change],
        *,
        batch_no: int | None = None,
        force_path: str | None = None,
    ) -> dict[str, Any]:
        if not changes:
            raise ColmvError("E_CHANGE", "batch has no changes", field="changes")
        self.batch_no = self.batch_no + 1 if batch_no is None else batch_no
        decision = self.decide_batch(changes)
        if force_path is not None:
            if force_path not in ("resident", "rescan"):
                raise ColmvError("E_USAGE", f"invalid forced path {force_path!r}", field="path")
            decision = PathDecision(
                path=force_path,
                reason=f"forced by replay ledger path {force_path}",
                metrics=decision.metrics,
                boundary_affected=decision.boundary_affected,
                forced_by="replay_ledger",
            )

        if self.last_path is not None and self.last_path != decision.path:
            self.switch_history.append(
                {
                    "batch_no": self.batch_no,
                    "from": self.last_path,
                    "to": decision.path,
                    "reason": decision.reason,
                    "metrics": decision.metrics,
                }
            )
        self.last_path = decision.path
        self.batch_decisions.append({"batch_no": self.batch_no, **decision.to_dict()})

        affected_pages: set[str] = set()
        affected_refs: dict[tuple[str, str], GroupRef] = {}
        batch_denied: dict[tuple[str, str], GroupRef] = {}
        denied_identities: set[tuple[str, str]] = set()
        replacement_events: list[dict[str, Any]] = []
        rescan_targets: dict[tuple[str, str], GroupRef] = {}
        batch_hits = 0
        batch_group_pages: dict[tuple[str, str], set[str]] = {}

        for change in changes:
            old_row = self.current_rows.get(change.row_id)
            old_groups = self._row_groups(old_row) if old_row is not None else []
            page_id = self._apply_change(change)
            affected_pages.add(page_id)
            new_row = self.current_rows.get(change.row_id)
            new_groups = self._row_groups(new_row) if new_row is not None else []
            touched = stable_group_refs(old_groups + new_groups)
            for ref in touched:
                affected_refs[ref.identity] = ref
                batch_group_pages.setdefault(ref.identity, set()).add(page_id)
                touch = self.cache.touch(ref, denied=denied_identities)
                if touch.hit:
                    batch_hits += 1
                if touch.evicted is not None:
                    batch_denied[touch.evicted.identity] = touch.evicted
                    rescan_targets[touch.evicted.identity] = touch.evicted
                    replacement_events.append(
                        self._replacement_event(
                            batch_no=self.batch_no,
                            victim=touch.evicted,
                            incoming=ref,
                            slot_index=int(touch.slot_index or 0),
                        )
                    )
                if touch.denied:
                    batch_denied[ref.identity] = ref
                    rescan_targets[ref.identity] = ref

        touched_refs = sorted(affected_refs.values(), key=lambda ref: (ref.view_id, ref.key_sort))

        self.total_hits += batch_hits
        events: list[dict[str, Any]] = list(replacement_events)
        rescan_events: list[dict[str, Any]] = []
        if decision.path == "rescan":
            for ref in sorted(rescan_targets.values(), key=lambda item: (item.view_id, item.key_sort)):
                pages = set(self.group_page_counts.get(ref.identity, {}).keys())
                pages.update(batch_group_pages.get(ref.identity, set()))
                if not pages:
                    pages = set(affected_pages)
                self._rebuild_group(ref, batch_group_pages)
                for page_id in sort_pages(pages):
                    rescan_events.append(
                        self._rescan_event(
                            batch_no=self.batch_no,
                            ref=ref,
                            page_id=page_id,
                            reason="slot_not_resident:affected_page_rescan",
                        )
                    )
            events.extend(rescan_events)

        resident_affected = [
            ref for ref in touched_refs if self.cache.entry_for(ref) is not None
        ]
        denied_sorted = sorted(
            batch_denied.values(), key=lambda ref: (ref.view_id, ref.key_sort)
        )
        denied_affected = [
            ref for ref in denied_sorted if ref.identity in {item.identity for item in touched_refs}
        ]
        evicted_other = [
            ref for ref in denied_sorted if ref.identity not in {item.identity for item in touched_refs}
        ]
        partition_complete = {
            ref.identity for ref in resident_affected
        } | {ref.identity for ref in denied_affected} == {
            ref.identity for ref in touched_refs
        }
        partition_disjoint = not (
            {ref.identity for ref in resident_affected}
            & {ref.identity for ref in denied_affected}
        )

        group_payload = self.group_state_payload()
        cursor = changes[-1].seq
        result = {
            "batch_no": self.batch_no,
            "start_seq": changes[0].seq,
            "end_seq": changes[-1].seq,
            "change_count": len(changes),
            "cursor": cursor,
            "path": decision.path,
            "decision": decision.to_dict(),
            "affected_pages": sort_pages(affected_pages),
            "affected_groups": [self._ref_payload(ref) for ref in touched_refs],
            "events": events,
            "replacement_events": replacement_events,
            "rescan_events": rescan_events,
            "rescan_cost": {
                "page_scan_events": len(rescan_events),
                "group_count": len(rescan_targets),
                "page_ids": sort_pages(event["page_id"] for event in rescan_events),
                "reason": "one ledger event per (group, page) physical page scan",
            },
            "hits": batch_hits,
            "slot_partition": {
                "resident_groups": [self._ref_payload(ref) for ref in resident_affected],
                "rescan_groups": [self._ref_payload(ref) for ref in denied_sorted],
                "rescan_affected_groups": [self._ref_payload(ref) for ref in denied_affected],
                "evicted_unaffected_groups": [self._ref_payload(ref) for ref in evicted_other],
                "complete_for_affected": partition_complete,
                "disjoint": partition_disjoint,
            },
            "resident_groups_final": [self._ref_payload(ref) for ref in self.cache.snapshot()],
            "groups": group_payload,
            "scheme_version": self.scheme_version,
            "scheme_fingerprint": self.config.scheme_fingerprint(self.budget),
            "fixed_point": self.contract.contract_dict(),
        }
        return result

    def _ref_payload(self, ref: GroupRef) -> dict[str, Any]:
        return {
            "view_id": ref.view_id,
            "group_key": [typed_group_key([value])[0] for value in ref.values],
            "group_key_values": list(ref.values),
        }

    # ------------------------------------------------------------------
    # Group estimates and error bounds
    # ------------------------------------------------------------------
    def _group_error_bounds(
        self,
        view: ViewDefinition,
        state: GroupAggregate,
    ) -> dict[str, int]:
        value_contributions = state.value_contributions()
        float_count = state.float_value_count()
        value_count = len(value_contributions)
        bounds: dict[str, int] = {}
        for aggregate in view.aggregate_order():
            if aggregate == "count":
                bounds["count"] = 0
            elif aggregate == "sum":
                bounds["sum"] = float_count
            elif aggregate == "avg":
                if value_count == 0:
                    bounds["avg"] = 0
                elif float_count == 0:
                    bounds["avg"] = 1
                else:
                    bounds["avg"] = 2
        return bounds

    def _group_result(self, view: ViewDefinition, state: GroupAggregate) -> dict[str, Any]:
        count = state.count
        value_contributions = state.value_contributions()
        value_count = len(value_contributions)
        sum_units = state.sum_unbounded()
        sum_fixed = saturate_int64(sum_units)
        avg_raw = trunc_div_toward_zero(sum_units, value_count) if value_count else 0
        avg_fixed = saturate_int64(avg_raw)
        count_fixed = saturate_int64(count * self.contract.scale)
        bounds = self._group_error_bounds(view, state)
        aggregates: dict[str, Any] = {}
        for aggregate in view.aggregate_order():
            if aggregate == "count":
                aggregates["count"] = {
                    "fixed": count_fixed,
                    "exact": count,
                    "saturated": count_fixed != count * self.contract.scale,
                }
            elif aggregate == "sum":
                aggregates["sum"] = {
                    "fixed": sum_fixed,
                    "defined": value_count > 0,
                    "non_null_values": value_count,
                    "saturated": sum_fixed != sum_units,
                }
            elif aggregate == "avg":
                aggregates["avg"] = {
                    "fixed": avg_fixed,
                    "defined": value_count > 0,
                    "non_null_values": value_count,
                    "saturated": avg_fixed != avg_raw,
                }
        return {
            "view_id": view.view_id,
            "group_key": list(state.ref.values),
            "group_key_typed": typed_group_key(state.ref.values),
            "row_count": count,
            "aggregates": aggregates,
            "error_bounds_fp_units": bounds,
            "intermediate": {
                "sum_fp_units_unbounded": str(sum_units),
                "non_null_value_count": value_count,
                "float_value_count": state.float_value_count(),
                "canonical_accumulation_order": "change_seq,initial_row_order,row_id; groups sorted by typed group key",
            },
        }

    def group_state_payload(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for view in self.views:
            candidates = [
                state for identity, state in self.group_states.items() if identity[0] == view.view_id
            ]
            candidates.sort(key=lambda state: state.ref.key_sort)
            for state in candidates:
                payload.append(self._group_result(view, state))
        return payload

    def _current_null_group_key_counts(self) -> dict[str, int]:
        counts = {view.view_id: 0 for view in self.views}
        for row_id in sorted(self.current_rows):
            row = self.current_rows[row_id]
            for view in self.views:
                if not self._filter_matches(row, view):
                    continue
                values = [row.get(column) for column in view.group_by]
                if any(value is None for value in values):
                    counts[view.view_id] += 1
        return counts

    def _observed_max_quantization_error(self) -> dict[str, Any]:
        best_numerator = 0
        best_denominator = 1
        best_decimal = "0"
        value_columns = {
            view.value_column
            for view in self.views
            if view.value_column is not None
            and self.table.column_map[view.value_column].type == "float"
        }
        for row_id in sorted(self.current_rows):
            row = self.current_rows[row_id]
            for column in sorted(value_columns):
                value = row.get(column)
                if not isinstance(value, float):
                    continue
                numerator, denominator, decimal_text = self.contract.quantization_error_units_fraction(value)
                if numerator * best_denominator > best_numerator * denominator:
                    best_numerator = numerator
                    best_denominator = denominator
                    best_decimal = decimal_text
        return {
            "numerator": str(best_numerator),
            "denominator": str(best_denominator),
            "decimal_fp_units": best_decimal,
        }

    def materialized_result(self) -> dict[str, Any]:
        view_payloads: list[dict[str, Any]] = []
        error_segments: list[dict[str, Any]] = []
        total_error_units = 0
        for view in self.views:
            states = [
                state
                for identity, state in self.group_states.items()
                if identity[0] == view.view_id
            ]
            states.sort(key=lambda state: state.ref.key_sort)
            groups = [self._group_result(view, state) for state in states]
            for group in groups:
                for aggregate, units in group["error_bounds_fp_units"].items():
                    total_error_units += int(units)
                    if units > 0:
                        error_segments.append(
                            {
                                "view_id": view.view_id,
                                "group_key": group["group_key"],
                                "group_key_typed": group["group_key_typed"],
                                "aggregate": aggregate,
                                "upper_bound_fp_units": int(units),
                                "upper_bound_decimal": self.contract.decimal_from_units(int(units)),
                            }
                        )
            view_payloads.append(
                {
                    "view_id": view.view_id,
                    "definition_digest": view.digest,
                    "definition": view.to_dict(),
                    "groups": groups,
                }
            )

        error_segments.sort(
            key=lambda item: (
                -int(item["upper_bound_fp_units"]),
                item["view_id"],
                group_sort_key(item["group_key"]),
                item["aggregate"],
            )
        )
        for segment in error_segments:
            segment["share_ppm"] = (
                int(segment["upper_bound_fp_units"]) * 1_000_000 // total_error_units
                if total_error_units
                else 0
            )
        dominant = error_segments[0] if error_segments else None
        null_counts = self._current_null_group_key_counts()
        observed_quantization = self._observed_max_quantization_error()
        result = {
            "format_version": 1,
            "kind": "materialized_view_result",
            "scheme": {
                "version": self.scheme_version,
                "fingerprint": self.config.scheme_fingerprint(self.budget),
                "budget": self.budget,
            },
            "fixed_point": self.contract.contract_dict(),
            "views": view_payloads,
            "excluded_null_group_key_rows": null_counts,
            "error_summary": {
                "single_value_quantization_error": {
                    "direction": "toward-zero",
                    "upper_bound_fp_units": 1,
                    "upper_bound_decimal": self.contract.decimal_from_units(1),
                    "integer_inputs_exact": True,
                    "observed_max_abs_error_fp_units": observed_quantization,
                },
                "total_upper_bound_fp_units": total_error_units,
                "segments": error_segments,
                "dominant_segment": dominant,
            },
        }
        return result

    def materialized_bytes(self) -> bytes:
        return canonical_bytes(self.materialized_result())

    def materialized_digest(self) -> str:
        return object_digest(self.materialized_result())

    def collect_groups_and_pages(self) -> dict[str, Any]:
        refs: dict[tuple[str, str], GroupRef] = {}
        pages: set[str] = set()
        for row_id in sorted(self.current_rows):
            row = self.current_rows[row_id]
            page_id = self.row_page[row_id]
            for ref in self._row_groups(row):
                refs[ref.identity] = ref
                pages.add(page_id)
        return {
            "groups": [self._ref_payload(ref) for ref in sorted(refs.values(), key=lambda item: (item.view_id, item.key_sort))],
            "pages": sort_pages(pages),
        }

    def collect_reference_index(self) -> dict[str, Any]:
        group_map: dict[tuple[str, str], dict[str, Any]] = {}
        page_to_groups: dict[str, set[tuple[str, str]]] = {}
        for row_id in sorted(self.current_rows):
            row = self.current_rows[row_id]
            page_id = self.row_page[row_id]
            for ref in self._row_groups(row):
                entry = group_map.setdefault(
                    ref.identity,
                    {
                        **self._ref_payload(ref),
                        "group_key_typed": typed_group_key(ref.values),
                        "pages": set(),
                    },
                )
                entry["pages"].add(page_id)
                page_to_groups.setdefault(page_id, set()).add(ref.identity)
        groups = []
        for identity in sorted(group_map):
            entry = group_map[identity]
            groups.append({**entry, "pages": sort_pages(entry["pages"])})
        pages = [
            {
                "page_id": page_id,
                "referenced_by_groups": [
                    {"view_id": view_id, "group_sort": group_sort}
                    for view_id, group_sort in sorted(page_to_groups.get(page_id, set()))
                ],
            }
            for page_id in sort_pages(page_to_groups)
        ]
        return {"groups": groups, "pages": pages}


def run_all(
    config: RuntimeConfig,
    changes: Sequence[Change],
    *,
    budget: int,
    batch_size: int,
    path_mode: str = "auto",
    scheme_version: int = 1,
) -> tuple[MaterializationEngine, list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ColmvError("E_USAGE", "batch_size must be positive", field="batch_size")
    engine = MaterializationEngine(
        config,
        budget=budget,
        path_mode=path_mode,
        scheme_version=scheme_version,
    )
    batches: list[dict[str, Any]] = []
    for offset in range(0, len(changes), batch_size):
        batches.append(engine.process_batch(changes[offset : offset + batch_size]))
    return engine, batches
