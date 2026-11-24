"""Invalidation propagation along source -> metric key -> window references.

Given a set of declaration / parameter changes, this module derives exactly
which windows must be recomputed and which can be reused. The derivation is
pure and deterministic: identical inputs always produce an identical,
stably-ordered propagation trail.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import PropagationEvent, SourceDeclaration, WindowResult

# Change kinds understood by the propagation engine.
CHANGE_FORMAT = "format_version"
CHANGE_MAPPING = "metric_mapping"
CHANGE_DEFAULT_KEY = "default_metric_key"
CHANGE_DEFAULT_SCALE = "default_scale"
CHANGE_SOURCE_ADDED = "source_added"
CHANGE_SOURCE_REMOVED = "source_removed"
CHANGE_WINDOW_SIZE = "window_size"
CHANGE_SLIDE_STEP = "slide_step"


@dataclass(frozen=True, slots=True)
class SourceChange:
    """One declaration difference for one source."""

    source_id: str
    field: str
    old_value: str | None
    new_value: str | None


@dataclass(frozen=True, slots=True)
class PropagationPlan:
    events: tuple[PropagationEvent, ...]
    # Fully affected (metric_key, window_index) set.
    affected: frozenset[tuple[str, int]]
    # metric keys touched by at least one source-level change.
    touched_metrics: frozenset[str]

    def affected_sorted(self) -> list[tuple[str, int]]:
        return sorted(self.affected)


def _metric_keys_for_source(source: SourceDeclaration | None,
                            fallback: SourceDeclaration | None) -> set[str]:
    keys: set[str] = set()
    for candidate in (source, fallback):
        if candidate is None:
            continue
        keys.add(candidate.default_metric_key)
        keys.update(candidate.metric_mapping.values())
        keys.update(candidate.metric_mapping.keys())
    return keys


def derive_propagation(
    old_sources: dict[str, SourceDeclaration],
    new_sources: dict[str, SourceDeclaration],
    old_windows: list[WindowResult],
    source_changes: list[SourceChange],
    *,
    window_size_changed: bool = False,
    slide_step_changed: bool = False,
) -> PropagationPlan:
    """Derive the ordered invalidation trail and affected window set."""
    events: list[PropagationEvent] = []
    affected: set[tuple[str, int]] = set()
    touched: set[str] = set()
    order = 0

    def window_refs(metric_keys: set[str]) -> list[tuple[str, int]]:
        refs = {
            (w.metric_key, w.window_index)
            for w in old_windows if w.metric_key in metric_keys
        }
        return sorted(refs)

    def emit(reason: str, source_id: str | None, metric_key: str | None,
             change: SourceChange | None, field: str,
             old_value: str | None, new_value: str | None,
             refs: list[tuple[str, int]], basis: str) -> None:
        nonlocal order
        order += 1
        events.append(PropagationEvent(
            order=order,
            reason=reason,
            source_id=source_id,
            metric_key=metric_key,
            changed_field=field,
            old_value=old_value,
            new_value=new_value,
            affected_windows=tuple(refs),
            basis=basis,
        ))
        affected.update(refs)

    # Source-level changes first, in stable source/field order.
    for change in sorted(source_changes, key=lambda c: (c.source_id, c.field)):
        old_src = old_sources.get(change.source_id)
        new_src = new_sources.get(change.source_id)
        if change.field in (CHANGE_SOURCE_ADDED, CHANGE_SOURCE_REMOVED):
            keys = _metric_keys_for_source(old_src, new_src)
            touched.update(keys)
            refs = window_refs(keys)
            emit(
                reason="source_set_change",
                source_id=change.source_id,
                metric_key=None,
                change=change,
                field=change.field,
                old_value=change.old_value,
                new_value=change.new_value,
                refs=refs,
                basis=(f"source {change.source_id!r} "
                       f"{'added' if change.field == CHANGE_SOURCE_ADDED else 'removed'}; "
                       "all windows fed by its metric keys are invalidated"),
            )
            continue

        keys = _metric_keys_for_source(old_src, new_src)
        touched.update(keys)
        refs = window_refs(keys)
        if change.field == CHANGE_FORMAT:
            basis = ("source format version changed; normalisation of every "
                     "record from this source changes, so all dependent "
                     "windows are invalidated")
        elif change.field in (CHANGE_MAPPING, CHANGE_DEFAULT_KEY):
            basis = ("metric key mapping changed; records may move between "
                     "metric keys, invalidating source and destination "
                     "windows")
        elif change.field == CHANGE_DEFAULT_SCALE:
            basis = ("record fixed-point scale changed; quantisation of this "
                     "source changes, invalidating dependent windows")
        else:
            basis = "source declaration change invalidates dependent windows"
        emit(
            reason="source_change",
            source_id=change.source_id,
            metric_key=None,
            change=change,
            field=change.field,
            old_value=change.old_value,
            new_value=change.new_value,
            refs=refs,
            basis=basis,
        )

    # Window parameter changes are global and rebuild the whole grid.
    if window_size_changed:
        refs = sorted({(w.metric_key, w.window_index) for w in old_windows})
        emit(
            reason="window_param_change",
            source_id=None,
            metric_key=None,
            change=None,
            field=CHANGE_WINDOW_SIZE,
            old_value=None,
            new_value=None,
            refs=refs,
            basis="window_size changed; the window grid is rebuilt for "
                  "every metric key",
        )
    if slide_step_changed:
        refs = sorted({(w.metric_key, w.window_index) for w in old_windows})
        emit(
            reason="window_param_change",
            source_id=None,
            metric_key=None,
            change=None,
            field=CHANGE_SLIDE_STEP,
            old_value=None,
            new_value=None,
            refs=refs,
            basis="slide_step changed; window boundaries shift for every "
                  "metric key",
        )

    return PropagationPlan(
        events=tuple(events),
        affected=frozenset(affected),
        touched_metrics=frozenset(touched),
    )


def diff_source_sets(
    old_sources: dict[str, SourceDeclaration],
    new_sources: dict[str, SourceDeclaration],
) -> list[SourceChange]:
    """Compute the stable list of declaration differences."""
    changes: list[SourceChange] = []
    for source_id in sorted(set(old_sources) | set(new_sources)):
        old = old_sources.get(source_id)
        new = new_sources.get(source_id)
        if old is None:
            changes.append(SourceChange(source_id, CHANGE_SOURCE_ADDED, None,
                                        str(new.format_version)))
            continue
        if new is None:
            changes.append(SourceChange(source_id, CHANGE_SOURCE_REMOVED,
                                        str(old.format_version), None))
            continue
        if old.format_version != new.format_version:
            changes.append(SourceChange(
                source_id, CHANGE_FORMAT, str(old.format_version),
                str(new.format_version)))
        if old.metric_mapping != new.metric_mapping:
            changes.append(SourceChange(
                source_id, CHANGE_MAPPING,
                repr(sorted(old.metric_mapping.items())),
                repr(sorted(new.metric_mapping.items()))))
        if old.default_metric_key != new.default_metric_key:
            changes.append(SourceChange(
                source_id, CHANGE_DEFAULT_KEY, old.default_metric_key,
                new.default_metric_key))
        if old.default_scale != new.default_scale:
            changes.append(SourceChange(
                source_id, CHANGE_DEFAULT_SCALE, str(old.default_scale),
                str(new.default_scale)))
    return changes
