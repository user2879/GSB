"""Trace-local causal chain reconstruction and sequence diagnostics."""

from __future__ import annotations

from typing import Any, Iterable

from .canonical import event_fingerprint
from .errors import (
    E_CROSS_TRACE,
    E_CYCLE,
    E_ORPHAN,
    E_SEQ_DUPLICATE,
    E_SEQ_GAP,
    E_SEQ_ROLLBACK,
    business_error,
)


def reconstruct_traces(event_states: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build deterministic parent/child forests for every trace.

    Returns ``(trace_chains, errors)``. Parent references are resolved against
    all traces, so missing and cross-trace references are distinguishable.
    """

    events = {
        state["event"]["event_id"]: state["event"]
        for state in event_states
        if state.get("event", {}).get("event_id")
    }
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for event in events.values():
        by_trace.setdefault(event["trace_id"], []).append(event)
    for trace_events in by_trace.values():
        trace_events.sort(key=lambda item: (item["seq"], item["event_id"]))

    errors: list[dict[str, Any]] = []
    children: dict[str, list[str]] = {event_id: [] for event_id in events}
    parent_of: dict[str, str | None] = {}
    reported_cycle_members: set[str] = set()

    for trace_id in sorted(by_trace):
        for event in by_trace[trace_id]:
            event_id = event["event_id"]
            parent_id = event.get("parent_event_id")
            if not parent_id:
                parent_of[event_id] = None
                continue
            if parent_id == event_id:
                parent_of[event_id] = None
                if event_id not in reported_cycle_members:
                    reported_cycle_members.add(event_id)
                    errors.append(
                        business_error(
                            E_CYCLE,
                            "parent_event_id references the event itself",
                            event_id=event_id,
                            trace_id=trace_id,
                            details={"cycle": [event_id]},
                        )
                    )
                continue
            parent = events.get(parent_id)
            if parent is None:
                parent_of[event_id] = None
                errors.append(
                    business_error(
                        E_ORPHAN,
                        "parent_event_id does not exist in any input stream",
                        event_id=event_id,
                        trace_id=trace_id,
                        details={"parent_event_id": parent_id},
                    )
                )
                continue
            if parent["trace_id"] != trace_id:
                parent_of[event_id] = None
                errors.append(
                    business_error(
                        E_CROSS_TRACE,
                        "parent_event_id resolves to a different trace",
                        event_id=event_id,
                        trace_id=trace_id,
                        details={
                            "parent_event_id": parent_id,
                            "parent_trace_id": parent["trace_id"],
                        },
                    )
                )
                continue
            parent_of[event_id] = parent_id
            children[parent_id].append(event_id)

    for event_id in children:
        children[event_id].sort(key=lambda child_id: (events[child_id]["seq"], child_id))

    for start_id in sorted(events):
        path: list[str] = []
        positions: dict[str, int] = {}
        current: str | None = start_id
        while current is not None and current not in positions:
            positions[current] = len(path)
            path.append(current)
            current = parent_of.get(current)
        if current is not None and current in positions:
            cycle = path[positions[current] :]
            for event_id in sorted(cycle):
                if event_id not in reported_cycle_members:
                    reported_cycle_members.add(event_id)
                    errors.append(
                        business_error(
                            E_CYCLE,
                            "cycle detected while resolving parent_event_id",
                            event_id=event_id,
                            trace_id=events[event_id]["trace_id"],
                            details={"cycle": sorted(cycle)},
                        )
                    )

    trace_chains: list[dict[str, Any]] = []
    for trace_id in sorted(by_trace):
        trace_events = by_trace[trace_id]
        trace_ids = {event["event_id"] for event in trace_events}
        roots = [
            event["event_id"]
            for event in trace_events
            if parent_of.get(event["event_id"]) is None
        ]
        roots.sort(key=lambda event_id: (events[event_id]["seq"], event_id))
        traversal: list[dict[str, Any]] = []
        seen: set[str] = set()

        def visit(event_id: str, depth: int) -> None:
            if event_id in seen:
                return
            seen.add(event_id)
            event = events[event_id]
            traversal.append(
                {
                    "event_id": event_id,
                    "parent_event_id": parent_of.get(event_id),
                    "children": list(children[event_id]),
                    "depth": depth,
                    "seq": event["seq"],
                    "ts": event["ts"],
                }
            )
            for child_id in children[event_id]:
                if child_id in trace_ids:
                    visit(child_id, depth + 1)

        for root_id in roots:
            visit(root_id, 0)
        for event in trace_events:
            visit(event["event_id"], 0)

        trace_chains.append(
            {
                "trace_id": trace_id,
                "root_event_ids": roots,
                "event_count": len(trace_events),
                "order": [node["event_id"] for node in traversal],
                "nodes": traversal,
            }
        )

    errors.sort(key=_error_sort_key)
    return trace_chains, errors


def _error_sort_key(error: dict[str, Any]) -> tuple[Any, ...]:
    return (
        error.get("trace_id", ""),
        error.get("event_id", ""),
        error.get("code", ""),
        str(error.get("details", "")),
    )


def analyze_sequences(
    occurrences: Iterable[dict[str, Any]],
    event_states: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Report arrival-order seq rollback and canonical-set seq gaps distinctly."""

    errors: list[dict[str, Any]] = []
    ordered = sorted(
        occurrences,
        key=lambda item: (
            item.get("source_order", 0),
            item.get("position_order", 0),
            item.get("sync_marker", ""),
        ),
    )

    seen_markers: set[tuple[str, str]] = set()
    previous_seq_by_trace: dict[str, int] = {}
    seen_fingerprints_by_trace: dict[str, set[str]] = {}
    for occurrence in ordered:
        marker = (occurrence.get("sync_marker", ""), occurrence.get("source", ""))
        if marker in seen_markers:
            continue
        seen_markers.add(marker)
        core = occurrence.get("event_core") or {}
        trace_id = core.get("trace_id")
        seq = core.get("seq")
        if not isinstance(trace_id, str) or not isinstance(seq, int):
            continue
        fingerprint = event_fingerprint(core)
        seen_fingerprints = seen_fingerprints_by_trace.setdefault(trace_id, set())
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        previous = previous_seq_by_trace.get(trace_id)
        if previous is not None and seq <= previous:
            errors.append(
                business_error(
                    E_SEQ_ROLLBACK,
                    "seq did not increase in deterministic arrival order",
                    source_format=occurrence.get("source_format"),
                    source=occurrence.get("source"),
                    position=occurrence.get("position"),
                    event_id=core.get("event_id"),
                    trace_id=trace_id,
                    details={"previous_seq": previous, "current_seq": seq},
                )
            )
        previous_seq_by_trace[trace_id] = seq

    by_trace: dict[str, list[dict[str, Any]]] = {}
    for state in event_states:
        event = state.get("event") or {}
        if event.get("event_id") and isinstance(event.get("seq"), int):
            by_trace.setdefault(event["trace_id"], []).append(event)

    marker_by_event: dict[tuple[str, str], dict[str, Any]] = {}
    for occurrence in ordered:
        core = occurrence.get("event_core") or {}
        key = (core.get("trace_id"), core.get("event_id"))
        marker_by_event.setdefault(key, occurrence)

    for trace_id in sorted(by_trace):
        events = sorted(
            by_trace[trace_id], key=lambda item: (item["seq"], item["event_id"])
        )
        for previous, current in zip(events, events[1:]):
            if current["seq"] == previous["seq"]:
                occurrence = marker_by_event.get((trace_id, current["event_id"]), {})
                errors.append(
                    business_error(
                        E_SEQ_DUPLICATE,
                        "two distinct events share a seq in the same trace",
                        source_format=occurrence.get("source_format"),
                        source=occurrence.get("source"),
                        position=occurrence.get("position"),
                        event_id=current["event_id"],
                        trace_id=trace_id,
                        details={
                            "seq": current["seq"],
                            "other_event_id": previous["event_id"],
                        },
                    )
                )
            elif current["seq"] - previous["seq"] > 1:
                occurrence = marker_by_event.get((trace_id, current["event_id"]), {})
                missing = list(range(previous["seq"] + 1, current["seq"]))
                errors.append(
                    business_error(
                        E_SEQ_GAP,
                        "seq gap detected in canonical trace state",
                        source_format=occurrence.get("source_format"),
                        source=occurrence.get("source"),
                        position=occurrence.get("position"),
                        event_id=current["event_id"],
                        trace_id=trace_id,
                        details={
                            "previous_seq": previous["seq"],
                            "current_seq": current["seq"],
                            "missing_seq": missing,
                        },
                    )
                )

    errors.sort(key=_error_sort_key)
    return errors
