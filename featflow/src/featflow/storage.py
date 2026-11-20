"""Deterministic event storage and sliding-window retrieval.

Events are kept per entity (user / item).  Within an entity they are sorted by
event_id lexicographically (the mandated accumulation order); numpy arrays
support O(log n) window boundary lookup.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import SIDE_ITEM, SIDE_USER
from .models import Event


@dataclass
class EntityWindow:
    event_ids: list[str]
    ts: np.ndarray                      # int64, aligned with event_ids order
    values: dict[str, np.ndarray]       # field -> float64 aligned
    behavior: list[str]


class EventStore:
    def __init__(self) -> None:
        self._by_user: dict[str, list[Event]] = {}
        self._by_item: dict[str, list[Event]] = {}
        self._all: dict[str, Event] = {}
        self._windows: dict[tuple[str, str], EntityWindow] = {}

    # ------------------------------------------------------------------ build
    def rebuild(self, events: list[Event]) -> None:
        self._by_user.clear()
        self._by_item.clear()
        self._all.clear()
        self._windows.clear()
        for ev in events:
            self._all[ev.event_id] = ev
            self._by_user.setdefault(ev.user_id, []).append(ev)
            self._by_item.setdefault(ev.item_id, []).append(ev)

    def all_events(self) -> list[Event]:
        return [self._all[k] for k in sorted(self._all)]

    def get(self, event_id: str) -> Event | None:
        return self._all.get(event_id)

    def cursor_ts(self) -> int:
        """Largest processed timestamp; events are appended in arrival order."""
        if not self._all:
            return -1
        return max(ev.ts_ms for ev in self._all.values())

    # ----------------------------------------------------------------- window
    def window(self, side: str, entity_id: str) -> EntityWindow:
        key = (side, entity_id)
        cached = self._windows.get(key)
        if cached is not None:
            return cached
        events = (self._by_user if side == SIDE_USER else self._by_item).get(entity_id, [])
        ordered = sorted(events, key=lambda e: e.event_id)
        fields = sorted({f for ev in ordered for f in ev.context})
        win = EntityWindow(
            event_ids=[ev.event_id for ev in ordered],
            ts=np.array([ev.ts_ms for ev in ordered], dtype=np.int64),
            values={
                f: np.array([
                    ev.context.get(f, np.nan) if f in ev.context else np.nan
                    for ev in ordered
                ], dtype=np.float64)
                for f in fields
            },
            behavior=[ev.behavior for ev in ordered],
        )
        self._windows[key] = win
        return win

    def select_window(self, side: str, entity_id: str, anchor_ts: int,
                      window_ms: int, source_field: str | None) -> tuple[list[str], np.ndarray | None]:
        """Return (event_ids in [anchor-window+1? no: ts >= anchor-window, anchor], values).

        The window includes both boundaries as required.  Order is event_id asc.
        """
        win = self.window(side, entity_id)
        if win.ts.size == 0:
            return [], None
        lower = anchor_ts - window_ms
        # boundaries inclusive: lower <= ts <= anchor_ts
        lo = int(np.searchsorted(win.ts, lower, side="left"))
        hi = int(np.searchsorted(win.ts, anchor_ts, side="right"))
        ids = win.event_ids[lo:hi]
        if source_field is None:
            return ids, None
        values = win.values.get(source_field)
        if values is None:
            return ids, None
        return ids, values[lo:hi]
