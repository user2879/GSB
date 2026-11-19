"""Interaction controller: drag previews, deterministic rejection, commit.

A drag session is the *only* place where candidate state exists. The committed
items are never mutated while dragging; ``cancel_drag`` restores the exact
pre-drag state and ``commit_drag`` solves a fresh layout and appends an
immutable snapshot. Every rejection returns one of the declared business
codes and leaves the committed state untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ErrorRecord, StaleVersionError, WorkbenchError
from .layout import LayoutConfig, solve_layout, validate_drag
from .models import SubtitleItem
from .snapshot import SnapshotStore


@dataclass(frozen=True)
class DragPreview:
    item_id: str
    candidate: SubtitleItem
    target_lane: int
    conflict_item_id: str | None
    error: ErrorRecord | None

    @property
    def accepted(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "candidate": self.candidate.to_dict(),
            "target_lane": self.target_lane,
            "conflict_item_id": self.conflict_item_id,
            "error": None if self.error is None else self.error.to_dict(),
        }


class WorkbenchController:
    def __init__(self, store: SnapshotStore,
                 config: LayoutConfig | None = None):
        self.store = store
        self.config = config or LayoutConfig()
        self._session: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    @property
    def committed_items(self) -> tuple[SubtitleItem, ...]:
        current = self.store.load_current()
        return tuple(current.items) if current else ()

    @property
    def current_version(self) -> int:
        current = self.store.load_current()
        return current.version if current else 0

    def begin_drag(self, item_id: str) -> DragPreview:
        items = self.committed_items
        original = self._find(item_id, items)
        current = self.store.load_current()
        lane = next((cue.lane_index for cue in current.layout.cues
                     if cue.item.item_id == item_id), 0)
        self._session = {
            "item_id": item_id,
            "original": original,
            "base_version": self.current_version,
            "last_preview": None,
        }
        return DragPreview(item_id, original, lane, None, None)

    def preview_drag(self, item_id: str, target_lane: int,
                     new_start: int, new_end: int) -> DragPreview:
        if self._session is None or self._session["item_id"] != item_id:
            raise WorkbenchError("no active drag session for " + item_id)
        original: SubtitleItem = self._session["original"]
        candidate = SubtitleItem(item_id, new_start, new_end, original.text,
                                 original.speaker, original.language,
                                 original.priority, original.is_ad)
        others = [it for it in self.committed_items if it.item_id != item_id]
        error: ErrorRecord | None = None
        conflict_id: str | None = None
        try:
            validate_drag(original, target_lane, new_start, new_end,
                          others, self.config)
        except WorkbenchError as exc:
            error = ErrorRecord(exc.code, str(exc), exc.conflict_item_id)
            conflict_id = exc.conflict_item_id
        preview = DragPreview(item_id, candidate, target_lane,
                              conflict_id, error)
        self._session["last_preview"] = preview
        return preview

    def commit_drag(self) -> tuple[DragPreview, Any]:
        """Solve and snapshot the accepted preview; reject otherwise."""
        if self._session is None:
            raise WorkbenchError("no active drag session")
        preview = self._session["last_preview"]
        if preview is None or not preview.accepted:
            return preview, None
        items = list(self.committed_items)
        for index, item in enumerate(items):
            if item.item_id == preview.item_id:
                items[index] = preview.candidate
                break
        layout = solve_layout(tuple(items), self.config)
        snapshot = self.store.commit(
            tuple(items), layout,
            expected_previous_version=self._session["base_version"])
        self._session = None
        return preview, snapshot

    def cancel_drag(self) -> SubtitleItem:
        """Discard every candidate preview and restore the committed item."""
        if self._session is None:
            raise WorkbenchError("no active drag session")
        original = self._session["original"]
        self._session = None
        restored = self._find(original.item_id, self.committed_items)
        if restored != original:
            raise WorkbenchError("cancel failed to restore committed state")
        return restored

    def undo(self, target_version: int) -> Any:
        """Restore an earlier immutable snapshot by version pointer update."""
        snapshot = self.store.load_version(target_version)
        if snapshot is None:
            raise StaleVersionError(f"version {target_version} does not exist")
        return self.store.restore(snapshot)

    # ------------------------------------------------------------------
    @staticmethod
    def _find(item_id: str, items) -> SubtitleItem:
        for item in items:
            if item.item_id == item_id:
                return item
        raise WorkbenchError("unknown item " + item_id)
