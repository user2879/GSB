"""PyQt5 viewport over immutable snapshots.

The viewport only ever reads the snapshot published by the worker.  It never
mutates point sets or meshes; degradation decisions affect draw parameters
alone.  This module is imported lazily so headless runs do not require a
display.
"""
from __future__ import annotations

from .framing import FrameBudgeter


class ViewportController:
    """GL-widget-agnostic controller keeping the UI thread non-blocking."""

    def __init__(self, worker, budget_ms: float, *, deterministic: bool = True):
        self.worker = worker
        self.budgeter = FrameBudgeter(budget_ms, deterministic=deterministic)
        self._displayed_version = worker.service.snapshot.version
        self._pending_base_version = self._displayed_version

    def request_edit(self, edit: dict, *, verify: bool = True) -> None:
        # Non-blocking: hand off to the worker thread at the current head.
        head = self.worker.service.snapshot.version
        self._pending_base_version = head
        self.worker.submit(edit, head, verify)

    def pump(self, interaction: str, active_patches: int, outcome=None) -> dict:
        """Called once per UI frame; applies the budget ladder."""
        if outcome is None:
            outcome = self.worker.poll_result()
        if outcome and outcome[0] == "ok":
            # Atomic whole-snapshot swap identified by version + hash.
            new_snapshot = outcome[1].snapshot
            if new_snapshot.version > self._displayed_version:
                self._displayed_version = new_snapshot.version
        snap = self.worker.service.snapshot
        record = self.budgeter.plan_frame(
            interaction=interaction,
            live_points=snap.points.shape[0],
            active_patches=active_patches,
            edit_version=self._displayed_version,
        )
        return record.to_dict()

    @property
    def displayed_version(self) -> int:
        return self._displayed_version
