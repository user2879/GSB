"""Diagnostic service boundary: recomputation happens here, never in the view."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .pipeline import PipelineResult, PipelineRunner
from .snapshot import DiagnosticSnapshot, SnapshotStore, make_snapshot, save_snapshot


class DiagnosticService:
    def __init__(self, runner: PipelineRunner, store: Optional[SnapshotStore] = None):
        self.runner = runner
        self.store = store or SnapshotStore()
        self.last_result: Optional[PipelineResult] = None
        self.last_snapshot: Optional[DiagnosticSnapshot] = None

    def recompute(
        self,
        version: int,
        output_dir: str | Path,
        *,
        verify_closure: bool = False,
    ) -> PipelineResult:
        result = self.runner.run(output_dir, verify_closure=verify_closure)
        self.last_result = result
        return result

    def build_and_publish_snapshot(
        self,
        version: int,
        snapshot_dir: str | Path,
    ) -> DiagnosticSnapshot:
        if self.last_result is None:
            raise RuntimeError("recompute must be called before snapshot publication")
        snapshot = make_snapshot(self.last_result, version)
        save_snapshot(snapshot, snapshot_dir)
        self.store.publish(snapshot)
        self.last_snapshot = snapshot
        return snapshot
