"""End-to-end deterministic application pipeline and report generation."""
from __future__ import annotations

import os

import numpy as np

from . import config
from .binary_format import read_array
from .errors import UndoEmptyError
from .framing import FrameBudgeter
from .hashing import read_json, write_canonical_json
from .ledger import Ledger
from .service import ReconstructionService, build_initial_snapshot, tile_hash
from .state import Snapshot

INTERACTION_CYCLE = ("idle", "box_select", "drag", "param_change", "idle")


class Application:
    def __init__(self, workspace: str, *, verify: bool = True):
        self.workspace = workspace
        self.data_dir = os.path.join(workspace, "data")
        self.report_dir = os.path.join(workspace, "reports")
        os.makedirs(self.report_dir, exist_ok=True)
        self.ledger = Ledger(workspace)
        self.recovery = self.ledger.recover()
        self.ledger.reset_to_initial()
        self.verify = verify
        self._version_stack: list[Snapshot] = []
        self._load_sample()
        self.service = ReconstructionService(self.initial_snapshot)
        self.budgeter = FrameBudgeter(config.FRAME_BUDGET_MS, deterministic=True)
        self.incremental_reports: list[dict] = []

    def _load_sample(self) -> None:
        points = read_array(os.path.join(self.data_dir, config.POINTS_FILE), 3)
        normals = read_array(os.path.join(self.data_dir, config.NORMALS_FILE), 4)
        if points.shape[0] != normals.shape[0]:
            from .errors import PointCloudFormatError

            raise PointCloudFormatError("points/normals record count mismatch")
        self.points = points
        self.normals = normals
        self.initial_snapshot = build_initial_snapshot(points, normals)
        self.manifest = read_json(os.path.join(self.data_dir, config.MANIFEST_FILE))
        self.edits_doc = read_json(os.path.join(self.data_dir, config.EDITS_FILE))

    def run_edit_sequence(self) -> dict:
        initial_hashes = self._snapshot_hashes(self.initial_snapshot)
        self.ledger.append_commit(0, {"type": "initial"}, initial_hashes, {"type": "initial"})
        self._version_stack.append(self.initial_snapshot)

        for edit in self.edits_doc["edits"]:
            base = self.service.snapshot.version
            result = self.service.apply_edit(edit, base_version=base, verify=self.verify)
            snap = result.snapshot
            hashes = self._snapshot_hashes(snap)
            self.ledger.append_commit(snap.version, edit, hashes, result.audit)
            self._version_stack.append(snap)
            self._plan_interaction_frames(edit, snap, result)
            self.incremental_reports.append(
                {
                    "op_id": edit.get("op_id"),
                    "edit_type": edit["type"],
                    "version": snap.version,
                    "invalidated_closure": [list(t) for t in result.invalidated],
                    "rebuilt_tiles": [list(t) for t in result.rebuilt],
                    "reused_tile_count": len(result.reused),
                    "closure_equals_rebuilt": set(result.invalidated) == set(result.rebuilt),
                    "consistency": result.consistency,
                    "audit": result.audit,
                    "hashes": hashes,
                }
            )
        return self._summarize()

    def _plan_interaction_frames(self, edit: dict, snap: Snapshot, result) -> None:
        # Three normal interaction frames per edit, using a fixed interaction
        # cycle, followed by one deterministic high-load stress frame that
        # exercises the full degradation ladder (defer patch refresh / skip).
        active_patches = len(result.rebuilt) or len(snap.tiles)
        for step in range(3):
            interaction = INTERACTION_CYCLE[(edit.get("op_id", 0) + step) % len(INTERACTION_CYCLE)]
            self.budgeter.plan_frame(
                interaction=interaction,
                live_points=snap.points.shape[0],
                active_patches=active_patches,
                edit_version=snap.version,
            )
        self.budgeter.plan_frame(
            interaction="box_select_stress",
            live_points=snap.points.shape[0] * 12,
            active_patches=len(snap.tiles),
            edit_version=snap.version,
        )

    def undo_last(self) -> Snapshot:
        if len(self._version_stack) < 2:
            raise UndoEmptyError("no committed edit to undo")
        current = self._version_stack.pop()
        target = self._version_stack[-1]
        restored = Snapshot(
            version=current.version + 1,
            points=target.points,
            normals=target.normals,
            params=target.params,
            tiles=target.tiles,
            merged=target.merged,
            tile_hashes=target.tile_hashes,
            op_history=target.op_history,
            edit_audit=target.edit_audit,
        )
        hashes = self._snapshot_hashes(restored)
        self.ledger.append_undo(restored.version, target.version, hashes)
        self.service._snapshot = restored  # service lock not needed in headless run
        return restored

    def _snapshot_hashes(self, snap: Snapshot) -> dict:
        return {
            "version": snap.version,
            "points_hash": snap.points_hash,
            "normals_hash": snap.normals_hash,
            "mesh_hash": snap.mesh_hash,
            "content_hash": snap.content_hash(),
            "tile_count": len(snap.tile_hashes),
        }

    def _summarize(self) -> dict:
        final = self.service.snapshot
        frame_report = self.budgeter.report()
        incremental_report = {
            "format": "pcrecon-incremental-report",
            "format_version": 1,
            "seed": self.manifest["seed"],
            "initial_point_count": int(self.points.shape[0]),
            "final_point_count": int(final.points.shape[0]),
            "edit_count": len(self.edits_doc["edits"]),
            "all_consistent": all(
                (r["consistency"] or {}).get("status") == "match"
                for r in self.incremental_reports
                if r["consistency"] is not None
            ),
            "edits": self.incremental_reports,
            "final_hashes": self._snapshot_hashes(final),
        }
        frame_doc = {
            "format": "pcrecon-frame-report",
            "format_version": 1,
            "budget_ms": config.FRAME_BUDGET_MS,
            "clock": "deterministic",
            **frame_report,
        }
        inc_path = os.path.join(self.report_dir, "incremental_report.json")
        frame_path = os.path.join(self.report_dir, "frame_report.json")
        inc_hash = write_canonical_json(inc_path, incremental_report)
        frame_hash = write_canonical_json(frame_path, frame_doc)
        return {
            "incremental_report": inc_path,
            "frame_report": frame_path,
            "incremental_report_hash": inc_hash,
            "frame_report_hash": frame_hash,
            "incremental": incremental_report,
            "frames": frame_doc,
        }

    def write_clean_exit_report(self) -> dict:
        head = self.service.snapshot
        report = {
            "format": "pcrecon-exit-report",
            "format_version": 1,
            "clean_exit": True,
            "head_version": head.version,
            "head_content_hash": head.content_hash(),
            "commits_in_ledger": len([c for c in self.ledger.read_commits() if c.get("record") == "commit"]),
        }
        self.ledger.write_exit_report(report)
        return report
