"""Snapshot durability, resume, tamper rejection and versioning tests."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from level_solver.errors import E_DIGEST, SolverError
from level_solver.parser import parse_spec
from level_solver.snapshot import SnapshotStore, build_version_id, parse_snapshot_file
from level_solver.solver import LevelSolver


def spec_dict(**overrides):
    spec = {
        "width": 14, "height": 12, "seed": 555,
        "blocked": [[0, 0], [13, 11]],
        "concurrency": 3, "memory_budget": 6, "max_rollback": 10,
        "alternative_limit": 24,
        "constraints": [
            {"id": "rooms", "type": "min_rooms",
             "params": {"count": 4, "min_room_w": 2, "min_room_h": 2,
                        "max_room_w": 4, "max_room_h": 4}},
            {"id": "spacing", "type": "room_spacing", "params": {"min_distance": 1}},
            {"id": "cap", "type": "max_region_area", "params": {"max_area": 100}},
            {"id": "joined", "type": "connectivity", "params": {}},
            {"id": "reach", "type": "reachability",
             "params": {"start": [1, 1], "end": [12, 10]}},
        ],
    }
    spec.update(overrides)
    return spec


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lvl-snap-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dir(self, name):
        return str(Path(self.tmp) / name)

    def test_snapshot_written_per_committed_constraint(self):
        spec = parse_spec(spec_dict())
        LevelSolver(spec, self._dir("a")).solve()
        store = SnapshotStore(self._dir("a"), build_version_id(spec))
        self.assertEqual(len(store.list_snapshot_paths()), len(spec.constraints) + 1)

    def test_interrupt_and_resume_matches_uninterrupted(self):
        raw = spec_dict()
        full = LevelSolver(parse_spec(raw), self._dir("full")).solve()

        snapshot_dir = self._dir("split")
        first = LevelSolver(parse_spec(raw), snapshot_dir)
        partial = first.solve(interrupt_after=3)
        self.assertTrue(partial.interrupted)
        self.assertTrue(os.path.exists(os.path.join(snapshot_dir, "current", "interrupted.json")))

        second = LevelSolver(parse_spec(raw), snapshot_dir)
        second.prepare()
        self.assertTrue(second.resumed)
        resumed = second.solve()
        self.assertEqual(resumed.grid.to_list(), full.grid.to_list())
        self.assertEqual(resumed.constraint_counts, full.constraint_counts)
        self.assertEqual([f.attempts for f in resumed.applied], [f.attempts for f in full.applied])
        self.assertFalse(os.path.exists(os.path.join(snapshot_dir, "current", "interrupted.json")))

    def test_resume_at_every_boundary_matches(self):
        raw = spec_dict()
        full = LevelSolver(parse_spec(raw), self._dir("reference")).solve()
        total = len(raw["constraints"])
        for boundary in range(1, total):
            directory = self._dir(f"boundary-{boundary}")
            first = LevelSolver(parse_spec(raw), directory)
            first.solve(interrupt_after=boundary)
            second = LevelSolver(parse_spec(raw), directory)
            second.prepare()
            result = second.solve()
            self.assertEqual(result.grid.to_list(), full.grid.to_list(), f"boundary {boundary}")
            self.assertEqual(result.constraint_counts, full.constraint_counts)

    def test_tampered_snapshot_rejected_with_digest_code(self):
        raw = spec_dict()
        directory = self._dir("tamper")
        first = LevelSolver(parse_spec(raw), directory)
        first.solve(interrupt_after=2)
        target = sorted(
            Path(directory, "current").glob("snapshot_*.json")
        )[-1]
        document = json.loads(target.read_text(encoding="utf-8"))
        document["content"]["grid"][0][0] = 2
        target.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(SolverError) as ctx:
            parse_snapshot_file(str(target))
        self.assertEqual(ctx.exception.code, E_DIGEST)

    def test_resume_falls_back_to_older_valid_snapshot(self):
        raw = spec_dict()
        directory = self._dir("fallback")
        first = LevelSolver(parse_spec(raw), directory)
        first.solve(interrupt_after=3)
        snapshots = sorted(Path(directory, "current").glob("snapshot_*.json"))
        newest = snapshots[-1]
        document = json.loads(newest.read_text(encoding="utf-8"))
        document["digest"] = "0" * 64
        newest.write_text(json.dumps(document), encoding="utf-8")

        solver = LevelSolver(parse_spec(raw), directory)
        solver.prepare()
        self.assertTrue(solver.resumed)
        self.assertEqual(solver.sequence, 2)  # newest intact snapshot
        self.assertTrue(any(w["code"] == E_DIGEST for w in solver.digest_warnings))
        full = LevelSolver(parse_spec(raw), self._dir("reference")).solve()
        result = solver.solve()
        self.assertEqual(result.grid.to_list(), full.grid.to_list())

    def test_missing_digest_field_invalidates_snapshot(self):
        raw = spec_dict()
        directory = self._dir("missing-digest")
        LevelSolver(parse_spec(raw), directory).solve(interrupt_after=1)
        target = sorted(Path(directory, "current").glob("snapshot_*.json"))[-1]
        document = json.loads(target.read_text(encoding="utf-8"))
        del document["digest"]
        target.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(SolverError) as ctx:
            parse_snapshot_file(str(target))
        self.assertEqual(ctx.exception.code, E_DIGEST)

    def test_version_change_archives_old_snapshots(self):
        raw = spec_dict()
        directory = self._dir("versioned")
        LevelSolver(parse_spec(raw), directory).solve(interrupt_after=2)
        changed = spec_dict(seed=raw["seed"] + 1)
        store = SnapshotStore(directory, build_version_id(parse_spec(changed)))
        store.initialize()
        self.assertTrue(Path(directory, "archive").exists())
        archives = list(Path(directory, "archive").iterdir())
        self.assertEqual(len(archives), 1)
        self.assertFalse(list(Path(directory, "current").glob("snapshot_*.json")))
        index = json.loads(Path(directory, "index.json").read_text())
        self.assertEqual(index["current_version"], build_version_id(parse_spec(changed)))

    def test_foreign_version_interrupt_marker_restarts_from_scratch(self):
        raw = spec_dict()
        directory = self._dir("foreign")
        first = LevelSolver(parse_spec(raw), directory)
        first.solve(interrupt_after=2)
        changed = parse_spec(spec_dict(seed=raw["seed"] + 1))
        solver = LevelSolver(changed, directory)
        solver.prepare()
        self.assertFalse(solver.resumed)
        self.assertEqual(solver.sequence, 0)

    def test_verify_command_reports_tampering(self):
        raw = spec_dict()
        directory = self._dir("verify")
        spec = parse_spec(raw)
        LevelSolver(spec, directory).solve(interrupt_after=2)
        store = SnapshotStore(directory, build_version_id(spec))
        target = sorted(Path(directory, "current").glob("snapshot_*.json"))[1]
        document = json.loads(target.read_text(encoding="utf-8"))
        document["content"]["rooms"] = []
        target.write_text(json.dumps(document), encoding="utf-8")
        reports = store.verify_all()
        self.assertFalse(all(report["ok"] for report in reports))
        self.assertTrue(any(report.get("code") == E_DIGEST for report in reports))


if __name__ == "__main__":
    unittest.main()
