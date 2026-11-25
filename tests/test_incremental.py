from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from mgirecon.cli import canonical_result
from mgirecon.commands import apply_commands
from mgirecon.config import config_hash, load_config, save_config
from mgirecon.engine import reconstruct
from mgirecon.errors import E_RECON_BUDGET_EXCEEDED
from mgirecon.snapshot import load_snapshot, write_snapshot

from tests.helpers import make_project


class IncrementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary, cls.root, cls.config = make_project("mgirecon-incremental-")
        cls.baseline = reconstruct(cls.root, cls.config, previous_state={})

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _incremental_and_full(self, root: Path, config: dict, previous: dict) -> tuple[dict, dict]:
        incremental = reconstruct(root, config, previous_state=previous)
        full = reconstruct(root, config, previous_state={})
        self.assertEqual(canonical_result(incremental), canonical_result(full))
        return incremental, full

    def test_replace_view_propagates_content_point_depth_and_voxel(self) -> None:
        temporary, root, config = make_project("mgirecon-replace-")
        try:
            baseline = reconstruct(root, config, previous_state={})
            updated, errors = apply_commands(root, config, [{"op": "replace_view", "view": "v02", "render_seed": 7777}])
            self.assertEqual(errors, [])
            incremental, _ = self._incremental_and_full(root, updated, baseline)
            invalidated = incremental["ledger"]["invalidated"]
            self.assertIn("feature", {entry["stage"] for entry in invalidated})
            self.assertIn("triangulated_point", {entry["stage"] for entry in invalidated})
            self.assertIn("depth_map", {entry["stage"] for entry in invalidated})
            self.assertIn("voxel", {entry["stage"] for entry in invalidated})
            relations = set()
            for entry in invalidated:
                relations.update(entry["relations"])
            self.assertIn("point", relations)
            self.assertIn("depth", relations)
            self.assertIn("hierarchy", relations)
        finally:
            temporary.cleanup()

    def test_extrinsics_and_threshold_reuse_unaffected_voxels(self) -> None:
        temporary, root, config = make_project("mgirecon-extrinsics-")
        try:
            baseline = reconstruct(root, config, previous_state={})
            changed = copy.deepcopy(config)
            changed["views"][2]["extrinsics"] = [[1, 0, 0, 12], [0, 1, 0, 6], [0, 0, 1, 196]]
            save_config(root, changed)
            first, _ = self._incremental_and_full(root, changed, baseline)
            self.assertIn("camera", {relation for entry in first["ledger"]["invalidated"] for relation in entry["relations"]})
            self.assertTrue(first["ledger"]["reused"])
            self.assertTrue(all("basis" in entry for entry in first["ledger"]["reused"]))
            changed_threshold = copy.deepcopy(changed)
            changed_threshold["reconstruction"]["residual_threshold"] = 9
            second, _ = self._incremental_and_full(root, changed_threshold, first)
            self.assertIn("hierarchy", {relation for entry in second["ledger"]["invalidated"] for relation in entry["relations"]})
        finally:
            temporary.cleanup()

    def test_snapshot_resume_is_identical_and_cache_hits(self) -> None:
        temporary, root, config = make_project("mgirecon-snapshot-")
        try:
            first = reconstruct(root, config, previous_state={})
            manifest = write_snapshot(root, first)
            snapshot = load_snapshot(root, config_hash(config))
            self.assertEqual(snapshot["resume_mode"], "resume")
            resumed = reconstruct(root, config, previous_state=snapshot["state"])
            self.assertEqual(canonical_result(first), canonical_result(resumed))
            self.assertGreater(len(resumed["ledger"]["reused"]), 0)
            self.assertEqual(resumed["ledger"]["invalidated"], [])
            self.assertEqual(snapshot["manifest"]["budget_consumed"], first["budget"]["global_consumed"])
        finally:
            temporary.cleanup()

    def test_budget_is_stable_and_retains_pending(self) -> None:
        limited = copy.deepcopy(self.config)
        limited["scheduler"]["global_point_budget"] = 3
        limited["scheduler"]["per_view_point_quota"] = 100
        first = reconstruct(self.root, limited, previous_state={})
        second = reconstruct(self.root, limited, previous_state={})
        self.assertIn(E_RECON_BUDGET_EXCEEDED, [error["code"] for error in first["errors"]])
        self.assertTrue(first["pending"])
        accepted = [point["stable_key"] for point in first["points"] if point["budget_accepted"]]
        self.assertEqual(len(accepted), 3)
        self.assertEqual(accepted, [point["stable_key"] for point in second["points"] if point["budget_accepted"]])
        self.assertEqual(first["pending"], second["pending"])
        full = reconstruct(self.root, self.config, previous_state={})
        unlimited_keys = [point["stable_key"] for point in full["points"] if point["residual_pass"] and point["budget_accepted"]]
        self.assertEqual(accepted, unlimited_keys[:3])

        quota_limited = copy.deepcopy(self.config)
        quota_limited["scheduler"]["global_point_budget"] = 100
        quota_limited["scheduler"]["per_view_point_quota"] = 1
        quota_result = reconstruct(self.root, quota_limited, previous_state={})
        self.assertTrue(quota_result["budget"]["per_view_consumed"])
        self.assertTrue(all(value <= 1 for value in quota_result["budget"]["per_view_consumed"].values()))
        self.assertTrue(any(item["reason"] == "per_view_point_quota" for item in quota_result["pending"]))

    def test_concurrency_is_deterministic_and_bounded(self) -> None:
        serial_config = copy.deepcopy(self.config)
        serial_config["scheduler"]["max_workers"] = 1
        parallel_config = copy.deepcopy(self.config)
        parallel_config["scheduler"]["max_workers"] = 4
        serial = reconstruct(self.root, serial_config, previous_state={})
        parallel = reconstruct(self.root, parallel_config, previous_state={})
        self.assertEqual(canonical_result(serial), canonical_result(parallel))
        for stats in (serial["scheduler"], parallel["scheduler"]):
            self.assertLessEqual(stats["max_active_observed"], stats["max_workers"])
            for stage in ("features", "matching", "voxels"):
                self.assertLessEqual(stats[stage]["max_active"], stats["max_workers"])

    def test_incremental_full_equality_with_budget_and_parallelism(self) -> None:
        temporary, root, config = make_project("mgirecon-budget-incremental-")
        try:
            constrained = copy.deepcopy(config)
            constrained["scheduler"]["max_workers"] = 4
            constrained["scheduler"]["global_point_budget"] = 3
            constrained["scheduler"]["per_view_point_quota"] = 2
            baseline = reconstruct(root, constrained, previous_state={})
            changed, errors = apply_commands(
                root,
                constrained,
                [{"op": "replace_view", "view": "v02", "render_seed": 4242}],
            )
            self.assertEqual(errors, [])
            incremental, full = self._incremental_and_full(root, changed, baseline)
            self.assertIn(E_RECON_BUDGET_EXCEEDED, [error["code"] for error in incremental["errors"]])
            self.assertEqual(incremental["pending"], full["pending"])
            self.assertEqual(
                [point["stable_key"] for point in incremental["points"] if point["budget_accepted"]],
                [point["stable_key"] for point in full["points"] if point["budget_accepted"]],
            )
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
