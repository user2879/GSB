"""Deterministic behavior, compatibility, recovery and isolation tests."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from aedflow.boundary import classify_frames, decide_boundaries
from aedflow.config import CANONICAL_CLASSES, config_fingerprint, load_config
from aedflow.loader import load_shards
from aedflow.matching import evaluate
from aedflow.pipeline import GracefulExit, Pipeline
from aedflow.trajectory import build_trajectories, smooth_boundaries

WORKSPACE = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKSPACE / "config" / "config.yaml"


class AedFlowTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="aedflow-test-"))
        cls.workspace = cls.temp_dir / "ws"
        shutil.copytree(WORKSPACE / "samples", cls.workspace / "samples")
        shutil.copytree(WORKSPACE / "config", cls.workspace / "config")
        cls.config = load_config(cls.workspace / "config" / "config.yaml")
        with (cls.workspace / "config" / "config.yaml").open("r", encoding="utf-8") as handle:
            cls.manifest = yaml.safe_load(handle)
        with (cls.workspace / cls.config["paths"]["manifest"]).open("r", encoding="utf-8") as handle:
            cls.manifest = yaml.safe_load(handle)
        with (cls.workspace / cls.config["paths"]["truth"]).open("r", encoding="utf-8") as handle:
            cls.truth = yaml.safe_load(handle)["events"]
        cls.loaded = load_shards(
            cls.workspace / cls.config["paths"]["shard_dir"], cls.manifest, cls.config
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def stages(self, max_workers: int):
        config = json.loads(json.dumps(self.config))
        config["runtime"]["max_workers"] = max_workers
        boundary = decide_boundaries(
            self.loaded["loaded_shards"], self.loaded["global_grid"], config
        )
        smoothing = smooth_boundaries(
            self.loaded["global_grid"], boundary["decisions"], config
        )
        labels, distribution = classify_frames(
            smoothing["grid"], config["thresholds"]["frame_posterior"]
        )
        built = build_trajectories(
            labels,
            self.loaded["frame_ownership"],
            self.loaded["loaded_shards"],
            boundary["decisions"],
            smoothing["smoothing_records"],
            config,
        )
        evaluation = evaluate(built["trajectories"], self.truth, config)
        return boundary, smoothing, built, evaluation, distribution


class DeterminismTests(AedFlowTestBase):
    def test_serial_vs_concurrent_field_identical(self) -> None:
        serial = self.stages(1)
        for workers in (2, 4, 8):
            parallel = self.stages(workers)
            serial_boundary = [row.to_dict() for row in serial[0]["decisions"]]
            parallel_boundary = [row.to_dict() for row in parallel[0]["decisions"]]
            self.assertEqual(serial_boundary, parallel_boundary)
            np.testing.assert_array_equal(serial[1]["grid"], parallel[1]["grid"])
            self.assertEqual(
                [row.to_dict() for row in serial[2]["trajectories"]],
                [row.to_dict() for row in parallel[2]["trajectories"]],
            )
            self.assertEqual(serial[3], parallel[3])
            self.assertEqual(serial[4], parallel[4])

    def test_repeated_runs_byte_identical_report(self) -> None:
        first = Pipeline(self.workspace, self.config, {})
        first.run()
        first_report = first.write_report().read_bytes()
        second = Pipeline(self.workspace, self.config, {})
        second.run()
        second_report = second.write_report().read_bytes()
        self.assertEqual(first_report, second_report)


class CompatibilityTests(AedFlowTestBase):
    def test_semantic_codebook_mapping(self) -> None:
        by_id = {record["shard_id"]: record for record in self.loaded["compat_records"]}
        v1 = by_id["shard_02"]
        self.assertEqual(v1["codebook_version"], "codebook_v1")
        self.assertEqual(
            v1["semantic_mapping"], {"0": "alarm", "1": "music", "2": "noise", "3": "speech"}
        )
        self.assertEqual(v1["unmapped_codes"], [])

    def test_unmapped_code_flagged(self) -> None:
        codes = {failure["code"] for failure in self.loaded["failures"]}
        self.assertIn("E_CLASS_MAP_UNMAPPED", codes)
        by_id = {record["shard_id"]: record for record in self.loaded["compat_records"]}
        self.assertEqual(by_id["shard_03"]["unmapped_codes"], ["50"])

    def test_foreign_grid_resampled_with_residuals(self) -> None:
        by_id = {record["shard_id"]: record for record in self.loaded["compat_records"]}
        record = by_id["shard_04"]
        self.assertTrue(record["resampled"])
        self.assertEqual(set(record["resample_residuals"]), set(CANONICAL_CLASSES))
        self.assertTrue(all(value >= 0.0 for value in record["resample_residuals"].values()))

    def test_misaligned_shard_failure_and_residual(self) -> None:
        failure_codes = {(f["subject_id"], f["code"]) for f in self.loaded["failures"]}
        self.assertIn(("shard_bad_align", "E_FRAME_GRID_MISALIGN"), failure_codes)
        residual = self.loaded["alignment_residuals"][0]
        self.assertEqual(residual["shard_id"], "shard_bad_align")
        self.assertEqual(residual["start_remainder_ms"], 5)

    def test_range_and_length_failures_isolated(self) -> None:
        failure_codes = {(f["subject_id"], f["code"]) for f in self.loaded["failures"]}
        self.assertIn(("shard_bad_prob", "E_PROBABILITY_RANGE"), failure_codes)
        self.assertIn(("shard_bad_len", "E_SHARD_LENGTH_MISMATCH"), failure_codes)
        self.assertEqual(len(self.loaded["loaded_shards"]), 5)



    def test_duplicate_shard_id_rejected(self) -> None:
        manifest = json.loads(json.dumps(self.manifest))
        manifest["shards"] = list(manifest["shards"]) + [
            {
                "shard_id": "shard_01",
                "start_ms": 0,
                "end_ms": 100,
                "header": {
                    "frame_grid_version": "frame_grid_v1",
                    "codebook_version": "codebook_v2",
                },
            }
        ]
        result = load_shards(
            self.workspace / self.config["paths"]["shard_dir"], manifest, self.config
        )
        codes = {(f["subject_id"], f["code"]) for f in result["failures"]}
        self.assertIn(("shard_01", "E_SHARD_ID_DUPLICATE"), codes)

class ReconstructionTests(AedFlowTestBase):
    def test_cross_boundary_trajectories(self) -> None:
        boundary, _, built, evaluation, _ = self.stages(4)
        crossed = [t for t in built["trajectories"] if t.crossed_boundaries]
        boundary_ids = {decision.boundary_id for decision in boundary["decisions"] if decision.continuous}
        self.assertEqual(len(crossed), len(boundary_ids))
        for trajectory in crossed:
            self.assertTrue(trajectory.smoothing["applied"])
            self.assertEqual(trajectory.smoothing["method"], "linear_crossfade")
            self.assertEqual(trajectory.boundary_evidence[0]["continuous"], True)
        self.assertEqual(evaluation["matched_trajectory_count"], 5)
        self.assertEqual(evaluation["unmatched_truth_events"], ["evt_06"])

    def test_frame_tie_break_and_background(self) -> None:
        grid = np.array(
            [
                [0.2, 0.2, 0.2, 0.2],
                [0.1, 0.1, 0.1, 0.1],
                [0.7, 0.2, 0.2, 0.2],
            ]
        )
        labels, distribution = classify_frames(grid, 0.15)
        self.assertEqual(labels[0], "speech")  # descending name tie-break
        self.assertEqual(labels[1], "background")
        self.assertEqual(labels[2], "alarm")
        self.assertEqual(distribution["background"], 1)


class SnapshotTests(AedFlowTestBase):
    def test_graceful_exit_then_resume_matches_fresh(self) -> None:
        snapshot_dir = self.workspace / "output" / "snapshot"
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        partial = Pipeline(self.workspace, self.config, {"exit_after_stage": "boundary_decision"})
        with self.assertRaises(GracefulExit):
            partial.run()
        manifest = json.loads((snapshot_dir / "snapshot_manifest.json").read_text())
        self.assertEqual(manifest["stage"], "boundary_decision")

        resumed = Pipeline(self.workspace, self.config, {})
        resumed.run()
        resumed_report = resumed.write_report().read_bytes()

        shutil.rmtree(snapshot_dir, ignore_errors=True)
        fresh = Pipeline(self.workspace, self.config, {})
        fresh.run()
        fresh_report = fresh.write_report().read_bytes()
        self.assertEqual(resumed_report, fresh_report)

    def test_stale_snapshot_rejected(self) -> None:
        from aedflow.errors import AedError
        from aedflow.snapshot import SnapshotStore

        pipeline = Pipeline(self.workspace, self.config, {})
        pipeline.run()
        identity = {
            "config_fingerprint": config_fingerprint(self.config),
            "header_version_set": ["frame_grid_v1|codebook_v3"],
            "global_version": 1,
        }
        store = SnapshotStore(snapshot_dir := self.workspace / "output" / "snapshot", identity)
        with self.assertRaises(AedError) as ctx:
            store.load_manifest()
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT_STALE")


class IsolationTests(AedFlowTestBase):
    def test_boundary_task_exception_isolated(self) -> None:
        boundary = decide_boundaries(
            self.loaded["loaded_shards"],
            self.loaded["global_grid"],
            self.config,
            inject_failure_boundary="b00_00800ms",
        )
        self.assertEqual(len(boundary["failures"]), 1)
        failure = boundary["failures"][0]
        self.assertEqual(failure["subject_id"], "b00_00800ms")
        self.assertEqual(failure["code"], "E_TASK_FAILURE")
        self.assertEqual(failure["exception_type"], "RuntimeError")
        self.assertEqual(len(boundary["decisions"]), 3)

    def test_smoothing_task_exception_isolated(self) -> None:
        boundary = decide_boundaries(
            self.loaded["loaded_shards"], self.loaded["global_grid"], self.config
        )
        smoothing = smooth_boundaries(
            self.loaded["global_grid"],
            boundary["decisions"],
            self.config,
            inject_failure_trajectory="b00_00800ms",
        )
        self.assertEqual(smoothing["failures"][0]["code"], "E_TASK_FAILURE")
        self.assertEqual(len(smoothing["smoothing_records"]), 2)


if __name__ == "__main__":
    unittest.main()
