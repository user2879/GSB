from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mdr.acceptance import setup_workspace
from mdr.commands import apply_commands
from mdr.depth import build_depth_map
from mdr.pipeline import Pipeline, canonical_fingerprint
from mdr.triangulation import triangulate_match
from mdr.utils import fingerprint, quantize_fixed, round_div
from mdr.validation import validate_dataset
from mdr.voxels import build_spatial_refs, run_voxel_stage


class FixedPointTests(unittest.TestCase):
    def test_round_div_ties_toward_zero(self) -> None:
        self.assertEqual(round_div(5, 2), (2, "down"))
        self.assertEqual(round_div(-5, 2), (-2, "up"))
        self.assertEqual(round_div(4, 2), (2, "exact"))

    def test_quantize_direction(self) -> None:
        self.assertEqual(quantize_fixed(9, 2), (8, "down"))
        self.assertEqual(quantize_fixed(10, 2), (8, "down"))
        self.assertEqual(quantize_fixed(15, 2), (16, "up"))
        self.assertEqual(quantize_fixed(14, 2), (12, "down"))


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="mdr-test-")
        self.root = Path(self.temp.name)
        setup_workspace(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_sample_has_no_input_errors(self) -> None:
        dataset = json.loads((self.root / "samples" / "dataset.json").read_text())
        config = json.loads((self.root / "configs" / "reconstruction.json").read_text())
        result = validate_dataset(dataset, self.root / "samples", config)
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.views), 4)
        self.assertGreaterEqual(len(result.pairs), 3)

    def test_bad_png_and_unknown_pair_are_skipped(self) -> None:
        from PIL import Image

        bad_image = self.root / "samples" / "views" / "bad-size.png"
        Image.new("RGB", (8, 8), (10, 20, 30)).save(bad_image)
        dataset = json.loads((self.root / "samples" / "dataset.json").read_text())
        config = json.loads((self.root / "configs" / "reconstruction.json").read_text())
        dataset["views"].append({"view_id": "bad", "sequence": 99, "image": "views/bad-size.png"})
        dataset["pairs"].append(["v0", "ghost"])
        result = validate_dataset(dataset, self.root / "samples", config)
        self.assertIn("v0", result.valid_view_ids)
        self.assertNotIn("bad", result.valid_view_ids)
        self.assertTrue(any(e["code"] == "E_INPUT_INVALID" for e in result.errors))
        self.assertFalse(any("ghost" in pair for pair in result.pairs))

    def test_serial_parallel_and_cached_runs_match(self) -> None:
        serial = Pipeline(self.root, parallel=False).run()
        serial_fp = canonical_fingerprint(serial)

        parallel = Pipeline(self.root, parallel=True).run()
        self.assertEqual(canonical_fingerprint(parallel), serial_fp)
        decisions = [event["decision"] for event in parallel["cache_events"] if event["stage"] == "feature"]
        self.assertTrue(decisions)
        self.assertTrue(all(decision == "hit" for decision in decisions))

    def test_invalid_commands_do_not_abort_stream(self) -> None:
        result = apply_commands(
            self.root,
            [
                {"op": "replace_view", "view_id": "missing", "variant": 1},
                {"op": "set_param", "key": "min_parallax", "value": -1},
                {"op": "replace_view", "view_id": "v1", "variant": 2},
            ],
        )
        self.assertEqual(len(result["errors"]), 2)
        state = Pipeline(self.root, command_errors=result["errors"]).run()
        self.assertEqual(state["business_code_counts"]["E_INPUT_INVALID"], 2)
        self.assertIn("v1", state["features"])

    def test_small_budget_serial_parallel_match(self) -> None:
        config_path = self.root / "configs" / "reconstruction.json"
        config = json.loads(config_path.read_text())
        config["budgets"]["global_point_budget"] = 4
        config["budgets"]["max_subdivisions_per_batch"] = 1
        config_path.write_text(json.dumps(config, sort_keys=True))

        serial_root = self.root / "serial"
        parallel_root = self.root / "parallel"
        setup_workspace(serial_root, config)
        setup_workspace(parallel_root, {**config, "budgets": {**config["budgets"], "max_workers": 4}})
        serial_state = Pipeline(serial_root, parallel=False).run()
        parallel_state = Pipeline(parallel_root, parallel=True).run()
        self.assertGreater(len(serial_state["pending"]), 0)
        self.assertEqual(canonical_fingerprint(serial_state), canonical_fingerprint(parallel_state))

    def test_depth_half_step_conflict_is_marked(self) -> None:
        config = json.loads((self.root / "configs" / "reconstruction.json").read_text())
        config["depth_radius"] = 1
        scale = 1 << int(config["frac_bits"])
        camera = {
            "K": [[108, 0, 48], [0, 108, 36], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [0, 0, 0],
        }
        points = [
            {
                "point_id": "pa",
                "cache_key": "ka",
                "raw_point_q": [0, 0, 60 * scale],
                "observations": {"v0": [48, 36]},
            },
            {
                "point_id": "pb",
                "cache_key": "kb",
                "raw_point_q": [0, 0, 60 * scale + (1 << int(config["quant_shifts"][config["quant_level"]])) // 2],
                "observations": {"v0": [48, 36]},
            },
        ]
        depth_map = build_depth_map("v0", {"view_id": "v0"}, points, {"v0": camera}, config)
        self.assertGreaterEqual(depth_map["conflict_count"], 1)
        center = next(row for row in depth_map["pixels"] if row[0] == 48 and row[1] == 36)
        self.assertEqual(center[4], 1)

    def test_min_parallax_rejection_has_reason(self) -> None:
        config = json.loads((self.root / "configs" / "reconstruction.json").read_text())
        config["min_parallax"] = 4
        camera = {
            "K": [[108, 0, 48], [0, 108, 36], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [0, 0, 0],
        }
        match = {
            "match_id": "m",
            "match_cache_key": "k",
            "left_view": "v0",
            "right_view": "v1",
            "left_xy": [48, 36],
            "right_xy": [49, 36],
        }
        record = triangulate_match(match, {"v0": camera, "v1": camera}, config)
        self.assertEqual(record["status"], "rejected")
        self.assertEqual(record["reject_reason"], "MIN_PARALLAX")

    def test_voxel_budget_sweep_matches_serial(self) -> None:
        state = Pipeline(self.root, parallel=False).run()
        dataset = json.loads((self.root / "samples" / "dataset.json").read_text())
        config = json.loads((self.root / "configs" / "reconstruction.json").read_text())
        validated = validate_dataset(dataset, self.root / "samples", config)
        spatial = build_spatial_refs(
            state["triangulation"],
            state["depth_maps"],
            validated.views,
            validated.cameras,
            config,
        )
        for budget in range(1, 8):
            serial = run_voxel_stage(spatial, config, max_subdivisions=budget, max_workers=1, parallel=False)
            parallel = run_voxel_stage(spatial, config, max_subdivisions=budget, max_workers=4, parallel=True)
            self.assertEqual(fingerprint(serial), fingerprint(parallel), f"budget {budget} differs")

    def test_voxel_pending_is_not_cached_as_final(self) -> None:
        config_path = self.root / "configs" / "reconstruction.json"
        config = json.loads(config_path.read_text())
        config["budgets"]["max_subdivisions_per_batch"] = 1
        config_path.write_text(json.dumps(config, sort_keys=True))
        first = Pipeline(self.root, parallel=False).run()
        self.assertGreaterEqual(first["voxels"]["budget"]["pending_count"], 1)

        apply_commands(self.root, [{"op": "set_param", "key": "max_subdivisions_per_batch", "value": 10000}])
        resumed = Pipeline(self.root, parallel=False).run()
        self.assertEqual(resumed["voxels"]["budget"]["pending_count"], 0)
        self.assertFalse(any(node["status"] == "pending" for node in resumed["voxels"]["nodes"].values()))

        full_temp = tempfile.TemporaryDirectory(prefix="mdr-voxel-resume-full-")
        self.addCleanup(full_temp.cleanup)
        full_root = Path(full_temp.name)
        setup_workspace(full_root, {"budgets": {"max_subdivisions_per_batch": 10000}})
        full = Pipeline(full_root, parallel=False).run()
        self.assertEqual(canonical_fingerprint(resumed), canonical_fingerprint(full))


if __name__ == "__main__":
    unittest.main()
