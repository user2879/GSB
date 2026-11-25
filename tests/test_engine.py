from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from layout_diag.editing import apply_edit_stream
from layout_diag.engine import run_from_paths
from layout_diag.generator import ImageSpec
from tests.helpers import make_dataset, read_config, write_config


def two_small_specs():
    return (
        ImageSpec("small_a.png", 2, 3, 0.0012, 5.0),
        ImageSpec("small_b.png", 2, 4, 0.0015, 4.5),
    )


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.samples = make_dataset(cls.root / "samples", specs=two_small_specs())
        config = read_config(cls.samples / "config.json")
        config["tier"] = "custom"
        config["window"] = {"width": 64, "height": 64, "step_x": 32, "step_y": 32}
        config["cc"]["merge_gap"] = 3
        config["precision"] = {"fixed_bits": 8, "quant_step": 2}
        config["point_budget"] = 200
        config["max_workers"] = 2
        write_config(cls.samples / "config.json", config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def case(self, name: str) -> Path:
        destination = self.root / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(self.samples, destination)
        return destination

    def test_fixed_point_precision_changes_coordinates_and_residuals(self) -> None:
        root = self.case("precision")
        config_path = root / "config.json"
        config = read_config(config_path)
        config["precision"] = {"fixed_bits": 10, "quant_step": 1}
        write_config(config_path, config)
        high, _ = run_from_paths(root, config_path, self.root / "precision-high", force_rebuild=True)
        config["precision"] = {"fixed_bits": 6, "quant_step": 4}
        write_config(config_path, config)
        low, _ = run_from_paths(root, config_path, self.root / "precision-low", force_rebuild=True)
        high_points = high.documents["keypoints"]["keypoints"]
        low_points = low.documents["keypoints"]["keypoints"]
        self.assertEqual(len(high_points), len(low_points))
        self.assertNotEqual((high_points[0]["x_q"], high_points[0]["y_q"]), (low_points[0]["x_q"], low_points[0]["y_q"]))
        self.assertNotEqual(
            (high_points[0]["residual_x_q"], high_points[0]["residual_y_q"]),
            (low_points[0]["residual_x_q"], low_points[0]["residual_y_q"]),
        )

    def test_explicit_tiers_change_candidates_and_keypoint_quantization(self) -> None:
        root = self.case("tiers")
        config_path = root / "config.json"
        config = read_config(config_path)
        config["tier"] = "small"
        write_config(config_path, config)
        small, _ = run_from_paths(root, config_path, self.root / "tier-small", force_rebuild=True)
        config["tier"] = "large"
        write_config(config_path, config)
        large, _ = run_from_paths(root, config_path, self.root / "tier-large", force_rebuild=True)
        self.assertNotEqual(
            small.documents["report"]["stages"]["candidate_count"],
            large.documents["report"]["stages"]["candidate_count"],
        )
        self.assertNotEqual(
            (small.documents["keypoints"]["keypoints"][0]["x_q"], small.documents["keypoints"]["keypoints"][0]["y_q"]),
            (large.documents["keypoints"]["keypoints"][0]["x_q"], large.documents["keypoints"]["keypoints"][0]["y_q"]),
        )
        self.assertEqual(len(small.documents["reading_order"]["sequence"]), len(large.documents["reading_order"]["sequence"]))

    def test_stage_specific_invalidation_is_precise(self) -> None:
        root = self.case("invalidation")
        config_path = root / "config.json"
        out = self.root / "invalidation-out"
        run_from_paths(root, config_path, out)
        config = read_config(config_path)
        config["cc"]["merge_gap"] = 4
        write_config(config_path, config)
        merged, _ = run_from_paths(root, config_path, out)
        stages = merged.documents["reuse"]["summary"]["by_stage"]
        self.assertEqual(stages["sliding"]["hit"], 2)
        self.assertEqual(stages["components"]["miss"], 2)
        self.assertEqual(stages["keypoints"]["hit"], 0)

        config["window"] = {"width": 32, "height": 32, "step_x": 16, "step_y": 16}
        write_config(config_path, config)
        windowed, _ = run_from_paths(root, config_path, out)
        stages = windowed.documents["reuse"]["summary"]["by_stage"]
        self.assertEqual(stages["sliding"]["miss"], 2)
        self.assertEqual(stages["components"]["hit"], 2)
        self.assertEqual(stages["keypoints"]["hit"], 0)

        config["precision"] = {"fixed_bits": 10, "quant_step": 1}
        write_config(config_path, config)
        quantized, _ = run_from_paths(root, config_path, out)
        stages = quantized.documents["reuse"]["summary"]["by_stage"]
        self.assertEqual(stages["sliding"]["hit"], 2)
        self.assertEqual(stages["components"]["hit"], 2)
        self.assertEqual(stages["keypoints"]["hit"], 0)

    def test_image_replacement_only_invalidates_referencing_image(self) -> None:
        root = self.case("replace")
        config_path = root / "config.json"
        out = self.root / "replace-out"
        run_from_paths(root, config_path, out)
        replacement = self.root / "replacement.png"
        source = Image.open(root / "images" / "small_a.png").convert("L")
        draw = ImageDraw.Draw(source)
        draw.rectangle((10, 900, 26, 916), fill=12)
        source.save(replacement)
        failures = apply_edit_stream(
            root,
            config_path,
            [{"op": "replace_image", "image": "small_a.png", "source": str(replacement)}],
        )
        self.assertFalse(failures.items)
        result, _ = run_from_paths(root, config_path, out)
        image_status = {
            (entry["image"], entry["stage"]): entry["status"]
            for entry in result.documents["reuse"]["entries"]
            if entry["scope"] == "image"
        }
        self.assertEqual(image_status[("small_a.png", "sliding")], "miss")
        self.assertEqual(image_status[("small_a.png", "components")], "miss")
        self.assertEqual(image_status[("small_b.png", "sliding")], "hit")
        self.assertEqual(image_status[("small_b.png", "components")], "hit")

    def test_later_image_reading_cache_survives_global_index_shift(self) -> None:
        root = self.case("index-shift")
        config_path = root / "config.json"
        out = self.root / "index-shift-out"
        before, _ = run_from_paths(root, config_path, out)
        blank = self.root / "blank.png"
        Image.new("L", (640, 960), 238).save(blank)
        failures = apply_edit_stream(
            root,
            config_path,
            [{"op": "replace_image", "image": "small_a.png", "source": str(blank)}],
        )
        self.assertFalse(failures.items)
        after, _ = run_from_paths(root, config_path, out)
        reading_hits = [
            entry
            for entry in after.documents["reuse"]["entries"]
            if entry["image"] == "small_b.png" and entry["scope"] == "reading" and entry["product_id"] == "sequence"
        ]
        self.assertEqual(reading_hits[0]["status"], "hit")
        before_b = [
            point for point in before.documents["keypoints"]["keypoints"] if point["image"] == "small_b.png"
        ]
        after_b = [
            point for point in after.documents["keypoints"]["keypoints"] if point["image"] == "small_b.png"
        ]
        self.assertEqual(len(before_b), len(after_b))
        self.assertEqual(
            [{key: point[key] for key in ("keypoint_id", "x_q", "y_q", "column")} for point in before_b],
            [{key: point[key] for key in ("keypoint_id", "x_q", "y_q", "column")} for point in after_b],
        )
        self.assertNotEqual(before_b[0]["keypoint_index"], after_b[0]["keypoint_index"])

    def test_threaded_and_serial_execution_match(self) -> None:
        root = self.case("concurrency")
        config_path = root / "config.json"
        threaded, _ = run_from_paths(root, config_path, self.root / "threaded", force_rebuild=True, serial=False)
        serial, _ = run_from_paths(root, config_path, self.root / "serial", force_rebuild=True, serial=True)
        self.assertEqual(
            threaded.canonical_results(include_reuse=True),
            serial.canonical_results(include_reuse=True),
        )
        self.assertLessEqual(threaded.documents["report"]["scheduling"]["max_workers_configured"], 2)

    def test_budget_pending_resume_and_snapshot_recovery(self) -> None:
        root = self.case("budget")
        config_path = root / "config.json"
        config = read_config(config_path)
        config["point_budget"] = 5
        write_config(config_path, config)
        out = self.root / "budget-out"
        result, failures = run_from_paths(root, config_path, out)
        self.assertEqual(len(result.documents["keypoints"]["keypoints"]), 5)
        self.assertGreater(result.manifest["pending_points"], 0)
        self.assertIn("E_POINT_BUDGET_PENDING", [failure.code for failure in failures.items])
        self.assertFalse(result.manifest["completed_images"])
        self.assertEqual(len(result.documents["schedule"]["pending"]), 3)
        self.assertEqual(sum(item["pending_points"] for item in result.documents["schedule"]["pending"]), 9)
        self.assertTrue(all(item["code"] == "E_POINT_BUDGET_PENDING" for item in result.documents["schedule"]["pending"]))

        resumed, _ = run_from_paths(root, config_path, out)
        self.assertEqual(
            resumed.documents["keypoints"]["keypoints"],
            result.documents["keypoints"]["keypoints"],
        )
        self.assertEqual(resumed.manifest["run_mode"], "resume")

        shutil.move(str(out / "current"), str(out / ".previous"))
        recovered, _ = run_from_paths(root, config_path, out)
        self.assertEqual(recovered.documents["report"]["snapshot_recovery"], "restored_previous")
        self.assertEqual(
            recovered.documents["failures"],
            result.documents["failures"],
        )

    def test_edit_sequence_incremental_matches_full_rebuild(self) -> None:
        root = self.case("edits")
        config_path = root / "config.json"
        out = self.root / "edits-incremental"
        run_from_paths(root, config_path, out)

        replacement = self.root / "edit-replacement.png"
        source = Image.open(root / "images" / "small_a.png").convert("L")
        ImageDraw.Draw(source).rectangle((15, 905, 35, 925), fill=30)
        source.save(replacement)
        sources = self.root / "edit-sources"
        if sources.exists():
            shutil.rmtree(sources)
        sources.mkdir()
        shutil.copy2(root / "images" / "small_b.png", sources / "small_b.png")
        shutil.copy2(root / "truth" / "small_b.json", sources / "small_b.json")
        shutil.copy2(root / "keypoints" / "v2" / "small_b.json", sources / "small_b_v2.json")
        shutil.copy2(root / "keypoints" / "v2" / "small_a.json", sources / "small_a_v2.json")
        config_commands = [
            {"op": "set_window", "width": 64, "height": 64, "step_x": 32, "step_y": 32},
            {"op": "set_merge_threshold", "value": 3},
            {"op": "set_quantization", "fixed_bits": 10, "quant_step": 1},
            {"op": "set_tier", "tier": "small"},
        ]
        image_commands = [
            {"op": "replace_image", "image": "small_a.png", "source": str(replacement)},
            {"op": "remove_image", "image": "small_b.png"},
            {
                "op": "add_image",
                "image": "small_b.png",
                "source": str(sources / "small_b.png"),
                "truth_source": str(sources / "small_b.json"),
                "version_source": str(sources / "small_b_v2.json"),
                "version": "v2",
            },
            {"op": "replace_keypoints", "image": "small_a.png", "version": "v2", "source": str(sources / "small_a_v2.json")},
            {"op": "set_keypoint_version", "version": "v2"},
        ]
        edit_failures = apply_edit_stream(root, config_path, config_commands)
        self.assertFalse(edit_failures.items, [failure.as_dict() for failure in edit_failures.items])
        run_from_paths(root, config_path, out)
        edit_failures = apply_edit_stream(root, config_path, image_commands)
        self.assertFalse(edit_failures.items, [failure.as_dict() for failure in edit_failures.items])
        incremental, _ = run_from_paths(root, config_path, out)
        full, _ = run_from_paths(root, config_path, self.root / "edits-full", force_rebuild=True)
        self.assertEqual(
            incremental.canonical_results(include_reuse=False),
            full.canonical_results(include_reuse=False),
        )
        left_keys = [
            (entry["image"], entry["scope"], entry["stage"], entry["product_id"], entry["key"])
            for entry in incremental.documents["reuse"]["entries"]
        ]
        right_keys = [
            (entry["image"], entry["scope"], entry["stage"], entry["product_id"], entry["key"])
            for entry in full.documents["reuse"]["entries"]
        ]
        self.assertEqual(left_keys, right_keys)
        self.assertGreater(incremental.documents["reuse"]["summary"]["hits"], 0)

    def test_keypoints_reference_window_component_and_content_keys(self) -> None:
        root = self.case("references")
        result, _ = run_from_paths(root, root / "config.json", self.root / "references-out", force_rebuild=True)
        for point in result.documents["keypoints"]["keypoints"]:
            self.assertTrue(point["source_window_id"].startswith("w"))
            self.assertTrue(point["source_component_id"].startswith("ccm"))
            self.assertEqual(len(point["source_component_ids"]), len(set(point["source_component_ids"])))
        for entry in result.documents["reuse"]["entries"]:
            self.assertEqual(len(entry["key"]), 64)

    def test_keypoint_version_replacement_updates_annotation_hash_without_localization_breakage(self) -> None:
        root = self.case("version-replace")
        config_path = root / "config.json"
        out = self.root / "version-replace-out"
        before, _ = run_from_paths(root, config_path, out)
        failures = apply_edit_stream(
            root,
            config_path,
            [
                {
                    "op": "replace_keypoints",
                    "image": "small_a.png",
                    "version": "v2",
                    "source": str(root / "keypoints" / "v2" / "small_a.json"),
                },
                {"op": "set_keypoint_version", "version": "v2"},
            ],
        )
        self.assertFalse(failures.items)
        after, _ = run_from_paths(root, config_path, out)
        self.assertEqual(after.documents["report"]["annotations"]["active_version"], "v2")
        before_annotation = next(
            item for item in before.documents["report"]["annotations"]["images"] if item["image"] == "small_a.png"
        )
        after_annotation = next(
            item for item in after.documents["report"]["annotations"]["images"] if item["image"] == "small_a.png"
        )
        self.assertNotEqual(before_annotation["content_hash"], after_annotation["content_hash"])
        self.assertEqual(
            before.documents["keypoints"],
            after.documents["keypoints"],
        )
        stage_summary = after.documents["reuse"]["summary"]["by_stage"]
        self.assertEqual(stage_summary["sliding"]["hit"], 2)
        self.assertEqual(stage_summary["components"]["hit"], 2)


if __name__ == "__main__":
    unittest.main()
