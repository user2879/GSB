from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from chart_workbench.edits import apply_edit_commands
from chart_workbench.engine import run_workbench
from chart_workbench.generate import generate_samples
from chart_workbench.loader import load_sample_directory


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = json.loads((ROOT / "config" / "calibration.json").read_text(encoding="utf-8"))


class WorkbenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="chart-workbench-test-")
        self.work = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config(self, *, samples: Path | None = None, **changes) -> Path:
        config = dict(BASE_CONFIG)
        config["sample_dir"] = str((samples or (ROOT / "samples")).resolve())
        config.update(changes)
        path = self.work / "calibration.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _run(self, config: Path, output: Path, *, serial: bool = False, stop_after: int | None = None):
        return run_workbench(config, output_override=output, serial=serial, stop_after_pages=stop_after)

    def test_generation_is_reproducible_and_inputs_validate(self) -> None:
        first = self.work / "first"
        second = self.work / "second"
        self.assertEqual(7, len(generate_samples(first)))
        self.assertEqual(7, len(generate_samples(second)))
        first_hashes = {
            path.name: path.read_bytes()
            for path in sorted(first.iterdir())
            if path.suffix in {".png", ".json"}
        }
        second_hashes = {
            path.name: path.read_bytes()
            for path in sorted(second.iterdir())
            if path.suffix in {".png", ".json"}
        }
        self.assertEqual(first_hashes, second_hashes)
        loaded = load_sample_directory(first)
        self.assertEqual(7, len(loaded.pages))
        self.assertEqual([], loaded.failures)
        self.assertEqual(14, len(first_hashes))

    def test_serial_parallel_reports_are_byte_identical(self) -> None:
        config = self._config(thread_pool_concurrency=7)
        serial = self._run(config, self.work / "serial", serial=True)
        parallel = self._run(config, self.work / "parallel", serial=False)
        self.assertEqual(serial, parallel)
        self.assertEqual(7, serial["summary"]["pages_ok"])
        self.assertEqual(42, serial["summary"]["data_points"])
        self.assertEqual(0, serial["summary"]["failures"])
        self.assertEqual(7, serial["summary"]["verification_passed_pages"])
        self.assertEqual(4, len(serial["budget"]["batches"]))
        self.assertEqual(3, len(serial["budget"]["pending_events"]))

        pages = {page["page"]: page for page in serial["pages"]}
        self.assertEqual("linear", pages["page_01_linear_bars.png"]["axis_mappings"]["y"]["mapping_type"])
        self.assertEqual("log10", pages["page_02_log_line.png"]["axis_mappings"]["y"]["mapping_type"])
        self.assertEqual("broken_linear", pages["page_03_broken_line.png"]["axis_mappings"]["y"]["mapping_type"])
        self.assertEqual(2, len(pages["page_03_broken_line.png"]["axis_mappings"]["y"]["segments"]))
        self.assertEqual(1, pages["page_04_linear_line_occluded.png"]["axis_mappings"]["y"]["excluded_tick_count"])
        self.assertEqual(1, pages["page_06_linear_bar_occluded.png"]["axis_mappings"]["x"]["excluded_tick_count"])
        self.assertEqual(11, len(pages["page_05_dense_bars.png"]["data_points"]))

    def test_stage_cache_invalidation_is_precise(self) -> None:
        config = self._config()
        output = self.work / "out"
        baseline = self._run(config, output, serial=True)
        baseline_values = [point["value"] for point in baseline["pages"][0]["data_points"]]

        config_data = json.loads(config.read_text(encoding="utf-8"))
        config_data["quantization_step"] = 0.5
        config.write_text(json.dumps(config_data), encoding="utf-8")
        quantized = self._run(config, output, serial=False)
        self.assertEqual(7, quantized["cache"]["stats"]["ticks_reused"])
        self.assertEqual(7, quantized["cache"]["stats"]["mapping_reused"])
        self.assertEqual(7, quantized["cache"]["stats"]["data_recomputed"])
        self.assertEqual({"data"}, {item["stage"] for item in quantized["cache"]["recompute"]})
        quantized_values = [point["value"] for point in quantized["pages"][0]["data_points"]]
        self.assertNotEqual(baseline_values, quantized_values)

        config_data["log_axis_detection_threshold"] = 1e-9
        config.write_text(json.dumps(config_data), encoding="utf-8")
        thresholded = self._run(config, output, serial=True)
        self.assertEqual(7, thresholded["cache"]["stats"]["ticks_reused"])
        self.assertEqual(7, thresholded["cache"]["stats"]["mapping_recomputed"])
        pages = {page["page"]: page for page in thresholded["pages"]}
        self.assertNotEqual("log10", pages["page_02_log_line.png"]["axis_mappings"]["y"]["mapping_type"])
        self.assertIn("page_02_log_line.png", thresholded["cache"]["config_changed_pages"])

        config_data["tick_min_length"] = 9
        config.write_text(json.dumps(config_data), encoding="utf-8")
        tick_changed = self._run(config, output, serial=True)
        self.assertEqual(7, tick_changed["cache"]["stats"]["ticks_recomputed"])
        self.assertIn("E_TICKS_NOT_FOUND", {failure["code"] for failure in tick_changed["failures"]})

    def test_page_replacement_add_and_remove_only_invalidates_affected_pages(self) -> None:
        sample_dir = self.work / "samples"
        shutil.copytree(ROOT / "samples", sample_dir)
        config = self._config(samples=sample_dir)
        output = self.work / "out"
        first = self._run(config, output, serial=True)

        shutil.copy2(ROOT / "variants" / "page_01_linear_bars.png", sample_dir / "page_01_linear_bars.png")
        shutil.copy2(ROOT / "variants" / "page_01_linear_bars.json", sample_dir / "page_01_linear_bars.json")
        shutil.copy2(ROOT / "variants" / "page_08_extra_line.png", sample_dir / "page_08_extra_line.png")
        shutil.copy2(ROOT / "variants" / "page_08_extra_line.json", sample_dir / "page_08_extra_line.json")
        (sample_dir / "page_06_linear_bar_occluded.png").unlink()
        (sample_dir / "page_06_linear_bar_occluded.json").unlink()
        second = self._run(config, output, serial=False)

        self.assertEqual(["page_08_extra_line.png"], second["cache"]["added_pages"])
        self.assertEqual(["page_06_linear_bar_occluded.png"], second["cache"]["removed_pages"])
        replaced = [item for item in second["cache"]["recompute"] if item["page"] == "page_01_linear_bars.png"]
        self.assertEqual(["ticks", "mapping", "data"], [item["stage"] for item in replaced])
        self.assertEqual({"image_changed"}, {item["reason"] for item in replaced})
        unaffected = [
            item
            for item in second["cache"]["recompute"]
            if item["page"] not in {"page_01_linear_bars.png", "page_08_extra_line.png"}
        ]
        self.assertEqual([], unaffected)
        self.assertNotEqual(first["pages"][0]["data_points"][0]["value"], second["pages"][0]["data_points"][0]["value"])

    def test_axis_switch_controls_change_real_outputs(self) -> None:
        baseline = self._run(self._config(), self.work / "base", serial=False)

        endpoints = self._run(self._config(inverse_mode="endpoints"), self.work / "endpoints", serial=False)
        self.assertNotEqual(baseline["pages"][0]["data_points"], endpoints["pages"][0]["data_points"])
        self.assertEqual("linear", endpoints["pages"][0]["axis_mappings"]["y"]["mapping_type"])

        high_break = self._run(
            self._config(broken_axis_detection_threshold=10.0),
            self.work / "high-break",
            serial=False,
        )
        self.assertEqual("broken_linear", baseline["pages"][2]["axis_mappings"]["y"]["mapping_type"])
        self.assertEqual("linear", high_break["pages"][2]["axis_mappings"]["y"]["mapping_type"])

        strict_tolerance = self._run(
            self._config(line_width_tolerance=0),
            self.work / "strict-width",
            serial=False,
        )
        self.assertEqual(7, strict_tolerance["cache"]["stats"]["ticks_recomputed"])
        self.assertTrue(all(page["status"] == "failed" for page in strict_tolerance["pages"]))
        self.assertIn("E_TICKS_NOT_FOUND", {failure["code"] for failure in strict_tolerance["failures"]})

        no_occlusion = self._run(
            self._config(label_occlusion_tolerance=0.0),
            self.work / "no-occlusion",
            serial=False,
        )
        self.assertEqual(
            {"page_04_linear_line_occluded.png", "page_06_linear_bar_occluded.png"},
            {failure["page"] for failure in no_occlusion["failures"] if failure["code"] == "E_LABEL_OCCLUSION"},
        )

    def test_edit_command_stream_applies_page_and_config_changes(self) -> None:
        sample_dir = self.work / "samples"
        shutil.copytree(ROOT / "samples", sample_dir)
        config = self._config(samples=sample_dir)
        commands = {
            "schema_version": 1,
            "commands": [
                {
                    "op": "replace_page",
                    "page": "page_01_linear_bars.png",
                    "source_png": str(ROOT / "variants" / "page_01_linear_bars.png"),
                    "source_truth": str(ROOT / "variants" / "page_01_linear_bars.json"),
                },
                {
                    "op": "add_page",
                    "page": "page_08_extra_line.png",
                    "source_png": str(ROOT / "variants" / "page_08_extra_line.png"),
                    "source_truth": str(ROOT / "variants" / "page_08_extra_line.json"),
                },
                {"op": "remove_page", "page": "page_06_linear_bar_occluded.png"},
                {"op": "set_config", "path": "quantization_step", "value": 0.5},
            ],
        }
        command_path = self.work / "commands.json"
        command_path.write_text(json.dumps(commands), encoding="utf-8")
        self._run(config, self.work / "out", serial=True)
        applied = apply_edit_commands(config, command_path)
        self.assertEqual(4, len(applied["applied"]))
        self.assertFalse((sample_dir / "page_06_linear_bar_occluded.png").exists())
        self.assertTrue((sample_dir / "page_08_extra_line.png").exists())
        self.assertEqual(0.5, json.loads(config.read_text(encoding="utf-8"))["quantization_step"])
        report = self._run(config, self.work / "out", serial=False)
        self.assertEqual(["page_08_extra_line.png"], report["cache"]["added_pages"])
        self.assertEqual(["page_06_linear_bar_occluded.png"], report["cache"]["removed_pages"])
        self.assertEqual(7, report["summary"]["pages_ok"])

    def test_budget_and_page_quota(self) -> None:
        low_budget = self._config(tick_comparison_budget=10, thread_pool_concurrency=3)
        serial = self._run(low_budget, self.work / "budget-serial", serial=True)
        parallel = self._run(low_budget, self.work / "budget-parallel", serial=False)
        self.assertEqual(serial, parallel)
        self.assertGreater(serial["summary"]["pages_pending"], 0)
        self.assertIn("E_CALIB_BUDGET_PENDING", {failure["code"] for failure in serial["failures"]})
        self.assertTrue(any(event["code"] == "E_CALIB_BUDGET_PENDING" for event in serial["budget"]["pending_events"]))

        quota_config = self._config(page_data_point_quota=3)
        quota = self._run(quota_config, self.work / "quota", serial=False)
        self.assertTrue(all(page["status"] == "partial" for page in quota["pages"]))
        self.assertTrue(all(len(page["data_points"]) == 3 for page in quota["pages"]))
        self.assertEqual(
            {page["page"] for page in quota["pages"]},
            {failure["page"] for failure in quota["failures"] if failure["code"] == "E_PAGE_QUOTA_EXCEEDED"},
        )

    def test_snapshot_resume_matches_fresh_core_results(self) -> None:
        config = self._config()
        resumed_output = self.work / "resumed"
        partial = self._run(config, resumed_output, serial=True, stop_after=2)
        self.assertTrue(partial["stopped"])
        self.assertEqual(2, partial["summary"]["pages_completed"])
        self.assertGreater(partial["summary"]["pages_pending"], 0)

        resumed = self._run(config, resumed_output, serial=False)
        fresh = self._run(config, self.work / "fresh", serial=False)
        self.assertEqual(fresh["pages"], resumed["pages"])
        self.assertEqual(fresh["failures"], resumed["failures"])
        self.assertEqual(fresh["budget"], resumed["budget"])
        self.assertEqual(fresh["result_digest"], resumed["result_digest"])
        self.assertEqual(7, resumed["summary"]["pages_completed"])
        self.assertEqual(0, resumed["summary"]["pages_pending"])

        unchanged = self._run(config, resumed_output, serial=True)
        self.assertEqual(21, unchanged["cache"]["stats"]["reused"])
        self.assertEqual(0, unchanged["cache"]["stats"]["recomputed"])
        self.assertEqual(resumed["snapshot"]["version"], unchanged["snapshot"]["version"])

    def test_invalid_pages_are_isolated_and_sorted(self) -> None:
        sample_dir = self.work / "samples"
        sample_dir.mkdir(parents=True)
        shutil.copy2(ROOT / "samples" / "page_01_linear_bars.png", sample_dir / "b_valid.png")
        shutil.copy2(ROOT / "samples" / "page_01_linear_bars.json", sample_dir / "b_valid.json")
        (sample_dir / "a_bad_header.png").write_bytes(b"not a png")
        (sample_dir / "a_bad_header.json").write_text("{}", encoding="utf-8")
        shutil.copy2(ROOT / "samples" / "page_01_linear_bars.png", sample_dir / "z_bad_truth.png")
        bad_truth = json.loads((ROOT / "samples" / "page_01_linear_bars.json").read_text(encoding="utf-8"))
        bad_truth["axes"]["y"]["values"] = [0, 1, 0, 3, 4, 5]
        (sample_dir / "z_bad_truth.json").write_text(json.dumps(bad_truth), encoding="utf-8")

        config = self._config(samples=sample_dir)
        report = self._run(config, self.work / "out", serial=False)
        self.assertEqual(["a_bad_header.png", "z_bad_truth.png"], [failure["page"] for failure in report["failures"]])
        self.assertEqual({"E_CHART_INVALID"}, {failure["code"] for failure in report["failures"]})
        statuses = {page["page"]: page["status"] for page in report["pages"]}
        self.assertEqual("ok", statuses["b_valid.png"])
        self.assertEqual("invalid", statuses["a_bad_header.png"])
        self.assertEqual("invalid", statuses["z_bad_truth.png"])


if __name__ == "__main__":
    unittest.main()
