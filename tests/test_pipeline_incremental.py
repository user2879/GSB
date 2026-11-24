from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from gsb_layout.config import load_config, with_overrides
from gsb_layout.pipeline import Pipeline, semantic_digest
from gsb_layout.sample_generator import generate_samples
from gsb_layout.verification import verify_incremental_equivalence


class TestPipelineIncremental(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.samples = self.root / "samples"
        generate_samples(self.samples)
        self.config = load_config(self.samples / "config.json")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_noop_reuses_every_stage_and_snapshot_manifest(self) -> None:
        state = self.root / "state"
        first = Pipeline(self.samples, self.config, state).run(full_rebuild=True)
        second = Pipeline(self.samples, self.config, state).run()
        self.assertEqual(first["semantic_digest"], second["semantic_digest"])
        actions = second["cache"]["reuse_recompute"]
        self.assertTrue(
            all(
                entry["action"] == "reuse"
                for entry in actions["columns"] + actions["characters"]
            )
        )
        manifest = json.loads(
            (state / "snapshot" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["global_version"], 2)
        self.assertEqual(manifest["config_hash"], self.config.fingerprint())
        self.assertEqual(
            set(manifest["stage_versions"]),
            {"columns", "segments", "boxes", "matches"},
        )
        self.assertEqual(
            manifest["completed_images"],
            ["page_001.png", "page_002.png", "page_003.png"],
        )
        self.assertEqual(
            set(p.name for p in (state / "snapshot" / "stages").glob("*.json")),
            {"boxes.json", "columns.json", "matches.json", "segments.json"},
        )
        checkpoints = list((state / "snapshot" / "checkpoints").glob("layout-*.json"))
        self.assertEqual(len(checkpoints), 3)

    def test_snapshot_version_mismatch_is_stale_and_rebuilds(self) -> None:
        state = self.root / "state"
        Pipeline(self.samples, self.config, state).run(full_rebuild=True)
        manifest_path = state / "snapshot" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["format_version"] = 999
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = Pipeline(self.samples, self.config, state).run()
        self.assertEqual(report["run"]["snapshot_stale_reason"], "format_version_mismatch")
        self.assertEqual(report["diagnostics"][0]["code"], "E_SNAPSHOT_STALE")
        rebuilt = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt["format_version"], 1)
        self.assertEqual(rebuilt["global_version"], 2)

    def test_config_edits_only_invalidate_dependent_stage_records(self) -> None:
        state = self.root / "state"
        Pipeline(self.samples, self.config, state).run(full_rebuild=True)
        changed = with_overrides(self.config, split_penalty=self.config.split_penalty + 13)
        report = Pipeline(self.samples, changed, state).run()
        self.assertEqual(report["run"]["snapshot_stale_reason"], "config_hash_mismatch")
        self.assertEqual(report["diagnostics"][0]["code"], "E_SNAPSHOT_STALE")
        trace = report["cache"]["reuse_recompute"]
        self.assertTrue(all(e["action"] == "reuse" for e in trace["columns"] if e["stage"] == "segments"))
        regression_actions = [
            e["action"] for e in trace["characters"] if e["stage"] == "regression"
        ]
        self.assertIn("recompute", regression_actions)
        self.assertIn("reuse", regression_actions)

        window_cfg = with_overrides(changed, row_projection_window=7)
        window_report = Pipeline(self.samples, window_cfg, state).run()
        window_full = Pipeline(
            self.samples, window_cfg, self.root / "window_full"
        ).run(full_rebuild=True)
        self.assertEqual(window_report["semantic_digest"], window_full["semantic_digest"])
        window_trace = window_report["cache"]["reuse_recompute"]
        self.assertTrue(
            any(
                entry["action"] == "recompute"
                for entry in window_trace["characters"]
                if entry["stage"] == "regression"
            )
        )

        metric_cfg = with_overrides(changed, match_metric="cosine")
        metric_report = Pipeline(self.samples, metric_cfg, state).run()
        metric_trace = metric_report["cache"]["reuse_recompute"]
        self.assertTrue(
            all(
                e["action"] == "reuse"
                for e in metric_trace["characters"]
                if e["stage"] == "regression"
            )
        )
        self.assertTrue(
            all(
                e["action"] == "recompute"
                for e in metric_trace["characters"]
                if e["stage"] == "template_match"
            )
        )

    def test_budget_resume_matches_single_full_run(self) -> None:
        budget_cfg = with_overrides(self.config, match_point_budget=4800)
        state = self.root / "budget_state"
        reports: list[dict] = []
        for index in range(20):
            report = Pipeline(self.samples, budget_cfg, state).run(full_rebuild=(index == 0))
            reports.append(report)
            sequence = [item["job_id"] for item in report["budget"]["sequence"]]
            self.assertEqual(sequence, sorted(sequence))
            if report["budget"]["pending_jobs"] == 0:
                break
            self.assertTrue(
                any(
                    failure["code"] == "E_MATCH_BUDGET_PENDING"
                    for failure in report["failures"]
                )
            )
            self.assertTrue(report["threading"]["pending_match_task_keys"])
        self.assertLess(len(reports), 20)
        self.assertEqual(reports[-1]["budget"]["pending_jobs"], 0)
        self.assertGreater(reports[0]["budget"]["pending_jobs"], 0)
        self.assertTrue(reports[-1]["budget"]["resumed_from_prior_pending"])

        full_state = self.root / "full_state"
        full = Pipeline(self.samples, self.config, full_state).run(full_rebuild=True)
        self.assertEqual(reports[-1]["semantic_digest"], full["semantic_digest"])
        self.assertEqual(
            semantic_digest(reports[-1]["images"]), semantic_digest(full["images"])
        )

    def test_thread_count_does_not_change_order_or_results(self) -> None:
        one = with_overrides(self.config, thread_count=1, match_point_budget=6000)
        four = with_overrides(self.config, thread_count=4, match_point_budget=6000)
        r1 = Pipeline(self.samples, one, self.root / "thread1").run(full_rebuild=True)
        r4 = Pipeline(self.samples, four, self.root / "thread4").run(full_rebuild=True)
        self.assertEqual(r1["semantic_digest"], r4["semantic_digest"])
        self.assertEqual(
            [image["name"] for image in r1["images"]],
            [image["name"] for image in r4["images"]],
        )
        self.assertEqual(
            [item["job_id"] for item in r1["budget"]["sequence"]],
            [item["job_id"] for item in r4["budget"]["sequence"]],
        )
        self.assertEqual(r1["budget"]["pending_order"], r4["budget"]["pending_order"])
        self.assertEqual(
            [failure["failure_id"] + failure["code"] for failure in r1["failures"]],
            [failure["failure_id"] + failure["code"] for failure in r4["failures"]],
        )
        self.assertLessEqual(r1["threading"]["column_workers"], 1)
        self.assertLessEqual(r4["threading"]["column_workers"], 4)
        self.assertLessEqual(r4["threading"]["match_workers"], 4)

    def test_single_image_replacement_recomputes_only_changed_image_family(self) -> None:
        state = self.root / "state"
        baseline = Pipeline(self.samples, self.config, state).run(full_rebuild=True)
        page = self.samples / "page_001.png"
        with Image.open(page) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        pixels[0, 0] = 241 if int(pixels[0, 0]) != 241 else 242
        Image.fromarray(pixels, mode="L").save(page, format="PNG", optimize=False)
        changed = Pipeline(self.samples, self.config, state).run()
        trace = changed["cache"]["reuse_recompute"]
        column_actions = {
            (entry["image"], entry["action"])
            for entry in trace["columns"]
            if entry["stage"] == "columns"
        }
        self.assertIn(("page_001.png", "recompute"), column_actions)
        self.assertIn(("page_002.png", "reuse"), column_actions)
        self.assertIn(("page_003.png", "reuse"), column_actions)
        # The one-pixel content mutation is outside all glyphs, so final semantic
        # fields remain equivalent to a fresh full build over the same inputs.
        fresh = Pipeline(self.samples, self.config, self.root / "fresh").run(full_rebuild=True)
        self.assertEqual(changed["semantic_digest"], fresh["semantic_digest"])
        self.assertNotEqual(baseline["images"][0]["image_sha256"], changed["images"][0]["image_sha256"])

    def test_full_verify_ladder(self) -> None:
        result = verify_incremental_equivalence(
            self.samples,
            self.config,
            self.root / "verify",
        )
        self.assertTrue(result["passed"], json.dumps(result["checks"], ensure_ascii=False))
        self.assertTrue(result["checks"]["all_step_semantics_equal"])
        self.assertTrue(result["checks"]["noop_all_cache_reused"])
        self.assertTrue(result["checks"]["replacement_invalidates_changed_image"])
        self.assertTrue(result["checks"]["replacement_reuses_unaffected_images"])


if __name__ == "__main__":
    unittest.main()
