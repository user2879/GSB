from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from renderlab.errors import E_PREVIEW_BUDGET_PENDING
from renderlab.preview import PreviewBroker
from renderlab.workbench import Workbench
from tests.helpers import write_case


class EngineIncrementalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.config = write_case(self.root / "case")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_cold_run_repeat_reuse_and_stable_output(self) -> None:
        workbench = Workbench(self.config, temp_dir=self.root / "cache-a", batch_size=2)
        first = workbench.run()
        self.assertEqual(first.status, "ok")
        first_payload = first.to_dict()
        self.assertGreater(len(first_payload["frames"]), 0)
        self.assertGreater(first_payload["concurrency"]["max_observed"], 1)
        self.assertLessEqual(
            first_payload["concurrency"]["max_observed"],
            first_payload["concurrency"]["configured"],
        )
        self.assertEqual(first_payload["cache"]["recompute_mismatches"], 0)
        self.assertGreater(first_payload["cache"]["recomputes"], 0)

        second = workbench.run()
        self.assertEqual(first.canonical_dict(), second.canonical_dict())
        summary = second.to_dict()["invalidation"]["summary"]
        self.assertTrue(all(bucket["computed"] == 0 for bucket in summary.values()))

        frames = second.to_dict()["frames"]
        order = [(frame["tier_rank"], frame["prompt_index"], frame["seed_index"]) for frame in frames]
        self.assertEqual(order, sorted(order))
        for frame in frames:
            path = (self.root / "cache-a") / frame["png_relative_path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), frame["png_sha256"])

    def test_guidance_edit_reuses_encoding_and_initialization(self) -> None:
        workbench = Workbench(self.config, temp_dir=self.root / "cache-guidance", batch_size=2)
        workbench.run()
        workbench.apply_edit({"op": "set_guidance", "value": 3.5})
        report = workbench.run()
        payload = report.to_dict()
        summary = payload["invalidation"]["summary"]
        self.assertEqual(summary["encode"]["computed"], 0)
        self.assertEqual(summary["init"]["computed"], 0)
        self.assertGreater(summary["step"]["computed"], 0)
        self.assertGreater(summary["decode"]["computed"], 0)
        computed_ids = {event["stage_id"] for event in payload["invalidation"]["computed_stages"]}
        self.assertFalse(any(stage_id.startswith("encode:") for stage_id in computed_ids))
        self.assertFalse(any(stage_id.startswith("init:") for stage_id in computed_ids))

        changed = payload["invalidation"]["changes_from_previous"]
        self.assertEqual(changed, [{"field": "guidance_scale", "before": 2.0, "after": 3.5}])

    def test_prompt_replacement_only_invalidates_that_prompt_chain(self) -> None:
        workbench = Workbench(self.config, temp_dir=self.root / "cache-prompt", batch_size=2)
        workbench.run()
        workbench.apply_edit({"op": "replace_prompt", "prompt_id": "p1", "tokens": ["glass", "river"]})
        report = workbench.run()
        payload = report.to_dict()
        computed_ids = {event["stage_id"] for event in payload["invalidation"]["computed_stages"]}
        self.assertIn("encode:p1", computed_ids)
        self.assertNotIn("encode:p0", computed_ids)
        self.assertTrue(any(stage_id.startswith("init:") and ":p1:" in stage_id for stage_id in computed_ids))
        self.assertFalse(any(stage_id.startswith("init:") and ":p0:" in stage_id for stage_id in computed_ids))

    def test_seed_and_tier_edits_recompute_only_required_stages(self) -> None:
        workbench = Workbench(self.config, temp_dir=self.root / "cache-seed-tier", batch_size=2)
        workbench.run()
        workbench.apply_edit({"op": "set_seeds", "seeds": [13, 17]})
        seed_report = workbench.run().to_dict()
        seed_summary = seed_report["invalidation"]["summary"]
        self.assertEqual(seed_summary["encode"]["computed"], 0)
        self.assertGreater(seed_summary["init"]["computed"], 0)
        self.assertGreater(seed_summary["step"]["computed"], 0)
        self.assertGreater(seed_summary["decode"]["computed"], 0)

        workbench.apply_edit(
            {
                "op": "set_resolution_tiers",
                "tiers": [
                    {"name": "tiny", "width": 8, "height": 8},
                    {"name": "small", "width": 16, "height": 16},
                    {"name": "medium", "width": 24, "height": 24},
                ],
            }
        )
        tier_report = workbench.run().to_dict()
        tier_summary = tier_report["invalidation"]["summary"]
        self.assertGreater(tier_summary["init"]["reused"], 0)
        self.assertGreater(tier_summary["init"]["computed"], 0)
        reused_step_ids = {
            event["stage_id"] for event in tier_report["invalidation"]["reused_stages"]
            if event["kind"] == "step"
        }
        computed_step_ids = {
            event["stage_id"] for event in tier_report["invalidation"]["computed_stages"]
            if event["kind"] == "step"
        }
        self.assertTrue(any(":tiny:" in stage_id for stage_id in reused_step_ids))
        self.assertTrue(any(":medium:" in stage_id for stage_id in computed_step_ids))

    def test_edit_sequence_matches_cold_rebuild(self) -> None:
        edits = [
            {"op": "set_batch_size", "value": 1},
            {"op": "set_guidance", "value": 2.75},
            {"op": "set_seeds", "seeds": [13, 17]},
            {"op": "replace_prompt", "prompt_id": "p0", "tokens": ["arch", "cloud"]},
            {
                "op": "set_resolution_tiers",
                "tiers": [{"name": "tiny", "width": 8, "height": 8}],
            },
            {
                "op": "set_resolution_tiers",
                "tiers": [
                    {"name": "tiny", "width": 8, "height": 8},
                    {"name": "small", "width": 16, "height": 16},
                ],
            },
            {
                "op": "patch_latent",
                "prompt_id": "p1",
                "seed": 17,
                "tier": "tiny",
                "start": 3,
                "length": 9,
                "delta": 0.025,
            },
        ]
        incremental = Workbench(self.config, temp_dir=self.root / "incremental", batch_size=2)
        reports = incremental.run_edit_sequence(edits)
        final_incremental = reports[-1]

        fresh = Workbench(self.config, temp_dir=self.root / "fresh", batch_size=2)
        fresh.apply_edits(edits)
        final_fresh = fresh.run()
        self.assertEqual(final_incremental.canonical_dict(), final_fresh.canonical_dict())
        for incremental_frame, fresh_frame in zip(
            final_incremental.to_dict()["frames"], final_fresh.to_dict()["frames"]
        ):
            incremental_bytes = (self.root / "incremental" / incremental_frame["png_relative_path"]).read_bytes()
            fresh_bytes = (self.root / "fresh" / fresh_frame["png_relative_path"]).read_bytes()
            self.assertEqual(incremental_bytes, fresh_bytes)

    def test_concurrent_and_serial_frame_outputs_match(self) -> None:
        concurrent = Workbench(self.config, temp_dir=self.root / "parallel", batch_size=2)
        parallel_report = concurrent.run()

        def serial_config(config: dict) -> None:
            config["runtime"]["concurrency"] = 1

        serial_path = write_case(self.root / "serial-case", serial_config)
        serial = Workbench(serial_path, temp_dir=self.root / "serial", batch_size=2)
        serial_report = serial.run()
        self.assertEqual(parallel_report.to_dict()["frames"], serial_report.to_dict()["frames"])


class PreviewBrokerTests(unittest.TestCase):
    def test_budget_pending_returns_latest_completed_and_then_current(self) -> None:
        broker = PreviewBroker(10)
        broker.begin_run("run-new", {"phase": "denoise", "frames_done": 0})
        response = broker.request_preview(0)
        self.assertEqual(response.status, "pending")
        self.assertEqual(response.business_code, E_PREVIEW_BUDGET_PENDING)
        self.assertIsNone(response.preview)

        broker.publish_preview(
            "run-new",
            b"png-bytes",
            tier="tiny",
            prompt_id="p0",
            seed=11,
            frame_index=0,
            relative_path="runs/run-new/tiny/p0/11.png",
            progress={"phase": "preview_ready", "frames_done": 1},
        )
        ready = broker.request_preview(0)
        self.assertEqual(ready.status, "ok")
        self.assertIsNone(ready.business_code)
        self.assertEqual(ready.preview["tier"], "tiny")
        self.assertEqual(ready.preview["png_sha256"], hashlib.sha256(b"png-bytes").hexdigest())

        broker.complete_run("run-new", {"phase": "complete"})
        broker.begin_run("run-next", {"phase": "denoise"})
        stale = broker.request_preview(0)
        self.assertEqual(stale.status, "pending")
        self.assertEqual(stale.business_code, E_PREVIEW_BUDGET_PENDING)
        self.assertTrue(stale.stale)
        self.assertEqual(stale.preview["parameter_signature"], "run-new")

        broker.publish_preview(
            "run-next",
            b"new-png",
            tier="tiny",
            prompt_id="p0",
            seed=13,
            frame_index=1,
            relative_path="runs/run-next/tiny/p0/13.png",
            progress={"phase": "preview_ready", "frames_done": 1},
        )
        current = broker.request_preview(0)
        self.assertEqual(current.status, "ok")
        self.assertEqual(current.preview["parameter_signature"], "run-next")


if __name__ == "__main__":
    unittest.main()
