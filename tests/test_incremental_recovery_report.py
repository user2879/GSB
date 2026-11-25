from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from edge_ledger.engine import build_workspace
from edge_ledger.errors import SimulatedCrash
from edge_ledger.report import compare_quick_view, write_reports
from edge_ledger.snapshot import SnapshotStore
from edge_ledger.verify import run_verification
from edge_ledger.workspace import WorkspaceEditor

from .helpers import make_sample_workspace


class IncrementalRecoveryReportTests(unittest.TestCase):
    def test_replace_threshold_and_quantization_invalidation_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_sample_workspace(Path(temporary) / "workspace")
            initial = build_workspace(root)
            original = next(row for row in initial.result["samples"] if row["sample_id"] == "S0003")
            editor = WorkspaceEditor.from_root(root)
            replacement = next(
                sample
                for batch in editor.sample_batches
                for sample in batch["samples"]
                if sample["sample_id"] == "S0003"
            )
            replacement["slot_fills"][0]["value"] = "changed topic value"
            editor.save()
            replaced = build_workspace(root)
            render_actions = {
                item["entity_id"]: item["action"]
                for item in replaced.result["execution"]["cache_ledger"]
                if item["stage"] == "render"
            }
            recomputed = sorted(key for key, action in render_actions.items() if action == "recompute")
            self.assertEqual(recomputed, ["S0003"])
            changed = next(row for row in replaced.result["samples"] if row["sample_id"] == "S0003")
            self.assertNotEqual(original["render_digest"], changed["render_digest"])

            editor = WorkspaceEditor.from_root(root)
            editor.config["dedupe"]["threshold_permille"] = 910
            editor.save()
            thresholded = build_workspace(root)
            threshold_render = [
                item
                for item in thresholded.result["execution"]["cache_ledger"]
                if item["stage"] == "render"
            ]
            self.assertTrue(all(item["action"] == "reuse" for item in threshold_render))
            dedupe_entry = next(
                item
                for item in thresholded.result["execution"]["cache_ledger"]
                if item["stage"] == "dedupe"
            )
            self.assertEqual(dedupe_entry["action"], "recompute")

            editor = WorkspaceEditor.from_root(root)
            editor.config["fixed_point"] = {"decimal_places": 6, "quantization": "Q10000"}
            editor.save()
            quantized = build_workspace(root)
            self.assertTrue(
                all(
                    item["action"] == "reuse"
                    for item in quantized.result["execution"]["cache_ledger"]
                    if item["stage"] == "render"
                )
            )
            self.assertTrue(
                any(
                    item["action"] == "recompute"
                    for item in quantized.result["execution"]["cache_ledger"]
                    if item["stage"] == "score"
                )
            )

    def test_incremental_equals_full_after_all_edits(self) -> None:
        verification = run_verification(Path(".").resolve())
        self.assertTrue(verification.passed)
        self.assertEqual(len(verification.report["edit_steps"]), 10)
        self.assertTrue(
            all(step["incremental_vs_full"] for step in verification.report["edit_steps"])
        )

    def test_snapshot_crash_keeps_previous_pointer_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_sample_workspace(Path(temporary) / "workspace")
            first = build_workspace(root)
            store = SnapshotStore(root / "state")
            before = store.load_current().snapshot.generation
            with self.assertRaises(SimulatedCrash):
                build_workspace(root, crash_after_stage="render", crash_mode="raise")
            after = store.load_current().snapshot.generation
            self.assertEqual(before, after)
            recovered = build_workspace(root)
            self.assertEqual(first.canonical_digest, recovered.canonical_digest)
            manifest = store.load_current().snapshot.manifest
            self.assertEqual(manifest["config_hash"], first.result["config_hash"])
            self.assertEqual(manifest["template_chain_fingerprint"], first.result["template_chain_fingerprint"])
            self.assertIn("render", manifest["stages"])

    def test_report_and_quick_view_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_sample_workspace(Path(temporary) / "workspace")
            result = build_workspace(root)
            comparison = compare_quick_view(result.result)
            self.assertTrue(comparison["matches"])
            paths = write_reports(result)
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["markdown"].exists())
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertIn("per_template_sample_counts", payload["report"])
            self.assertIn("render_points_consumed", payload["report"])
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("Per-template counts", markdown)
            self.assertIn("Dedupe removed", markdown)

    def test_output_ordering_is_root_path_and_concurrency_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first_root = make_sample_workspace(Path(temporary) / "one" / "workspace")
            second_root = Path(temporary) / "two" / "workspace"
            shutil.copytree(first_root, second_root)
            import yaml

            first_config = yaml.safe_load((first_root / "config" / "pipeline.yaml").read_text(encoding="utf-8"))
            first_config["scheduler"]["max_workers"] = 1
            (first_root / "config" / "pipeline.yaml").write_text(
                yaml.safe_dump(first_config, sort_keys=False), encoding="utf-8"
            )
            second_config = yaml.safe_load((second_root / "config" / "pipeline.yaml").read_text(encoding="utf-8"))
            second_config["scheduler"]["max_workers"] = 4
            (second_root / "config" / "pipeline.yaml").write_text(
                yaml.safe_dump(second_config, sort_keys=False), encoding="utf-8"
            )
            single = build_workspace(first_root)
            parallel = build_workspace(second_root)
            self.assertEqual(single.result["samples"], parallel.result["samples"])
            self.assertEqual(single.result["clusters"], parallel.result["clusters"])
            self.assertEqual(single.result["failures"], parallel.result["failures"])
            self.assertEqual(single.result["pending_queue"], parallel.result["pending_queue"])

    def test_pending_queue_is_persisted_in_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_sample_workspace(Path(temporary) / "workspace")
            import yaml

            config = yaml.safe_load((root / "config" / "pipeline.yaml").read_text(encoding="utf-8"))
            config["scheduler"]["render_point_budget"] = 3
            (root / "config" / "pipeline.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            result = build_workspace(root)
            self.assertGreater(len(result.result["pending_queue"]), 0)
            snapshot = SnapshotStore(root / "state").load_current().snapshot
            self.assertEqual(snapshot.pending_queue, result.result["pending_queue"])
            self.assertTrue(all(item["code"] == "E_RENDER_BUDGET_PENDING" for item in snapshot.pending_queue))


if __name__ == "__main__":
    unittest.main()
