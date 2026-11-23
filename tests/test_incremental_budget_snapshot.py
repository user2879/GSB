from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from timbre_workbench.config import load_config, load_labels
from timbre_workbench.engine import (
    Workbench,
    load_or_initialize_workbench,
    semantic_projection,
)
from timbre_workbench.synth import generate_demo
from timbre_workbench.util import canonical_json


class IncrementalEqualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.demo = self.root / "demo"
        generate_demo(self.demo)
        self.base_config = load_config(
            self.demo / "config.json",
            overlay={
                "runtime": {
                    "first_response_budget_ms": 25,
                    "max_workers": 3,
                    "snapshot_after_each_command": False,
                }
            },
        )
        self.labels = load_labels(self.demo / "labels.json")
        self.commands = json.loads((self.demo / "commands.json").read_text(encoding="utf-8"))["commands"]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_incremental_is_precise_and_matches_full_rebuild(self) -> None:
        commands = list(self.commands) + [
            {"op": "set_assignment_metric", "metric": "normalized_l1", "rbf_gamma": 6.5},
            {"op": "noop"},
        ]
        incremental = Workbench.from_directory(
            self.demo / "audio",
            config=self.base_config,
            labels=self.labels,
            snapshot_manager=None,
            command_base_dir=self.demo,
        )
        incremental.rebuild_full(commit=False)
        incremental.run_commands(commands[:1], command_base_dir=self.demo)
        split_report = incremental.refresh_report()
        split_events = split_report["diagnostics"]["cache_events"]
        total_segments = split_report["summary"]["completed_segments"]
        self.assertGreaterEqual(split_events["reused_count"], total_segments - 2)
        self.assertLessEqual(split_events["recomputed_keys"]["feature"].__len__(), 2)
        self.assertGreater(split_events["invalidated_count"], 0)

        responses = incremental.run_commands(commands, command_base_dir=self.demo)
        incremental_report = incremental.refresh_report()
        self.assertTrue(
            all("E_COMMAND_INVALID" not in response["error_codes"] for response in responses)
        )
        self.assertEqual(incremental_report["summary"]["failure_count"], 0)
        final_events = incremental_report["diagnostics"]["cache_events"]
        self.assertEqual(final_events["recomputed_count"], 0)
        self.assertGreater(final_events["reused_count"], 0)
        self.assertLessEqual(
            incremental_report["diagnostics"]["threading"]["peak_workers_observed"],
            self.base_config["runtime"]["max_workers"],
        )

        full = Workbench.from_directory(
            self.demo / "audio",
            config=self.base_config,
            labels=self.labels,
            snapshot_manager=None,
            command_base_dir=self.demo,
        )
        full.run_commands(commands, command_base_dir=self.demo, force_full=True)
        full_report = full.refresh_report()
        self.assertEqual(
            canonical_json(semantic_projection(incremental_report)),
            canonical_json(semantic_projection(full_report)),
        )
        self.assertEqual(incremental_report["canonical_digest"], full_report["canonical_digest"])
        self.assertEqual(
            [row["sort_key"] for row in incremental_report["failures"]],
            [row["sort_key"] for row in full_report["failures"]],
        )


class BudgetAndSnapshotTests(unittest.TestCase):
    def test_pending_first_response_contains_latest_completed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            demo = root / "demo"
            generate_demo(demo)
            config = load_config(
                demo / "config.json",
                overlay={
                    "runtime": {
                        "first_response_budget_ms": 0,
                        "first_response_scheduling_tolerance_ms": 50.0,
                        "max_workers": 2,
                        "snapshot_after_each_command": False,
                    }
                },
            )
            labels = load_labels(demo / "labels.json")
            commands = json.loads((demo / "commands.json").read_text(encoding="utf-8"))["commands"]
            workbench = Workbench.from_directory(
                demo / "audio", config=config, labels=labels, command_base_dir=demo
            )
            responses = workbench.run_commands(commands[:2], command_base_dir=demo)
            self.assertTrue(all(response["status"] == "pending" for response in responses))
            for response in responses:
                self.assertEqual(response["first_response"]["error_code"], "E_FIRST_RESPONSE_PENDING")
                summary = response["first_response"]["summary"]
                for key in ("completed_segments", "speaker_count", "vector_dimension_count", "progress"):
                    self.assertIn(key, summary)
                self.assertIn("affected_segment_ids", summary)
                self.assertTrue(response["first_response"]["within_budget"])
            self.assertGreater(responses[1]["first_response"]["summary"]["completed_segments"], 0)

    def test_snapshot_resume_matches_full_and_stale_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            demo = root / "demo"
            generate_demo(demo)
            config = load_config(
                demo / "config.json",
                overlay={
                    "runtime": {
                        "first_response_budget_ms": 25,
                        "max_workers": 3,
                        "snapshot_after_each_command": True,
                        "snapshot_keep": 3,
                    }
                },
            )
            labels = load_labels(demo / "labels.json")
            commands = json.loads((demo / "commands.json").read_text(encoding="utf-8"))["commands"]
            work_dir = root / "work"
            first, warnings = load_or_initialize_workbench(
                input_dir=demo / "audio",
                config=config,
                labels=labels,
                commands=commands,
                work_dir=work_dir,
                resume=False,
                command_base_dir=demo,
            )
            self.assertEqual(warnings, [])
            first.run_commands(commands[:3], command_base_dir=demo)
            self.assertEqual(first.command_cursor, 3)
            self.assertIsNotNone(first.last_snapshot)
            self.assertTrue((work_dir / "CURRENT.json").exists())

            resumed, warnings = load_or_initialize_workbench(
                input_dir=demo / "audio",
                config=config,
                labels=labels,
                commands=commands,
                work_dir=work_dir,
                resume=True,
                command_base_dir=demo,
            )
            self.assertEqual(warnings, [])
            self.assertEqual(resumed.command_cursor, 3)
            resumed.run_commands(commands, command_base_dir=demo)
            resumed_report = resumed.refresh_report()

            full = Workbench.from_directory(
                demo / "audio",
                config=config,
                labels=labels,
                snapshot_manager=None,
                command_base_dir=demo,
            )
            full.run_commands(commands, command_base_dir=demo, force_full=True)
            full_report = full.refresh_report()
            self.assertEqual(resumed_report["canonical_digest"], full_report["canonical_digest"])
            self.assertEqual(
                canonical_json(semantic_projection(resumed_report)),
                canonical_json(semantic_projection(full_report)),
            )

            _, stale_warnings = load_or_initialize_workbench(
                input_dir=demo / "audio",
                config=config,
                labels=labels,
                commands=commands,
                work_dir=work_dir,
                resume=True,
                expected_global_version=999,
                command_base_dir=demo,
            )
            self.assertEqual([row["error_code"] for row in stale_warnings], ["E_SNAPSHOT_STALE"])
            snapshot_pointer = json.loads((work_dir / "CURRENT.json").read_text(encoding="utf-8"))
            self.assertTrue((work_dir / "snapshots" / snapshot_pointer["snapshot"] / "snapshot.json").exists())
            snapshot_manifest = json.loads(
                (work_dir / "snapshots" / snapshot_pointer["snapshot"] / "snapshot.json").read_text(encoding="utf-8")
            )
            self.assertIn("quantization", snapshot_manifest)
            self.assertIn("global_version", snapshot_manifest)
            self.assertIn("product_versions", snapshot_manifest)

            changed_config = load_config(
                demo / "config.json",
                overlay={
                    "assignment": {"similarity_threshold": 0.91},
                    "runtime": {
                        "first_response_budget_ms": 25,
                        "max_workers": 3,
                        "snapshot_after_each_command": True,
                    },
                },
            )
            _, config_stale = load_or_initialize_workbench(
                input_dir=demo / "audio",
                config=changed_config,
                labels=labels,
                commands=commands,
                work_dir=work_dir,
                resume=True,
                command_base_dir=demo,
            )
            self.assertEqual([row["error_code"] for row in config_stale], ["E_SNAPSHOT_STALE"])


if __name__ == "__main__":
    unittest.main()
