from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from replica_repair.sample import (
    init_sample,
    run_demo,
    sample_ledger_path,
    sample_manifest_path,
)
from replica_repair.model import Manifest
from replica_repair.util import is_readonly


def strip_volatile(value):
    if isinstance(value, dict):
        return {
            key: strip_volatile(item)
            for key, item in value.items()
            if key != "ports"
        }
    if isinstance(value, list):
        return [strip_volatile(item) for item in value]
    return value


class EndToEndTests(unittest.TestCase):
    def test_three_version_demo_repair_staleness_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_demo(root)
            second = run_demo(root)
            self.assertEqual(
                json.dumps(strip_volatile(first), sort_keys=True, ensure_ascii=False),
                json.dumps(strip_volatile(second), sort_keys=True, ensure_ascii=False),
            )
            groups = {group["group_id"]: group for group in first["groups"]}
            self.assertEqual(set(groups), {"interop-v1", "interop-v2", "interop-v3"})
            self.assertEqual(groups["interop-v1"]["upgrade"]["source_version"], 1)
            self.assertEqual(groups["interop-v2"]["upgrade"]["source_version"], 2)
            self.assertEqual(groups["interop-v3"]["upgrade"]["source_version"], 3)
            for group in groups.values():
                self.assertEqual(group["upgrade"]["target_version"], 3)
                self.assertEqual(group["after"]["mismatches"], [])
                self.assertTrue(group["replay"]["ok"])
                manifest = Manifest.from_file(sample_manifest_path(root, group["group_id"]))
                self.assertEqual(manifest.format_version, 3)
            self.assertEqual(len(groups["interop-v3"]["before"]["mismatches"]), 4)
            self.assertEqual(groups["interop-v3"]["plan"]["total_repair_kb"], 10)
            self.assertGreaterEqual(len(groups["interop-v3"]["plan"]["queue_events"]), 1)
            self.assertTrue(
                all(event["reason"] == "per_node_quota" for event in groups["interop-v3"]["plan"]["queue_events"])
            )
            self.assertTrue(is_readonly(Path(str(sample_manifest_path(root, "interop-v1")) + ".v1.readonly.bak")))
            self.assertTrue(is_readonly(Path(str(sample_manifest_path(root, "interop-v2")) + ".v2.readonly.bak")))

            stale = first["stale_protection"]
            self.assertTrue(stale["evolution"]["version_incremented"])
            self.assertEqual(stale["evolution"]["manifest_version"], 3)
            self.assertEqual(stale["run"]["recovery"]["affected_nodes"], ["v3-c"])
            self.assertEqual(
                stale["run"]["recovery"]["affected_blocks"],
                ["blk-0001", "blk-0002", "blk-0003"],
            )
            self.assertTrue(stale["run"]["recovery"]["replaced"])
            self.assertTrue(stale["run"]["replay"]["ok"])
            for group_id in ("interop-v1", "interop-v2", "interop-v3"):
                self.assertTrue(sample_ledger_path(root, group_id).exists())

    def test_cli_init_run_report_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init_sample(root, force=True)
            manifest = sample_manifest_path(root, "interop-v3")
            ledger = sample_ledger_path(root, "interop-v3")
            from tests.helpers import run_cli

            run = run_cli(
                [
                    "run",
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest),
                    "--ledger",
                    str(ledger),
                    "--max-parallel",
                    "3",
                    "--per-node-kb",
                    "3",
                ],
                timeout=30,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            payload = json.loads(run.stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(all(not block["mismatches"] for block in payload["after_truth"]["blocks"]))
            report = run_cli(["report", "--ledger", str(ledger)])
            self.assertEqual(report.returncode, 0, report.stderr)
            report_payload = json.loads(report.stdout)
            self.assertIsNotNone(report_payload["current"])
            self.assertEqual(report_payload["current"]["phase"], "after")
            replay = run_cli(
                [
                    "replay",
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest),
                    "--ledger",
                    str(ledger),
                ]
            )
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertTrue(json.loads(replay.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
