from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cm_fusion.engine import RetrievalEngine
from cm_fusion.sample_data import generate_samples
from cm_fusion.verification import compare_snapshots


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.samples = cls.root / "samples"
        generate_samples(cls.samples, 20261120)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_incremental_matches_full_and_selective_reuse(self) -> None:
        incremental = RetrievalEngine(self.root / "incremental_state")
        incremental.build(
            self.samples / "docs.jsonl",
            self.samples / "queries.jsonl",
            self.samples / "config.json",
            self.samples / "compat.json",
        )
        outcomes = {}
        for line in (self.samples / "edits.jsonl").read_text(encoding="utf-8").splitlines():
            command = json.loads(line)
            outcomes[command["command_id"]] = incremental.apply_edit(command, budget_ms=120000)
        tier_switch = outcomes["E003"].result
        position_switch = outcomes["E004"].result
        weights_switch = outcomes["E005"].result
        format_switch = outcomes["E006"].result
        self.assertGreater(tier_switch["index"]["reuse_report"]["rebuilt_block_count"], 0)
        self.assertEqual(tier_switch["index"]["reuse_report"]["reused_block_count"], 0)
        self.assertEqual(tier_switch["query_reuse"]["reused_query_count"], 8)
        self.assertEqual(position_switch["index"]["reuse_report"]["reused_block_count"], 0)
        self.assertEqual(position_switch["query_reuse"]["reused_query_count"], 8)
        self.assertGreater(weights_switch["index"]["reuse_report"]["reused_block_count"], 0)
        self.assertEqual(weights_switch["index"]["reuse_report"]["rebuilt_block_count"], 0)
        self.assertEqual(weights_switch["query_reuse"]["rebuilt_query_count"], 8)
        self.assertEqual(format_switch["index"]["reuse_report"]["reused_block_count"], 0)
        self.assertIn("format_version_changed", format_switch["invalidation"]["index_reasons"])

        full = RetrievalEngine(self.root / "full_state")
        full.build(
            self.samples / "final" / "docs.jsonl",
            self.samples / "final" / "queries.jsonl",
            self.samples / "final" / "config.json",
            self.samples / "final" / "compat.json",
        )
        incremental_current = incremental.manager.load_current()
        full_current = full.manager.load_current()
        self.assertIsNotNone(incremental_current)
        self.assertIsNotNone(full_current)
        comparison = compare_snapshots(incremental_current, full_current)
        self.assertTrue(comparison["ok"], comparison["mismatches"])
        self.assertEqual(comparison["incremental"]["index_fingerprint"], comparison["full"]["index_fingerprint"])

    def test_budget_pending_and_resume_mismatch(self) -> None:
        engine = RetrievalEngine(self.root / "budget_state")
        build = engine.build(
            self.samples / "docs.jsonl",
            self.samples / "queries.jsonl",
            self.samples / "config.json",
            self.samples / "compat.json",
        )
        self.assertEqual(build.status, "OK")
        before = engine.manager.load_current()
        pending = engine.apply_edit(
            {"command_id": "B001", "op": "set_config", "patch": {"top_k": 7}},
            budget_ms=0,
        )
        after = engine.manager.load_current()
        self.assertEqual(pending.status, "E_INTERACT_BUDGET_PENDING")
        self.assertEqual(pending.result["completed_stage"], "command_received")
        self.assertEqual(pending.result["summary"]["completed_queries"], 8)
        self.assertEqual(before.version, after.version)
        self.assertTrue((engine.state_dir / "budget_events.jsonl").is_file())
        resumed = engine.resume(expected_version=before.version)
        self.assertEqual(resumed.version, before.version)
        orphan = engine.state_dir / "snapshots" / "v99999999"
        orphan.mkdir(parents=True)
        resumed = engine.resume(expected_version=before.version)
        self.assertFalse(orphan.exists())
        with self.assertRaises(Exception) as mismatch:
            engine.resume(expected_config_path=self.samples / "final" / "config.json")
        self.assertEqual(getattr(mismatch.exception, "code", None), "E_CONFIG_MISMATCH")


if __name__ == "__main__":
    unittest.main()
