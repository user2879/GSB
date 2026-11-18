from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mixedprec_diag.config import parse_config
from mixedprec_diag.engine import DiagnosticSession, RunRequest
from mixedprec_diag.errors import E_DATA, DiagnosticError
from mixedprec_diag.versioning import ConfigVersionManager
from tests.helpers import config_payload


class VersionManagerTests(unittest.TestCase):
    def test_relevant_change_increments_and_replacement_is_stable(self) -> None:
        manager = ConfigVersionManager()
        first = parse_config(config_payload(batch_size_limit=4, precision="float32"))
        pending = manager.begin(first)
        replacement = manager.commit(pending, first, ["B0002", "B0001"])
        self.assertEqual(replacement["previous_version"], None)
        self.assertEqual(replacement["current_version"], 1)
        self.assertTrue(replacement["replaced_whole"])

        second = parse_config(config_payload(batch_size_limit=6, precision="float32"))
        pending = manager.begin(second)
        replacement = manager.commit(pending, second, ["B0001", "B0002"])
        self.assertEqual(replacement["previous_version"], 1)
        self.assertEqual(replacement["current_version"], 2)
        self.assertEqual(replacement["changed_fields"], ["batch_size_limit"])
        self.assertEqual(replacement["affected_batches"], ["B0001", "B0002"])
        self.assertEqual(replacement["stale_batch_ids"], ["B0001", "B0002"])

        same = parse_config(config_payload(batch_size_limit=6, precision="float32"))
        pending = manager.begin(same)
        replacement = manager.commit(pending, same, ["B0001", "B0002"])
        self.assertEqual(replacement["current_version"], 2)
        self.assertEqual(replacement["changed_fields"], [])

    def test_preallocation_change_also_evolves_version(self) -> None:
        manager = ConfigVersionManager()
        first = parse_config(config_payload(preallocation="per_batch"))
        manager.commit(manager.begin(first), first, ["B0001"])
        second = parse_config(config_payload(preallocation="block_pool"))
        replacement = manager.commit(manager.begin(second), second, ["B0001"])
        self.assertEqual(replacement["current_version"], 2)
        self.assertIn("preallocation", replacement["changed_fields"])


class SessionReplacementTests(unittest.TestCase):
    def test_digest_mismatch_aborts_with_report_and_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text('{"data_sha256":"' + "0" * 64 + '"}', encoding="utf-8")
            session = DiagnosticSession(
                RunRequest(seed=1, sample_count=4, input_dim=4, hidden_dim=4),
                report_dir=root / "reports",
                snapshot_dir=root / "snapshots",
            )
            config = parse_config(config_payload())
            with self.assertRaises(DiagnosticError) as raised:
                session.run_config(config, expected_manifest_path=manifest)
            self.assertEqual(raised.exception.code, E_DATA)
            self.assertEqual(raised.exception.field_path, "$.data_sha256")
            self.assertIsNone(session.current_report)
            aborted = root / "reports" / "aborted_E_DATA.json"
            self.assertTrue(aborted.exists())
            import json

            payload = json.loads(aborted.read_text(encoding="utf-8"))
            self.assertFalse(payload["inference_started"])
            self.assertEqual(payload["summary"]["codes"], [E_DATA])
            self.assertEqual(payload["summary"]["error"]["field_path"], "$.data_sha256")
            self.assertEqual(len(payload["artifacts"]["data_sha256"]), 64)

    def test_three_versions_whole_replace_and_exclude_stale_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = DiagnosticSession(
                RunRequest(seed=44, sample_count=8, input_dim=6, hidden_dim=8),
                report_dir=root / "reports",
                snapshot_dir=root / "snapshots",
            )
            configs = [
                parse_config(config_payload(batch_size_limit=4, precision="float32", preallocation="per_batch")),
                parse_config(config_payload(batch_size_limit=4, precision="fixed", quant_scale_power=4)),
                parse_config(
                    config_payload(
                        memory_budget_mib=2,
                        batch_size_limit=3,
                        precision="fixed",
                        quant_scale_power=5,
                    )
                ),
            ]
            reports = [session.run_config(config) for config in configs]
            self.assertEqual([report["config_version"] for report in reports], [1, 2, 3])
            self.assertEqual(reports[1]["replacement"]["previous_version"], 1)
            self.assertIn("precision", reports[1]["replacement"]["changed_fields"])
            self.assertEqual(reports[2]["replacement"]["previous_version"], 2)
            self.assertIn("batch_size_limit", reports[2]["replacement"]["changed_fields"])
            self.assertIn("quant_scale_power", reports[2]["replacement"]["changed_fields"])
            self.assertIn("memory_budget_mib", reports[2]["replacement"]["changed_fields"])
            self.assertTrue(reports[2]["replacement"]["replaced_whole"])
            self.assertEqual(reports[2]["replacement"]["stale_marker"], "STALE")
            self.assertNotIn("stale", reports[2])

            stale_path = root / "snapshots" / "version_0002_stale.json"
            self.assertTrue(stale_path.exists())
            original_snapshot = root / "snapshots" / "version_0002_fixed.json"
            self.assertTrue(original_snapshot.exists())
            stale_report = root / "reports" / "version_0002.json"
            self.assertTrue(stale_report.exists())
            import json

            stale = json.loads(stale_path.read_text(encoding="utf-8"))
            self.assertEqual(stale["stale"]["marker"], "STALE")
            self.assertEqual(stale["stale"]["stale_batch_ids"], ["B0001", "B0002"])
            self.assertNotEqual(stale["run_id"], session.current_report["run_id"])
            self.assertEqual(json.loads(stale_report.read_text(encoding="utf-8"))["stale"]["marker"], "STALE")
            self.assertEqual(json.loads(original_snapshot.read_text(encoding="utf-8"))["stale"]["marker"], "STALE")

            session.query_view()
            session.query_batch(session.current_report["batches"][0]["batch_id"])
            self.assertEqual(session.current_report["ui_latency"]["query_count"], 2)
            self.assertEqual(session.current_report["config_version"], 3)

            session_summary = session.write_session_summary()
            statuses = {entry["version"]: entry["status"] for entry in session_summary["history"]}
            self.assertEqual(statuses[1], "STALE")
            self.assertEqual(statuses[2], "STALE")
            self.assertEqual(statuses[3], "CURRENT")
            self.assertTrue((root / "reports" / "current.json").exists())
            self.assertTrue((root / "reports" / "version_0003.json").exists())


if __name__ == "__main__":
    unittest.main()
