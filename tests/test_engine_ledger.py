from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from config_arbiter.engine import ArbiterEngine
from config_arbiter.ledger import FORMAT_VERSION, Ledger
from config_arbiter.spec import load_specs
from config_arbiter.util import canonical_bytes
from tests.support import ROOT, build_engine, read_json, write_json


class InvalidationTests(unittest.TestCase):
    def test_source_change_invalidates_only_dependent_key_and_preserves_unaffected_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, ledger, _, environment_path, views_path = build_engine(root)
            try:
                engine.run_all()
                before_ops = ledger.current_view("ops-authority")
                self.assertIsNotNone(before_ops)
                db_before = canonical_bytes(before_ops["result"]["keys"]["db.pool_size"])
                _http_before = canonical_bytes(before_ops["result"]["keys"]["http.timeout_ms"])

                environment_document = read_json(environment_path)
                for environment in environment_document["environments"]:
                    if environment["environment_id"] != "prod-cn":
                        continue
                    for source in environment["sources"]:
                        if source["source_id"] != "prod-cn/override":
                            continue
                        for observation in source["observations"]:
                            if observation["key"] == "http.timeout_ms":
                                observation["value"] = 3008.0
                write_json(environment_path, environment_document)

                specs = load_specs(environment_path, views_path)
                second = ArbiterEngine(specs, ledger)
                report = second.run_all()
                ops_report = next(item for item in report["views"] if item["view_id"] == "ops-authority")
                self.assertEqual(ops_report["action"], "committed")
                self.assertEqual(ops_report["version"], 2)
                self.assertEqual(ops_report["affected_keys"], ["http.timeout_ms"])

                after_ops = ledger.current_view("ops-authority")
                self.assertIsNotNone(after_ops)
                self.assertEqual(after_ops["current_version"], 2)
                self.assertEqual(canonical_bytes(after_ops["result"]["keys"]["db.pool_size"]), db_before)
                self.assertNotEqual(canonical_bytes(after_ops["result"]["keys"]["http.timeout_ms"]), _http_before)

                stale = ledger.stale_findings("ops-authority", 2)
                self.assertEqual([item["key"] for item in stale], ["http.timeout_ms"])
                self.assertEqual(stale[0]["old_version"], 1)
                self.assertEqual(stale[0]["reason"], "source_chain_changed")
                self.assertTrue(any(".sources." in item for item in stale[0]["trace"]["changed_paths"]))

                history = ledger.view_history("ops-authority")
                self.assertEqual([item["version"] for item in history], [2, 1])
                self.assertEqual(history[1]["status"], "stale")
                self.assertEqual(ledger.current_view("release-guard")["current_version"], 1)

                view_document = read_json(views_path)
                for view in view_document["views"]:
                    if view["view_id"] == "ops-authority":
                        view["key_policies"]["db.pool_size"]["tolerance"] = 1
                write_json(views_path, view_document)
                third = ArbiterEngine(load_specs(environment_path, views_path), ledger)
                third.run_all()
                self.assertEqual(ledger.current_view("ops-authority")["current_version"], 3)
                policy_stale = ledger.stale_findings("ops-authority", 3)
                self.assertEqual([item["key"] for item in policy_stale], ["db.pool_size"])
                self.assertEqual(policy_stale[0]["reason"], "arbitration_policy_changed")
                self.assertEqual(
                    canonical_bytes(ledger.current_view("ops-authority")["result"]["keys"]["http.timeout_ms"]),
                    canonical_bytes(after_ops["result"]["keys"]["http.timeout_ms"]),
                )
            finally:
                ledger.close()

    def test_ledger_has_integer_format_version_and_stable_content_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, ledger, _, _, _ = build_engine(root)
            try:
                engine.run_all()
                meta = ledger.meta()
                self.assertEqual(int(meta["format_version"]), FORMAT_VERSION)
                first_digest = ledger.verify_integrity()
                self.assertEqual(ledger.verify_integrity(), first_digest)
                self.assertEqual(meta["content_digest"], first_digest)
            finally:
                ledger.close()

    def test_scale_policy_version_change_increments_version_and_changes_error_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, ledger, _, environment_path, views_path = build_engine(root)
            try:
                engine.run_all()
                first = ledger.current_view("ops-authority")
                self.assertEqual(first["result"]["scale_policy_version"], "scale-policy/1000.1")
                document = read_json(views_path)
                document["session"]["scale_power"] = 2
                document["session"]["scale_policy_version"] = "scale-policy/100.2"
                write_json(views_path, document)
                second = ArbiterEngine(load_specs(environment_path, views_path), ledger)
                second.run_all()
                current = ledger.current_view("ops-authority")
                self.assertEqual(current["current_version"], 2)
                self.assertEqual(current["result"]["scale_policy_version"], "scale-policy/100.2")
                self.assertEqual(current["result"]["scale_policy"]["scale_factor"], 100)
                self.assertEqual(current["result"]["keys"]["edge.latency_ms"]["authoritative"]["value_fixed"], 10000)
                self.assertNotEqual(
                    current["result"]["keys"]["edge.latency_ms"]["error_model"],
                    first["result"]["keys"]["edge.latency_ms"]["error_model"],
                )
                self.assertGreater(
                    current["result"]["view_aggregation_error_upper_bound_fixed_units"],
                    -1,
                )
            finally:
                ledger.close()


class ForcedTerminationTests(unittest.TestCase):
    def test_sigkill_keeps_only_committed_view_and_marks_rest_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment_path = ROOT / "samples" / "environments.json"
            views_path = ROOT / "samples" / "views.json"
            db_path = root / "ledger.sqlite"
            snapshot_path = root / "snapshot.json"
            log_path = root / "arbiter.jsonl"
            command = [
                sys.executable,
                "-m",
                "config_arbiter",
                "run",
                "--env",
                str(environment_path),
                "--views",
                str(views_path),
                "--db",
                str(db_path),
                "--snapshot",
                str(snapshot_path),
                "--log",
                str(log_path),
            ]
            environment = os.environ.copy()
            environment["CONFIG_ARBITER_TEST_CRASH_AFTER_VIEW"] = "1"
            crashed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(crashed.returncode, 0)
            self.assertFalse(snapshot_path.exists())
            self.assertNotIn("completed_views", crashed.stdout)

            ledger = Ledger(db_path)
            try:
                ledger.verify_integrity()
                specs = load_specs(environment_path, views_path)
                status = ArbiterEngine(specs, ledger).status_report()
                self.assertEqual(len(status["completed_views"]), 1)
                self.assertEqual(len(status["unknown_views"]), 2)
                self.assertTrue(all(item["requires_rerun"] for item in status["views"] if item["status"] != "complete"))
            finally:
                ledger.close()

            resumed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            ledger = Ledger(db_path)
            try:
                specs = load_specs(environment_path, views_path)
                status = ArbiterEngine(specs, ledger).status_report()
                self.assertEqual(status["unknown_views"], [])
                self.assertEqual(len(status["completed_views"]), 3)
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
