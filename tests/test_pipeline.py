from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from qplan.ledger import Ledger
from qplan.pipeline import run_pipeline
from qplan.spec import load_spec
from qplan.spec import apply_overrides
from qplan.verify import verify_ledger

from tests.helpers import small_spec_dict, write_spec


class PipelineTests(unittest.TestCase):
    def test_run_verify_and_result_digests(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec_path = write_spec(root / "spec.json")
            spec = load_spec(spec_path)
            report = run_pipeline(
                spec,
                spec_path=spec_path,
                ledger_path=root / "ledger.sqlite",
                db_path=root / "sample.sqlite",
                batch_size=1,
            )
            self.assertEqual(report["session"]["status"], "completed")
            self.assertEqual(len(report["batches"]), 2)
            self.assertTrue(all(query["result_match"] for query in report["queries"]))
            self.assertTrue(all(query["all_paths_match_baseline"] for query in report["queries"]))
            verification = verify_ledger(
                spec_path=spec_path,
                ledger_path=root / "ledger.sqlite",
            )
            self.assertEqual(verification["code"], "OK")
            self.assertEqual(verification["queries_checked"], 2)
            self.assertGreater(verification["paths_checked"], 0)

    def test_invalidation_creates_new_plan_version_and_replaces_view(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec_path = write_spec(root / "spec.json")
            ledger_path = root / "ledger.sqlite"
            db_path = root / "sample.sqlite"
            first_spec = load_spec(spec_path)
            first = run_pipeline(
                first_spec,
                spec_path=spec_path,
                ledger_path=ledger_path,
                db_path=db_path,
                batch_size=2,
            )
            changed_data = small_spec_dict()
            changed_data["tables"][0]["columns"][1]["selectivity_hint"] = 0.31
            changed_path = write_spec(root / "spec-changed.json", changed_data)
            changed_spec = load_spec(changed_path)
            second = run_pipeline(
                changed_spec,
                spec_path=changed_path,
                ledger_path=ledger_path,
                db_path=db_path,
                batch_size=2,
            )
            self.assertEqual(second["snapshot"]["snapshot_version"], first["snapshot"]["snapshot_version"] + 1)
            self.assertEqual(second["snapshot"]["plan_version"], first["snapshot"]["plan_version"] + 1)
            self.assertEqual(second["invalidation"]["change_kind"], "content_changed")
            affected_ids = [item["query_id"] for item in second["invalidation"]["affected_queries"]]
            self.assertIn("Q_events", affected_ids)
            self.assertIn("Q_users", affected_ids)
            with sqlite3.connect(ledger_path) as connection:
                statuses = dict(
                    connection.execute("SELECT snapshot_version, status FROM snapshots ORDER BY snapshot_version")
                )
                old_batch_statuses = [
                    row[0]
                    for row in connection.execute(
                        "SELECT status FROM batches WHERE snapshot_version=?",
                        (first["snapshot"]["snapshot_version"],),
                    )
                ]
            self.assertEqual(statuses[first["snapshot"]["snapshot_version"]], "stale")
            self.assertEqual(statuses[second["snapshot"]["snapshot_version"]], "current")
            self.assertTrue(old_batch_statuses)
            self.assertTrue(all(status == "stale" for status in old_batch_statuses))
            view = Ledger.read_only(ledger_path)
            try:
                current = view.load_current_view()
            finally:
                view.close()
            current_ids = {
                query["query_id"]
                for batch in current["batches"]
                for query in batch["payload"]["queries"]
            }
            self.assertEqual(current_ids, {"Q_events", "Q_users"})
            verification = verify_ledger(spec_path=changed_path, ledger_path=ledger_path)
            self.assertEqual(verification["code"], "OK")

    def test_verify_uses_ledger_session_weights_without_cli_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec_path = write_spec(root / "spec.json")
            ledger_path = root / "ledger.sqlite"
            spec = apply_overrides(
                load_spec(spec_path),
                weights={"rows": 2, "comparisons": 3, "lookups": 5, "io_pages": 7},
            )
            run_pipeline(
                spec,
                spec_path=spec_path,
                ledger_path=ledger_path,
                db_path=root / "sample.sqlite",
                batch_size=2,
            )
            result = verify_ledger(spec_path=spec_path, ledger_path=ledger_path)
            self.assertEqual(result["code"], "OK")
            self.assertIn("cost_weights", result["declaration_session_field_differences"])

    def test_expected_snapshot_mismatch_replaces_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec_path = write_spec(root / "spec.json")
            ledger_path = root / "ledger.sqlite"
            spec = load_spec(spec_path)
            first = run_pipeline(
                spec,
                spec_path=spec_path,
                ledger_path=ledger_path,
                db_path=root / "sample.sqlite",
                batch_size=2,
            )
            second = run_pipeline(
                spec,
                spec_path=spec_path,
                ledger_path=ledger_path,
                db_path=root / "sample.sqlite",
                batch_size=2,
                expected_snapshot_version=first["snapshot"]["snapshot_version"] + 99,
            )
            self.assertEqual(second["snapshot"]["snapshot_version"], first["snapshot"]["snapshot_version"] + 1)
            self.assertEqual(
                second["snapshot"]["replacement_reason"],
                "expected_snapshot_version_or_digest_mismatch",
            )
            self.assertEqual(verify_ledger(spec_path=spec_path, ledger_path=ledger_path)["code"], "OK")

    def test_forced_kill_recovery_and_partial_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec_path = write_spec(root / "spec.json")
            ledger_path = root / "ledger.sqlite"
            db_path = root / "sample.sqlite"
            command = [
                sys.executable,
                "-m",
                "qplan.cli",
                "run",
                "--spec",
                str(spec_path),
                "--ledger",
                str(ledger_path),
                "--db",
                str(db_path),
                "--batch-size",
                "1",
                "--crash-after-batches",
                "1",
            ]
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 75, completed.stderr)
            ledger = Ledger(ledger_path)
            try:
                recovery = ledger.recover()
                view = ledger.load_current_view()
            finally:
                ledger.close()
            self.assertEqual(recovery["status"], "recovered_partial")
            self.assertEqual(len(recovery["committed_batches"]), 1)
            self.assertEqual(len(recovery["unknown_batches"]), 1)
            self.assertEqual(view["code"], "E_RECOVERY")
            blocked = verify_ledger(spec_path=spec_path, ledger_path=ledger_path)
            self.assertEqual(blocked["code"], "E_RECOVERY")
            partial = verify_ledger(spec_path=spec_path, ledger_path=ledger_path, allow_partial=True)
            self.assertEqual(partial["code"], "OK")
            self.assertEqual(partial["queries_checked"], 1)
            self.assertEqual(len(partial["skipped_queries"]), 1)

    def test_ledger_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec_path = write_spec(root / "spec.json")
            ledger_path = root / "ledger.sqlite"
            spec = load_spec(spec_path)
            run_pipeline(
                spec,
                spec_path=spec_path,
                ledger_path=ledger_path,
                db_path=root / "sample.sqlite",
                batch_size=2,
            )
            with sqlite3.connect(ledger_path) as connection:
                payload_json = connection.execute(
                    "SELECT payload_json FROM batches ORDER BY batch_no LIMIT 1"
                ).fetchone()[0]
                payload = json.loads(payload_json)
                payload["queries"][0]["path_estimates"][next(iter(payload["queries"][0]["path_estimates"]))]["cost"] += 1
                connection.execute(
                    "UPDATE batches SET payload_json=? WHERE batch_no=(SELECT MIN(batch_no) FROM batches)",
                    (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
                )
                connection.commit()
            result = verify_ledger(spec_path=spec_path, ledger_path=ledger_path)
            self.assertEqual(result["code"], "E_LEDGER")
            replacement = run_pipeline(
                spec,
                spec_path=spec_path,
                ledger_path=ledger_path,
                db_path=root / "sample.sqlite",
                batch_size=2,
            )
            self.assertEqual(replacement["snapshot"]["snapshot_version"], 2)
            self.assertEqual(
                replacement["snapshot"]["replacement_reason"], "snapshot_or_digest_mismatch"
            )
            self.assertTrue(replacement["replacement_detected_errors"])
            self.assertEqual(verify_ledger(spec_path=spec_path, ledger_path=ledger_path)["code"], "OK")


if __name__ == "__main__":
    unittest.main()
