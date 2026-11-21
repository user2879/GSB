import json
import os
import subprocess
import sys
import tempfile
import unittest

from emergency_drill.cli import main
from emergency_drill.repository import Ledger
from emergency_drill.sample_data import generate_samples


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        generate_samples(self.dir)
        self.plan = os.path.join(self.dir, "plan_valid.json")
        self.events = os.path.join(self.dir, "events_valid.json")
        self.ledger = os.path.join(self.dir, "drill.sqlite3")
        self.run_output = os.path.join(self.dir, "run.json")
        self.edit_output = os.path.join(self.dir, "edit.json")
        self.snapshot = os.path.join(self.dir, "snapshot.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _cli(self, *args):
        return main(list(args))

    def test_full_pipeline_incremental_consistency_and_snapshot(self):
        self.assertEqual(self._cli("run", "--plan", self.plan, "--events", self.events,
                                   "--ledger", self.ledger, "--output", self.run_output), 0)
        changeset = {
            "operations": [
                {"op": "update_duration", "stage_id": "recover", "duration_ms": 1700},
                {"op": "rewrite_criterion", "stage_id": "recover",
                 "criterion": {"mode": "all_of", "required_actions": ["a_service_restore"]}},
            ]
        }
        changeset_path = os.path.join(self.dir, "changeset.json")
        with open(changeset_path, "w", encoding="utf-8") as handle:
            json.dump(changeset, handle)
        self.assertEqual(self._cli("edit", "--ledger", self.ledger,
                                   "--changeset", changeset_path,
                                   "--output", self.edit_output), 0)
        edit_payload = json.load(open(self.edit_output, encoding="utf-8"))
        self.assertEqual(edit_payload["invalid_stage_ids"], ["recover"])
        self.assertEqual(edit_payload["frozen_stage_ids"],
                         ["detect", "isolate", "failover", "verify"])
        self.assertTrue(edit_payload["verification"]["matched"])

        before = {item["stage_id"]: item for item in json.load(open(self.run_output, encoding="utf-8"))["outcomes"]}
        after = {item["stage_id"]: item for item in edit_payload["outcomes"]}
        for sid in edit_payload["frozen_stage_ids"]:
            before_frozen = {k: v for k, v in before[sid].items() if k != "plan_version"}
            after_frozen = {k: v for k, v in after[sid].items() if k != "plan_version"}
            self.assertEqual(json.dumps(before_frozen, sort_keys=True),
                             json.dumps(after_frozen, sort_keys=True))

        self.assertEqual(self._cli("verify-incr", "--ledger", self.ledger), 0)
        self.assertEqual(self._cli("snapshot", "--ledger", self.ledger,
                                   "--output", self.snapshot), 0)
        self.assertEqual(self._cli("replay", "--ledger", self.ledger,
                                   "--snapshot", self.snapshot), 0)

        ledger = Ledger(self.ledger)
        self.assertEqual(ledger.latest_version(), 2)
        self.assertEqual(len(ledger.list_versions()), 2)
        ledger.close()

    def test_e_plan_and_e_cycle_cli_codes_and_structured_log(self):
        log_path = os.path.join(self.dir, "log.jsonl")
        corrupt = os.path.join(self.dir, "plan_corrupt.json")
        code = main(["--log", log_path, "validate", "--plan", corrupt,
                     "--events", self.events])
        self.assertEqual(code, 2)
        cycle = os.path.join(self.dir, "plan_cycle.json")
        code = main(["--log", log_path, "validate", "--plan", cycle,
                     "--events", self.events, "--inject", "cycle"])
        self.assertEqual(code, 2)
        with open(log_path, encoding="utf-8") as handle:
            codes = {json.loads(line)["code"] for line in handle}
        self.assertIn("E_PLAN", codes)
        self.assertIn("E_CYCLE", codes)

    def test_sigkill_subprocess_recovery_unknown_and_rerun(self):
        crash_ledger = os.path.join(self.dir, "crash.sqlite3")
        worker = [
            sys.executable, "-m", "emergency_drill.crash_worker",
            "--plan", self.plan, "--events", self.events,
            "--ledger", crash_ledger, "--crash-after-stage", "detect",
        ]
        proc = subprocess.run(worker, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("simulated_sigkill", proc.stdout)

        recover_output = os.path.join(self.dir, "recover.json")
        self.assertEqual(main(["recover", "--ledger", crash_ledger,
                               "--output", recover_output]), 2)
        with open(recover_output, encoding="utf-8") as handle:
            recover_payload = json.load(handle)
        self.assertEqual(recover_payload["unknown_stages"], ["detect"])
        self.assertFalse(recover_payload["can_build_exit_report"])
        self.assertIn("REFUSED", recover_payload["exit_report"])

        rerun_output = os.path.join(self.dir, "rerun.json")
        self.assertEqual(main(["rerun-after-crash", "--ledger", crash_ledger,
                               "--output", rerun_output]), 0)
        ledger = Ledger(crash_ledger)
        self.assertEqual(ledger.stats(1)["unknown"], 0)
        self.assertEqual(ledger.recover(1)["can_build_exit_report"], True)
        ledger.close()

    def test_new_round_lists_replaced_stale_identifiers(self):
        self.assertEqual(main(["run", "--plan", self.plan, "--events", self.events,
                               "--ledger", self.ledger, "--output", self.run_output]), 0)
        code = main(["new-round", "--plan", self.plan, "--events", self.events,
                     "--ledger", self.ledger])
        self.assertEqual(code, 0)
        ledger = Ledger(self.ledger)
        self.assertEqual(ledger.latest_version(), 2)
        stale = ledger.conn.execute(
            "SELECT COUNT(*) AS n FROM outcomes WHERE stale=1"
        ).fetchone()["n"]
        self.assertEqual(stale, 5)
        ledger.close()


if __name__ == "__main__":
    unittest.main()
