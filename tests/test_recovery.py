from __future__ import annotations

import subprocess
import sys
import unittest

from mlpipe_workbench.engine import Engine
from mlpipe_workbench.errors import E_REPLAY, WorkbenchError
from mlpipe_workbench.ledger import Ledger
from mlpipe_workbench.recovery import RecoveryManager

from tests.helpers import DATA, PIPELINE, ROOT, TempDirectoryMixin, prepared_base


class RecoveryTests(TempDirectoryMixin, unittest.TestCase):
    def _crash_run(self, crash_stage: str, db_name: str) -> subprocess.CompletedProcess[str]:
        db = self.tempdir / db_name
        command = [
            sys.executable,
            "-m",
            "mlpipe_workbench",
            "run",
            "--data",
            str(DATA),
            "--pipeline",
            str(PIPELINE),
            "--db",
            str(db),
            "--log",
            str(self.tempdir / (db_name + ".jsonl")),
            "--run-id",
            "crashed",
            "--batch-size",
            "16",
            "--crash-after-stage",
            crash_stage,
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 75, result.stderr)
        return result

    def test_forced_termination_recovers_only_committed_stages_then_replays(self) -> None:
        self._crash_run("fill_missing", "crash.db")
        db = self.tempdir / "crash.db"
        with Ledger(db) as ledger:
            before = ledger.get_run("crashed")
            self.assertEqual(before["status"], "RUNNING")
            report = RecoveryManager(ledger).recover("crashed")
            after = ledger.get_run("crashed")
            attempts = ledger.stage_attempts("crashed")
            unknown_rows = [row for rows in attempts.values() for row in rows if row["status"] == "UNKNOWN"]
            conclusion = ledger.current_conclusion()
        inspection = report["inspection"]
        self.assertFalse(inspection["exit_report_available"])
        self.assertEqual(
            [stage["stage_id"] for stage in inspection["recovered_stages"]],
            ["canonicalize", "fill_missing"],
        )
        self.assertEqual(
            [stage["stage_id"] for stage in inspection["unknown_stages"]],
            ["normalize_amount", "clip_amount", "pack"],
        )
        self.assertEqual(
            inspection["rerun_plan"],
            ["normalize_amount", "clip_amount", "pack"],
        )
        self.assertEqual(report["replay"]["status"], "OK")
        self.assertEqual(report["replay"]["stages_verified"], 5)
        self.assertEqual(report["replay"]["shards_verified"], 5)
        self.assertEqual(after["status"], "RECOVERED_COMPLETE")
        self.assertTrue(unknown_rows)
        self.assertIsNotNone(conclusion)
        self.assertEqual(conclusion["run_id"], "crashed")

    def test_crash_after_final_stage_still_completes_ledger_transaction(self) -> None:
        self._crash_run("pack", "crash_pack.db")
        db = self.tempdir / "crash_pack.db"
        with Ledger(db) as ledger:
            before = ledger.get_run("crashed")
            self.assertEqual(before["status"], "RUNNING")
            self.assertIsNone(before["final_data_digest"])
            report = RecoveryManager(ledger).recover("crashed")
        self.assertEqual(report["inspection"]["unknown_stages"], [])
        self.assertEqual(report["inspection"]["rerun_plan"], [])
        self.assertEqual(report["replay"]["status"], "OK")

    def test_replay_uses_the_normalize_scale_recorded_on_the_run(self) -> None:
        with Ledger(self.tempdir / "scale.db") as ledger:
            result = Engine(ledger).run(
                prepared_base(),
                run_id="scale",
                batch_size=16,
                normalize_scale=4,
            )
            view = ledger.current_conclusion()["view"]
            replay = RecoveryManager(ledger).replay("scale")
        self.assertEqual(view["normalize_scale"], 4)
        self.assertEqual(replay["status"], "OK")
        self.assertEqual(replay["final_data_digest"], result.final_data_digest)

    def test_tampered_stage_observation_raises_e_replay_with_first_field(self) -> None:
        with Ledger(self.tempdir / "tamper.db") as ledger:
            Engine(ledger).run(prepared_base(), run_id="tamper", batch_size=16)
            ledger.conn.execute(
                "UPDATE stage_runs SET output_digest=? WHERE run_id=? AND stage_id=? AND attempt=1",
                ("0" * 64, "tamper", "fill_missing"),
            )
            with self.assertRaises(WorkbenchError) as captured:
                RecoveryManager(ledger).replay("tamper")
            errors = ledger.list_errors(run_id="tamper")
            report = ledger.get_report("tamper", "replay")
        error = captured.exception
        self.assertEqual(error.code, E_REPLAY)
        self.assertEqual(error.stage_id, "fill_missing")
        self.assertEqual(error.param, "output_digest")
        self.assertEqual(error.details["first_stage_id"], "fill_missing")
        self.assertEqual(error.details["first_field"], "output_digest")
        self.assertTrue(errors)
        self.assertEqual(errors[0]["code"], E_REPLAY)
        self.assertEqual(report["payload"]["status"], "FAILED")

    def test_tampered_shard_fingerprint_raises_e_replay(self) -> None:
        with Ledger(self.tempdir / "shard_tamper.db") as ledger:
            Engine(ledger).run(prepared_base(), run_id="tamper", batch_size=16)
            shard_rows = ledger.shards_for_run("tamper")
            target = shard_rows[1]
            ledger.conn.execute(
                "UPDATE shards SET shard_fingerprint=? WHERE run_id=? AND stage_id=? "
                "AND stage_attempt=? AND shard_index=?",
                (
                    "0" * 64,
                    "tamper",
                    target["stage_id"],
                    target["stage_attempt"],
                    target["shard_index"],
                ),
            )
            with self.assertRaises(WorkbenchError) as captured:
                RecoveryManager(ledger).replay("tamper")
        error = captured.exception
        self.assertEqual(error.code, E_REPLAY)
        self.assertEqual(error.stage_id, "pack")
        self.assertEqual(error.details["first_field"], "shards[1].shard_fingerprint")


if __name__ == "__main__":
    unittest.main()
