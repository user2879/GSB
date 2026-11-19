from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from batch_audit.audit import run_audit
from batch_audit.engine import Scheduler, WorkspacePaths
from batch_audit.ledger import Ledger

from .helpers import assignment_map, make_workspace, run_cli


class CrashRecoveryTests(unittest.TestCase):
    def test_force_exit_keeps_only_committed_slots_and_resume_redoes_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ws, _ = make_workspace(Path(temp))
            paths = WorkspacePaths.under(ws)
            crashed = run_cli(["run", "--workspace", str(ws), "--crash-after-commits", "2"])
            self.assertEqual(crashed.returncode, 99, crashed.stderr + crashed.stdout)
            self.assertFalse(paths.report.exists(), "an aborted process must not leave an exit report")

            with Ledger(paths.ledger) as ledger:
                generation = ledger.latest_generation()
                assert generation is not None
                slots = ledger.slots(generation["generation_id"])
                self.assertEqual([row["status"] for row in slots], ["committed", "committed", "pending", "pending", "pending"])
                before = {
                    str(item["sample_id"]): (
                        int(item["batch_slot"]),
                        int(item["rank"]),
                        str(item["assignment_id"]),
                        str(item["batch_output_digest"]),
                    )
                    for item in ledger.assignments(generation["generation_id"])
                    if int(item["batch_slot"]) in {0, 1}
                }

            partial = run_audit(ledger_path=paths.ledger, feed_path=paths.feed)
            self.assertEqual(partial["code"], "AUDIT_PARTIAL")
            self.assertEqual(partial["auditable_slots"], [0, 1])
            self.assertEqual([item["batch_slot"] for item in partial["not_auditable_slots"]], [2, 3, 4])

            resumed = Scheduler(paths).run()
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["recovery"]["unknown_slots_before_redo"], [2, 3, 4])
            self.assertEqual(resumed["recovery"]["redone_slots"], [2, 3, 4])
            self.assertTrue(paths.report.exists())
            after_map = assignment_map(resumed)
            for sample_id, value in before.items():
                self.assertEqual(after_map[sample_id], value)
            final = run_audit(ledger_path=paths.ledger, feed_path=paths.feed)
            self.assertEqual(final["code"], "AUDIT_OK")
            self.assertEqual(final["audit_scope"], "full")


if __name__ == "__main__":
    unittest.main()
