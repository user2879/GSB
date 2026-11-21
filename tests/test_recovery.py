from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from memdiag.errors import RecoveryRequired
from memdiag.ledger import Ledger
from memdiag.services import recover_session, replay_session, resume_session, run_full_session
from tests.helpers import ROOT, TemporaryDirectoryTestCase, fixture_plan, fixture_samples


class ForcedTerminationRecoveryTests(TemporaryDirectoryTestCase):
    def test_crash_then_recover_and_replay(self):
        db = self.temp_dir / "crash.sqlite3"
        snapshots = self.temp_dir / "snapshots"
        command = [
            sys.executable,
            "-m",
            "memdiag.cli",
            "run",
            "--plan",
            str(ROOT / "data" / "plan.json"),
            "--samples",
            str(ROOT / "data" / "samples.json"),
            "--db",
            str(db),
            "--snapshots",
            str(snapshots),
            "--session-id",
            "crash-session",
            "--crash-after",
            "2",
            "--no-snapshot",
            "--report",
            str(self.temp_dir / "never-written.json"),
        ]
        process = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(process.returncode, 137, process.stderr)
        self.assertFalse((self.temp_dir / "never-written.json").exists())

        ledger = Ledger(db)
        try:
            scan = recover_session(ledger, "crash-session")
            self.assertFalse(scan["exit_report_available"])
            self.assertEqual(scan["status"], "needs_rerun")
            self.assertEqual(scan["committed_layers"], ["stem", "branch_a"])
            self.assertEqual(scan["unknown_layers"], ["branch_b", "merge", "head"])
            with self.assertRaises(RecoveryRequired):
                replay_session(ledger, "crash-session")
            resumed = resume_session(
                ledger,
                "crash-session",
                save_snapshot=True,
            )
            self.assertTrue(resumed["exit_report_available"])
            self.assertEqual(resumed["replay_verification"]["checked_layers"], ["stem", "branch_a"])
            self.assertEqual(resumed["rerun_layers"], ["branch_b", "merge", "head"])
            replay = replay_session(ledger, "crash-session")
            self.assertEqual(
                replay["checked_layers"],
                ["stem", "branch_a", "branch_b", "merge", "head"],
            )
            self.assertTrue(replay["exit_report_available"])
        finally:
            ledger.close()

    def test_unknown_batches_are_explicit_after_partial_commit(self):
        plan = fixture_plan()
        samples = fixture_samples()
        ledger = Ledger(self.temp_dir / "partial.sqlite3")
        try:
            from memdiag.engine import run_initial

            reference = run_initial(plan, samples)
            session = ledger.start_session(plan, samples, session_id="manual-partial")
            from memdiag.engine import _build_batch_groups, _layer_batch_commits

            groups = _build_batch_groups(samples, plan.max_batch_size)
            layer_id = "stem"
            result = reference.layers[layer_id]
            from memdiag.engine import MemoryBlockPool

            pool = MemoryBlockPool(
                plan.capacity_blocks,
                plan.block_size_bytes,
                reference.allocations,
                reference.allocation_modes,
            )
            batches = _layer_batch_commits(
                plan.layer_map[layer_id],
                result.sample_output_digests,
                result.sample_float_digests,
                pool.allocation(layer_id),
                groups,
                pool,
            )
            ledger.commit_layer(session, 0, result, batches)
            scan = recover_session(ledger, "manual-partial")
            self.assertEqual(scan["committed_layers"], ["stem"])
            self.assertEqual(scan["unknown_batches"][0]["layer_id"], "branch_a")
            self.assertFalse(scan["exit_report_available"])
        finally:
            ledger.close()


if __name__ == "__main__":
    unittest.main()
