from __future__ import annotations

import os
import subprocess
import sys
import unittest

from config_diag.clock import ManualClock
from config_diag.compare import compare_incremental_and_full
from config_diag.engine import ConfigEngine
from config_diag.errors import DiagnosticError
from config_diag.recovery import recover_and_replay
from config_diag.structured_log import JsonlLogger, NullLogger
from config_diag.validation import load_bundle

from .support import ROOT, SAMPLES, local_temp


class CompareAndRecoveryTests(unittest.TestCase):
    def test_incremental_and_full_paths_are_key_identical(self) -> None:
        with local_temp("compare-ok-") as tmp:
            result = compare_incremental_and_full(SAMPLES, tmp, now_ns=1_000_000_000)
        self.assertTrue(result["ok"])
        self.assertEqual(result["keys_compared"], 11)
        self.assertEqual(result["operations_compared"], 4)
        self.assertTrue(result["invalidation_sets_equal"])

    def test_compare_injection_raises_first_e_diverge(self) -> None:
        with local_temp("compare-bad-") as tmp:
            with self.assertRaises(DiagnosticError) as raised:
                compare_incremental_and_full(
                    SAMPLES,
                    tmp,
                    inject_diverge=True,
                    now_ns=1_000_000_000,
                )
        exc = raised.exception
        self.assertEqual(exc.code, "E_DIVERGE")
        self.assertIsNotNone(exc.key)
        self.assertEqual(exc.field, "value")
        self.assertIn("incremental", exc.context)
        self.assertIn("full_rebuild", exc.context)

    def test_clean_recovery_replays_values_digests_versions_and_traces(self) -> None:
        spec, clients, changes = load_bundle(SAMPLES)
        with local_temp("replay-clean-") as tmp:
            clock = ManualClock(1_000_000_000)
            engine = ConfigEngine(tmp, logger=NullLogger(), clock=clock)
            try:
                engine.initialize(spec, clients, now_ns=1_000_000_000)
                for operation in changes.operations:
                    engine.apply_operation(operation, now_ns=1_000_000_000)
                clock.advance_ms(181)
                engine.collect_expired()
                result = recover_and_replay(engine, tmp)
            finally:
                engine.close()
        self.assertTrue(result["ok"])
        self.assertFalse(result["needs_rerun"])
        self.assertFalse(result["exit_report_available"])
        self.assertEqual(result["keys_compared"], 11)
        self.assertEqual(result["operations_replayed"], 5)
        self.assertEqual(result["invalidation_traces_compared"], 5)

    def test_forced_termination_leaves_unknown_and_requires_rerun(self) -> None:
        spec, clients, changes = load_bundle(SAMPLES)
        with local_temp("crash-") as tmp:
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "PYTHONIOENCODING": "utf-8",
            }
            base = [
                sys.executable,
                "-m",
                "config_diag",
                "--workspace",
                str(tmp),
                "--samples",
                str(SAMPLES),
            ]
            init_run = subprocess.run(base + ["init"], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(init_run.returncode, 0, init_run.stderr)
            first_run = subprocess.run(base + ["apply", "--sequence", "1"], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(first_run.returncode, 0, first_run.stderr)
            crash_run = subprocess.run(
                base + ["apply", "--sequence", "2", "--crash-point", "after-intent"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(crash_run.returncode, 137)
            self.assertEqual(crash_run.stdout.strip(), "")

            engine = ConfigEngine(tmp, logger=JsonlLogger(tmp / "logs" / "diagnostics.jsonl"), clock=ManualClock(2_000_000_000))
            try:
                self.assertTrue(any(event["code"] == "E_RECOVERY" for event in engine.open_events))
                recovered = recover_and_replay(engine, tmp)
                self.assertTrue(recovered["needs_rerun"])
                self.assertFalse(recovered["exit_report_available"])
                self.assertEqual(recovered["committed_change_cursor"], 1)
                self.assertEqual(recovered["unknown_transactions"][0]["operation_id"], "op-002-group")
                records = engine.ledger.records()
                self.assertEqual(records["tax.rate_permille"]["value"], 8)
                self.assertEqual(records["ops.batch_size"]["value"], 25)
                rerun = engine.apply_operation(changes.operations[1], now_ns=2_100_000_000)
                self.assertEqual(rerun["journal_id"], recovered["unknown_transactions"][0]["journal_id"])
                rechecked = recover_and_replay(engine, tmp)
                self.assertFalse(rechecked["needs_rerun"])
                self.assertEqual(rechecked["committed_change_cursor"], 2)
                self.assertEqual(engine.ledger.records()["tax.rate_permille"]["value"], 10)
            finally:
                engine.close()

    def test_before_commit_crash_rolls_back_keys_but_keeps_intent(self) -> None:
        with local_temp("crash-before-commit-") as tmp:
            env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
            base = [
                sys.executable,
                "-m",
                "config_diag",
                "--workspace",
                str(tmp),
                "--samples",
                str(SAMPLES),
            ]
            self.assertEqual(subprocess.run(base + ["init"], cwd=ROOT, env=env, capture_output=True).returncode, 0)
            self.assertEqual(
                subprocess.run(base + ["apply", "--sequence", "1"], cwd=ROOT, env=env, capture_output=True).returncode,
                0,
            )
            crashed = subprocess.run(
                base + ["apply", "--sequence", "2", "--crash-point", "before-commit"],
                cwd=ROOT,
                env=env,
                capture_output=True,
            )
            self.assertEqual(crashed.returncode, 137)
            engine = ConfigEngine(tmp, logger=NullLogger(), clock=ManualClock(2_000_000_000))
            try:
                self.assertTrue(any(event["code"] == "E_RECOVERY" for event in engine.open_events))
                self.assertEqual(engine.ledger.meta()["ledger_version"], 2)
                records = engine.ledger.records()
                self.assertEqual(records["tax.rate_permille"]["value"], 8)
                self.assertEqual(records["checkout.total"]["value"], 1350)
                unknown = engine.ledger.unresolved_unknown()
                self.assertEqual(unknown[0]["operation_id"], "op-002-group")
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
