from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from rts_pipeline.canonical import atomic_write_json
from rts_pipeline.errors import PipelineError
from rts_pipeline.ledger import load_and_verify_ledger, verify_ledger
from rts_pipeline.runner import recover_workspace, run_workspace
from rts_pipeline.verifier import compare_ledgers

from .helpers import make_workspace, mutate_tick_state


class VerificationAndPersistenceTests(unittest.TestCase):
    def test_full_match_first_divergence_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            make_workspace(Path(root), seed=501, ticks=12, chunk_ticks=3)
            run = run_workspace(
                Path(root),
                threads=4,
                memory_budget_bytes=8 * 1024 * 1024,
                chunk_ticks=3,
            )
            self.assertEqual(run["status"], "ok")
            parallel = load_and_verify_ledger(Path(root) / "output" / "parallel_ledger.json")
            serial = load_and_verify_ledger(Path(root) / "output" / "serial_ledger.json")
            full = compare_ledgers(parallel, serial)
            self.assertTrue(full["full_consistency"])
            self.assertIsNone(full["first_divergence"])
            self.assertTrue(full["explicit_conclusion"].startswith("FULL_MATCH"))
            self.assertGreater(full["compared"]["faction_tick_pairs"], 0)

            divergent = copy.deepcopy(serial)
            mutate_tick_state(divergent, 4, "B")
            comparison = compare_ledgers(parallel, divergent)
            self.assertFalse(comparison["full_consistency"])
            self.assertEqual(comparison["first_divergence"]["tick"], 4)
            self.assertEqual(comparison["first_divergence"]["faction"], "B")
            self.assertIsNotNone(comparison["first_divergence"]["boundary_snapshot"])
            self.assertIn("participating_events", comparison["first_divergence"])

    def test_rng_drift_and_ledger_hash_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            make_workspace(Path(root), seed=601, ticks=8, chunk_ticks=4)
            run_workspace(Path(root), threads=2, memory_budget_bytes=8 * 1024 * 1024)
            ledger = load_and_verify_ledger(Path(root) / "output" / "parallel_ledger.json")

            rng_drift = copy.deepcopy(ledger)
            rng_drift["rng_records"][0]["value_u64"] ^= 1
            with self.assertRaises(PipelineError) as caught:
                verify_ledger(rng_drift)
            self.assertEqual(caught.exception.code, "E_RNG_DRIFT")

            hash_drift = copy.deepcopy(ledger)
            hash_drift["tick_records"][0]["snapshot"]["factions"]["A"]["resources"] += 2
            with self.assertRaises(PipelineError) as caught:
                verify_ledger(hash_drift)
            self.assertEqual(caught.exception.code, "E_LEDGER_HASH")

    def test_recovery_uses_only_committed_ledger_and_never_reconstructs_exit(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            make_workspace(Path(root), seed=701, ticks=8, chunk_ticks=4)
            run_workspace(Path(root), threads=2, memory_budget_bytes=8 * 1024 * 1024)
            output = Path(root) / "output"
            (output / "exit_report.json").unlink()
            (output / ".parallel_ledger.json.tmp-dead").write_text("partial", encoding="ascii")
            report = recover_workspace(Path(root))
            self.assertEqual(report["status"], "recovered_from_committed_ledger")
            self.assertFalse(report["exit_report_reconstructed"])
            self.assertFalse((output / "exit_report.json").exists())
            self.assertTrue((output / "recovery_report.json").exists())

            ledger_path = output / "parallel_ledger.json"
            corrupted = json.loads(ledger_path.read_text(encoding="utf-8"))
            corrupted["ledger_hash"] = "0" * 64
            atomic_write_json(ledger_path, corrupted)
            (output / "recovery_report.json").unlink()
            with self.assertRaises(PipelineError) as caught:
                recover_workspace(Path(root))
            self.assertEqual(caught.exception.code, "E_LEDGER_HASH")
            self.assertFalse((output / "recovery_report.json").exists())
            self.assertFalse((output / "exit_report.json").exists())


if __name__ == "__main__":
    unittest.main()

