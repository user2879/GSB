from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mixedmem.errors import DiagnosticError
from mixedmem.ledger import Ledger
from mixedmem.runtime import InferenceRuntime, recover_and_resume
from mixedmem.schema import parse_plan
from mixedmem.snapshot import build_snapshot, load_snapshot, save_snapshot
from mixedmem.util import atomic_write_json

from .helpers import PLAN_PATH, ROOT, plan_copy


class SnapshotRecoveryTests(unittest.TestCase):
    def test_snapshot_roundtrip_and_mismatch(self) -> None:
        plan = parse_plan(plan_copy())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root / "ledger.db")
            runtime = InferenceRuntime(plan, 8, batch_size=4, ledger=ledger)
            runtime.run(worker_count=2)
            payload = runtime.snapshot_payload(cursor={"batch_id": "b0001", "layer_id": "head"})
            path = save_snapshot(root / "snapshots", payload)
            loaded = load_snapshot(path, current_plan=runtime.plan)
            self.assertTrue(loaded.replaced_whole_state)
            self.assertEqual(loaded.plan.version, 1)
            self.assertEqual(loaded.layer_summaries, runtime.layer_summaries())
            runtime.apply_change("hidden", {"scale": 256})
            loaded_old = load_snapshot(
                path,
                current_plan=runtime.plan,
                current_ledger_digest=ledger.content_digest(),
            )
            self.assertTrue(any(item["item"].endswith("hidden.scale") for item in loaded_old.differences))
            bad = root / "bad.json"
            corrupt = dict(payload)
            corrupt["content_digest"] = "0" * 64
            atomic_write_json(bad, corrupt)
            with self.assertRaises(DiagnosticError) as context:
                load_snapshot(bad, current_plan=runtime.plan)
            self.assertEqual(context.exception.code, "E_SNAPSHOT")
            ledger.close()

    def test_quantization_scale_version_must_align(self) -> None:
        plan = parse_plan(plan_copy())
        other = replace(plan, quantization_scale_version=2)
        with tempfile.TemporaryDirectory() as directory:
            payload = build_snapshot(
                other,
                sample_count=8,
                dataset_digest=None,
                lease_records=[],
                layer_summaries={},
                processed_batch_cursor=None,
                ledger_content_digest=None,
            )
            path = save_snapshot(Path(directory), payload)
            with self.assertRaises(DiagnosticError) as context:
                load_snapshot(path, current_plan=plan)
            self.assertEqual(context.exception.code, "E_SNAPSHOT")

    def test_in_memory_snapshot_replacement(self) -> None:
        plan = parse_plan(plan_copy())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root / "ledger.db")
            runtime = InferenceRuntime(plan, 8, batch_size=4, ledger=ledger)
            runtime.run(worker_count=2)
            original_summaries = runtime.layer_summaries()
            path = save_snapshot(root / "snapshots", runtime.snapshot_payload())
            runtime.apply_change("hidden", {"scale": 256})
            self.assertEqual(runtime.plan.version, 2)
            replacement = runtime.restore_snapshot(
                path, current_ledger_digest=ledger.content_digest()
            )
            self.assertTrue(replacement["memory_replaced"])
            self.assertEqual(runtime.plan.version, 1)
            self.assertEqual(runtime.layer_summaries(), original_summaries)
            self.assertEqual(runtime.results, {})
            ledger.close()

    def test_ledger_digest_detects_tampering(self) -> None:
        plan = parse_plan(plan_copy())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.db"
            ledger = Ledger(path)
            runtime = InferenceRuntime(plan, 4, batch_size=2, ledger=ledger)
            runtime.run(worker_count=1)
            ledger.verify_integrity()
            ledger._conn.execute(
                "UPDATE layer_results SET float_digest='tampered' WHERE 1=1"
            )
            # Deliberately do not update the stored aggregate digest.
            with self.assertRaises(DiagnosticError) as context:
                ledger.verify_integrity()
            self.assertEqual(context.exception.code, "E_REPLAY")
            ledger.close()

    def test_forced_termination_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "ledger.db"
            snapshots = root / "snapshots"
            snapshots.mkdir()
            command = [
                sys.executable,
                "-m",
                "mixedmem",
                "run",
                "--plan",
                str(PLAN_PATH),
                "--db",
                str(db),
                "--snapshots",
                str(snapshots),
                "--samples",
                "8",
                "--batch-size",
                "4",
                "--workers",
                "1",
                "--crash-after-committed-layers",
                "5",
            ]
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 91)
            ledger = Ledger(db)
            self.assertTrue(ledger.verify_integrity()["ok"])
            result = recover_and_resume(
                plan=parse_plan(plan_copy()),
                sample_count=8,
                batch_size=4,
                ledger=ledger,
                worker_count=2,
            )
            self.assertFalse(result["uncommitted_exit_report_available"])
            self.assertGreaterEqual(len(result["unknown_batches"]), 1)
            self.assertGreaterEqual(len(result["unknown_leases"]), 1)
            self.assertTrue(result["replay_verification"]["ok"])
            self.assertTrue(result["complete_after_resume"])
            self.assertTrue(ledger.verify_integrity()["ok"])
            ledger.close()


if __name__ == "__main__":
    unittest.main()
