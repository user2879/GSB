from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from blockledger.blocks import initial_block_bytes
from blockledger.canonical import content_digest, spec_identity
from blockledger.engine import apply_events, replay_spec
from blockledger.errors import SpecError, VerifyError
from blockledger.invalidation import analyze_change
from blockledger.ledger import Ledger, open_session_ledger
from blockledger.pipeline import execute_run, recover_and_rerun_unknown
from blockledger.samplegen import build_sample_spec, sample_changes
from blockledger.sampling import is_selected
from blockledger.spec import normalize_spec
from blockledger.verify import verify_ledger
from blockledger.verification import verify_block_with_subprocess


ROOT = Path(__file__).resolve().parents[1]


def valid_spec() -> dict:
    return build_sample_spec()


class SpecValidationTests(unittest.TestCase):
    def test_sample_is_valid(self) -> None:
        spec = valid_spec()
        self.assertEqual(3, len(spec["volumes"]))
        self.assertEqual(4, len(spec["snapshots"]))
        self.assertEqual(14, len(spec["events"]))

    def test_duplicate_volume_reports_first_id_and_field(self) -> None:
        raw = copy.deepcopy(valid_spec())
        raw["volumes"].append(copy.deepcopy(raw["volumes"][0]))
        with self.assertRaises(SpecError) as ctx:
            normalize_spec(raw)
        self.assertEqual("E_SPEC", ctx.exception.code)
        self.assertEqual("vol-alpha", ctx.exception.entry_id)
        self.assertEqual("id", ctx.exception.field)

    def test_unsupported_block_size(self) -> None:
        raw = copy.deepcopy(valid_spec())
        raw["volumes"][0]["block_size"] = 1000
        with self.assertRaises(SpecError) as ctx:
            normalize_spec(raw)
        self.assertEqual("block_size", ctx.exception.field)
        self.assertEqual("vol-alpha", ctx.exception.entry_id)

    def test_snapshot_interval_out_of_range(self) -> None:
        raw = copy.deepcopy(valid_spec())
        raw["snapshots"][0]["intervals"] = [[0, 999]]
        with self.assertRaises(SpecError) as ctx:
            normalize_spec(raw)
        self.assertEqual("intervals", ctx.exception.field)
        self.assertEqual("snap-alpha-10", ctx.exception.entry_id)

    def test_event_seq_must_be_strict(self) -> None:
        raw = copy.deepcopy(valid_spec())
        raw["events"][1]["seq"] = raw["events"][0]["seq"]
        with self.assertRaises(SpecError) as ctx:
            normalize_spec(raw)
        self.assertEqual("seq", ctx.exception.field)


class DeterminismTests(unittest.TestCase):
    def test_content_digest_binds_metadata_and_bytes(self) -> None:
        volume = valid_spec()["volumes"][0]
        content = initial_block_bytes(volume, 3)
        digest = content_digest("vol-alpha", 3, 1, content)
        self.assertEqual(64, len(digest))
        self.assertNotEqual(digest, content_digest("vol-beta", 3, 1, content))
        self.assertNotEqual(digest, content_digest("vol-alpha", 4, 1, content))
        self.assertNotEqual(digest, content_digest("vol-alpha", 3, 2, content))
        self.assertNotEqual(digest, content_digest("vol-alpha", 3, 1, content + b"x"))

    def test_split_replay_and_shuffled_order_match(self) -> None:
        spec = valid_spec()
        whole = replay_spec(spec)

        for split_at in range(len(spec["events"]) + 1):
            partial_spec = {**spec, "events": spec["events"][:split_at]}
            split = replay_spec(partial_spec)
            apply_events(split, spec["events"][split_at:])
            self.assertEqual(whole.fingerprint(), split.fingerprint(), split_at)

        shuffled = copy.deepcopy(spec)
        shuffled["volumes"] = list(reversed(shuffled["volumes"]))
        shuffled["snapshots"] = list(reversed(shuffled["snapshots"]))
        normalized = normalize_spec(shuffled)
        self.assertEqual(whole.fingerprint(), replay_spec(normalized).fingerprint())

    def test_sample_selection_is_stable_and_bounded(self) -> None:
        values = [is_selected("rb", "v", i, 500) for i in range(100)]
        self.assertTrue(any(values))
        self.assertFalse(all(values))
        self.assertEqual(10, sum(is_selected("rb", "v", i, 1000) for i in range(10)))
        self.assertEqual(0, sum(is_selected("rb", "v", i, 0) for i in range(10)))


class InvalidationTests(unittest.TestCase):
    def test_write_subset_matches_full_rebuild(self) -> None:
        spec = valid_spec()
        report = analyze_change(spec, sample_changes()["change-write.json"])
        self.assertTrue(report["full_rebuild_consistency"]["match"])
        self.assertIn("snap-alpha-11", report["affected_snapshot_ids"])
        self.assertIn("rb-alpha-11", report["affected_rollback_ids"])
        self.assertTrue(
            any(item["block_index"] == 4 for item in report["changed_snapshot_fields"])
        )

    def test_delete_subset_matches_full_rebuild(self) -> None:
        spec = valid_spec()
        report = analyze_change(spec, sample_changes()["change-delete.json"])
        self.assertTrue(report["full_rebuild_consistency"]["match"])
        ledger_keys = {(item["rollback_id"], item["block_index"]) for item in report["ledger_keys"]}
        self.assertIn(("rb-alpha-11", 4), ledger_keys)
        self.assertIn(("rb-alpha-11", 9), ledger_keys)


class SubprocessVerificationTests(unittest.TestCase):
    def test_pass_corrupt_and_force_kill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "block.bin"
            content = b"deterministic-block" * 32
            path.write_bytes(content)
            digest = content_digest("vol", 7, 1, content)
            ok = verify_block_with_subprocess(
                block_file=path,
                volume_id="vol",
                block_index=7,
                baseline_version=1,
                expected_digest=digest,
            )
            self.assertEqual("pass", ok["verification_status"])

            bad = verify_block_with_subprocess(
                block_file=path,
                volume_id="vol",
                block_index=7,
                baseline_version=1,
                expected_digest=digest,
                corrupt_in_memory=True,
            )
            self.assertEqual("corrupt", bad["verification_status"])
            self.assertNotEqual(digest, bad["restored_digest"])

            killed = verify_block_with_subprocess(
                block_file=path,
                volume_id="vol",
                block_index=7,
                baseline_version=1,
                expected_digest=digest,
                force_kill=True,
                timeout_ms=30,
            )
            self.assertEqual("unknown", killed["verification_status"])
            self.assertTrue(killed["force_killed"])
            self.assertEqual(-signal.SIGKILL, killed["worker_returncode"])


class LedgerPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.spec_path = self.root / "spec.json"
        self.spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
        self.ledger = self.root / "ledger" / "test.sqlite"
        self.session = self.root / "ledger" / "session.json"
        self.content = self.root / "snapshots" / "content"
        self.live = self.root / "snapshots" / "live"
        self.work = self.root / "snapshots" / "work"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, **kwargs):
        options = dict(
            ledger_path=self.ledger,
            session_path=self.session,
            content_root=self.content,
            live_root=self.live,
            work_root=self.work,
            python_executable=sys.executable,
        )
        options.update(kwargs)
        return execute_run(valid_spec(), **options)

    def test_run_and_independent_verify(self) -> None:
        report = self._run()
        self.assertEqual("ok", report["status"])
        self.assertEqual(8, report["task_count"])
        verify = verify_ledger(valid_spec(), self.ledger)
        self.assertTrue(verify["ok"])
        self.assertEqual(28, verify["checked_blocks"])
        self.assertGreater(report["verification_before_invalidation"]["passed_blocks"], 0)
        self.assertGreater(report["verification_before_invalidation"]["unchecked_blocks"], 0)

        for metric in report["wait_metrics"]:
            occupancy = metric["quota_occupancy"]
            self.assertLessEqual(occupancy["global_active_after_acquire"], 2)
            self.assertLessEqual(occupancy["volume_active_after_acquire"], 1)
            self.assertIsNotNone(metric["wait_started_ms"])
            self.assertIsNotNone(metric["wait_ended_ms"])

    def test_verify_detects_tampered_status(self) -> None:
        self._run()
        with Ledger(self.ledger) as ledger:
            ledger.conn.execute(
                "UPDATE block_state_latest SET verification_status='pass' "
                "WHERE rollback_id='rb-beta-20' AND block_index=0"
            )
            ledger.conn.commit()
        with self.assertRaises(VerifyError) as ctx:
            verify_ledger(valid_spec(), self.ledger)
        self.assertEqual("E_VERIFY", ctx.exception.code)
        self.assertEqual("vol-beta", ctx.exception.volume_id)
        self.assertEqual(0, ctx.exception.block_index)
        self.assertEqual("verification_status", ctx.exception.field)

    def test_interrupted_lease_recovery_marks_all_blocks_unknown(self) -> None:
        ledger = Ledger(self.ledger)
        ledger.initialize_identity(spec_identity(valid_spec()))
        task = {
            "batch_id": "rb-beta-20:a01-b999",
            "rollback_id": "rb-beta-20",
            "event_seq": 12,
            "volume_id": "vol-beta",
            "snapshot_id": "snap-beta-20",
            "batch_index": 999,
            "blocks": [
                {"block_index": 0, "baseline_version": 1, "expected_digest": "a" * 64, "replay_block_version": 2},
                {"block_index": 1, "baseline_version": 1, "expected_digest": "b" * 64, "replay_block_version": 2},
            ],
            "max_blocks": 4,
            "global_concurrency": 2,
            "per_volume_concurrency": 1,
            "wait_started_ms": 1,
            "wait_ended_ms": 2,
            "_global_active": 1,
            "_volume_active": 1,
        }
        ledger.start_batch_lease(task, 1, 1)
        recovered = ledger.recover_interrupted()
        self.assertEqual(1, len(recovered))
        self.assertEqual(2, recovered[0]["blocks"])
        self.assertEqual(2, len(ledger.unknown_tasks()))
        ledger.close()

        report = recover_and_rerun_unknown(
            valid_spec(),
            ledger_path=self.ledger,
            session_path=self.session,
            content_root=self.content,
            work_root=self.work,
            rerun=True,
            python_executable=sys.executable,
        )
        self.assertEqual(2, report["unknown_before_rerun"])
        self.assertEqual(1, report["rerun_batch_count"])

    def test_session_identity_mismatch_creates_readonly_backup(self) -> None:
        self._run()
        changed = valid_spec()
        changed["budget"]["verification_sample_permille"] = 1000
        report = self._run(spec=changed) if False else execute_run(
            changed,
            ledger_path=self.ledger,
            session_path=self.session,
            content_root=self.content,
            live_root=self.live,
            work_root=self.work,
            python_executable=sys.executable,
        )
        self.assertIsNotNone(report["session_replacement"])
        diff = report["session_replacement"]
        backup = Path(diff["backup_database"])
        self.assertTrue(backup.exists())
        mode = backup.stat().st_mode
        self.assertFalse(mode & 0o200)
        self.assertTrue(Path(diff["differences"][0] and self.root / "ledger" / "backups").exists())

    def test_sigkill_cli_then_recover_full_verify(self) -> None:
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        command = [
            sys.executable,
            "-m",
            "blockledger.cli",
            "--root",
            str(self.root),
            "run",
            "--spec",
            str(self.spec_path),
            "--only-rollback",
            "rb-beta-20",
            "--crash-after-lease-batch",
            "rb-beta-20:a01-b000",
        ]
        crashed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(-signal.SIGKILL, crashed.returncode, crashed.stderr)

        recovery = recover_and_rerun_unknown(
            valid_spec(),
            ledger_path=self.root / "ledger" / "blockledger.sqlite",
            session_path=self.root / "ledger" / "session.json",
            content_root=self.root / "snapshots" / "content",
            work_root=self.root / "snapshots" / "work",
            rerun=True,
            python_executable=sys.executable,
        )
        self.assertEqual(1, len(recovery["recovered_interrupted"]))
        self.assertEqual(4, recovery["unknown_before_rerun"])

        execute_run(
            valid_spec(),
            ledger_path=self.root / "ledger" / "blockledger.sqlite",
            session_path=self.root / "ledger" / "session.json",
            content_root=self.root / "snapshots" / "content",
            live_root=self.root / "snapshots" / "live",
            work_root=self.root / "snapshots" / "work",
            attempt=2,
            python_executable=sys.executable,
        )
        verify = verify_ledger(valid_spec(), self.root / "ledger" / "blockledger.sqlite")
        self.assertTrue(verify["ok"])


if __name__ == "__main__":
    unittest.main()
