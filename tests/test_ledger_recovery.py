from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gsb.canonical import canonical_json
from gsb.errors import DiagnosticError
from gsb.engine import run_session
from gsb.invalidation import compare_ledger_with_rebuild
from gsb.ledger import BatchLedgerRecord, Ledger
from gsb.model import BatchSpec, CheckpointSpec, EventSpec
from gsb.params import compute_batch_summary, get_parameter_set
from gsb.leases import LeaseManager
from gsb.scheduler import SchedulerJob
from tests.helpers import make_session, make_shard


class LedgerRecoveryTests(unittest.TestCase):
    def _build_session(self):
        shard = make_shard(parameter_version=2)
        base = CheckpointSpec("base", shard.shard_id, 1, "1" * 64)
        v2 = CheckpointSpec("v2", shard.shard_id, 2, "2" * 64)
        batches = (
            BatchSpec("b1", shard.shard_id, (1, 2), 1, 1, 100),
            BatchSpec("b2", shard.shard_id, (3,), 1, 2, 100),
            BatchSpec("b3", shard.shard_id, (4,), 1, 30, 100),
        )
        events = (
            EventSpec(10, "commit", shard.shard_id, checkpoint_id="v2"),
            EventSpec(20, "rollback", shard.shard_id, checkpoint_id="base"),
            EventSpec(40, "commit", shard.shard_id, checkpoint_id="base"),
        )
        return make_session(
            (shard,), checkpoints=(base, v2), batches=batches, events=events
        )

    def test_run_replay_and_full_rebuild(self) -> None:
        session = self._build_session()
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "ledger.sqlite"
            result = run_session(session, ledger_path=ledger_path, startup_timeout_s=5)
            self.assertTrue(result["ok"])
            self.assertEqual(result["checkpoint_state"]["current_versions"][session.shards[0].shard_id], 1)
            self.assertEqual(result["checkpoint_state"]["generations"][session.shards[0].shard_id], 1)
            self.assertTrue(all(batch["status"] == "committed" for batch in result["batches"]))
            ledger = Ledger.open(ledger_path, session)
            try:
                formats = {
                    row["format_version"]
                    for row in ledger.connection.execute(
                        "SELECT format_version FROM batches"
                    ).fetchall()
                }
                self.assertEqual(formats, {1})
                self.assertEqual(ledger.replay()["checked_batches"], 3)
                consistency = compare_ledger_with_rebuild(session, ledger)
                self.assertTrue(consistency["consistent"])
            finally:
                ledger.close()

    def test_two_runs_have_byte_identical_checkpoint_state(self) -> None:
        session = self._build_session()
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = run_session(
                session, ledger_path=Path(first_dir) / "ledger.sqlite", startup_timeout_s=5
            )
            second = run_session(
                session, ledger_path=Path(second_dir) / "ledger.sqlite", startup_timeout_s=5
            )
            self.assertEqual(
                canonical_json(first["checkpoint_state"]),
                canonical_json(second["checkpoint_state"]),
            )

    def test_force_kill_recovery_marks_unknown_and_reruns(self) -> None:
        shard = make_shard()
        batch_committed = BatchSpec("b1", shard.shard_id, (1,), 1, 1, 100)
        batch_pending = BatchSpec("b2", shard.shard_id, (2, 3), 1, 2, 100)
        session = make_session((shard,), batches=(batch_committed, batch_pending))
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "ledger.sqlite"
            ledger = Ledger.open(ledger_path, session)
            version = shard.parameter_version
            digest = get_parameter_set(shard, version).digest
            job_committed = SchedulerJob(
                batch=batch_committed,
                shard=shard,
                parameter_version=version,
                parameter_digest=digest,
                error_bound=1,
            )
            lease_committed = LeaseManager().issue(shard.shard_id, 1000)
            ledger.start_batch(job_committed, lease_committed.to_dict(), now_ms=1)
            summary = compute_batch_summary(
                shard, batch_committed.batch_id, batch_committed.sample_ids, version
            )
            ledger.commit_batch(
                BatchLedgerRecord(
                    sequence=batch_committed.order,
                    batch_id=batch_committed.batch_id,
                    shard_id=shard.shard_id,
                    batch_order=batch_committed.order,
                    parameter_version=version,
                    parameter_digest=digest,
                    loss_value=str(summary["loss"]),
                    loss_digest=summary["loss_digest"],
                    sample_ids=batch_committed.sample_ids,
                    lease_id=lease_committed.lease_id,
                    lease_version=lease_committed.lease_version,
                    lease_owner=lease_committed.owner,
                    checkpoint_state={"parameter_version": version, "generation": 0, "checkpoints": {}},
                    progress_cursor=batch_committed.order,
                    committed_at_ms=2,
                )
            )
            job_pending = SchedulerJob(
                batch=batch_pending,
                shard=shard,
                parameter_version=version,
                parameter_digest=digest,
                error_bound=1,
            )
            lease_pending = LeaseManager().issue(shard.shard_id, 1000)
            ledger.start_batch(job_pending, lease_pending.to_dict(), now_ms=3)
            # The child is really SIGKILLed; the pending attempt is the only
            # durable evidence left for recovery.
            from gsb.process import WorkerPool

            pool = WorkerPool(session, startup_timeout_s=5)
            try:
                pool.start()
                pool.kill(shard.shard_id)
            finally:
                pool.stop(force=True)
            ledger.close()

            reopened = Ledger.open(ledger_path, session)
            try:
                report = reopened.recover(now_ms=2)
                self.assertEqual(report["requires_rerun"], ["b2"])
                self.assertEqual(report["committed_batches"], ["b1"])
                state = reopened.connection.execute(
                    "SELECT state FROM leases WHERE lease_id=?", (lease_pending.lease_id,)
                ).fetchone()
                self.assertEqual(state["state"], "reclaimed")
            finally:
                reopened.close()
            rerun = run_session(session, ledger_path=ledger_path, startup_timeout_s=5)
            self.assertTrue(rerun["ok"])
            self.assertEqual(
                [item["batch_id"] for item in rerun["batches"]],
                ["b1", "b2"],
            )
            self.assertEqual(rerun["batches"][0]["status"], "already_committed")
            self.assertEqual(rerun["batches"][1]["status"], "committed")
            final = Ledger.open(ledger_path, session)
            try:
                self.assertTrue(final.replay()["consistent"])
            finally:
                final.close()

    def test_snapshot_identity_mismatch_replaces_whole_snapshot(self) -> None:
        shard = make_shard()
        session = make_session((shard,), batches=(BatchSpec("b", shard.shard_id, (1,), 1, 1),))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            first = Ledger.open(path, session)
            first.close()
            # A second config with a different shard set must replace, not merge.
            other_shard = make_shard("other", port=shard.port + 1)
            other_session = make_session((other_shard,))
            original = Ledger.open(path, session)
            epoch_before = original.snapshot_info()["epoch"]
            original.close()
            replacement = Ledger.open(path, other_session)
            try:
                info = replacement.snapshot_info()
                self.assertGreater(info["epoch"], epoch_before)
                self.assertIn("shard_set_version mismatch", info["replaced_reason"])
                self.assertEqual(replacement.committed_records(), ())
            finally:
                replacement.close()

    def test_content_digest_mismatch_replaces_snapshot(self) -> None:
        shard = make_shard()
        session = make_session((shard,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            ledger = Ledger.open(path, session)
            ledger.connection.execute(
                "UPDATE meta SET value=? WHERE key='content_digest'", ("0" * 64,)
            )
            epoch = ledger.snapshot_info()["epoch"]
            ledger.close()
            reopened = Ledger.open(path, session)
            try:
                info = reopened.snapshot_info()
                self.assertGreater(info["epoch"], epoch)
                self.assertIn("content_digest", info["replaced_reason"])
            finally:
                reopened.close()

    def test_replay_reports_first_batch_and_field(self) -> None:
        session = self._build_session()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            run_session(session, ledger_path=path, startup_timeout_s=5)
            ledger = Ledger.open(path, session)
            try:
                ledger.connection.execute(
                    "UPDATE batches SET loss_digest='bad' WHERE batch_id='b1'"
                )
                with self.assertRaises(DiagnosticError) as caught:
                    ledger.replay()
                error = caught.exception
                self.assertEqual(error.code, "E_REPLAY")
                self.assertEqual(error.details["batch_id"], "b1")
                self.assertEqual(error.details["shard_id"], session.shards[0].shard_id)
                self.assertEqual(error.details["field"], "content_digest")
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
