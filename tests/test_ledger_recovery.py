import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from coljoin.canonical import digest
from coljoin.errors import ReplayError
from coljoin.ledger import LEDGER_FORMAT_VERSION, Ledger

RECEIPT = {
    "shard_id": "P1:0000",
    "aggregate": {"1": {"count": 3}},
}
COST = {"vector": {"total_cost": 100, "access_rows": 5, "compares": 7, "peak_memory": 9}}


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "ledger", "coljoin.db")
        self.snapshot_dir = os.path.join(self.tmp, "snapshots")
        self.ledger = Ledger(self.db_path, self.snapshot_dir)

    def tearDown(self):
        self.ledger.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _commit(self, ordinal, shard_id="P1:0000", version=1):
        receipt = dict(RECEIPT, shard_id=shard_id)
        self.ledger.commit_batch(
            plan_version=version,
            plan_id="P1",
            ordinal=ordinal,
            executor_id="executor-0",
            receipt=receipt,
            cost_summary=COST,
            shard_digest_value=digest({"ordinal": ordinal}),
            invalidation_rows=[
                {
                    "plan_id": "P1",
                    "kind": "cost_input_change",
                    "basis": "test",
                    "block_groups": [],
                }
            ],
            total_shards=3,
        )

    def test_atomic_commit_and_cursor(self):
        for ordinal in range(3):
            self._commit(ordinal, f"P1:{ordinal:04d}")
        rows = self.ledger.committed_shards(1)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["status"] == "committed" for row in rows))
        cursor = self.ledger.conn.execute(
            "SELECT committed_shards, total_shards FROM cursors"
        ).fetchone()
        self.assertEqual(tuple(cursor), (3, 3))
        self.assertTrue(all(row["format_version"] == LEDGER_FORMAT_VERSION for row in rows))

    def test_snapshot_mismatch_replaces_and_keeps_readonly_backup(self):
        self._commit(0)
        path = self.ledger.write_snapshot(1)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"tampered": true}')
        backup = self.ledger.verify_snapshot(1)
        self.assertIsNotNone(backup)
        self.assertTrue(os.access(backup, os.R_OK))
        self.assertFalse(os.access(backup, os.W_OK))
        self.assertIsNone(self.ledger.verify_snapshot(1))

    def test_unknown_shards_requiring_rerun(self):
        self._commit(0)
        unknown = self.ledger.unknown_shards(
            ["P1:0000", "P1:0001", "P1:0002"], plan_version=1
        )
        self.assertEqual(unknown, ["P1:0001", "P1:0002"])

    def test_replay_mismatch_locates_field(self):
        self._commit(0)
        expected_receipts = {
            "P1:0000": dict(RECEIPT, aggregate={"1": {"count": 999}})
        }
        with self.assertRaises(ReplayError) as ctx:
            self.ledger.replay_verify(1, {"P1": COST}, expected_receipts)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.object_id, "P1:0000")
        self.assertIn("aggregate", ctx.exception.field)

    def test_replay_pass_on_identical_inputs(self):
        self._commit(0)
        self.ledger.replay_verify(
            1, {"P1": COST}, {"P1:0000": dict(RECEIPT)}
        )


class CrashRecoveryTest(unittest.TestCase):
    def test_forced_kill_only_committed_batches_survive(self):
        workspace = tempfile.mkdtemp()
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.generate_sample",
                    "--workspace",
                    workspace,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            crash_script = os.path.join(workspace, "crash_runner.py")
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            with open(crash_script, "w", encoding="utf-8") as handle:
                handle.write(
                    "import sys\n"
                    "sys.path.insert(0, %r)\n"
                    "from coljoin.pipeline import Pipeline\n"
                    "p = Pipeline(%r, executor_count=2, max_in_flight=2, "
                    "per_executor_slots=1, shard_rows=20)\n"
                    "p.run(plan_version=1, crash_after=2)\n"
                    % (repo_root, workspace)
                )
            proc = subprocess.run(
                [sys.executable, crash_script],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            db_path = os.path.join(workspace, "ledger", "coljoin.db")
            self.assertTrue(os.path.exists(db_path))
            conn = sqlite3.connect(db_path)
            committed = conn.execute(
                "SELECT COUNT(*) FROM batches WHERE status='committed'"
            ).fetchone()[0]
            total_batches = conn.execute(
                "SELECT total_shards FROM cursors"
            ).fetchone()[0]
            self.assertGreaterEqual(committed, 1)
            self.assertLess(committed, total_batches)
            conn.close()

            # 恢复：重跑完整批次，账本只承认已完整提交记录，未知分片要求重跑。
            ledger = Ledger(
                db_path, os.path.join(workspace, "snapshots")
            )
            known = {
                row["shard_id"]
                for row in ledger.conn.execute(
                    "SELECT shard_id FROM batches WHERE plan_version=1"
                ).fetchall()
            }
            from coljoin.pipeline import Pipeline

            pipeline = Pipeline(
                workspace,
                executor_count=2,
                max_in_flight=2,
                per_executor_slots=1,
                shard_rows=20,
            )
            plans, _, _ = pipeline.plan()
            chosen = pipeline.plan()[2]["chosen_plan_id"]
            from coljoin.sharding import split_batches

            chosen_plan = next(item for item in plans if item["plan_id"] == chosen)
            driver = next(
                group
                for group in pipeline.groups
                if group.bg_id == chosen_plan["body"]["order"][0]
            )
            all_shards = [
                batch["shard_id"]
                for batch in split_batches(chosen_plan, driver, 20)
            ]
            unknown = [shard for shard in all_shards if shard not in known]
            self.assertTrue(unknown)
            ledger.close()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
