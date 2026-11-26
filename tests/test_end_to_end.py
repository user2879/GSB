import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path

from gsb.batching import plan_batches
from gsb.bus import Bus
from gsb.client import ProcessorPool
from gsb.gen_sample import generate
from gsb.ledger import Ledger, diff_replay
from gsb.receipts import canonical_normalized_text
from gsb.spec import load_spec


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self._kill_stray_processors()
        self.base = Path("/tmp/gsb_e2e")
        if self.base.exists():
            shutil.rmtree(self.base)
        generate(self.base / "sample", n_events=24)
        self.spec = load_spec(
            self.base / "sample" / "spec.json", self.base / "sample" / "events.ndjson"
        )

    @staticmethod
    def _kill_stray_processors():
        result = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "gsb.processor" in line and "grep" not in line:
                pid = int(line.split()[1])
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_full_run_tri_format_consistency_and_trajectory(self):
        run_dir = self.base / "run"
        bus = Bus(self.spec, run_dir)
        bus.write_manifest()
        with ProcessorPool(self.spec, processor_dir=run_dir / "processors") as pool:
            bus.run(pool)
            stale = bus.stale_renewal_demo(pool, bus.batches[0].batch_id)
        bus.write_outputs()
        bus.close()

        self.assertEqual(stale["code"], "E_LEASE")
        self.assertEqual(len(bus.adoptions), len(bus.batches))

        conn = sqlite3.connect(run_dir / "ledger.sqlite3")
        # Three receipts per batch, all committed in atomic transactions.
        counts = conn.execute(
            "SELECT batch_id, COUNT(*) FROM receipts GROUP BY batch_id"
        ).fetchall()
        self.assertTrue(all(count == 3 for _, count in counts))

        # The three normalized receipts share a byte-identical adoption core.
        for (batch_id,) in conn.execute("SELECT DISTINCT batch_id FROM receipts"):
            rows = conn.execute(
                "SELECT normalized FROM receipts WHERE batch_id=?", (batch_id,)
            ).fetchall()
            cores = set()
            for (normalized_text,) in rows:
                normalized = json.loads(normalized_text)
                cores.add(canonical_normalized_text(normalized))
            self.assertEqual(len(cores), 1, batch_id)

        trajectory = json.loads((run_dir / "trajectory.json").read_text())
        self.assertTrue(any(snap["queued_length"] > 0 for snap in trajectory))
        self.assertTrue(any(snap["global_leases"] > 0 for snap in trajectory))

        # Backpressure attribution populated with all three segments.
        decisions = json.loads((run_dir / "adoptions.json").read_text())
        for decision in decisions.values():
            segments = decision["attribution"]["segments_ms"]
            self.assertEqual(
                set(segments), {"queue_ms", "backfill_ms", "processing_ms"}
            )

    def test_forced_termination_recovery_and_first_mismatch(self):
        crash_dir = self.base / "crash"
        total = len(plan_batches(self.spec))
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gsb.cli",
                "run",
                "--spec",
                str(self.base / "sample" / "spec.json"),
                "--events",
                str(self.base / "sample" / "events.ndjson"),
                "--run-dir",
                str(crash_dir),
                "--crash-after",
                "2",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=60)
            self.assertEqual(proc.returncode, 137)
            self.assertNotIn("exit report", out.lower())
        finally:
            # Reap the whole group so hard-exited parents do not leave
            # orphaned processor subprocesses holding loopback ports.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._kill_stray_processors()

        ledger = Ledger(crash_dir / "ledger.sqlite3", self.spec.spec_digest)
        committed = ledger.committed_batch_ids()
        self.assertLess(len(committed), total)
        records = [ledger.lookup_dispatch(b.batch_id) for b in plan_batches(self.spec)]
        finding = diff_replay(
            plan_batches(self.spec), [r for r in records if r], ledger
        )
        ledger.close()
        self.assertIsNotNone(finding)
        self.assertIn(finding["kind"], {"unknown", "mismatch"})
        self.assertTrue(finding["batch_id"])

    def test_snapshot_mismatch_replaces_whole_store(self):
        run_dir = self.base / "snap"
        ledger_a = Ledger(run_dir / "ledger.sqlite3", "digest-a")
        ledger_a.record_snapshot("session", {"v": 1})
        ledger_a.close()
        ledger_b = Ledger(run_dir / "ledger.sqlite3", "digest-b")
        self.assertTrue(ledger_b.replaced)
        self.assertIsNone(ledger_b.conn.execute("SELECT * FROM snapshots").fetchone())
        ledger_b.close()


if __name__ == "__main__":
    unittest.main()
