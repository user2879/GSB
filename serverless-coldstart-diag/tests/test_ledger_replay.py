import json
import tempfile
import unittest
from pathlib import Path

from coldstart_diag.demo import run_demo
from coldstart_diag.engine import Engine
from coldstart_diag.errors import ReplayError
from coldstart_diag.ledger import Ledger
from coldstart_diag.replay import replay_run
from coldstart_diag.samples import base_manifest, write_samples


class LedgerReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.samples = write_samples(self.root / "samples")
        self.fid = "fn-order-processor"

    def tearDown(self):
        self.tmp.cleanup()

    def _manifest(self, name):
        from coldstart_diag.spec import load_manifest
        return load_manifest(self.root / "samples" / name)

    def test_crash_recovery_only_committed_minutes(self):
        m = self._manifest("manifest_v1.json")
        db = self.root / "ledger.sqlite"
        with Ledger(db) as ledger:
            ledger.start_run("r", m, len(m["call_sequence"]))
            engine = Engine(m, self.fid, 1)
            for i, item in enumerate(sorted(m["call_sequence"], key=lambda x: x["minute"])):
                r = engine.run_minute(item)
                if i < 4:  # 模拟第 4 分钟后被 kill
                    ledger.append_minute("r", r)
            rec = ledger.recover("r")
        self.assertEqual(rec["recovered_minutes"], 4)
        self.assertEqual(rec["pending_minutes"], list(range(4, len(m["call_sequence"]))))
        self.assertEqual(rec["status"], "INTERRUPTED")

    def test_replay_reproducible_and_tamper_detection(self):
        m = self._manifest("manifest_v1.json")
        db = self.root / "ledger.sqlite"
        with Ledger(db) as ledger:
            engine = Engine(m, self.fid, 1)
            ledger.start_run("r", m, len(m["call_sequence"]))
            for item in sorted(m["call_sequence"], key=lambda x: x["minute"]):
                ledger.append_minute("r", engine.run_minute(item))
            ledger.finalize_run("r")
            ok = replay_run(self.root / "samples" / "manifest_v1.json", ledger, "r",
                            m["call_sequence"], self.fid, 1)
            self.assertEqual(ok["status"], "REPRODUCIBLE")

            # 篡改第 2 分钟的冷启动计数 -> E_REPLAY 定位首个不一致字段
            row = ledger.conn.execute(
                "SELECT payload_json FROM minutes WHERE run_id='r' AND minute=2").fetchone()
            payload = json.loads(row[0])
            payload["cold_starts"] += 1
            from coldstart_diag.ledger import canonical_digest
            ledger.conn.execute(
                "UPDATE minutes SET payload_json=?, digest=? WHERE run_id='r' AND minute=2",
                (json.dumps(payload, sort_keys=True), canonical_digest(payload)),
            )
            ledger.conn.commit()
            with self.assertRaises(ReplayError) as ctx:
                replay_run(self.root / "samples" / "manifest_v1.json", ledger, "r",
                           m["call_sequence"], self.fid, 1)
            self.assertEqual(ctx.exception.code, "E_REPLAY")
            self.assertIn("minute[2].cold_starts", ctx.exception.path)

    def test_scale_mismatch_reported_separately(self):
        m1 = self._manifest("manifest_v1.json")
        m3 = self._manifest("manifest_v3.json")
        db = self.root / "ledger.sqlite"
        with Ledger(db) as ledger:
            engine = Engine(m1, self.fid, 1)
            ledger.start_run("r1", m1, len(m1["call_sequence"]))
            for item in m1["call_sequence"]:
                ledger.append_minute("r1", engine.run_minute(item))
            ledger.finalize_run("r1")
            with self.assertRaises(ReplayError) as ctx:
                replay_run(self.root / "samples" / "manifest_v3.json", ledger, "r1",
                           m3["call_sequence"], self.fid, 1)
            self.assertEqual(ctx.exception.code, "E_REPLAY")
            self.assertEqual(ctx.exception.path, "config")
            self.assertTrue(any(d["field"] == "scale" for d in ctx.exception.details))


class StaleVersionTest(unittest.TestCase):
    def test_mark_stale_and_full_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            summary, out = run_demo(ws)
            self.assertTrue(out.exists())
            s2 = next(s for s in summary["scenarios"] if "S2" in s["scenario"])
            self.assertGreaterEqual(s2["stale_minutes_marked"], 1)
            self.assertIn(0, s2["changed_minutes"])
            s3 = next(s for s in summary["scenarios"] if "S3" in s["scenario"])
            self.assertEqual(s3["config_mismatch_report"]["code"], "E_REPLAY")
            self.assertEqual(s3["replay_current"]["status"], "REPRODUCIBLE")


if __name__ == "__main__":
    unittest.main()
