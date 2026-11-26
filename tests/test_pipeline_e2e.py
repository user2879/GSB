import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import unittest

from src.core import load_config
from src.errors import ReplayError
from src.ledger import Ledger
from src.pipeline import run_pipeline
from src.replay import replay
from src.verify import compare
from tests.util import TmpDir, make_config, make_ports


def _alt_config(tmp, **changes):
    cfg_path = make_config(tmp)
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    if "worker_seed" in changes:
        for w in cfg["workers"]:
            if w["worker_id"] == changes["worker_seed"][0]:
                w["seed"] = changes["worker_seed"][1]
    if "batch_size" in changes:
        cfg["batch_size"] = changes["batch_size"]
    alt = os.path.join(tmp, "alt.json")
    with open(alt, "w") as fh:
        json.dump(cfg, fh)
    return alt


class TestPipelineE2E(unittest.TestCase):
    def test_incremental_recompute_and_verify_and_replay(self):
        with TmpDir() as tmp:
            cfg = make_config(tmp)
            ports = make_ports(tmp, base=24601)
            ledger_path = os.path.join(tmp, "ledger.db")
            s1 = run_pipeline(cfg, ports, ledger_path, use_workers=False)
            self.assertEqual(len(s1["committed"]), 4)
            ledger = Ledger(ledger_path)
            before = {r["batch_id"]: r["adopted_digest"]
                      for r in ledger.query("SELECT batch_id, adopted_digest"
                                            " FROM batches")}
            ledger.close()

            alt = _alt_config(tmp, worker_seed=("w1", 999))
            s2 = run_pipeline(alt, ports, ledger_path, use_workers=False)
            self.assertEqual(s2["invalidated_batches"], ["batch-0001", "batch-0003"])
            self.assertEqual(sorted(s2["committed"]), ["batch-0001", "batch-0003"])
            ledger = Ledger(ledger_path)
            after = {r["batch_id"]: r["adopted_digest"]
                     for r in ledger.query("SELECT batch_id, adopted_digest"
                                           " FROM batches")}
            ledger.close()
            for bid in ("batch-0000", "batch-0002"):
                self.assertEqual(before[bid], after[bid])
            for bid in ("batch-0001", "batch-0003"):
                self.assertNotEqual(before[bid], after[bid])

            result = compare(ledger_path, alt, ports)
            self.assertEqual(result["result"], "IDENTICAL")
            replay_result = replay(ledger_path)
            self.assertEqual(len(replay_result["checked_batches"]), 4)

    def test_batch_size_change_then_verify(self):
        with TmpDir() as tmp:
            cfg = make_config(tmp)
            ports = make_ports(tmp, base=24611)
            ledger_path = os.path.join(tmp, "ledger.db")
            run_pipeline(cfg, ports, ledger_path, use_workers=False)
            alt = _alt_config(tmp, batch_size=3)
            s2 = run_pipeline(alt, ports, ledger_path, use_workers=False)
            self.assertIn("batch-0000", s2["invalidated_batches"])
            result = compare(ledger_path, alt, ports)
            self.assertEqual(result["result"], "IDENTICAL")
            replay(ledger_path)

    def test_replay_detects_tampered_digest(self):
        with TmpDir() as tmp:
            cfg = make_config(tmp)
            ports = make_ports(tmp, base=24621)
            ledger_path = os.path.join(tmp, "ledger.db")
            run_pipeline(cfg, ports, ledger_path, use_workers=False)
            ledger = Ledger(ledger_path)
            ledger.commit_batch(
                {"batch_id": "batch-0000", "worker_id": "w0", "param_version": 1,
                 "adopted_digest": "0" * 64, "adopted_format": 3,
                 "request_ids": ["req-0000", "req-0001"]},
                [], [])
            ledger.close()
            with self.assertRaises(ReplayError) as ctx:
                replay(ledger_path)
            self.assertEqual(ctx.exception.code, "E_REPLAY")
            self.assertEqual(ctx.exception.loc["batch"], "batch-0000")
            self.assertEqual(ctx.exception.loc["field"], "adopted_digest")

    def test_kill9_recovery_marks_unknown_and_reruns(self):
        with TmpDir() as tmp:
            cfg = make_config(tmp, request_count=12, batch_size=2)
            ports = make_ports(tmp, base=24631)
            ledger_path = os.path.join(tmp, "kill.db")
            env = dict(os.environ)
            proc = subprocess.Popen(
                [sys.executable, "-m", "src.pipeline", "run",
                 "--config", cfg, "--ports", ports, "--ledger", ledger_path,
                 "--compute-delay-ms", "800"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
            deadline = time.monotonic() + 30
            done = 0
            while time.monotonic() < deadline:
                if os.path.exists(ledger_path) and os.path.getsize(ledger_path) > 0:
                    try:
                        conn = sqlite3.connect(ledger_path)
                        done = conn.execute(
                            "SELECT COUNT(*) FROM batches WHERE status='done'"
                        ).fetchone()[0]
                        conn.close()
                    except sqlite3.Error:
                        done = 0
                    if done >= 1:
                        break
                time.sleep(0.05)
            proc.send_signal(signal.SIGKILL)
            proc.wait()
            self.assertGreaterEqual(done, 1)
            ledger = Ledger(ledger_path)
            unknown = [r["batch_id"] for r in ledger.query(
                "SELECT batch_id FROM batches WHERE status='unknown'")]
            ledger.close()
            self.assertTrue(unknown)
            s = run_pipeline(cfg, ports, ledger_path, use_workers=False)
            self.assertEqual(sorted(s["committed"]), sorted(unknown))
            replay(ledger_path)


if __name__ == "__main__":
    unittest.main()
