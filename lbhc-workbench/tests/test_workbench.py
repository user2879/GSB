"""End-to-end and unit tests for the LB health-check workbench."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lbhc import health as hm                       # noqa: E402
from lbhc.engine import run_engine                  # noqa: E402
from lbhc.errors import (LedgerError, PortError,    # noqa: E402
                         ReplayError, SpecError)
from lbhc.ledger import Ledger                      # noqa: E402
from lbhc.replay import replay_verify               # noqa: E402
from lbhc.sample import sample_spec                 # noqa: E402
from lbhc.spec import load_spec, write_spec         # noqa: E402

BASE_PORT = 18601


def _ports_free():
    import socket
    for p in range(BASE_PORT, BASE_PORT + 25):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", p))
        except OSError:
            return False
        finally:
            s.close()
    return True


class TmpDirMixin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lbhc-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def write_spec(self, doc, name="spec.json"):
        p = self.path(name)
        write_spec(doc, p)
        return p


class SpecValidationTests(TmpDirMixin):
    def expect_spec_error(self, mutate, expect_path):
        doc = sample_spec()
        mutate(doc)
        spec_path = self.write_spec(doc)
        with self.assertRaises(SpecError) as ctx:
            load_spec(spec_path)
        self.assertEqual(expect_path, ctx.exception.path)

    def test_valid_sample_loads(self):
        spec = load_spec(self.write_spec(sample_spec()))
        self.assertEqual(2, spec.pool_count)
        self.assertEqual(10, spec.instance_count)

    def test_broken_json(self):
        p = self.path("broken.json")
        with open(p, "w") as fh:
            fh.write('{"version": 1, "pools": [')
        with self.assertRaises(SpecError) as ctx:
            load_spec(p)
        self.assertEqual("$", ctx.exception.path)

    def test_duplicate_pool_id(self):
        def m(doc):
            doc["pools"][1]["id"] = doc["pools"][0]["id"]
        self.expect_spec_error(m, "pools[1].id")

    def test_duplicate_instance_id(self):
        def m(doc):
            doc["pools"][1]["backends"][0]["id"] = "h1"
        self.expect_spec_error(m, "pools[1].backends[0].id")

    def test_non_loopback_host(self):
        def m(doc):
            doc["pools"][0]["backends"][0]["host"] = "192.168.1.10"
        self.expect_spec_error(m, "pools[0].backends[0].host")

    def test_duplicate_port(self):
        def m(doc):
            doc["pools"][1]["backends"][0]["port"] = \
                doc["pools"][0]["backends"][0]["port"]
        self.expect_spec_error(m, "pools[1].backends[0].port")

    def test_zero_threshold(self):
        def m(doc):
            doc["pools"][0]["success_threshold"] = 0
        self.expect_spec_error(m, "pools[0].success_threshold")

    def test_zero_quota(self):
        def m(doc):
            doc["pools"][1]["concurrency_quota"] = 0
        self.expect_spec_error(m, "pools[1].concurrency_quota")

    def test_zero_budget(self):
        def m(doc):
            doc["global"]["concurrency_budget"] = 0
        self.expect_spec_error(m, "global.concurrency_budget")

    def test_bool_is_not_int(self):
        def m(doc):
            doc["pools"][0]["timeout_ms"] = True
        self.expect_spec_error(m, "pools[0].timeout_ms")

    def test_bad_protocol(self):
        def m(doc):
            doc["pools"][0]["protocol"] = "udp"
        self.expect_spec_error(m, "pools[0].protocol")


class HealthMachineTests(unittest.TestCase):
    def test_success_threshold_and_counter_reset(self):
        h = hm.initial_health("active")
        self.assertIsNone(hm.apply_outcome(h, "success", "", 2, 2))
        self.assertEqual((1, 0), (h.consec_success, h.consec_failure))
        t = hm.apply_outcome(h, "success", "", 2, 2)
        self.assertEqual(("unknown", "healthy", "threshold_met"), t)
        hm.apply_outcome(h, "failure", "timeout", 2, 2)
        self.assertEqual((0, 1), (h.consec_success, h.consec_failure))
        t = hm.apply_outcome(h, "failure", "timeout", 2, 2)
        self.assertEqual(("healthy", "unhealthy", "timeout"), t)

    def test_draining_never_transitions(self):
        h = hm.initial_health("draining")
        for _ in range(5):
            self.assertIsNone(hm.apply_outcome(h, "success", "", 1, 1))
        self.assertEqual("draining", h.state)


class EngineTests(TmpDirMixin):
    @classmethod
    def setUpClass(cls):
        if not _ports_free():
            raise unittest.SkipTest("demo ports busy")

    def run_engine(self, doc, name="ledger.db", **kw):
        spec_path = self.write_spec(doc, name + ".json")
        return run_engine(spec_path, self.path(name), log_name="t", **kw)

    def test_budget_quota_and_rate_constraints(self):
        doc = sample_spec()
        doc["global"]["rate_limit_per_sec"] = 6
        rep = self.run_engine(doc, rounds=2)
        budget = rep["budget"]
        self.assertLessEqual(budget["peak_global_inflight"],
                             budget["global_budget"])
        for pid, peak in budget["peak_pool_inflight"].items():
            self.assertLessEqual(peak, budget["pool_quotas"][pid])
        # rate: no 1s window may contain more starts than the limit
        db = sqlite3.connect(self.path("ledger.db"))
        starts = [r[0] for r in db.execute(
            "SELECT started_ms FROM probes ORDER BY started_ms")]
        db.close()
        for i, t0 in enumerate(starts):
            window = [t for t in starts[i:] if t - t0 < 1000.0]
            self.assertLessEqual(len(window), 6)
        # queueing happened and wait times were recorded
        waits = [d["wait_ms"] for r in rep["rounds"]
                 for d in r["dispatched"]]
        self.assertTrue(any(w > 1.0 for w in waits))

    def test_deterministic_initiated_sequence(self):
        doc = sample_spec()
        r1 = self.run_engine(doc, "a.db", rounds=2)
        r2 = self.run_engine(doc, "b.db", rounds=2)
        self.assertEqual(r1["initiated_sequence"],
                         r2["initiated_sequence"])

    def test_dual_path_byte_identical(self):
        doc = sample_spec()
        r1 = self.run_engine(doc, "full.db", rounds=4, path="full")
        r2 = self.run_engine(doc, "strat.db", rounds=4, path="stratified")

        def canon(rep):
            states = {r["instance_id"]: (r["state"], r["last_reason"])
                      for r in rep["final_view"]}
            trans = [(t["instance_id"], t["from_state"], t["to_state"],
                      t["reason_category"]) for t in rep["transitions"]]
            return json.dumps({"s": states, "t": trans},
                              sort_keys=True).encode()
        self.assertEqual(canon(r1), canon(r2))
        # stratified path probed strictly less overall
        n1 = sum(len(r["dispatched"]) for r in r1["rounds"])
        n2 = sum(len(r["dispatched"]) for r in r2["rounds"])
        self.assertLess(n2, n1)

    def test_reason_categories_from_real_sockets(self):
        doc = sample_spec()
        rep = self.run_engine(doc, rounds=3)
        reasons = {t["reason_category"] for t in rep["transitions"]}
        self.assertEqual({"threshold_met", "timeout", "conn_refused",
                          "digest_mismatch"}, reasons)

    def test_stale_guard_and_wholesale_replace(self):
        v1 = sample_spec(version=1)
        self.run_engine(v1, "evo.db", rounds=2)
        v2 = sample_spec(version=2)
        v2["global"]["concurrency_budget"] = 3
        v2["pools"][0]["backends"] = [
            b for b in v2["pools"][0]["backends"] if b["id"] != "h5"]
        rep2 = self.run_engine(v2, "evo.db", rounds=2)
        changes = {e["instance_id"]: e["change"]
                   for e in rep2["stale_diff"]}
        self.assertEqual("removed", changes["h5"])
        db = sqlite3.connect(self.path("evo.db"))
        stale = db.execute("SELECT COUNT(*) FROM instance_view "
                           "WHERE version=1 AND stale=1").fetchone()[0]
        current = db.execute("SELECT COUNT(*) FROM instance_view "
                             "WHERE version=2 AND stale=0").fetchone()[0]
        db.close()
        self.assertEqual(10, stale)
        self.assertEqual(9, current)
        # non-monotonic version is rejected
        with self.assertRaises(SpecError):
            self.run_engine(sample_spec(version=1), "evo.db", rounds=1)

    def test_crash_recovery_and_replay(self):
        doc = sample_spec()
        spec_path = self.write_spec(doc)
        ledger = self.path("crash.db")
        env = dict(os.environ, LBHC_CRASH_AFTER="10", PYTHONPATH=ROOT)
        proc = subprocess.run(
            [sys.executable, "-m", "lbhc.cli", "run", "--spec", spec_path,
             "--ledger", ledger, "--rounds", "5"],
            env=env, capture_output=True, text=True)
        self.assertEqual(137, proc.returncode)
        self.assertFalse(os.path.exists(self.path("report.json")))
        db = sqlite3.connect(ledger)
        committed = db.execute("SELECT COUNT(*) FROM probes").fetchone()[0]
        inflight = db.execute("SELECT COUNT(*) FROM inflight").fetchone()[0]
        db.close()
        self.assertEqual(10, committed)
        self.assertGreater(inflight, 0)
        time.sleep(0.4)  # let orphaned backends notice the closed pipe
        rep = run_engine(spec_path, ledger, rounds=5, log_name="t")
        self.assertTrue(rep["resumed"])
        self.assertEqual(0, len(rep["stale_diff"]))
        self.assertGreater(rep["replay"]["probes_replayed"], committed)

    def test_replay_detects_view_tampering(self):
        doc = sample_spec()
        self.run_engine(doc, "tamper.db", rounds=2)
        db = sqlite3.connect(self.path("tamper.db"))
        db.execute("UPDATE instance_view SET state='healthy' "
                   "WHERE instance_id='h4'")
        db.commit()
        db.close()
        spec = load_spec(self.path("tamper.db.json"))
        ledger = Ledger(self.path("tamper.db"))
        with self.assertRaises(ReplayError) as ctx:
            replay_verify(ledger, spec)
        ledger.close()
        self.assertEqual("h4", ctx.exception.detail["instance_id"])
        self.assertEqual("state", ctx.exception.detail["field"])

    def test_ledger_chain_detects_tampering(self):
        doc = sample_spec()
        self.run_engine(doc, "chain.db", rounds=1)
        db = sqlite3.connect(self.path("chain.db"))
        db.execute("UPDATE events SET payload=payload||'x' "
                   "WHERE id=(SELECT MIN(id) FROM events)")
        db.commit()
        db.close()
        ledger = Ledger(self.path("chain.db"))
        with self.assertRaises(LedgerError):
            ledger.verify_chain()
        ledger.close()

    def test_port_occupied_raises_e_port(self):
        import socket
        doc = sample_spec()
        spec_path = self.write_spec(doc)
        blocker = socket.socket()
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", BASE_PORT))
        blocker.listen(1)
        try:
            with self.assertRaises(PortError) as ctx:
                run_engine(spec_path, self.path("port.db"), log_name="t")
            self.assertEqual("pools[0].backends[0].port",
                             ctx.exception.path)
        finally:
            blocker.close()

    def test_probe_timeout_is_real(self):
        # a 'delay' backend must produce a real socket timeout that takes
        # at least the declared timeout to surface
        doc = sample_spec()
        doc["pools"][0]["timeout_ms"] = 150
        rep = self.run_engine(doc, rounds=1)
        db = sqlite3.connect(self.path("ledger.db"))
        row = db.execute(
            "SELECT started_ms,finished_ms,outcome,reason FROM probes "
            "WHERE instance_id='t3'").fetchone()
        db.close()
        self.assertEqual("failure", row[2])
        self.assertEqual("timeout", row[3])
        self.assertGreaterEqual(row[1] - row[0], 140.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
