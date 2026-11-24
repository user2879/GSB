import json
import os
import shutil
import tempfile
import unittest

from fppdiag.cli import main
from fppdiag.errors import DivergeError, ReplayError
from fppdiag.ledger import Ledger
from fppdiag.orchestrator import Orchestrator
from fppdiag.samplegen import build_sample_doc, write_sample
from fppdiag.spec import load_spec_file
from fppdiag.util import canonical_json


class EndToEndBase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="fppe2e_")
        self.spec_path = write_sample(self.root, rows=120)
        self.work = os.path.join(self.root, "work")
        self.spec = load_spec_file(self.spec_path)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def orch(self, **kw):
        return Orchestrator(load_spec_file(self.spec_path), self.work, **kw)


class TestPersistenceAndRecovery(EndToEndBase):
    def test_full_run_persists_all_tables(self):
        o = self.orch()
        report = o.run_full()
        self.assertEqual(len(report["features"]), len(self.spec.stages))
        ledger = Ledger(os.path.join(self.work, "ledger.sqlite"))
        committed = ledger.committed_stages(self.spec.version)
        self.assertEqual(committed, {s.stage_id for s in self.spec.stages})
        for sid in committed:
            summary, digest = ledger.get_summary(self.spec.version, sid)
            self.assertTrue(digest)
            self.assertEqual(len(digest), 64)
        ledger.close()
        o.close()

    def test_crash_leaves_unknown_then_recovery_completes(self):
        o = self.orch()
        crash = o.run_full(crash_after_stage="log_size")
        self.assertEqual(crash, {"crashed": "log_size"})
        o.close()
        ledger = Ledger(os.path.join(self.work, "ledger.sqlite"))
        self.assertIn("log_size", ledger.unknown_stages(self.spec.version))
        self.assertNotIn("log_size",
                         ledger.committed_stages(self.spec.version))
        ledger.close()
        # Reopen with a brand-new orchestrator and recover.
        o2 = self.orch()
        report = o2.recover()
        self.assertEqual(len(report["features"]), len(self.spec.stages))
        self.assertFalse(
            Ledger(os.path.join(self.work, "ledger.sqlite"))
            .unknown_stages(self.spec.version))
        o2.close()

    def test_recovery_commits_downstream_stages_never_started(self):
        # Crash at an interaction stage; its downstream truncate never began a
        # batch and must still be inserted and committed during recovery.
        o = self.orch()
        o.run_full(crash_after_stage="interact_pv")
        o.close()
        o2 = self.orch()
        report = o2.recover()
        o2.close()
        ledger = Ledger(os.path.join(self.work, "ledger.sqlite"))
        self.assertEqual(
            ledger.committed_stages(self.spec.version),
            {s.stage_id for s in self.spec.stages})
        self.assertEqual(len(report["features"]), len(self.spec.stages))
        ledger.close()

    def test_replay_detects_tampered_ledger(self):
        o = self.orch()
        o.run_full()
        o.close()
        ledger = Ledger(os.path.join(self.work, "ledger.sqlite"))
        # Tamper a committed summary digest without touching source code.
        summary, _ = ledger.get_summary(self.spec.version, "std_price")
        bad = dict(summary)
        bad["nulls"] = bad["nulls"] + 1
        ledger.conn.execute(
            "UPDATE stage_batches SET summary_json=? WHERE spec_version=? AND stage_id='std_price'",
            (canonical_json(bad), self.spec.version))
        ledger.conn.commit()
        ledger.close()
        o2 = self.orch()
        with self.assertRaises(ReplayError) as ctx:
            o2.recover()
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.location, "std_price")
        o2.close()


class TestIncremental(EndToEndBase):
    def test_affected_set_forward_propagation(self):
        o = self.orch()
        affected = o.affected_stages({"std_price"}, set())
        # std_price feeds bucket_price and interact_pv; product feeds clip.
        self.assertIn("bucket_price", affected)
        self.assertIn("interact_pv", affected)
        self.assertIn("clip_interact", affected)
        self.assertNotIn("std_vol", affected)
        self.assertNotIn("log_size", affected)
        o.close()

    def test_source_change_propagates_to_fed_stages(self):
        o = self.orch()
        affected = o.affected_stages(set(), {"src_vol"})
        self.assertIn("std_vol", affected)
        self.assertIn("interact_pv", affected)
        o.close()

    def test_verify_matches_full_shadow(self):
        rc = main(["verify", "--spec", self.spec_path, "--workdir", self.work])
        self.assertEqual(rc, 0)

    def test_diverge_detected_when_ledger_wrong(self):
        o = self.orch()
        o.run_full()
        o.close()
        # Corrupt a verdict drift value to force an incremental/full mismatch.
        ledger = Ledger(os.path.join(self.work, "ledger.sqlite"))
        verdict, _ = ledger.get_verdict(self.spec.version, "std_vol")
        verdict = json.loads(json.dumps(verdict))
        verdict["drift_bps"] = verdict["drift_bps"] + 999
        ledger.put_verdict(self.spec.version, "std_vol", verdict)
        ledger.close()
        shadow = os.path.join(self.work, "_full_shadow")
        shutil.rmtree(shadow, ignore_errors=True)
        full_o = Orchestrator(load_spec_file(self.spec_path), shadow)
        full_report = full_o.run_full()
        inc_o = self.orch()
        inc_report = inc_o.run_full()  # keeps tampered committed verdict
        with self.assertRaises(DivergeError) as ctx:
            inc_o.verify_incremental(full_report, inc_report)
        self.assertEqual(ctx.exception.code, "E_DIVERGE")
        self.assertEqual(ctx.exception.location, "std_vol")
        full_o.close()
        inc_o.close()


class TestFaultInjection(EndToEndBase):
    def test_data_corruption_e_data(self):
        rc = main(["inject", "--fault", "data_corrupt", "--spec",
                   self.spec_path, "--workdir", self.work])
        self.assertEqual(rc, 11)

    def test_param_overflow_e_param(self):
        rc = main(["inject", "--fault", "param_overflow", "--spec",
                   self.spec_path, "--workdir", self.work,
                   "--stage", "std_price"])
        self.assertEqual(rc, 12)

    def test_snapshot_digest_e_snapshot(self):
        rc = main(["inject", "--fault", "snapshot_digest", "--spec",
                   self.spec_path, "--workdir", self.work])
        self.assertEqual(rc, 15)

    def test_structured_log_written(self):
        main(["inject", "--fault", "param_overflow", "--spec",
              self.spec_path, "--workdir", self.work, "--stage", "std_price"])
        log_path = os.path.join(self.work, "events.log.jsonl")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        self.assertTrue(any(r.get("event") == "E_PARAM" for r in records))


class TestSnapshotReplacement(EndToEndBase):
    def test_identity_mismatch_replaces_whole_ledger(self):
        o = self.orch()
        o.run_full()
        o.close()
        ledger = Ledger(os.path.join(self.work, "ledger.sqlite"))
        before = len(ledger.committed_stages(self.spec.version))
        ledger.close()
        # Bump spec version and digest -> snapshot identity mismatch.
        doc = build_sample_doc(rows=120, version=self.spec.version + 1)
        new_path = os.path.join(self.root, "pipeline_v2.json")
        with open(new_path, "w", encoding="utf-8") as fh:
            fh.write(canonical_json(doc))
        spec2 = load_spec_file(new_path)
        o2 = Orchestrator(spec2, self.work)
        ledger = Ledger(os.path.join(self.work, "ledger.sqlite"))
        # Old version rows must have been removed (whole replacement).
        self.assertEqual(ledger.committed_stages(self.spec.version), set())
        ledger.close()
        o2.close()
        self.assertEqual(before, len(self.spec.stages))


if __name__ == "__main__":
    unittest.main()
