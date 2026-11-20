"""验收测试: 混存归一、升级一致、快照重放、三类注入、断点续做、格式校验."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ledger_tool import app
from ledger_tool.errors import (E_CHAIN, E_FORMAT, E_INTEGRITY, E_REPLAY,
                                E_SNAPSHOT, E_UPGRADE, LedgerError)
from ledger_tool.ledger import Ledger
from ledger_tool.pipeline import Pipeline
from ledger_tool import records as records_mod


def run_quiet(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(*args, **kwargs)
    return code, buf.getvalue()


class WorkspaceCase(unittest.TestCase):
    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="ledger-ws-")
        self.addCleanup(shutil.rmtree, self.ws, True)
        run_quiet(app.cmd_prepare, self.ws)
        self.p = app._paths(self.ws)

    def read_log_codes(self):
        path = os.path.join(self.p["logs"], "ledger.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(line)["code"] for line in fh if line.strip()]


class TestDeterminism(unittest.TestCase):
    def test_bit_identical_digests(self):
        cfg = app.ensure_config(tempfile.mkdtemp(prefix="ledger-cfg-"))
        a = Pipeline(cfg)
        b = Pipeline(cfg)
        self.assertEqual(a.input_digest, b.input_digest)
        self.assertEqual(a.param_digest, b.param_digest)
        self.assertEqual(a.backend_fingerprint, b.backend_fingerprint)
        va = {v["request_id"]: v["output_digest"] for v in a.ordered_views()}
        vb = {v["request_id"]: v["output_digest"] for v in b.ordered_views()}
        self.assertEqual(va, vb)


class TestFormatValidation(unittest.TestCase):
    def setUp(self):
        cfg = app.ensure_config(tempfile.mkdtemp(prefix="ledger-cfg-"))
        self.pipeline = Pipeline(cfg)
        self.view = self.pipeline.ordered_views()[0]
        self.slots = self.pipeline.supported_slots

    def decode(self, fmt, **mut):
        rec = records_mod.encode(fmt, self.view)
        rec.update(mut)
        return records_mod.decode(rec, self.slots)

    def test_valid_records(self):
        for fmt in ("r1", "r2", "r3"):
            self.assertEqual(self.decode(fmt)["fmt"], fmt)

    def test_bad_slot_first_field(self):
        with self.assertRaises(LedgerError) as ctx:
            self.decode("r1", batch_slot=99)
        self.assertEqual(ctx.exception.code, E_FORMAT)
        self.assertEqual(ctx.exception.field, "batch_slot")

    def test_bad_digest(self):
        with self.assertRaises(LedgerError) as ctx:
            self.decode("r1", output_digest="zz" + "0" * 62)
        self.assertEqual(ctx.exception.code, E_FORMAT)
        self.assertEqual(ctx.exception.field, "output_digest")

    def test_negative_arrival(self):
        with self.assertRaises(LedgerError) as ctx:
            self.decode("r2", arrival_seq=-1)
        self.assertEqual(ctx.exception.code, E_FORMAT)
        self.assertEqual(ctx.exception.field, "arrival_seq")

    def test_extra_field_rejected(self):
        with self.assertRaises(LedgerError) as ctx:
            self.decode("r1", group_rank=0)
        self.assertEqual(ctx.exception.code, E_FORMAT)
        self.assertEqual(ctx.exception.field, "group_rank")


class TestMixedNormalize(WorkspaceCase):
    def test_mixed_read_equals_per_format_read(self):
        code, out = run_quiet(app.cmd_run, self.ws)
        self.assertEqual(code, 0, out)
        report = app._read_json(self.p["report"])
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["record_count"], 12)
        fmts = {r["fmt"] for r in report["records"]}
        self.assertEqual(fmts, {"r1", "r2", "r3"})
        r1 = next(r for r in report["records"] if r["fmt"] == "r1")
        self.assertEqual(set(r1["filled"]),
                         {"arrival_seq", "backend_fingerprint", "latency_ms",
                          "group_rank", "superseded_refs", "fingerprint_snapshot"})
        r3 = next(r for r in report["records"] if r["fmt"] == "r3")
        self.assertEqual(r3["filled"], {})

    def test_superseded_refs_traceable(self):
        cfg = app.ensure_config(self.ws)
        pipeline = Pipeline(cfg)
        self.assertEqual(pipeline.superseded["req-0003"], [12])
        self.assertEqual(pipeline.superseded["req-0007"], [13, 14])


class TestUpgradeConsistency(WorkspaceCase):
    def test_upgrade_paths_match_direct(self):
        code, out = run_quiet(app.cmd_run, self.ws)
        self.assertEqual(code, 0, out)
        report = app._read_json(self.p["report"])
        cons = report["consistency"]
        self.assertEqual(cons["status"], "ok")
        self.assertEqual(cons["count"], 12)
        self.assertEqual(cons["paths"]["r1->r3"]["digest"], cons["digest"])
        self.assertEqual(cons["paths"]["r2->r3"]["digest"], cons["digest"])
        ledger = Ledger(self.p["ledger_mixed"])
        self.assertEqual(ledger.formats(), ["r3"])
        ledger.close()
        backups = os.listdir(self.p["backup"])
        self.assertEqual(len(backups), len(report["upgrade_trail"]))
        for name in backups:
            mode = os.stat(os.path.join(self.p["backup"], name)).st_mode
            self.assertFalse(mode & 0o222, "backup must be read-only")

    def test_resume_after_interrupt(self):
        code, _ = run_quiet(app.cmd_run, self.ws, max_upgrades=2)
        self.assertEqual(code, 0)
        trail1 = app._read_json(self.p["trail"])
        self.assertEqual(len(trail1), 2)
        code, out = run_quiet(app.cmd_run, self.ws)
        self.assertEqual(code, 0, out)
        trail2 = app._read_json(self.p["trail"])
        ids = [t["request_id"] for t in trail2]
        self.assertEqual(len(ids), len(set(ids)), "no record upgraded twice")
        self.assertEqual(len(os.listdir(self.p["backup"])), len(ids))
        code, out = run_quiet(app.cmd_run, self.ws)
        self.assertEqual(code, 0)
        trail3 = app._read_json(self.p["trail"])
        self.assertEqual(len(trail3), len(trail2))


class TestSnapshot(WorkspaceCase):
    def test_replay_reproducible(self):
        self.assertEqual(run_quiet(app.cmd_run, self.ws)[0], 0)
        code, out = run_quiet(app.cmd_verify, self.ws)
        self.assertEqual(code, 0, out)

    def test_stale_snapshot_detected(self):
        self.assertEqual(run_quiet(app.cmd_run, self.ws)[0], 0)
        ledger = Ledger(self.p["ledger_mixed"])
        cfg = app.ensure_config(self.ws)
        pipeline = Pipeline(cfg)
        view = pipeline.ordered_views()[0]
        ledger.insert("req-9999", "r3", records_mod.encode("r3", view))
        ledger.close()
        code, out = run_quiet(app.cmd_verify, self.ws)
        self.assertEqual(code, 2)
        self.assertIn(E_SNAPSHOT, self.read_log_codes())

    def test_replay_mismatch_detected(self):
        self.assertEqual(run_quiet(app.cmd_run, self.ws)[0], 0)
        from ledger_tool.snapshot import load_latest
        snap_path, snap = load_latest(self.p["snapshots"])
        rid = sorted(snap["record_digests"])[0]
        snap["record_digests"][rid] = "0" * 64
        with open(snap_path, "w", encoding="utf-8") as fh:
            json.dump(snap, fh)
        code, out = run_quiet(app.cmd_verify, self.ws)
        self.assertEqual(code, 2)
        self.assertIn(E_REPLAY, self.read_log_codes())


class TestInjections(WorkspaceCase):
    def test_digest_corruption(self):
        run_quiet(app.cmd_inject, self.ws, "digest")
        code, out = run_quiet(app.cmd_run, self.ws)
        self.assertEqual(code, 2)
        self.assertIn(E_INTEGRITY, self.read_log_codes())
        self.assertIn(E_INTEGRITY, out)

    def test_missing_superseded_ref(self):
        run_quiet(app.cmd_inject, self.ws, "chain")
        code, out = run_quiet(app.cmd_run, self.ws)
        self.assertEqual(code, 2)
        self.assertIn(E_CHAIN, self.read_log_codes())
        self.assertIn(E_CHAIN, out)

    def test_checkpoint_mismatch(self):
        run_quiet(app.cmd_inject, self.ws, "checkpoint")
        code, out = run_quiet(app.cmd_run, self.ws)
        self.assertEqual(code, 0, "run recovers from boundary and completes")
        self.assertIn(E_UPGRADE, self.read_log_codes())
        report = app._read_json(self.p["report"])
        events = report["upgrade_events"]
        self.assertTrue(any(e.get("code") == E_UPGRADE for e in events))
        redo = next(e["redo"] for e in events if e.get("code") == E_UPGRADE)
        self.assertTrue(redo, "redo list must name record ids")


class TestAcceptanceSequence(WorkspaceCase):
    def test_full_acceptance(self):
        code, out = run_quiet(app.cmd_acceptance, self.ws)
        self.assertEqual(code, 0, out)
        self.assertIn('"PASS"', out)


if __name__ == "__main__":
    unittest.main()
