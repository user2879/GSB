"""配额仲裁工作台验收测试（unittest，本地确定性样本）。"""

import copy
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from quota_workbench.arbitrate import RULE_VERSION, arbitrate_volume
from quota_workbench.errors import WorkbenchError
from quota_workbench.events import evaluate_events, load_events
from quota_workbench.invalidate import affected_domains, replace_view
from quota_workbench.ledger import Ledger
from quota_workbench.loader import load_declaration
from quota_workbench.models import canonical
from quota_workbench.normalize import normalize_volume_sources
from quota_workbench.pipeline import (all_rows, build_view, load_rules,
                                      scheme_meta, volume_batch_rows)

SAMPLES = os.path.join(ROOT, "samples")


def _view():
    decl = load_declaration(os.path.join(SAMPLES, "declaration.json"))
    events = load_events(os.path.join(SAMPLES, "events.ndjson"))
    rules = load_rules(os.path.join(SAMPLES, "rules.json"))
    return decl, build_view(decl, events, rules["target_profile"],
                            rules["rule_version"]), rules


class TestFieldValidation(unittest.TestCase):
    def _write_decl(self, mutate):
        with open(os.path.join(SAMPLES, "declaration.json"),
                  encoding="utf-8") as fh:
            data = json.load(fh)
        mutate(data)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                          encoding="utf-8")
        json.dump(data, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_unknown_version_rejected(self):
        def mutate(data):
            data["sources"][0]["format_version"] = 99
        path = self._write_decl(mutate)
        with self.assertRaises(WorkbenchError) as ctx:
            load_declaration(path)
        self.assertEqual(ctx.exception.code, "E_FIELD")
        self.assertEqual(ctx.exception.details["field"], "format_version")
        self.assertIn("source_type", ctx.exception.details)
        self.assertIn("source_id", ctx.exception.details)

    def test_negative_capacity_rejected(self):
        def mutate(data):
            for rec in data["sources"]:
                if rec["source_type"] == "capacity_claim":
                    rec["fields"]["capacity_limit_bytes"] = -1
                    return
        path = self._write_decl(mutate)
        with self.assertRaises(WorkbenchError) as ctx:
            load_declaration(path)
        self.assertEqual(ctx.exception.code, "E_FIELD")
        self.assertEqual(ctx.exception.details["field"], "capacity_limit_bytes")

    def test_missing_required_field_rejected(self):
        def mutate(data):
            for rec in data["sources"]:
                if rec["source_type"] == "rate_claim" and rec["format_version"] == 2:
                    del rec["fields"]["burst_bucket_capacity"]
                    rec["field_count"] -= 1
                    return
        path = self._write_decl(mutate)
        with self.assertRaises(WorkbenchError) as ctx:
            load_declaration(path)
        self.assertEqual(ctx.exception.details["field"], "burst_bucket_capacity")

    def test_duplicate_volume_rejected(self):
        def mutate(data):
            data["volumes"].append(copy.deepcopy(data["volumes"][0]))
        path = self._write_decl(mutate)
        with self.assertRaises(WorkbenchError) as ctx:
            load_declaration(path)
        self.assertEqual(ctx.exception.details["field"], "volume_id")

    def test_field_count_inconsistent_rejected(self):
        decl, _, _ = _view()
        raw = copy.deepcopy(decl["sources"]["vol-01-cap2"])
        raw["field_count"] = 99
        with self.assertRaises(WorkbenchError) as ctx:
            normalize_volume_sources("vol-01", [raw])
        self.assertEqual(ctx.exception.details["field"], "field_count")


class TestNormalize(unittest.TestCase):
    def test_low_version_defaults_filled_and_version_kept(self):
        decl, _, _ = _view()
        records = normalize_volume_sources(
            "vol-01", [decl["sources"][s] for s in decl["volumes"]["vol-01"]])
        rate1 = next(r for r in records if r.source_id.endswith("rate1"))
        self.assertEqual(rate1.format_version, 1)
        self.assertIn("burst_bucket_capacity", rate1.defaulted_fields)
        self.assertIn("scale_factor", rate1.defaulted_fields)
        ops = rate1.extra["raw_ops_per_second"]
        self.assertEqual(rate1.metrics["burst_bucket_capacity"], ops)
        der1 = next(r for r in records if r.source_id.endswith("der1"))
        self.assertEqual(der1.format_version, 1)
        self.assertIn("divisor", der1.defaulted_fields)
        self.assertIn("rounding", der1.defaulted_fields)


class TestArbitration(unittest.TestCase):
    def test_unique_conclusion_per_metric(self):
        _, view, _ = _view()
        for vid, concl in view["arbitration"].items():
            self.assertEqual(set(concl), {"capacity_limit_bytes",
                                          "ops_per_second",
                                          "burst_bucket_capacity"})

    def test_order_independence(self):
        decl, _, _ = _view()
        vid = "vol-01"
        raws = [decl["sources"][s] for s in decl["volumes"][vid]]
        results = set()
        for seed in range(8):
            shuffled = list(raws)
            random.Random(seed).shuffle(shuffled)
            records = normalize_volume_sources(vid, shuffled)
            results.add(canonical(arbitrate_volume(vid, records)))
        self.assertEqual(len(results), 1)

    def test_rejected_evidence_kept_with_reason(self):
        _, view, _ = _view()
        concl = view["arbitration"]["vol-01"]["capacity_limit_bytes"]
        self.assertEqual(len(concl["rejected"]), 2)
        reasons = {r["reason"] for r in concl["rejected"]}
        self.assertTrue(reasons <= {"source_type_bias", "older_format_version",
                                    "source_id_order"})
        self.assertEqual(concl["rule_version"], RULE_VERSION)


class TestEvents(unittest.TestCase):
    def test_timestamp_regression_rejected(self):
        _, view, _ = _view()
        events = [
            {"volume_id": "vol-01", "ts_ms": 100, "op_count": 1,
             "op_bytes": 1, "seq": 1},
            {"volume_id": "vol-01", "ts_ms": 50, "op_count": 1,
             "op_bytes": 1, "seq": 2},
        ]
        with self.assertRaises(WorkbenchError) as ctx:
            evaluate_events(events, view["arbitration"])
        self.assertEqual(ctx.exception.code, "E_ORDER")
        self.assertEqual(ctx.exception.details["volume_id"], "vol-01")

    def test_verdicts_cover_all_kinds(self):
        _, view, _ = _view()
        kinds = set()
        for s in view["events_summary"]:
            kinds |= {k for k, v in s["verdict_counts"].items() if v}
        self.assertTrue({"allow", "throttle", "over_quota",
                         "indeterminate_boundary"} <= kinds)


class TestMigrationAttribution(unittest.TestCase):
    def test_conflicts_have_attribution_with_intermediates(self):
        _, view, _ = _view()
        conflicts = [r for rows in view["migration"].values() for r in rows
                     if r["status"] == "conflict"]
        self.assertTrue(conflicts)
        attr = [r for rows in view["attribution"].values() for r in rows]
        self.assertEqual(len(attr), len(conflicts) * 3)
        for row in attr:
            self.assertIn("v_current", row["intermediates"])
            self.assertIn("v_step3", row["intermediates"])

    def test_attribution_stable_order(self):
        _, view, _ = _view()
        for vid, rows in view["attribution"].items():
            keys = [(-(r["contribution"] if r["contribution"] is not None else -1),
                     r["metric"]) for r in rows]
            self.assertEqual(keys, sorted(keys))

    def test_consistent_when_profile_identical(self):
        decl, _, _ = _view()
        vid = "vol-01"
        records = normalize_volume_sources(
            vid, [decl["sources"][s] for s in decl["volumes"][vid]])
        current = arbitrate_volume(vid, records)
        from quota_workbench.migrate import migration_review
        profile = {"profile_id": "same", "profile_version": 1,
                   "allowed_source_types": ["capacity_claim", "rate_claim",
                                            "derived_claim"],
                   "min_format_version": 1}
        rows = migration_review(vid, current, records, profile, RULE_VERSION)
        self.assertTrue(all(r["status"] == "consistent" for r in rows))


class TestInvalidation(unittest.TestCase):
    def test_unaffected_rows_byte_identical(self):
        decl, view, rules = _view()
        old_sources = decl["sources"]
        new_sources = copy.deepcopy(old_sources)
        new_sources["vol-01-cap1"]["fields"]["capacity_limit_bytes"] += 4096
        affected = affected_domains(old_sources, new_sources, decl["volumes"])
        self.assertEqual(affected, {("vol-01", "capacity_limit_bytes")})
        new_decl = {"volumes": decl["volumes"], "sources": new_sources}
        events = load_events(os.path.join(SAMPLES, "events.ndjson"))
        new_view = build_view(new_decl, events, rules["target_profile"],
                              rules["rule_version"])
        old_flat = [c for v in view["arbitration"].values() for c in v.values()]
        new_flat = [c for v in new_view["arbitration"].values()
                    for c in v.values()]
        replaced, retired, added, intact = replace_view(old_flat, new_flat,
                                                        affected)
        self.assertTrue(intact)
        self.assertEqual(len(retired), 1)
        self.assertEqual(len(added), 1)
        self.assertNotEqual(canonical(retired[0]), canonical(added[0]))
        kept_old = {canonical(r) for r in old_flat
                    if (r["volume_id"], r["metric"]) not in affected}
        kept_new = {canonical(r) for r in replaced
                    if (r["volume_id"], r["metric"]) not in affected}
        self.assertEqual(kept_old, kept_new)


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ledger_path = os.path.join(self.tmp.name, "ledger.sqlite")

    def _commit_all(self):
        decl, view, rules = _view()
        ledger = Ledger(self.ledger_path)
        meta = scheme_meta(decl, rules["rule_version"], 1)
        for seq, vid in enumerate(sorted(view["arbitration"]), 1):
            ledger.commit_batch(seq, vid, volume_batch_rows(view, vid), meta)
        return decl, view, rules, ledger

    def test_commit_and_replay_ok(self):
        decl, view, rules, ledger = self._commit_all()
        self.assertEqual(ledger.check_snapshot(
            scheme_meta(decl, rules["rule_version"], 1)["sources_digest"],
            rules["rule_version"]), "ok")
        self.assertTrue(ledger.replay_verify(all_rows(view)))
        ledger.close()

    def test_snapshot_mismatch_rejected_and_full_replace(self):
        decl, view, rules, ledger = self._commit_all()
        with self.assertRaises(WorkbenchError) as ctx:
            ledger.check_snapshot("0" * 64, rules["rule_version"])
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
        meta = scheme_meta(decl, rules["rule_version"], 2)
        ledger.full_replace(all_rows(view), meta)
        self.assertEqual(ledger.read_meta()["scheme_version"], "2")
        self.assertTrue(ledger.replay_verify(all_rows(view)))
        ledger.close()

    def test_replay_locates_first_difference(self):
        decl, view, rules, ledger = self._commit_all()
        row = ledger.conn.execute(
            "SELECT domain, volume_id, metric, payload FROM conclusions"
            " WHERE domain='arbitration' ORDER BY volume_id LIMIT 1").fetchone()
        payload = json.loads(row[3])
        payload["value"] += 1
        ledger.conn.execute(
            "DELETE FROM conclusions WHERE domain=? AND volume_id=? AND metric=?",
            (row[0], row[1], row[2]))
        ledger.conn.execute(
            "INSERT INTO conclusions VALUES (?, ?, ?, ?)",
            (row[0], row[1], row[2], canonical(payload)))
        ledger.conn.commit()
        with self.assertRaises(WorkbenchError) as ctx:
            ledger.replay_verify(all_rows(view))
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.details["volume_id"], row[1])
        self.assertEqual(ctx.exception.details["metric"], row[2])
        self.assertEqual(ctx.exception.details["field"], "value")
        ledger.close()


class TestKillRecovery(unittest.TestCase):
    def test_kill_between_batches_then_resume(self):
        import shutil
        workdir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(workdir, ignore_errors=True))
        for name in ("declaration.json", "events.ndjson", "rules.json"):
            shutil.copyfile(os.path.join(SAMPLES, name),
                            os.path.join(workdir, name))
        env = dict(os.environ)
        env["QW_SAMPLES_DIR"] = workdir
        script = os.path.join(ROOT, "scripts", "kill_recovery_test.py")
        first = subprocess.run(
            [sys.executable, script, "--kill-after", "5"],
            capture_output=True, text=True, env=env)
        self.assertEqual(first.returncode, 42, first.stderr)
        second = subprocess.run([sys.executable, script],
                                capture_output=True, text=True, env=env)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("重放校验通过", second.stdout)
        self.assertTrue(os.path.exists(
            os.path.join(workdir, "backup", "ledger.sqlite.bak")))
        self.assertTrue(os.path.exists(os.path.join(workdir, "checkpoint.txt")))


if __name__ == "__main__":
    unittest.main()
