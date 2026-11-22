import json
import tempfile
import unittest
from pathlib import Path

from distlock.errors import (ReferenceError_, SpecError, TimingSpecError)
from distlock.spec import load_events, load_spec

VALID = {
    "session": {"domain_set_version": 1, "scale_factor": 10000,
                "global_concurrency": 2, "per_holder_concurrency": 1,
                "retry_backoff_ms": [5]},
    "domains": [
        {"domain_id": "d1", "capacity": 1, "lease_duration_ms": 100,
         "renew_interval_ms": 20, "queue_limit": 2, "version": 1}],
    "holders": [
        {"holder_id": "h1", "priority": 5, "target_domains": ["d1"]}],
}


def write_tmp(obj, name="spec.json"):
    d = tempfile.mkdtemp()
    p = Path(d) / name
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p


class SpecTests(unittest.TestCase):
    def test_valid_loads(self):
        domains, holders, session = load_spec(write_tmp(VALID))
        self.assertEqual(set(domains), {"d1"})
        self.assertEqual(holders["h1"].priority, 5)
        self.assertEqual(session["scale_factor"], 10000)

    def test_timing_violation_is_e_timing(self):
        bad = json.loads(json.dumps(VALID))
        bad["domains"][0]["renew_interval_ms"] = 100  # equal, not strictly less
        with self.assertRaises(TimingSpecError) as ctx:
            load_spec(write_tmp(bad))
        self.assertEqual(ctx.exception.code, "E_TIMING")
        self.assertEqual(ctx.exception.context["field"], "renew_interval_ms")
        self.assertIn("line", ctx.exception.context)

    def test_missing_domain_reference_is_e_ref(self):
        bad = json.loads(json.dumps(VALID))
        bad["holders"][0]["target_domains"] = ["d1", "ghost"]
        with self.assertRaises(ReferenceError_) as ctx:
            load_spec(write_tmp(bad))
        self.assertEqual(ctx.exception.code, "E_REF")
        self.assertEqual(ctx.exception.context["unknown_domain"], "ghost")

    def test_non_integer_rejected(self):
        bad = json.loads(json.dumps(VALID))
        bad["domains"][0]["capacity"] = "two"
        with self.assertRaises(SpecError) as ctx:
            load_spec(write_tmp(bad))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.context["field"], "capacity")

    def test_event_seq_gap_and_timestamp_regression(self):
        domains, holders, _ = load_spec(write_tmp(VALID))
        d = tempfile.mkdtemp()
        good = Path(d) / "e.ndjson"
        good.write_text("\n".join([
            json.dumps({"seq": 1, "timestamp_ms": 1, "holder_id": "h1",
                        "domain_id": "d1", "action": "申请"}),
            json.dumps({"seq": 3, "timestamp_ms": 2, "holder_id": "h1",
                        "domain_id": "d1", "action": "释放"}),
        ]), encoding="utf-8")
        with self.assertRaises(SpecError) as ctx:
            load_events(good, domains, holders)
        self.assertEqual(ctx.exception.context["field"], "seq")

        bad_ts = Path(d) / "e2.ndjson"
        bad_ts.write_text("\n".join([
            json.dumps({"seq": 1, "timestamp_ms": 5, "holder_id": "h1",
                        "domain_id": "d1", "action": "申请"}),
            json.dumps({"seq": 2, "timestamp_ms": 4, "holder_id": "h1",
                        "domain_id": "d1", "action": "释放"}),
        ]), encoding="utf-8")
        with self.assertRaises(SpecError) as ctx:
            load_events(bad_ts, domains, holders)
        self.assertEqual(ctx.exception.context["field"], "timestamp_ms")

    def test_bad_action_rejected(self):
        domains, holders, _ = load_spec(write_tmp(VALID))
        d = tempfile.mkdtemp()
        p = Path(d) / "e.ndjson"
        p.write_text(json.dumps(
            {"seq": 1, "timestamp_ms": 1, "holder_id": "h1",
             "domain_id": "d1", "action": "explode"}), encoding="utf-8")
        with self.assertRaises(SpecError) as ctx:
            load_events(p, domains, holders)
        self.assertEqual(ctx.exception.context["field"], "action")


if __name__ == "__main__":
    unittest.main()
