import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from graymirror.engine.errors import E_REPLAY, E_VERSION, GrayError
from graymirror.engine.loader import load_rules, load_stream
from graymirror.engine.replay import verify_replay
from graymirror.engine.router import RoutingEngine
from graymirror.engine.session import Session, restore_session

RULES = "samples/rules.json"
STREAM = "samples/stream.ndjson"


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_rules(RULES)
        cls.requests = load_stream(STREAM, cls.rules)

    def test_snapshot_restore_replay_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snap.json"
            session = Session()
            session.replace(self.rules, 2, self.requests)
            session.snapshot(snapshot)
            restored, payload, _ = restore_session(RULES, STREAM, snapshot)
            verdict = verify_replay(restored.current, payload)
            self.assertEqual(verdict["status"], "MATCH")
            self.assertEqual(verdict["checked_decisions"], 200)

    def test_version_mismatch_raises_e_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snap.json"
            session = Session()
            session.replace(self.rules, 2, self.requests)
            session.snapshot(snapshot)
            changed = [
                replace(rule, gray_weight=rule.gray_weight + 1)
                if rule.rule_id == "rule-users" else rule
                for rule in self.rules
            ]
            rule_path = Path(tmp) / "rules.json"
            rule_path.write_text(
                json.dumps({"rules": [
                    {k: v for k, v in r.__dict__.items() if k != "source_line"}
                    for r in changed]}))
            with self.assertRaises(GrayError) as ctx:
                restore_session(rule_path, STREAM, snapshot)
            self.assertEqual(ctx.exception.code, E_VERSION)
            self.assertIn("rule_fingerprint", ctx.exception.field_path)

    def test_tampered_snapshot_raises_e_replay_with_first_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snap.json"
            session = Session()
            session.replace(self.rules, 2, self.requests)
            session.snapshot(snapshot)
            payload = json.loads(snapshot.read_text())
            target = payload["decisions"][7]
            target["channel"] = "gray" if target["channel"] == "stable" else "stable"
            snapshot.write_text(json.dumps(payload))
            restored, snap, _ = restore_session(RULES, STREAM, snapshot)
            with self.assertRaises(GrayError) as ctx:
                verify_replay(restored.current, snap)
            self.assertEqual(ctx.exception.code, E_REPLAY)
            self.assertEqual(ctx.exception.details["first_request_id"],
                             payload["decisions"][7]["request_id"])

    def test_replacement_marks_old_views_stale_and_versions_monotonic(self):
        session = Session()
        first = session.replace(self.rules, 2, self.requests)
        changed = [
            replace(rule, mirror_rate=1000) if rule.rule_id == "rule-orders"
            else rule for rule in self.rules
        ]
        second = session.replace(changed, 2, self.requests)
        self.assertEqual(first.version, 1)
        self.assertEqual(second.version, 2)
        self.assertTrue(first.stale)
        self.assertFalse(second.stale)
        self.assertTrue(any(
            item["field"] == "mirror_rate" for item in second.change_ledger))

    def test_request_permutation_does_not_change_report(self):
        first = Session().replace(self.rules, 2, self.requests)
        reversed_stream = list(reversed(self.requests))
        second = Session().replace(self.rules, 2, reversed_stream)
        self.assertEqual(first.statistics, second.statistics)
        self.assertEqual(
            first.statistics["global"]["contributions"],
            second.statistics["global"]["contributions"])

    def test_mirror_records_never_change_main_routing(self):
        engine = RoutingEngine(self.rules, 100)
        baseline = {d.request_id: d.channel
                    for d in [engine.decide_route(r)[1] for r in self.requests]}
        result = engine.run(self.requests)
        for decision in result["decisions"]:
            self.assertEqual(decision.channel, baseline[decision.request_id])
        # every mirror record references an existing request but is stored apart
        mirrored_ids = {m.request_id for m in result["mirrors"]}
        decision_ids = {d.request_id for d in result["decisions"]}
        self.assertTrue(mirrored_ids <= decision_ids)


if __name__ == "__main__":
    unittest.main()
