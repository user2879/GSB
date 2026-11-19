import json
import os
import tempfile
import unittest

from rts_replay.errors import ErrorCode, RtsError
from rts_replay.events import event_from_dict, load_events
from rts_replay.rules import FactionRules, RuleConfig
from rts_replay.state import SNAPSHOT_FORMAT_VERSION, build_snapshot, snapshot_from_dict
from rts_replay.generator import PROFILES, build_rules, generate_events
from rts_replay.engine import TickEngine


class ErrorCodeTests(unittest.TestCase):
    def test_malformed_missing_field(self):
        with self.assertRaises(RtsError) as caught:
            event_from_dict({"tick": 1, "actor": "red"})
        self.assertEqual(caught.exception.code, ErrorCode.E_INPUT_MALFORMED)

    def test_malformed_wrong_type(self):
        good = {
            "tick": 1,
            "actor": "red",
            "action": "collect",
            "target": "mine_a",
            "amount": 1,
            "rng_seed_delta": 1,
            "seq": 0,
        }
        bad = dict(good, amount="10")
        with self.assertRaises(RtsError) as caught:
            event_from_dict(bad)
        self.assertEqual(caught.exception.code, ErrorCode.E_INPUT_MALFORMED)

    def test_malformed_unknown_action(self):
        with self.assertRaises(RtsError) as caught:
            event_from_dict(
                {
                    "tick": 1,
                    "actor": "red",
                    "action": "dance",
                    "target": "mine_a",
                    "amount": 1,
                    "rng_seed_delta": 1,
                    "seq": 0,
                }
            )
        self.assertEqual(caught.exception.code, ErrorCode.E_INPUT_MALFORMED)

    def test_malformed_jsonl_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "events.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{not valid json}\n")
            with self.assertRaises(RtsError) as caught:
                load_events(path)
            self.assertEqual(caught.exception.code, ErrorCode.E_INPUT_MALFORMED)
            self.assertEqual(caught.exception.context["line_no"], 1)

    def test_rule_range_invalid_value(self):
        rules = RuleConfig(
            match_seed=1,
            rng_chain_version=1,
            queue_ticks=2,
            factions=(FactionRules("red", 0, 10, 5, 1),),
        )
        with self.assertRaises(RtsError) as caught:
            rules.validate()
        self.assertEqual(caught.exception.code, ErrorCode.E_RULE_RANGE)

    def test_rule_range_cost_above_cap(self):
        rules = RuleConfig(
            match_seed=1,
            rng_chain_version=1,
            queue_ticks=2,
            factions=(FactionRules("red", 10, 10, 50, 1),),
        )
        with self.assertRaises(RtsError) as caught:
            rules.validate()
        self.assertEqual(caught.exception.code, ErrorCode.E_RULE_RANGE)

    def test_snapshot_stale_version(self):
        profile = PROFILES[0]
        rules = build_rules(profile)
        state = TickEngine(rules, generate_events(profile, rules)).state
        snapshot = build_snapshot(state)
        raw = snapshot.to_dict()
        raw["format_version"] = SNAPSHOT_FORMAT_VERSION - 1
        with self.assertRaises(RtsError) as caught:
            snapshot_from_dict(raw)
        self.assertEqual(caught.exception.code, ErrorCode.E_SNAPSHOT_STALE)

    def test_snapshot_tampered_hash(self):
        profile = PROFILES[0]
        rules = build_rules(profile)
        state = TickEngine(rules, generate_events(profile, rules)).state
        raw = build_snapshot(state).to_dict()
        raw["state"]["factions"][rules.faction_names()[0]]["resources"] += 1
        with self.assertRaises(RtsError) as caught:
            snapshot_from_dict(raw)
        self.assertEqual(caught.exception.code, ErrorCode.E_SNAPSHOT_STALE)

    def test_error_payload_is_locatable(self):
        try:
            raise RtsError(
                ErrorCode.E_SEQ_GAP,
                "gap",
                tick=7,
                actor="red",
                action="collect",
                context={"expected": 3, "actual": 5},
            )
        except RtsError as exc:
            payload = exc.to_dict()
        self.assertEqual(payload["code"], "E_SEQ_GAP")
        self.assertEqual(payload["tick"], 7)
        self.assertEqual(payload["actor"], "red")
        self.assertEqual(payload["context"]["actual"], 5)


if __name__ == "__main__":
    unittest.main()
