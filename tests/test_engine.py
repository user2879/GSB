import unittest

from rts_replay.engine import TickEngine
from rts_replay.errors import ErrorCode
from rts_replay.events import Event
from rts_replay.generator import PROFILES, build_rules, generate_events
from rts_replay.rules import FactionRules, RuleConfig
from rts_replay.state import GameState


def simple_rules(seed=1):
    return RuleConfig(
        match_seed=seed,
        rng_chain_version=1,
        queue_ticks=2,
        factions=(
            FactionRules("red", 100, 50, 20, 1),
            FactionRules("blue", 100, 50, 20, 2),
        ),
    )


class EngineTests(unittest.TestCase):
    def test_deterministic_tick_hashes_on_generated_samples(self):
        for profile in PROFILES:
            rules = build_rules(profile)
            events = generate_events(profile, rules)
            engine_a = TickEngine(rules, events)
            engine_b = TickEngine(rules, list(events))
            engine_a.advance_to(engine_a.last_tick)
            engine_b.advance_to(engine_b.last_tick)
            self.assertEqual(
                [r.state_hash for r in engine_a.records],
                [r.state_hash for r in engine_b.records],
                profile.name,
            )

    def test_snapshot_every_tick_matches_incremental(self):
        profile = PROFILES[0]
        rules = build_rules(profile)
        events = generate_events(profile, rules)
        incremental = TickEngine(rules, events, snapshot_interval=1)
        incremental.advance_to(incremental.last_tick)
        one_shot = TickEngine(rules, events)
        for record in incremental.records:
            one_shot.advance_to(record.tick)
            self.assertEqual(one_shot.state.state_hash(), record.state_hash)

    def test_no_lookahead_prefix_independence(self):
        profile = PROFILES[0]
        rules = build_rules(profile)
        events = generate_events(profile, rules)
        full = TickEngine(rules, events)
        full.advance_to(20)
        prefix_events = [event for event in events if event.tick <= 20]
        prefix = TickEngine(rules, prefix_events)
        prefix.advance_to(20)
        self.assertEqual(full.state.state_hash(), prefix.state.state_hash())

    def test_resource_overflow_collected_and_flagged(self):
        rules = simple_rules()
        events = [
            Event(1, "red", "collect", "mine_a", 500, 1, 0),
            Event(2, "red", "collect", "mine_a", 500, 2, 1),
        ]
        engine = TickEngine(rules, events)
        engine.advance_to(engine.last_tick)
        self.assertEqual(engine.state.factions["red"].resources, 100)
        codes = {error.code for error in engine.all_errors()}
        self.assertIn(ErrorCode.E_RESOURCE_OVERFLOW.value, codes)

    def test_queue_blocked_on_concurrency_and_resources(self):
        rules = simple_rules()
        events = [
            Event(1, "red", "collect", "mine_a", 40, 1, 0),
            Event(1, "red", "produce", "base", 1, 2, 1),
            Event(1, "red", "produce", "base", 1, 3, 2),
            Event(2, "red", "produce", "base", 5, 4, 3),
        ]
        engine = TickEngine(rules, events)
        engine.advance_to(engine.last_tick)
        codes = {error.code for error in engine.all_errors()}
        self.assertIn(ErrorCode.E_QUEUE_BLOCKED.value, codes)

    def test_sequence_gap_detected(self):
        rules = simple_rules()
        events = [
            Event(1, "red", "collect", "mine_a", 10, 1, 0),
            Event(1, "red", "collect", "mine_a", 10, 2, 2),
        ]
        engine = TickEngine(rules, events)
        engine.advance_to(engine.last_tick)
        gap_errors = [e for e in engine.all_errors() if e.code == ErrorCode.E_SEQ_GAP.value]
        self.assertEqual(len(gap_errors), 1)
        self.assertEqual(gap_errors[0].context["expected"], 1)
        self.assertEqual(gap_errors[0].context["actual"], 2)

    def test_queue_advances_into_idle_units(self):
        rules = simple_rules()
        events = [
            Event(1, "red", "collect", "mine_a", 40, 1, 0),
            Event(1, "red", "produce", "base", 2, 2, 1),
            Event(3, "red", "collect", "mine_a", 5, 3, 2),
        ]
        engine = TickEngine(rules, events)
        engine.advance_to(1)
        self.assertEqual(engine.state.factions["red"].idle_units, 0)
        engine.advance_to(3)
        self.assertEqual(engine.state.factions["red"].idle_units, 2)
        self.assertEqual(engine.state.factions["red"].queue, [])

    def test_unknown_actor_is_malformed(self):
        rules = simple_rules()
        events = [Event(1, "ghost", "collect", "mine_a", 10, 1, 0)]
        with self.assertRaises(Exception) as caught:
            TickEngine(rules, events)
        self.assertEqual(caught.exception.code, ErrorCode.E_INPUT_MALFORMED)


if __name__ == "__main__":
    unittest.main()
