import unittest

from rts_replay.engine import TickEngine
from rts_replay.generator import PROFILES, build_rules, generate_events
from rts_replay.rng import RNG_CHAIN_VERSION, derive_seed, make_event_rng


class RngChainTests(unittest.TestCase):
    def test_seed_derivation_is_order_independent(self):
        seed_a = derive_seed(7, 3, 11)
        seed_b = derive_seed(7, 3, 11)
        self.assertEqual(seed_a, seed_b)
        self.assertNotEqual(derive_seed(7, 3, 12), seed_a)
        self.assertNotEqual(derive_seed(7, 4, 11), seed_a)

    def test_same_events_same_random_values(self):
        profile = PROFILES[0]
        rules = build_rules(profile)
        events = generate_events(profile, rules)
        engine_a = TickEngine(rules, events)
        engine_b = TickEngine(rules, list(reversed(events)))
        engine_a.advance_to(engine_a.last_tick)
        engine_b.advance_to(engine_b.last_tick)
        ledger_a = [entry.to_dict() for entry in engine_a.ledger]
        ledger_b = [entry.to_dict() for entry in engine_b.ledger]
        self.assertEqual(
            [(item["tick"], item["actor"], item["seq"]) for item in ledger_a],
            [(item["tick"], item["actor"], item["seq"]) for item in ledger_b],
        )
        self.assertEqual(
            [item["rng_value"] for item in ledger_a],
            [item["rng_value"] for item in ledger_b],
        )

    def test_chain_version_change_drifts_hashes(self):
        profile = PROFILES[0]
        rules = build_rules(profile)
        events = generate_events(profile, rules)
        baseline = TickEngine(rules, events)
        drifted = TickEngine(rules, events, chain_version=RNG_CHAIN_VERSION + 1)
        baseline.advance_to(baseline.last_tick)
        drifted.advance_to(drifted.last_tick)
        baseline_hashes = {r.tick: r.state_hash for r in baseline.records}
        drifted_hashes = {r.tick: r.state_hash for r in drifted.records}
        changed = [tick for tick in baseline_hashes if baseline_hashes[tick] != drifted_hashes[tick]]
        self.assertTrue(changed)
        self.assertEqual(changed[0], 1)

    def test_independent_substreams(self):
        rng_one = make_event_rng(1, 5, 9)
        rng_two = make_event_rng(1, 6, 9)
        self.assertNotEqual(
            rng_one.integers(0, 1_000_000, size=8).tolist(),
            rng_two.integers(0, 1_000_000, size=8).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
