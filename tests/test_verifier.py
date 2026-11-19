import unittest

from rts_replay.generator import PROFILES, build_rules, generate_events
from rts_replay.offline import run_offline
from rts_replay.verifier import (
    error_code_audit,
    reconcile,
    verify_jump_equivalence,
)


class VerifierTests(unittest.TestCase):
    def test_all_sample_paths_reconcile(self):
        for profile in PROFILES:
            rules = build_rules(profile)
            events = generate_events(profile, rules)
            offline = run_offline(rules, events)
            result = reconcile(rules, events, offline)
            self.assertTrue(result.ledger_match, profile.name)
            self.assertIsNone(result.first_mismatch, profile.name)
            self.assertIsNotNone(result.rng_drift_tick, profile.name)

    def test_seq_gap_sample_reports_code_and_location(self):
        profile = next(item for item in PROFILES if item.kind == "seq_gap")
        rules = build_rules(profile)
        events = generate_events(profile, rules)
        offline = run_offline(rules, events)
        audit = error_code_audit(offline)
        self.assertIn("E_SEQ_GAP", audit["error_counts"])
        location = audit["first_locations"]["E_SEQ_GAP"]
        self.assertIsNotNone(location["tick"])
        self.assertIsNotNone(location["actor"])

    def test_overflow_and_contention_samples(self):
        for kind in ("overflow", "contention"):
            profile = next(item for item in PROFILES if item.kind == kind)
            rules = build_rules(profile)
            events = generate_events(profile, rules)
            offline = run_offline(rules, events)
            audit = error_code_audit(offline)
            self.assertTrue(audit["error_counts"], profile.name)

    def test_jump_equivalence_across_samples(self):
        for profile in PROFILES:
            rules = build_rules(profile)
            events = generate_events(profile, rules)
            offline = run_offline(rules, events)
            targets = [0, 1, profile.ticks // 2, profile.ticks - 1]
            outcome = verify_jump_equivalence(rules, events, offline, targets)
            self.assertTrue(outcome["all_match"], profile.name)


if __name__ == "__main__":
    unittest.main()
