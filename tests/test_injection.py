from __future__ import annotations

import copy
import unittest

from story_workbench.injection import (
    apply_injections,
    locate_first_flip,
    parse_injection_config,
)
from story_workbench.samples import injection_flip_config, injection_suite_config
from story_workbench.timeline import NarrativeSession

try:
    from tests.helpers import injection, normal
except ModuleNotFoundError:  # supports `unittest discover -s tests`
    from helpers import injection, normal


class InjectionTests(unittest.TestCase):
    def test_threshold_injection_locates_first_flip(self):
        spec = injection()
        baseline = NarrativeSession(spec, "injection_path").run()
        config = parse_injection_config(injection_flip_config())
        injected_spec, records = apply_injections(spec, config)
        injected = NarrativeSession(injected_spec, "injection_path").run()
        report = locate_first_flip(
            baseline,
            injected,
            injection_records=records,
            injection_config=config,
        )
        self.assertTrue(report["flip_detected"])
        self.assertTrue(report["ending_changed"])
        self.assertEqual(report["baseline"]["ending"], "good")
        self.assertEqual(report["injected"]["ending"], "blocked:i_gate")
        flip = report["first_decision_flip"]
        self.assertEqual(flip["condition_id"], "c_trust_good")
        self.assertEqual(flip["before_margin"], 0.09999999999999998)
        self.assertEqual(flip["after_margin"], -0.010000000000000009)
        self.assertTrue(flip["margin_interval"]["contains_zero"])
        self.assertEqual(report["error_interval"], flip["margin_interval"])
        self.assertEqual(report["injection_config"]["id"], "flip-threshold")

    def test_all_injection_kinds_are_serializable_and_effective(self):
        spec = normal()
        config = parse_injection_config(injection_suite_config())
        injected, records = apply_injections(spec, config)
        self.assertEqual(
            [record["operation"]["kind"] for record in records],
            ["coefficient_perturb", "threshold_shift", "initial_offset", "remove_effects"],
        )
        self.assertNotEqual(spec.spec_hash, injected.spec_hash)
        self.assertEqual(injected.initial_state[2], 0.01)
        fail_edge = injected.edge_by_id("e_fail")
        self.assertEqual(fail_edge.effects, ())
        rep_effect = injected.edge_by_id("e_intro").effects[3]
        self.assertEqual(rep_effect.expression, "(0.15000000000000002 + 0.2)")
        wait_condition = injected.edge_by_id("e_wait").conditions[0]
        self.assertEqual(wait_condition.threshold, 0.5)
        self.assertEqual(parse_injection_config(config.canonical_json()), config)
        roundtrip_again, records_again = apply_injections(spec, config)
        self.assertEqual(roundtrip_again.spec_hash, injected.spec_hash)
        self.assertEqual(records_again, records)

    def test_injection_does_not_mutate_baseline_spec(self):
        spec = normal()
        original = copy.deepcopy(spec.canonical_dict())
        config = parse_injection_config(injection_suite_config())
        apply_injections(spec, config)
        self.assertEqual(spec.canonical_dict(), original)

    def test_remove_effects_is_applied_as_input_parameter(self):
        spec = normal()
        config = parse_injection_config(
            {
                "schema": "narrative-workbench/injections/v1",
                "id": "drop",
                "operations": [{"kind": "remove_effects", "edge_id": "e_intro"}],
            }
        )
        injected, records = apply_injections(spec, config)
        self.assertEqual(injected.edge_by_id("e_intro").effects, ())
        self.assertEqual(records[0]["before"], [effect.id for effect in spec.edge_by_id("e_intro").effects])
        baseline = NarrativeSession(spec, "normal_path").run()
        changed = NarrativeSession(injected, "normal_path").run()
        self.assertEqual(baseline.ending, "alliance")
        self.assertEqual(changed.ending, "blocked:s_choice")


if __name__ == "__main__":
    unittest.main()
