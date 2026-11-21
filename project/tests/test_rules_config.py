from __future__ import annotations

import unittest

from arbitration_workbench.config import parse_config
from arbitration_workbench.errors import ConfigValidationError, RuleValidationError
from arbitration_workbench.rules import parse_rules


def base_rules() -> dict:
    return {
        "evidence_type_weights": {"a": "0.5"},
        "role_trust": {"r": "0.9"},
        "relations": [],
        "priority_bands": [{"level": 0, "name": "P0", "min_score": "0"}],
    }


class RuleAndConfigTests(unittest.TestCase):
    def test_self_reflexive_relation_is_rejected(self):
        raw = base_rules()
        raw["relations"] = [
            {
                "id": "self",
                "kind": "exclusive",
                "left": {"evidence_type": "a", "source_roles": ["r"]},
                "right": {"evidence_type": "a", "source_roles": ["r"]},
            }
        ]
        with self.assertRaises(RuleValidationError) as raised:
            parse_rules(raw)
        self.assertEqual("E_JSON_INVALID", raised.exception.code)
        self.assertIn("self-reflexive", raised.exception.message)

    def test_relation_unknown_type_and_role_are_rejected(self):
        raw = base_rules()
        raw["relations"] = [
            {
                "id": "bad-type",
                "kind": "exclusive",
                "left": {"evidence_type": "missing"},
                "right": {"evidence_type": "a"},
            }
        ]
        with self.assertRaises(RuleValidationError) as raised:
            parse_rules(raw)
        self.assertEqual("E_EVIDENCE_TYPE_UNKNOWN", raised.exception.code)
        raw["relations"][0]["left"]["evidence_type"] = "a"
        raw["relations"][0]["left"]["source_roles"] = ["missing-role"]
        with self.assertRaises(RuleValidationError) as raised:
            parse_rules(raw)
        self.assertEqual("E_ROLE_UNKNOWN", raised.exception.code)

    def test_invalid_quantization_and_half_life_are_rejected(self):
        raw = {
            "fixed_decimal_places": 6,
            "quantization_units": 0,
            "decay_half_life": 0,
            "exclusion_strength": "0.1",
            "fusion": "sum",
            "decision_threshold": "1.0",
            "interaction_budget_ms": 10,
            "worker_threads": 2,
            "snapshot_commit_interval": 1,
            "snapshot_dir": "snapshots",
            "temp_dir": "tmp",
        }
        with self.assertRaises(ConfigValidationError):
            parse_config(raw)


if __name__ == "__main__":
    unittest.main()

