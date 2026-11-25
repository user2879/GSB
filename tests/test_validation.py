from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evidence_diag.cli import main
from evidence_diag.engine import DiagnosticEngine
from evidence_diag.models import AppConfig, Session
from evidence_diag.validation import validate_commands, validate_config, validate_session

from .helpers import sample_objects


class ValidationTests(unittest.TestCase):
    def test_valid_sample(self) -> None:
        config, session, commands = sample_objects()
        self.assertEqual(validate_config(config), [])
        self.assertEqual(validate_session(config, session), [])
        self.assertEqual(validate_commands(commands), [])

    def test_invalid_channel_weight_and_fixed_point(self) -> None:
        config_raw, _, _ = sample_objects()
        raw = config_raw.to_dict()
        raw["channel_weights"]["dense"] = "-1.0"
        raw["fixed_point"]["fractional_bits"] = 10
        raw["fixed_point"]["quantization"] = 3
        raw["limits"]["evidence_budget"] = 0
        issues = validate_config(AppConfig.from_dict(raw))
        paths = {item["path"] for item in issues}
        self.assertEqual({item["code"] for item in issues}, {"E_INPUT_INVALID"})
        self.assertIn("channel_weights.dense", paths)
        self.assertIn("fixed_point.fractional_bits", paths)
        self.assertIn("fixed_point.quantization", paths)
        self.assertIn("limits.evidence_budget", paths)

    def test_duplicate_evidence_and_turn_gap(self) -> None:
        config, session, _ = sample_objects()
        raw = session.to_dict()
        raw["evidence"][1]["evidence_id"] = raw["evidence"][0]["evidence_id"]
        raw["turns"][0]["turn_id"] = 2
        issues = validate_session(config, Session.from_dict(raw))
        self.assertTrue(
            any("duplicate global evidence_id" in item["message"] for item in issues)
        )
        self.assertTrue(
            any("expected contiguous turn_id" in item["message"] for item in issues)
        )

    def test_command_sequence_gap(self) -> None:
        _, _, commands = sample_objects()
        broken = copy.deepcopy(commands)
        broken[3]["seq"] = 99
        issues = validate_commands(broken)
        self.assertTrue(issues)
        self.assertEqual(issues[0]["code"], "E_INPUT_INVALID")

    def test_invalid_command_rolls_back_and_is_isolated(self) -> None:
        config, session, _ = sample_objects()
        engine = DiagnosticEngine(config, session)
        engine.compute()
        before_config = engine.config.to_dict()
        invalid = {
            "seq": 1,
            "op": "set_channel_weight",
            "channel": "dense",
            "weight": "-3",
        }
        outcome = engine.apply_command(invalid)
        self.assertFalse(outcome["valid"])
        self.assertTrue(outcome["rollback"])
        self.assertEqual(engine.config.to_dict(), before_config)
        self.assertEqual(engine.failures[-1]["code"], "E_INPUT_INVALID")
        self.assertTrue(engine.failures[-1]["isolated"])

        valid = {"seq": 2, "op": "set_topic_threshold", "value_bp": 1300}
        outcome = engine.apply_command(valid)
        self.assertTrue(outcome["valid"])
        self.assertEqual(engine.command_cursor, 2)
        result = engine.compute().result
        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual(result["failures"][0]["seq"], 1)

    def test_invalid_initial_input_is_persisted_and_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config").mkdir()
            (root / "sample").mkdir()
            (root / "config" / "demo.json").write_text(
                json.dumps(
                    {
                        "channel_weights": {"dense": "-1"},
                        "fixed_point": {
                            "fractional_bits": 12,
                            "quantization": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "sample" / "session.json").write_text(
                json.dumps(
                    {
                        "session_id": "bad",
                        "turns": [],
                        "evidence": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "sample" / "commands.jsonl").write_text(
                "", encoding="utf-8"
            )
            with redirect_stdout(io.StringIO()):
                code = main(["--root", str(root), "run"])
            self.assertEqual(code, 2)
            failure = json.loads(
                (root / "reports" / "input_failures.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(failure["code"], "E_INPUT_INVALID")
            self.assertTrue(failure["isolated_failures"])


if __name__ == "__main__":
    unittest.main()
