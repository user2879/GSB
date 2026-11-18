from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mixedprec_diag.config import load_config, parse_config
from mixedprec_diag.errors import E_CONFIG, DiagnosticError
from tests.helpers import config_payload


class ConfigTests(unittest.TestCase):
    def test_valid_config_normalized_units(self) -> None:
        config = parse_config(config_payload())
        self.assertEqual(config.total_blocks, 64)
        self.assertEqual(config.effective_budget_bytes, 1024 * 1024)
        self.assertEqual(config.quantization_scale, 10000.0)

    def test_unsupported_block_size_has_field_path(self) -> None:
        with self.assertRaises(DiagnosticError) as raised:
            parse_config(config_payload(block_size_kb=7))
        self.assertEqual(raised.exception.code, E_CONFIG)
        self.assertEqual(raised.exception.field_path, "$.block_size_kb")

    def test_budget_rounding_reject_and_floor(self) -> None:
        with self.assertRaises(DiagnosticError) as raised:
            parse_config(
                config_payload(memory_budget_mib=2, block_size_kb=1536, budget_rounding="reject")
            )
        self.assertEqual(raised.exception.code, E_CONFIG)
        self.assertEqual(raised.exception.field_path, "$.memory_budget_mib")

        config = parse_config(
            config_payload(memory_budget_mib=2, block_size_kb=1536, budget_rounding="floor_to_block")
        )
        self.assertEqual(config.total_blocks, 1)
        self.assertEqual(config.effective_budget_bytes, 1536 * 1024)

    def test_bad_power_batch_and_unknown_field(self) -> None:
        cases = [
            (config_payload(quant_scale_power=2.5), "$.quant_scale_power"),
            (config_payload(quant_scale_power=10), "$.quant_scale_power"),
            (config_payload(batch_size_limit=0), "$.batch_size_limit"),
            ({**config_payload(), "extra": 1}, "$.extra"),
            (config_payload(precision="float16"), "$.precision"),
            (config_payload(preallocation="eager"), "$.preallocation"),
            (config_payload(ui_query_budget_ms=0), "$.ui_query_budget_ms"),
            (config_payload(pressure_threshold_ms=-1), "$.pressure_threshold_ms"),
        ]
        for payload, expected_path in cases:
            with self.subTest(expected_path=expected_path):
                with self.assertRaises(DiagnosticError) as raised:
                    parse_config(payload)
                self.assertEqual(raised.exception.code, E_CONFIG)
                self.assertEqual(raised.exception.field_path, expected_path)

    def test_boolean_is_not_accepted_as_integer(self) -> None:
        with self.assertRaises(DiagnosticError) as raised:
            parse_config(config_payload(batch_size_limit=True))
        self.assertEqual(raised.exception.code, E_CONFIG)
        self.assertEqual(raised.exception.field_path, "$.batch_size_limit")

    def test_duplicate_json_key_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text(
                '{"schema_version":1,"memory_budget_mib":1,"memory_budget_mib":2}',
                encoding="utf-8",
            )
            with self.assertRaises(DiagnosticError) as raised:
                load_config(path)
            self.assertEqual(raised.exception.code, E_CONFIG)
            self.assertEqual(raised.exception.field_path, "$.memory_budget_mib")

    def test_invalid_json_reports_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.json"
            path.write_text('{"schema_version": 1,', encoding="utf-8")
            with self.assertRaises(DiagnosticError) as raised:
                load_config(path)
            self.assertEqual(raised.exception.code, E_CONFIG)
            self.assertIn("line 1", raised.exception.field_path or "")

    def test_all_demo_configs_validate(self) -> None:
        for name in ("float32.json", "fixed.json", "scaled.json"):
            with self.subTest(name=name):
                config = load_config(Path("configs") / name)
                self.assertGreaterEqual(config.total_blocks, 1)
                json.dumps(config.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
