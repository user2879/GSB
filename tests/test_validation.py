from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from fixedseason.commands import CommandProcessor
from fixedseason.loader import load_config, load_series_jsonl
from fixedseason.model import FitConfig, SeriesRecord


class ValidationTests(unittest.TestCase):
    def test_series_invalid_lines_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "series.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"series_id":"a","timestamps":[0,1,2],"values":[1.0,null,3.0]}',
                        '{"series_id":"a","timestamps":[0,1,2],"values":[1,2,3]}',
                        '{"series_id":"b","timestamps":[0,1,3],"values":[1,2,3]}',
                        '{"series_id":"c","timestamps":[0,1,2],"values":[1,"bad",3]}',
                        '{"series_id":"d","timestamps":[0,1,2],"values":[1,Infinity,3]}',
                        '{"series_id":"e","timestamps":[0,1,2],"values":[null,null,null]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            series, issues = load_series_jsonl(path)
            self.assertEqual([record.series_id for record in series], ["a"])
            self.assertEqual(len(issues), 5)
            self.assertTrue(all(issue.code == "E_SERIES_INVALID" for issue in issues))

    def test_config_ranges_and_cpu_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.yaml"
            path.write_text(
                """\
version: 1
model:
  period: 12
  harmonic_order: 9
  regularization_strength: -1
fixed_point:
  bits: 31
  quantization_shift: 7
  pivot_threshold: -3
tolerance:
  relative: 0.001
  absolute: 0.001
budget:
  scale_profile: standard
runtime:
  thread_pool_workers: 99999
""",
                encoding="utf-8",
            )
            config, issues = load_config(path)
            self.assertIsNone(config)
            messages = "\n".join(issue.message for issue in issues)
            self.assertIn("harmonic_order", messages)
            self.assertIn("regularization_strength", messages)
            self.assertIn("fixed_point.bits", messages)
            self.assertIn("thread_pool_workers", messages)

    def test_valid_config_and_minimal_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ok.yaml"
            path.write_text(
                """\
version: 4
model:
  period: 12
  harmonic_order: 2   # comment
  regularization_strength: 0.25
fixed_point:
  bits: 32
  quantization_shift: 12
  pivot_threshold: 100
tolerance:
  relative: 1.0e-5
  absolute: 2.0e-6
budget:
  scale_profile: small
  global_fit_points: 500
  per_series_quota: 100
  batch_observation_points: 80
runtime:
  thread_pool_workers: 2
diagnostics:
  comparison_bit_widths: [32, 48]
""",
                encoding="utf-8",
            )
            config, issues = load_config(path)
            self.assertEqual(issues, [])
            assert config is not None
            self.assertEqual(config.version, 4)
            self.assertEqual(config.harmonic_order, 2)
            self.assertEqual(config.comparison_bit_widths, (32, 48))
            self.assertLessEqual(config.thread_pool_workers, os.cpu_count() or 1)

    def test_invalid_commands_do_not_stop_later_commands(self) -> None:
        record = SeriesRecord(
            "a", (0.0, 1.0, 2.0, 3.0), (1.0, 2.0, None, 4.0)
        )
        processor = CommandProcessor([record], FitConfig(harmonic_order=1))
        self.assertFalse(processor.apply({"op": "nope"}, line=1))
        self.assertFalse(
            processor.apply(
                {"op": "replace_observation", "series_id": "missing", "index": 0, "value": 1},
                line=2,
            )
        )
        self.assertTrue(
            processor.apply({"op": "set_harmonic_order", "value": 2}, line=3)
        )
        self.assertEqual(processor.state.config.harmonic_order, 2)
        self.assertEqual(len(processor.issues), 2)
        self.assertTrue(
            all(issue.code == "E_COMMAND_INVALID" for issue in processor.issues)
        )


if __name__ == "__main__":
    unittest.main()
