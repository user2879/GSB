import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from graymirror.cli import main

RULES = "samples/rules.json"
STREAM = "samples/stream.ndjson"


def expected_grid_step_fraction(exponent: int) -> Fraction:
    # delta = 10000 / 10**k expressed in per-10000 grid units
    return Fraction(10_000, 10 ** exponent)


class CliTests(unittest.TestCase):
    def test_run_across_exponents_and_bounds(self):
        for exponent in (0, 1, 2, 4, 6):
            with tempfile.TemporaryDirectory() as tmp:
                report = Path(tmp) / "report.json"
                log = Path(tmp) / "events.jsonl"
                code = main([
                    "run", "--rules", RULES, "--stream", STREAM,
                    "--scale-exponent", str(exponent),
                    "--report", str(report), "--log", str(log),
                ])
                self.assertEqual(code, 0)
                payload = json.loads(report.read_text())
                self.assertTrue(payload["ok"])
                self.assertEqual(len(payload["decisions"]), 200)
                self.assertTrue(log.exists())
                expected = expected_grid_step_fraction(exponent)
                for rule in payload["quantized_rules"]:
                    bound = Fraction(
                        rule["gray_quantization"]["error_bound_per_10000"])
                    self.assertEqual(bound, expected)

    def test_invalid_exponent_returns_e_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "events.jsonl"
            code = main([
                "run", "--rules", RULES, "--stream", STREAM,
                "--scale-exponent", "9", "--log", str(log),
            ])
            self.assertEqual(code, 2)
            records = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(records[0]["code"], "E_GRID")

    def test_injection_commands(self):
        for kind, expected in (
            ("weight_out_of_range", "E_RULE"),
            ("timestamp_regression", "E_STREAM"),
            ("snapshot_scale_missing", "E_GRID"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                log = Path(tmp) / "events.jsonl"
                code = main([
                    "inject", "--kind", kind,
                    "--workdir", str(Path(tmp) / "inj"),
                    "--log", str(log),
                ])
                self.assertEqual(code, 2)
                record = json.loads(log.read_text().splitlines()[0])
                self.assertEqual(record["actual"], expected)


if __name__ == "__main__":
    unittest.main()
