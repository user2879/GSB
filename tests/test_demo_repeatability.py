from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pickworkbench.canonical import content_hash


ROOT = Path(__file__).resolve().parents[1]


def run_demo(target: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-m", "pickworkbench", "demo", "--out", str(target)],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )


def read_tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class DemoRepeatabilityTests(unittest.TestCase):
    def test_demo_command_is_byte_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first_root = Path(temporary) / "first"
            second_root = Path(temporary) / "second"
            first = run_demo(first_root)
            second = run_demo(second_root)
            self.assertEqual(first.returncode, 0, first.stderr.decode("utf-8", errors="replace"))
            self.assertEqual(second.returncode, 0, second.stderr.decode("utf-8", errors="replace"))
            self.assertEqual(first.stdout, second.stdout)
            self.assertEqual(first.stderr, b"")
            self.assertEqual(second.stderr, b"")
            first_tree = read_tree(first_root)
            second_tree = read_tree(second_root)
            self.assertEqual(first_tree, second_tree)
            self.assertEqual(len(first_tree), 42)
            self.assertIn(b"repeat_check=identical", first.stdout)

            report = json.loads(first_tree["report.json"])
            self.assertEqual(len(report["cases"]), 7)
            self.assertTrue(all(case["status"] == "pass" for case in report["cases"]))
            self.assertEqual(report["generation_repeat_check"]["independent_generations_identical"], True)
            expected_overall = content_hash(
                {
                    "source_summary": report["source_summary"],
                    "cases": report["cases"],
                }
            )
            self.assertEqual(report["overall_hash"], expected_overall)

            for case in report["cases"]:
                report_path = f"cases/{case['name']}/report.json"
                case_report = json.loads(first_tree[report_path])
                self.assertEqual(case_report["content_hash"], case["case_report_hash"])
                if case["name"] == "00_normal":
                    self.assertEqual(case_report["records_hash"], case["records_hash"])
                    self.assertEqual(case["summary"]["passed"], 22)
                    self.assertEqual(case["summary"]["failed"], 0)
                elif case["name"] == "06_invalid_inject_config":
                    self.assertNotIn("records_hash", case_report)
                    self.assertEqual(case["detected_error_codes"], ["E_INJECT_CONFIG"])
                else:
                    self.assertEqual(case_report["records_hash"], case["records_hash"])
                    self.assertGreater(case["summary"]["failed"], 0)
                    self.assertTrue(case["witness_minimal"])

    def test_witness_cli_reproduces_first_affected_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "demo"
            demo = run_demo(target)
            self.assertEqual(demo.returncode, 0, demo.stderr.decode("utf-8", errors="replace"))
            witness_path = target / "cases" / "03_duplicate_id" / "witness.json"
            result = subprocess.run(
                [sys.executable, "-m", "pickworkbench", "witness", "--witness", str(witness_path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
            self.assertIn(b"reproduced=true", result.stdout)
            self.assertIn(b"error_code=E_OBJECT_DUP_ID", result.stdout)
            self.assertIn(b"query_id=q-000-ray", result.stdout)

    def test_generate_and_validate_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = subprocess.run(
                [sys.executable, "-m", "pickworkbench", "generate", "--out", str(root / "sample")],
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr.decode("utf-8", errors="replace"))
            self.assertIn(b"objects=12", generated.stdout)
            self.assertIn(b"queries=22", generated.stdout)

            validated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pickworkbench",
                    "validate",
                    "--scene",
                    str(root / "sample" / "scene.json"),
                    "--queries",
                    str(root / "sample" / "queries.json"),
                    "--out",
                    str(root / "report.json"),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr.decode("utf-8", errors="replace"))
            self.assertIn(b"status=pass", validated.stdout)
            self.assertIn(b"queries=22 passed=22 failed=0", validated.stdout)


if __name__ == "__main__":
    unittest.main()
