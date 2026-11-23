from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from timbre_workbench.cli import main


class CliSmokeTests(unittest.TestCase):
    def test_demo_run_and_compare_exit_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            demo_dir = root / "demo"
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["demo", "--out", str(demo_dir)]), 0)
            report = root / "report.json"
            html = root / "report.html"
            responses = root / "responses.json"
            with redirect_stdout(StringIO()):
                run_code = main(
                    [
                        "run",
                        "--input",
                        str(demo_dir / "audio"),
                        "--config",
                        str(demo_dir / "config.json"),
                        "--labels",
                        str(demo_dir / "labels.json"),
                        "--commands",
                        str(demo_dir / "commands.json"),
                        "--work",
                        str(root / "work"),
                        "--report",
                        str(report),
                        "--html",
                        str(html),
                        "--responses",
                        str(responses),
                    ]
                )
            self.assertEqual(run_code, 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "complete")
            self.assertIn("segments", payload)
            self.assertIn("canonical_digest", payload)
            self.assertIsNotNone(payload["diagnostics"]["first_response"])
            self.assertIn(payload["diagnostics"]["first_response"]["status"], {"pending", "complete"})
            self.assertTrue(html.read_text(encoding="utf-8").lower().startswith("<!doctype html>"))
            response_payload = json.loads(responses.read_text(encoding="utf-8"))
            self.assertEqual(len(response_payload["responses"]), 7)
            self.assertIn("stable_state", response_payload["responses"][-1])
            self.assertIn("speaker_vectors", response_payload["responses"][-1]["stable_state"])
            with redirect_stdout(StringIO()):
                compare_code = main(
                    [
                        "compare",
                        "--input",
                        str(demo_dir / "audio"),
                        "--config",
                        str(demo_dir / "config.json"),
                        "--labels",
                        str(demo_dir / "labels.json"),
                        "--commands",
                        str(demo_dir / "commands.json"),
                        "--output",
                        str(root / "compare.json"),
                    ]
                )
            self.assertEqual(compare_code, 0)
            comparison = json.loads((root / "compare.json").read_text(encoding="utf-8"))
            self.assertTrue(comparison["equivalent"])


if __name__ == "__main__":
    unittest.main()
