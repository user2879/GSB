from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from h2mux.cli import main


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_validate_run_snapshot_and_replay(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            base = Path(directory)
            validate_output = base / "validate.json"
            code = main(
                [
                    "validate",
                    "--declaration",
                    str(ROOT / "samples" / "session.json"),
                    "--script",
                    str(ROOT / "samples" / "frames.json"),
                    "--output",
                    str(validate_output),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(validate_output.read_text())["code"], "OK")

            snapshot = base / "stage.json"
            run_output = base / "run.json"
            code = main(
                [
                    "run",
                    "--declaration",
                    str(ROOT / "samples" / "session.json"),
                    "--script",
                    str(ROOT / "samples" / "frames.json"),
                    "--strategy",
                    "weighted_round_robin",
                    "--snapshot-at",
                    "12",
                    "--snapshot",
                    str(snapshot),
                    "--output",
                    str(run_output),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(run_output.read_text())["summary"]["completion"]["complete"])
            self.assertTrue(snapshot.exists())

            replay_output = base / "replay.json"
            code = main(
                [
                    "replay",
                    "--declaration",
                    str(ROOT / "samples" / "session.json"),
                    "--script",
                    str(ROOT / "samples" / "frames.json"),
                    "--snapshot",
                    str(snapshot),
                    "--strategy",
                    "weighted_round_robin",
                    "--output",
                    str(replay_output),
                ]
            )
            self.assertEqual(code, 0)
            report = json.loads(replay_output.read_text())
            self.assertEqual(report["status"], "MATCH")
            self.assertGreater(report["compared_record_count"], 0)


if __name__ == "__main__":
    unittest.main()
