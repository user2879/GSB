from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .helpers import make_workspace, run_cli


class CliTests(unittest.TestCase):
    def test_run_uses_supported_slots_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ws = Path(temp) / "custom"
            initialized = run_cli(
                [
                    "init",
                    "--workspace",
                    str(ws),
                    "--samples",
                    "6",
                    "--slots",
                    "0,2",
                ]
            )
            self.assertEqual(initialized.returncode, 0, initialized.stderr + initialized.stdout)
            completed = run_cli(["run", "--workspace", str(ws)])
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            report = json.loads(completed.stdout)
            self.assertEqual([slot["batch_slot"] for slot in report["slots"]], [0, 2])
            audited = run_cli(["audit", "--workspace", str(ws)])
            self.assertEqual(audited.returncode, 0, audited.stderr + audited.stdout)
            self.assertEqual(json.loads(audited.stdout)["code"], "AUDIT_OK")

    def test_invalid_feed_exits_with_efeed_and_first_error_location(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ws, _ = make_workspace(Path(temp))
            feed_path = ws / "feed.json"
            feed = json.loads(feed_path.read_text(encoding="utf-8"))
            first = feed["entries"][0]
            second = feed["entries"][1]
            second["arrival_seq"] = first["arrival_seq"]
            feed_path.write_text(
                json.dumps(feed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            completed = run_cli(["run", "--workspace", str(ws)])
            self.assertEqual(completed.returncode, 2, completed.stderr + completed.stdout)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["code"], "E_FEED")
            self.assertEqual(payload["arrival_seq"], first["arrival_seq"])
            self.assertIn(".arrival_seq", payload["field_path"])


if __name__ == "__main__":
    unittest.main()
