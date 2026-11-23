from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from planner_app.cli import main


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "samples" / "workspace.json"


class InjectionCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.log = self.base / "injections.jsonl"

    def run_injection(self, kind: str) -> int:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(
                [
                    "inject",
                    "--kind",
                    kind,
                    "--workspace",
                    str(WORKSPACE),
                    "--ledger",
                    str(self.base / f"{kind}.sqlite"),
                    "--snapshots",
                    str(self.base / "snapshots"),
                    "--log",
                    str(self.log),
                    "--reset-ledger",
                ]
            )
        self.last_payload = json.loads(stdout.getvalue())
        return code

    def test_three_required_business_codes_are_emitted_and_logged(self) -> None:
        expectations = {
            "syntax": "E_SYNTAX",
            "bad-ref": "E_REF",
            "snapshot-digest": "E_SNAPSHOT",
        }
        for kind, expected in expectations.items():
            with self.subTest(kind=kind):
                self.assertEqual(self.run_injection(kind), 0)
                self.assertEqual(self.last_payload["code"], expected)
                self.assertTrue(self.last_payload["ok"])
                self.assertIsNotNone(self.last_payload.get("edit_seq"))
                self.assertGreater(self.last_payload["edit_seq"], 0)
        records = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        matching = {record["code"]: record for record in records if record["code"] in expectations.values()}
        for expected in expectations.values():
            self.assertIn(expected, matching)
            self.assertGreater(matching[expected]["mono_ns"], 0)
            self.assertGreater(matching[expected]["edit_seq"], 0)


if __name__ == "__main__":
    unittest.main()
