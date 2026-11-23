from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from cacheproof.cli import main
from cacheproof.spec import load_spec
from cacheproof.store import CacheProofStore


ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "cache_spec.json"


class InjectionTests(unittest.TestCase):
    def _workspace(self, parent: str, name: str) -> Path:
        workspace = Path(parent) / name
        with CacheProofStore(
            workspace,
            spec=load_spec(SAMPLE),
            create=True,
            auto_snapshot=False,
        ):
            pass
        return workspace

    def _diagnose(self, workspace: Path, kind: str) -> tuple[int, dict, dict]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = main(
                [
                    "diagnose",
                    "--workspace",
                    str(workspace),
                    "--inject",
                    kind,
                    "--json",
                ]
            )
        result = json.loads(output.getvalue())
        log_lines = (workspace / "logs" / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        error_events = [
            json.loads(line)
            for line in log_lines
            if json.loads(line).get("event") == "business_error"
        ]
        self.assertTrue(error_events)
        event = error_events[-1]
        self.assertIsInstance(event["monotonic_ns"], int)
        self.assertIsInstance(event["operation_seq"], int)
        return code, result, event

    def test_truncate_is_e_integrity_and_logged(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cacheproof-inject-") as temp:
            workspace = self._workspace(temp, "truncate")
            code, result, event = self._diagnose(workspace, "truncate")
            self.assertEqual(code, 4)
            self.assertEqual(result["code"], "E_INTEGRITY")
            self.assertEqual(event["business_code"], "E_INTEGRITY")

    def test_capacity_is_e_capacity_and_logged(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cacheproof-inject-") as temp:
            workspace = self._workspace(temp, "capacity")
            code, result, event = self._diagnose(workspace, "capacity")
            self.assertEqual(code, 3)
            self.assertEqual(result["code"], "E_CAPACITY")
            self.assertEqual(result["segment_id"], "seg-alpha")
            self.assertEqual(event["business_code"], "E_CAPACITY")

    def test_ledger_digest_is_e_snapshot_and_logged(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cacheproof-inject-") as temp:
            workspace = self._workspace(temp, "ledger")
            code, result, event = self._diagnose(workspace, "ledger-digest")
            self.assertEqual(code, 5)
            self.assertEqual(result["code"], "E_SNAPSHOT")
            self.assertEqual(result["field"], "ledger.content_digest")
            self.assertEqual(event["business_code"], "E_SNAPSHOT")


if __name__ == "__main__":
    unittest.main()

