from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from quota_ledger.cli import main

from .helpers import make_sample


class InjectionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sample, _ = make_sample(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run(self, *args: str) -> int:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(list(args))
        self.last_output = output.getvalue()
        return code

    def _last_json(self):
        return json.loads(self.last_output.strip().splitlines()[-1])

    def test_corrupt_node_declaration_reports_e_spec_and_logs(self) -> None:
        workdir = self.root / "corrupt"
        code = self._run("inject", "--kind", "corrupt-nodes", "--sample", str(self.sample), "--workdir", str(workdir))
        self.assertEqual(code, 0)
        payload = self._last_json()
        self.assertEqual(payload["code"], "E_SPEC")
        self.assertIn("nodes.json:", payload["position"])
        log = [json.loads(line) for line in (workdir / "structured.jsonl").read_text().splitlines()]
        self.assertEqual(log[-1]["code"], "E_SPEC")
        self.assertIn("monotonic_ms", log[-1])
        self.assertIn("shard_id", log[-1])

    def test_cycle_reports_e_graph_and_stable_cycle(self) -> None:
        workdir = self.root / "cycle"
        code = self._run("inject", "--kind", "cycle", "--sample", str(self.sample), "--workdir", str(workdir))
        self.assertEqual(code, 0)
        payload = self._last_json()
        self.assertEqual(payload["code"], "E_GRAPH")
        self.assertEqual(payload["shard_id"], "s01")
        self.assertEqual(payload["detail"]["cycle"], ["s01", "s10", "s01"])
        log = [json.loads(line) for line in (workdir / "structured.jsonl").read_text().splitlines()]
        self.assertEqual(log[-1]["code"], "E_GRAPH")
        self.assertEqual(log[-1]["shard_id"], "s01")

    def test_snapshot_tamper_reports_e_snapshot_and_logs(self) -> None:
        workdir = self.root / "snapshot"
        code = self._run("snapshot-check", "--workdir", str(workdir), "--inject")
        self.assertEqual(code, 0)
        payload = self._last_json()
        self.assertEqual(payload["code"], "E_SNAPSHOT")
        self.assertEqual(payload["field"], "content_digest")
        log = [json.loads(line) for line in (workdir / "structured.jsonl").read_text().splitlines()]
        self.assertEqual(log[-1]["code"], "E_SNAPSHOT")
        self.assertIn("monotonic_ms", log[-1])
        self.assertIn("shard_id", log[-1])


if __name__ == "__main__":
    unittest.main()
