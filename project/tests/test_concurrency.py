from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from arbitration_workbench.config import parse_config
from arbitration_workbench.engine import ArbitrationEngine
from arbitration_workbench.numeric import read_json
from arbitration_workbench.rules import parse_rules

from .helpers import ROOT


class ConcurrencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = parse_rules(read_json(ROOT / "samples" / "rules.json"))
        cls.config = parse_config(read_json(ROOT / "samples" / "config.json"))
        cls.evidence_lines = (ROOT / "samples" / "evidence.jsonl").read_text(encoding="utf-8").splitlines()

    def _engine(self, root: Path, workers: int, output_name: str, injections=None) -> ArbitrationEngine:
        return ArbitrationEngine(
            rules=self.rules,
            config=replace(self.config, worker_threads=workers),
            evidence_lines=self.evidence_lines,
            workspace_root=root,
            cache_store=None,
            snapshot_store=None,
            exception_injections=injections or [],
            output_dir=root / output_name,
            write_outputs=True,
        )

    def test_single_thread_and_four_thread_results_are_field_equal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = self._engine(root, 1, "single").run(use_cache=False, mode="full")
            parallel = self._engine(root, 4, "parallel").run(use_cache=False, mode="full")
            self.assertEqual(single.comparison_payload(), parallel.comparison_payload())
            self.assertEqual(sorted(ticket["ticket_id"] for ticket in parallel.tickets), [t["ticket_id"] for t in parallel.tickets])
            self.assertTrue(parallel.diagnostics["concurrency"]["worker_limit_respected"])
            self.assertTrue(parallel.diagnostics["concurrency"]["final_order_is_ticket_id_sorted"])
            self.assertTrue(parallel.diagnostics["concurrency"]["stable_task_keys_unique"])
            for ticket in parallel.tickets:
                keys = [
                    (item["relation_id"], item["left_seq"], item["right_seq"], item["kind"])
                    for item in ticket["relation_contributions"]
                ]
                self.assertEqual(sorted(keys), keys)

    def test_each_stage_exception_is_isolated_and_other_tickets_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in ("score", "relations", "decision", "convert"):
                with self.subTest(stage=stage):
                    injection = {
                        "active": True,
                        "ticket_id": "T0048",
                        "stage": stage,
                        "exception_type": f"Synthetic{stage.title()}Failure",
                        "message": f"injected {stage}",
                    }
                    engine = self._engine(root, 4, f"fault-{stage}", [injection])
                    result = engine.run(use_cache=False, mode="full")
                    self.assertEqual(47, len(result.tickets))
                    self.assertNotIn("T0048", {ticket["ticket_id"] for ticket in result.tickets})
                    self.assertEqual(1, len(result.failures))
                    failure = result.failures[0]
                    self.assertEqual("T0048", failure["ticket_id"])
                    self.assertEqual(stage, failure["stage"])
                    self.assertEqual("E_STAGE_EXCEPTION", failure["code"])
                    self.assertEqual(f"Synthetic{stage.title()}Failure", failure["exception_type"])
                    self.assertEqual(
                        sorted(ticket["ticket_id"] for ticket in result.tickets),
                        [ticket["ticket_id"] for ticket in result.tickets],
                    )
                    self.assertEqual(1, result.diagnostics["concurrency"]["failed_ticket_count"])


if __name__ == "__main__":
    unittest.main()

