import json
import tempfile
import unittest
from pathlib import Path

from elastic_scaler.engine import ScalingEngine
from elastic_scaler.errors import BudgetError, LoadError, VersionConflictError
from elastic_scaler.fault_injection import (
    inject_budget_violation,
    inject_load_gap,
    inject_snapshot_version_conflict,
)
from elastic_scaler.logging_util import StructuredLogger
from elastic_scaler.spec import load_spec, parse_spec

V1 = "samples/cluster_v1.json"


class FaultInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.raw = json.loads(Path(V1).read_text(encoding="utf-8"))
        self.log_path = root / "events.jsonl"
        if self.log_path.exists():
            self.log_path.unlink()
        self.logger = StructuredLogger(self.log_path)

    def test_load_gap_raises_e_load_with_field_path(self):
        with self.assertRaises(LoadError) as caught:
            mutated = inject_load_gap(self.raw, "llm", 4)
            parse_spec(mutated)
        self.assertEqual(caught.exception.code, "E_LOAD")
        self.assertIn("requests[4]", caught.exception.fields[0])
        self.logger.log_error(caught.exception)

    def test_bad_budget_raises_e_budget(self):
        with self.assertRaises(BudgetError) as caught:
            inject_budget_violation(self.raw, -3)
        self.assertEqual(caught.exception.code, "E_BUDGET")
        self.logger.log_error(caught.exception)

    def test_snapshot_conflict_raises_e_version_and_all_errors_are_logged(self):
        db = Path(self.tmp.name) / "ledger.sqlite3"
        with ScalingEngine(str(db), str(self.log_path)) as engine:
            spec = load_spec(V1)
            version, _ = engine.initialize(spec)
            engine.advance_minute(0)
            # Re-log the first two injected faults through this engine's logger
            # so all three structured error events share one append stream.
            try:
                inject_load_gap(self.raw, "llm", 4)
            except LoadError as load_error:
                engine.logger.log_error(load_error)
            try:
                inject_budget_violation(self.raw, -3)
            except BudgetError as budget_error:
                engine.logger.log_error(budget_error)
            with self.assertRaises(VersionConflictError) as caught:
                inject_snapshot_version_conflict(
                    engine.ledger, "stale-snap", engine.run_id, version + 1, 0
                )
            self.assertEqual(caught.exception.code, "E_VERSION")
            engine.logger.log_error(caught.exception)
        codes = [entry.get("code") for entry in self.logger.read_all()
                 if entry.get("event") == "error"]
        self.assertEqual(codes, ["E_LOAD", "E_BUDGET", "E_VERSION"])


if __name__ == "__main__":
    unittest.main()
