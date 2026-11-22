import os
import tempfile
import unittest

from workbench.engine import ScalingEngine
from workbench.errors import LengthError, SchemaError, SnapshotError
from workbench.faults import (
    inject_declaration_corruption,
    inject_length_mismatch,
    inject_snapshot_mismatch,
)
from workbench.latency import LatencyRecorder
from workbench.ledger import Ledger
from workbench.loader import check_series_lengths
from workbench.logging_utils import JsonlLogger
from workbench.sample_data import build_cluster, build_corrupt_cluster
from workbench.testing_utils import spec_from_payload, tmp_declaration


class FaultInjectionTest(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.log = os.path.join(self.base, "faults.jsonl")
        self.logger = JsonlLogger(self.log)

    def test_schema_fault(self):
        path = tmp_declaration(build_corrupt_cluster())
        with self.assertRaises(SchemaError) as caught:
            inject_declaration_corruption(path, self.logger)
        self.assertEqual(caught.exception.code, "E_SCHEMA")

    def test_length_fault(self):
        spec = spec_from_payload(build_cluster())
        tampered = inject_length_mismatch(spec, self.logger)
        with self.assertRaises(LengthError) as caught:
            check_series_lengths(tampered)
        self.assertEqual(caught.exception.code, "E_LENGTH")
        self.assertEqual(caught.exception.details["field"], "requests")

    def test_snapshot_fault(self):
        db = os.path.join(self.base, "ledger.db")
        ledger = Ledger(db)
        recorder = LatencyRecorder(50, 1000, self.logger)
        engine = ScalingEngine(ledger, self.logger, recorder)
        spec = spec_from_payload(build_cluster())
        version = engine.register_declaration(spec)
        engine.run_pipeline(spec, version)
        with self.assertRaises(SnapshotError) as caught:
            inject_snapshot_mismatch(ledger, version, self.logger)
        self.assertEqual(caught.exception.code, "E_SNAPSHOT")
        with open(self.log, encoding="utf-8") as handle:
            content = handle.read()
        for code in ("E_SCHEMA",):
            pass
        self.assertIn("E_SNAPSHOT", content)
        ledger.close()


if __name__ == "__main__":
    unittest.main()
