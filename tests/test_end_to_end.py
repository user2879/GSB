import json
import os
import tempfile
import unittest

from workbench.cli import main
from workbench.gui import build_view_model
from workbench.latency import LatencyRecorder
from workbench.ledger import Ledger
from workbench.logging_utils import JsonlLogger
from workbench.models import canonical_bytes
from workbench.sample_data import write_samples
from workbench.testing_utils import spec_from_payload
from workbench.sample_data import build_cluster


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.sample_dir = os.path.join(self.base, "samples")
        self.db = os.path.join(self.base, "data", "workbench.db")
        self.log = os.path.join(self.base, "logs", "workbench.jsonl")
        write_samples(self.sample_dir)
        self.declaration = os.path.join(self.sample_dir, "cluster.json")

    def _args(self, *extra):
        return ["--db", self.db, "--log", self.log, *extra]

    def test_full_cli_lifecycle_and_error_codes(self):
        code = main(self._args("run", self.declaration))
        self.assertEqual(code, 0)
        code = main(self._args("replay", self.declaration))
        self.assertEqual(code, 0)
        code = main(self._args("recover", self.declaration))
        self.assertEqual(code, 0)
        code = main(self._args("gui", "--headless"))
        self.assertEqual(code, 0)

        corrupt = os.path.join(self.sample_dir, "cluster_corrupt_schema.json")
        code = main(self._args("fault", "schema", corrupt))
        self.assertEqual(code, 2)  # E_SCHEMA is first declared error code
        code = main(self._args("fault", "length", self.declaration))
        self.assertEqual(code, 3)  # E_LENGTH
        code = main(self._args("fault", "snapshot"))
        self.assertEqual(code, 8)  # E_SNAPSHOT

        with open(self.log, encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
        logged_codes = {event.get("code") for event in events if "code" in event}
        self.assertIn("E_SCHEMA", logged_codes)
        self.assertIn("E_LENGTH", logged_codes)
        self.assertIn("E_SNAPSHOT", logged_codes)

    def test_view_model_render_order_is_stable(self):
        isolated = tempfile.mkdtemp()
        db = os.path.join(isolated, "view.db")
        main(["--db", db, "--log", os.path.join(isolated, "view.jsonl"),
              "run", self.declaration])
        logger = JsonlLogger(self.log)
        ledger = Ledger(db)
        version = ledger.active_version()
        start_seq = ledger.next_latency_seq(version) - 1
        recorder = LatencyRecorder(50, 1000, logger, start_seq=start_seq)
        for row in ledger.latency_samples(version):
            recorder.samples.append(__import__(
                'workbench.latency', fromlist=['LatencySample']
            ).LatencySample(
                seq=row['seq'], kind=row['kind'], service_id=row['service_id'],
                elapsed_ms=row['elapsed_ms'], budget_ms=row['budget_ms'],
                over_budget=bool(row['over_budget']),
            ))
        view = build_view_model(ledger, version, recorder)
        rows = view.decision_rows()
        self.assertEqual(len(rows), 24 * 4)
        keys = [(row.minute, row.service_id) for row in rows]
        self.assertEqual(keys, sorted(keys))
        curve = view.budget_curve()
        self.assertEqual([point.minute for point in curve], list(range(24)))
        self.assertTrue(all(point.total_allocated <= point.budget for point in curve))
        snapshot_rows = view.snapshot_rows()
        self.assertEqual(len(snapshot_rows), 24)
        ledger.close()

    def test_committed_payloads_are_canonical_and_digestible(self):
        isolated = tempfile.mkdtemp()
        db = os.path.join(isolated, "canon.db")
        main(["--db", db, "--log", os.path.join(isolated, "canon.jsonl"),
              "run", self.declaration])
        ledger = Ledger(db)
        version = ledger.active_version()
        for snapshot in ledger.load_snapshots(version):
            payload = json.loads(snapshot["payload_json"])
            self.assertEqual(
                snapshot["payload_json"].encode("utf-8"),
                canonical_bytes(payload),
            )
        ledger.close()

    def test_incremental_window_fixed_point_outputs(self):
        from workbench.estimator import IncrementalEstimator

        spec = spec_from_payload(build_cluster())
        incremental = IncrementalEstimator(spec.window_size, spec.fixed_point_scale)
        incremental.run(spec)
        minute, window_sum, window_mean = incremental.window_history[0]
        self.assertEqual(minute, 0)
        self.assertEqual(len(window_sum), len(spec.services))
        self.assertTrue(all(value >= 0 for value in window_mean))


if __name__ == "__main__":
    unittest.main()
