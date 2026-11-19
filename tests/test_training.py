from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .common import adapter_payloads, copy_samples, read_report, run_workspace


class TrainingTests(unittest.TestCase):
    def test_concurrency_matches_serial_field_by_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            serial_samples = copy_samples(
                root / "serial-samples", [("concurrency: 3", "concurrency: 1")]
            )
            parallel_samples = copy_samples(
                root / "parallel-samples", [("concurrency: 3", "concurrency: 4")]
            )
            serial_work = root / "serial-work"
            parallel_work = root / "parallel-work"
            self.assertEqual(run_workspace(serial_samples, serial_work).exit_code, 0)
            self.assertEqual(run_workspace(parallel_samples, parallel_work).exit_code, 0)
            serial = read_report(serial_work)
            parallel = read_report(parallel_work)
            for key in (
                "compatibility",
                "compatibility_hash",
                "reuse",
                "retrain",
                "training_executed",
                "failures",
                "business_codes",
            ):
                self.assertEqual(serial[key], parallel[key], key)
            serial_adapters = adapter_payloads(serial, serial_work)
            parallel_adapters = adapter_payloads(parallel, parallel_work)
            self.assertEqual(sorted(serial_adapters), sorted(parallel_adapters))
            for task_name in serial_adapters:
                left = serial_adapters[task_name]
                right = parallel_adapters[task_name]
                self.assertEqual(left["A"], right["A"])
                self.assertEqual(left["B"], right["B"])
                self.assertEqual(left["loss_trajectory"], right["loss_trajectory"])
                self.assertEqual(left["content_hash"], right["content_hash"])
            self.assertEqual(
                serial["merge"]["merged_tasks"], parallel["merge"]["merged_tasks"]
            )
            self.assertEqual(serial["merge"]["budget"], parallel["merge"]["budget"])
            self.assertEqual(
                serial["merge"]["restore_max_abs_error"],
                parallel["merge"]["restore_max_abs_error"],
            )


if __name__ == "__main__":
    unittest.main()

