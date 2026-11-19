from __future__ import annotations

import unittest

from streamsim.errors import BusinessError
from streamsim.prefetch import run_prefetch_validation


def jobs(count: int, size_bytes: int = 100_000) -> list[dict]:
    return [
        {
            "job_seq": index + 1,
            "segment_index": index,
            "rep_index": 0,
            "bitrate_bps": "400000",
            "size_bytes": str(size_bytes),
            "duration_ms": 2000,
        }
        for index in range(count)
    ]


class PrefetchTests(unittest.TestCase):
    def test_memory_budget_serializes_batches_by_job_order(self) -> None:
        result = run_prefetch_validation(
            jobs(4, size_bytes=4_000_000),
            concurrency=2,
            memory_budget_bytes=5_000_000,
        )
        self.assertEqual(result["batch_count"], 4)
        self.assertEqual(result["peak_active_tasks"], 1)
        self.assertEqual(result["peak_active_bytes"], 4_000_000)
        self.assertEqual(result["executed_jobs"], 4)

    def test_parallel_batches_respect_concurrency(self) -> None:
        result = run_prefetch_validation(
            jobs(5, size_bytes=1_000_000),
            concurrency=2,
            memory_budget_bytes=5_000_000,
        )
        self.assertEqual(result["batch_count"], 3)
        self.assertEqual(result["peak_active_tasks"], 2)
        self.assertLessEqual(result["peak_active_bytes"], 5_000_000)
        self.assertEqual(result["executed_jobs"], 5)

    def test_first_job_error_is_returned_and_remaining_work_is_cancelled(self) -> None:
        def fail_second(job: dict) -> None:
            if job["job_seq"] == 2:
                raise RuntimeError("injected download failure")

        with self.assertRaises(BusinessError) as caught:
            run_prefetch_validation(
                jobs(6, size_bytes=1_000_000),
                concurrency=3,
                memory_budget_bytes=6_000_000,
                task_hook=fail_second,
            )
        error = caught.exception
        self.assertEqual(error.code, "E_PREFETCH")
        self.assertEqual(error.details["first_error_job_seq"], 2)

    def test_oversized_job_is_rejected_without_exceeding_budget(self) -> None:
        with self.assertRaises(BusinessError) as caught:
            run_prefetch_validation(
                jobs(1, size_bytes=6_000_000),
                concurrency=2,
                memory_budget_bytes=5_000_000,
            )
        self.assertEqual(caught.exception.code, "E_PREFETCH")


if __name__ == "__main__":
    unittest.main()
