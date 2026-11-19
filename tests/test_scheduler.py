from __future__ import annotations

import time
import unittest

from vecflow.errors import AppError
from vecflow.model import parse_data_payload, parse_tasks_payload, workspace_state_from_parts
from vecflow.planner import compile_workspace
from vecflow.scheduler import build_work_items, run_schedule
from tests.helpers import data_payload, tasks_payload


def _plan(*, shards=None, budget=4, threads=3, quota=2):
    payload = tasks_payload(budget=budget, threads=threads, quota=quota, shards=shards)
    state = workspace_state_from_parts(parse_data_payload(data_payload()), parse_tasks_payload(payload))
    return compile_workspace(state)


class SchedulerTests(unittest.TestCase):
    def test_global_budget_and_task_quota_are_never_exceeded(self) -> None:
        plan = _plan(
            shards=[{"start": index, "end": index + 1, "resources": 1} for index in range(8)],
            budget=2,
            threads=4,
            quota=2,
        )
        items = build_work_items(plan)

        def execute(item):
            time.sleep(0.03)
            return None

        report = run_schedule(
            items,
            global_budget=2,
            task_quotas={"t": 2},
            max_workers=4,
            execute_fn=execute,
        )
        self.assertEqual(report.peak_resources, 2)
        self.assertEqual(report.task_peak_concurrency["t"], 2)
        self.assertTrue(all(record.wait_duration_ms is not None for record in report.records))
        self.assertGreater(max(record.wait_duration_ms for record in report.records), 0)
        self.assertEqual([record.sequence for record in report.records], list(range(8)))

    def test_budget_exhaustion_is_e_budget(self) -> None:
        plan = _plan(shards=[{"start": 0, "end": 8, "resources": 2}], budget=0)
        with self.assertRaises(AppError) as raised:
            run_schedule(
                build_work_items(plan),
                global_budget=0,
                task_quotas={"t": 1},
                max_workers=2,
                execute_fn=lambda item: None,
            )
        self.assertEqual(raised.exception.code, "E_BUDGET")
        self.assertEqual(raised.exception.details["blocked_shards"], ["t#0000"])

    def test_multiple_tasks_two_dimensional_constraints(self) -> None:
        data = parse_data_payload(data_payload())
        payload = tasks_payload(budget=4, threads=4, quota=1)
        second = dict(payload["tasks"][0])
        second["id"] = "u"
        second["quota"] = 2
        payload["tasks"].append(second)
        state = workspace_state_from_parts(data, parse_tasks_payload(payload))
        plan = compile_workspace(state)

        def execute(item):
            time.sleep(0.025)
            return None

        report = run_schedule(
            build_work_items(plan),
            global_budget=4,
            task_quotas={"t": 1, "u": 2},
            max_workers=4,
            execute_fn=execute,
        )
        self.assertLessEqual(report.peak_resources, 4)
        self.assertLessEqual(report.task_peak_concurrency["t"], 1)
        self.assertLessEqual(report.task_peak_concurrency["u"], 2)


if __name__ == "__main__":
    unittest.main()
