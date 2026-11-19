from __future__ import annotations

import unittest

from vecflow.engine import BlockTensorBank, aggregate_task, evaluate_shard
from vecflow.model import parse_data_payload, parse_tasks_payload, workspace_state_from_parts
from vecflow.planner import compile_workspace
from tests.helpers import data_payload, tasks_payload


def _run(expression: str, aggregation: str = "concat"):
    state = workspace_state_from_parts(
        parse_data_payload(data_payload()),
        parse_tasks_payload(tasks_payload(expression=expression, aggregation=aggregation)),
    )
    plan = compile_workspace(state)
    bank = BlockTensorBank(state)
    results = [evaluate_shard(plan.tasks[0], shard, bank) for shard in plan.tasks[0].shards]
    return plan.tasks[0], results, aggregate_task(plan.tasks[0], results, 8)


class EngineTests(unittest.TestCase):
    def test_null_and_division_by_zero_propagation(self) -> None:
        task, shard_results, result = _run("flag ? x / d : 0.0")
        payload = result.payload["payload"]
        self.assertEqual(payload["null_mask"], [1, 0, 1, 0, 0, 0, 1, 0])
        self.assertEqual(payload["division_by_zero_mask"], [1, 0, 0, 0, 0, 0, 1, 0])
        self.assertEqual(payload["values"], [None, 0.0, None, 0.0, 2.5, 0.0, None, 0.0])
        self.assertEqual(sum(sum(result.division_by_zero_mask) for result in shard_results), 2)

    def test_concat_merges_by_row_range_not_completion_order(self) -> None:
        task, shard_results, ordered = _run("x + y")
        shuffled = list(reversed(shard_results))
        remerged = aggregate_task(task, shuffled, 8)
        self.assertEqual(ordered.canonical_bytes, remerged.canonical_bytes)
        self.assertEqual(ordered.payload["payload"]["values"], [1, 3, None, 7, 9, 11, 13, 15])

    def test_sum_aggregation_skips_nulls(self) -> None:
        _, _, result = _run("x + y", "sum")
        self.assertEqual(result.payload["payload"]["value"], 59)
        self.assertFalse(result.payload["payload"]["null"])

    def test_ternary_does_not_leak_unselected_branch_hazard(self) -> None:
        _, _, result = _run("flag ? 1 : x / 0.0")
        payload = result.payload["payload"]
        self.assertEqual(payload["null_mask"], [0, 1, 0, 1, 0, 1, 0, 1])
        self.assertEqual(payload["division_by_zero_mask"], [0, 1, 0, 1, 0, 1, 0, 1])

    def test_canonical_task_bytes_are_stable(self) -> None:
        _, _, first = _run("(x + y) * 2", "concat")
        _, _, second = _run("((x+y)*2)", "concat")
        self.assertEqual(first.canonical_bytes, second.canonical_bytes)
        self.assertEqual(first.result_digest, second.result_digest)


if __name__ == "__main__":
    unittest.main()
