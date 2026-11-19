from __future__ import annotations

import unittest

from vecflow.errors import AppError
from vecflow.model import parse_data_payload, parse_tasks_payload, workspace_state_from_parts
from vecflow.planner import compile_workspace
from tests.helpers import data_payload, tasks_payload


def _state(shards: list[dict]):
    return workspace_state_from_parts(parse_data_payload(data_payload()), parse_tasks_payload(tasks_payload(shards=shards)))


class PlannerTests(unittest.TestCase):
    def test_plan_is_contiguous_and_deterministic(self) -> None:
        state = _state(
            [
                {"start": 6, "end": 8, "resources": 1},
                {"start": 0, "end": 3, "resources": 1},
                {"start": 3, "end": 6, "resources": 2},
            ]
        )
        first = compile_workspace(state)
        second = compile_workspace(state)
        self.assertEqual(first.plan_digest, second.plan_digest)
        shards = first.tasks[0].shards
        self.assertEqual([(shard.start_row, shard.end_row) for shard in shards], [(0, 3), (3, 6), (6, 8)])
        self.assertEqual([shard.shard_id for shard in shards], [shard.shard_id for shard in second.tasks[0].shards])

    def test_out_of_range_shard_is_e_range(self) -> None:
        with self.assertRaises(AppError) as raised:
            compile_workspace(_state([{"start": 0, "end": 9, "resources": 1}]))
        self.assertEqual(raised.exception.code, "E_RANGE")
        self.assertEqual(raised.exception.details["end"], 9)

    def test_negative_start_is_e_range(self) -> None:
        with self.assertRaises(AppError) as raised:
            compile_workspace(_state([{"start": -1, "end": 8, "resources": 1}]))
        self.assertEqual(raised.exception.code, "E_RANGE")

    def test_gap_and_overlap_are_e_range(self) -> None:
        for shards in (
            [{"start": 0, "end": 3, "resources": 1}, {"start": 4, "end": 8, "resources": 1}],
            [{"start": 0, "end": 4, "resources": 1}, {"start": 3, "end": 8, "resources": 1}],
        ):
            with self.subTest(shards=shards), self.assertRaises(AppError) as raised:
                compile_workspace(_state(shards))
            self.assertEqual(raised.exception.code, "E_RANGE")

    def test_generated_shards_cover_all_rows(self) -> None:
        payload = tasks_payload()
        task = payload["tasks"][0]
        task.pop("shards")
        task["shard_rows"] = 3
        task["resources_per_shard"] = [1, 2, 1]
        state = workspace_state_from_parts(parse_data_payload(data_payload()), parse_tasks_payload(payload))
        plan = compile_workspace(state)
        self.assertEqual([(shard.start_row, shard.end_row) for shard in plan.tasks[0].shards], [(0, 3), (3, 6), (6, 8)])


if __name__ == "__main__":
    unittest.main()
