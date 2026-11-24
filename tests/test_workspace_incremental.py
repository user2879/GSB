from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from columnar_diag.models import BudgetConfig
from columnar_diag.predicate import evaluate_node, parse_where
from columnar_diag.sample import build_sample_spec, sample_budget, sample_changes
from columnar_diag.verifier import authoritative_replay
from columnar_diag.workspace import Workspace


class IncrementalUpdateTests(unittest.TestCase):
    def test_only_direct_block_reencoded_and_only_affected_tasks_rescanned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace.create(
                Path(directory),
                build_sample_spec(),
                budget=BudgetConfig(**sample_budget()),
                overwrite=True,
            )
            try:
                workspace.run_full_scan(batch_size=5, delay_ms=1.0)
                expected_by_query: dict[str, list[list[int]]] = {}
                for query in workspace.spec.queries:
                    ast = parse_where(query.where, query_id=query.id)
                    rows: list[list[int]] = []
                    for group in workspace.spec.row_groups:
                        for index in range(group.row_count):
                            row = {
                                block.column: block.values[index]
                                for block in workspace.spec.blocks
                                if block.row_group == group.id
                            }
                            if evaluate_node(row, ast):
                                rows.append([row[column] for column in query.select])
                    expected_by_query[query.id] = rows
                actual_by_query = {query_id: [] for query_id in expected_by_query}
                for result in workspace.ledger.current_query_results():
                    actual_by_query[result["query_id"]].extend(result["rows"])
                self.assertEqual(actual_by_query, expected_by_query)
                before_blocks = {
                    block_id: encoded.record_bytes
                    for block_id, encoded in workspace.image.blocks.items()
                }
                before_tasks = {
                    row["task_id"]: (
                        row["result_digest"],
                        row["result_json"],
                        row["block_digest"],
                    )
                    for row in workspace.ledger.current_task_rows()
                }
                result = workspace.apply_changeset(
                    sample_changes()["changes"], delay_ms=1.0
                )
                self.assertEqual(result["invalid_blocks"], ["amount@rg1"])
                self.assertEqual(result["recomputed_encoding_blocks"], ["amount@rg1"])
                self.assertTrue(result["all_preserved_blocks_byte_identical"])
                self.assertTrue(result["all_preserved_tasks_identical"])
                self.assertEqual(
                    result["affected_queries"], ["q1", "q2", "q3", "q5"]
                )
                after_blocks = {
                    block_id: encoded.record_bytes
                    for block_id, encoded in workspace.image.blocks.items()
                }
                changed_blocks = [
                    block_id
                    for block_id in before_blocks
                    if before_blocks[block_id] != after_blocks[block_id]
                ]
                self.assertEqual(changed_blocks, ["amount@rg1"])
                after_tasks = {
                    row["task_id"]: (
                        row["result_digest"],
                        row["result_json"],
                        row["block_digest"],
                    )
                    for row in workspace.ledger.current_task_rows()
                }
                for task_id, value in before_tasks.items():
                    if task_id not in result["affected_tasks"]:
                        self.assertEqual(after_tasks[task_id], value)
                replay = authoritative_replay(workspace)
                self.assertTrue(replay["ok"])
                self.assertEqual(workspace.versions["encoding_selection_version"], 2)
                generations = workspace.ledger.generation_rows()
                self.assertEqual(sum(int(row["is_current"]) for row in generations), 1)
                self.assertTrue(
                    all(int(row["is_current"]) == 0 for row in generations[:-1])
                )
            finally:
                workspace.close()
            reopened = Workspace(Path(directory))
            try:
                self.assertEqual(reopened.versions["encoding_selection_version"], 2)
                self.assertTrue(authoritative_replay(reopened)["ok"])
            finally:
                reopened.close()

    def test_value_only_update_persists_segment_digest_and_reopens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace.create(
                Path(directory),
                build_sample_spec(),
                budget=BudgetConfig(**sample_budget()),
                overwrite=True,
            )
            try:
                workspace.run_full_scan(batch_size=7)
                old_version = workspace.versions["encoding_selection_version"]
                result = workspace.apply_changeset(
                    [
                        {
                            "column": "qty",
                            "row_group": "rg0",
                            "values": [7, 7, 2, 2, 2, 4, 4, 1],
                        }
                    ]
                )
                self.assertFalse(result["encoding_version_bumped"])
                self.assertEqual(
                    workspace.versions["encoding_selection_version"], old_version
                )
                generation = workspace.ledger.current_generation()
                self.assertEqual(
                    generation["segment_digest"], workspace.image.file_digest
                )
                self.assertTrue(authoritative_replay(workspace)["ok"])
            finally:
                workspace.close()
            reopened = Workspace(Path(directory))
            try:
                self.assertTrue(authoritative_replay(reopened)["ok"])
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
