from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from mesh_audit.concurrency import BoundedOrderedEvaluator
from mesh_audit.errors import BusinessError, ErrorCode
from tests.helpers import make_mesh, make_script, run_audit


class ConcurrencyTests(unittest.TestCase):
    def test_worker_count_does_not_change_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_script = make_script({"concurrency": 1, "memory_budget_nodes": 4096})
            second_script = make_script({"concurrency": 8, "memory_budget_nodes": 4096})
            first, _ = run_audit(
                root,
                script=first_script,
                state_name="state-c1",
                output_name="output-c1",
            )
            second, _ = run_audit(
                root,
                script=second_script,
                state_name="state-c8",
                output_name="output-c8",
            )
            for left, right in zip(first.report["steps"], second.report["steps"]):
                self.assertEqual(left["execution_count"], right["execution_count"])
                self.assertEqual(left["operator_effect"], right["operator_effect"])
                self.assertEqual(left["error"]["vertex_errors"], right["error"]["vertex_errors"])
                self.assertEqual(
                    left["propagation"]["source_vertices"],
                    right["propagation"]["source_vertices"],
                )
                self.assertEqual(
                    left["propagation"]["decay_counts"],
                    right["propagation"]["decay_counts"],
                )
                for l_trace, r_trace in zip(
                    self._payload(first, left["step"])["propagation_sources"],
                    self._payload(second, right["step"])["propagation_sources"],
                ):
                    self.assertEqual(l_trace["layers"], r_trace["layers"])
                    self.assertEqual(
                        l_trace["affected_vertices"], r_trace["affected_vertices"]
                    )
                for key in (
                    "vertex_count",
                    "face_count",
                    "edge_count",
                    "nonmanifold_edge_count",
                    "boundary_edge_count",
                    "boundary_loop_count",
                    "degenerate_face_count",
                    "isolated_vertex_count",
                    "orientation_conflict_edge_count",
                    "euler_characteristic",
                    "rescan_matches",
                ):
                    self.assertEqual(
                        left["diagnostics"][key], right["diagnostics"][key]
                    )
            self.assertEqual(
                first.report["summary"]["fixed_point_verdict"],
                second.report["summary"]["fixed_point_verdict"],
            )
            second_evaluation = second.report["steps"][4]["propagation"]["evaluation"]
            self.assertEqual(second_evaluation["concurrency"], 8)
            self.assertLessEqual(
                second_evaluation["peak_inflight_nodes"],
                second_evaluation["budget_capacity_nodes"],
            )

    def test_tight_budget_waits_without_changing_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tight, _ = run_audit(
                root,
                script=make_script({"concurrency": 4, "memory_budget_nodes": 20}),
                state_name="state-tight",
                output_name="output-tight",
            )
            normal, _ = run_audit(
                root,
                script=make_script({"concurrency": 4, "memory_budget_nodes": 4096}),
                state_name="state-normal",
                output_name="output-normal",
            )
            evaluation = tight.report["steps"][4]["propagation"]["evaluation"]
            self.assertLessEqual(evaluation["peak_inflight_nodes"], 20)
            self.assertEqual(
                tight.report["steps"][4]["error"]["vertex_errors"],
                normal.report["steps"][4]["error"]["vertex_errors"],
            )
            self.assertEqual(
                tight.report["steps"][4]["propagation"]["decay_counts"],
                normal.report["steps"][4]["propagation"]["decay_counts"],
            )

    def test_budget_too_small_for_one_trace_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            from mesh_audit.engine import RunEngine

            engine = RunEngine(
                make_mesh(),
                make_script({"concurrency": 2, "memory_budget_nodes": 8}),
                root / "state-budget",
                root / "output-budget",
            )
            with self.assertRaises(BusinessError) as caught:
                engine.run()
            self.assertEqual(caught.exception.code, ErrorCode.E_BUDGET)
            self.assertFalse((root / "output-budget" / "report.json").exists())

    def test_budget_admission_waits_and_merges_by_index(self) -> None:
        evaluator = BoundedOrderedEvaluator[int](concurrency=4, memory_budget_nodes=6)

        def task(value: int):
            def execute() -> int:
                time.sleep(0.02)
                return value * 10

            return execute

        results = evaluator.map_indexed(
            [(3, task(3), 6), (1, task(1), 6), (2, task(2), 6)]
        )
        self.assertEqual(results, [10, 20, 30])
        metrics = evaluator.metrics
        self.assertLessEqual(metrics.peak_inflight_nodes, 6)
        self.assertGreater(metrics.waited_tasks, 0)

    def _payload(self, result, step: int) -> dict:
        record = next(
            record for record in result.state_inspection.snapshots if record.step == step
        )
        assert record.payload is not None
        return record.payload


if __name__ == "__main__":
    unittest.main()
