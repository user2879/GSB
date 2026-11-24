from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gpu_scheduler.errors import DependencyCycleError, SpecError
from gpu_scheduler.executor import execute_batch
from gpu_scheduler.graph import DependencyGraph
from gpu_scheduler.quant import fixed_metric
from gpu_scheduler.scheduler import SchedulerEngine
from gpu_scheduler.snapshot import compare_outcomes
from gpu_scheduler.spec import load_spec

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


class SpecAndQuantTests(unittest.TestCase):
    def test_first_invalid_field_is_located(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            raw = json.loads((SAMPLES / "platform.json").read_text(encoding="utf-8"))
            raw["pools"][0]["host"] = "localhost"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(SpecError) as ctx:
                load_spec(path)
            self.assertEqual(ctx.exception.code, "E_SPEC")
            self.assertEqual(ctx.exception.locator["entity_id"], "alpha")
            self.assertEqual(ctx.exception.locator["field"], "host")

    def test_cycle_uses_e_cycle(self) -> None:
        with self.assertRaises(DependencyCycleError) as ctx:
            load_spec(SAMPLES / "cycle_injection.json")
        self.assertEqual(ctx.exception.code, "E_CYCLE")
        self.assertIn("cyclic#0", ctx.exception.locator["cycle"])

    def test_fixed_metric_truncates_down(self) -> None:
        metric = fixed_metric(1, 3, 3)
        self.assertEqual(metric["ratio_fixed"], 333)
        self.assertEqual(metric["remaining"], 2)
        self.assertEqual(metric["quant_error_bound"]["fixed_units"], 1)


class GraphTests(unittest.TestCase):
    def test_propagation_order_is_stable(self) -> None:
        spec = load_spec(SAMPLES / "propagation.json")
        graph = DependencyGraph.build(spec)
        affected = graph.descendants({"etl#0"})
        self.assertEqual(sorted(affected), ["etl#0", "etl#1", "etl#2", "mart#0"])
        order = graph.stable_topological_order(affected)
        self.assertEqual(order, ["etl#0", "etl#1", "etl#2", "mart#0"])


class ExecutorBatchTests(unittest.TestCase):
    def test_batch_returns_deterministic_digests(self) -> None:
        batch = {
            "batch_id": "unit-batch",
            "shards": [
                {"shard_id": "x#0", "seed": 7, "compute_ms": 0, "gpu_count": 1, "memory_mib": 1, "integer_units": 2},
                {"shard_id": "x#1", "seed": 9, "compute_ms": 0, "gpu_count": 1, "memory_mib": 1, "integer_units": 2},
            ],
        }
        first = execute_batch(batch)
        second = execute_batch(batch)
        self.assertEqual([r["status"] for r in first["results"]], ["success", "success"])
        self.assertEqual(
            [r["outcome_digest"] for r in first["results"]],
            [r["outcome_digest"] for r in second["results"]],
        )
        self.assertEqual(first["results"][0]["compute"]["checksum"], second["results"][0]["compute"]["checksum"])
        self.assertEqual(first["results"][0]["caliber"]["name"], "numpy-int64-nonnegative-floor-dot")


class SnapshotCompareTests(unittest.TestCase):
    def test_first_outcome_difference_locates_field(self) -> None:
        old = [{"shard_id": "a#0", "outcome_status": "completed", "receipt_digest": "r1", "outcome_digest": "o1", "resource_digest": "g1", "checksum": 1}]
        new = [{"shard_id": "a#0", "outcome_status": "failed", "receipt_digest": "r2", "outcome_digest": "o2", "resource_digest": "g2", "checksum": 2}]
        result = compare_outcomes(old, new)
        self.assertFalse(result["match"])
        self.assertEqual(result["first_difference"]["shard_id"], "a#0")
        self.assertEqual(result["first_difference"]["field"], "outcome_status")


class VictimStabilityTests(unittest.TestCase):
    def test_victim_choice_independent_of_pool_order(self) -> None:
        spec = load_spec(SAMPLES / "platform.json")
        with tempfile.TemporaryDirectory() as tmp:
            engine = SchedulerEngine(spec, tmp, global_gpu_quota=8)
            try:
                engine.active["alpha-old#0"] = {
                    "task_id": "alpha-old",
                    "pool_id": "alpha",
                    "gpu_count": 4,
                    "memory_mib": 8192,
                }
                first = engine._choose_victims("alpha-parallel#0")
                self.assertEqual(first, ["alpha-old"])
                reversed_pools = tuple(reversed(spec.pools))
                object.__setattr__(engine.spec, "pools", reversed_pools)
                second = engine._choose_victims("alpha-parallel#0")
                self.assertEqual(second, ["alpha-old"])
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
