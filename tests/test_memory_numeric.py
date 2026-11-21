from __future__ import annotations

import unittest
import copy

import numpy as np

from memdiag.engine import MemoryBlockPool, rank_attribution, run_initial
from memdiag.errors import OOMError
from memdiag.models import SampleSet
from memdiag.plan import plan_from_dict, with_batch_size
from memdiag.quant import (
    INT32_MAX,
    INT32_MIN,
    fixed_point_layer_forward,
    float32_layer_forward,
    quantize,
)
from tests.helpers import tiny_plan_dict, tiny_samples


class MemoryPoolTests(unittest.TestCase):
    def test_contiguous_then_single_block_fallback(self):
        pool = MemoryBlockPool(6, 1024)
        self.assertEqual(pool.allocate("a", 2).blocks, (0, 1))
        self.assertEqual(pool.allocate("b", 2).blocks, (2, 3))
        self.assertEqual(pool.allocate("c", 1).mode, "contiguous")
        pool.free("a")
        pool.free("c")
        allocation = pool.allocate("d", 3)
        self.assertEqual(allocation.blocks, (0, 1, 4))
        self.assertEqual(allocation.mode, "single_block")
        self.assertEqual(pool.fragmentation_blocks, 0)
        self.assertEqual(pool.occupied_blocks, 5)

    def test_oom_reports_needed_and_free(self):
        pool = MemoryBlockPool(4, 1024)
        pool.allocate("a", 3)
        with self.assertRaises(OOMError) as caught:
            pool.allocate("b", 2)
        self.assertEqual(caught.exception.code, "E_OOM")
        self.assertEqual(caught.exception.details["needed_blocks"], 2)
        self.assertEqual(caught.exception.details["free_blocks"], 1)

    def test_contiguous_request_rejects_fragmented_pool(self):
        pool = MemoryBlockPool(6, 1024)
        pool.allocate("a", 2)
        pool.allocate("b", 2)
        pool.allocate("c", 1)
        pool.free("a")
        pool.free("c")
        with self.assertRaises(OOMError) as caught:
            pool.allocate("d", 3, strategy="contiguous")
        self.assertEqual(caught.exception.details["largest_contiguous_free_blocks"], 2)

    def test_engine_oom_after_budget_check(self):
        document = tiny_plan_dict(budget_mib=1)
        document["layers"][0]["param_count"] = 2_000_000
        plan = plan_from_dict(document)
        with self.assertRaises(OOMError) as caught:
            run_initial(plan, tiny_samples())
        self.assertEqual(caught.exception.code, "E_OOM")
        self.assertGreater(caught.exception.details["needed_blocks"], 0)


class NumericTests(unittest.TestCase):
    def test_quantize_ties_to_even_and_saturates(self):
        values = np.array([0.25, 0.75, -0.25, -0.75], dtype=np.float32)
        self.assertEqual(quantize(values, 1).tolist(), [0, 2, 0, -2])
        saturated = quantize(np.array([1e20, -1e20], dtype=np.float64), 6)
        self.assertEqual(saturated.tolist(), [INT32_MAX, INT32_MIN])

    def test_fixed_path_is_explicit_and_close_to_float(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0, 0.2, 8).astype(np.float32)
        w = rng.normal(0, 0.2, (8, 6)).astype(np.float32)
        b = rng.normal(0, 0.02, 6).astype(np.float32)
        reference = float32_layer_forward(x, w, b)
        fixed, details = fixed_point_layer_forward(x, w, b, 5)
        sequential, _ = fixed_point_layer_forward(x, w, b, 5, accumulation_order="sequential")
        self.assertEqual(fixed.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(fixed - reference))), 0.2)
        self.assertLessEqual(float(np.max(np.abs(fixed - sequential))), 0.05)
        self.assertEqual(details["q_inputs"].dtype, np.int32)
        self.assertEqual(details["accumulator"].dtype, np.int64)

    def test_batch_schedule_does_not_change_sample_outputs(self):
        plan = plan_from_dict(tiny_plan_dict())
        samples = tiny_samples(7)
        baseline = run_initial(plan, samples)
        reordered_samples = SampleSet(
            format_version=samples.format_version,
            seed=samples.seed,
            dim=samples.dim,
            samples=tuple(reversed(samples.samples)),
        )
        reordered = run_initial(plan, reordered_samples)
        alternate_batch = run_initial(with_batch_size(plan, 3), samples)
        for layer_id in baseline.layers:
            self.assertEqual(
                baseline.layers[layer_id].sample_output_digests,
                reordered.layers[layer_id].sample_output_digests,
            )
            self.assertEqual(
                baseline.layers[layer_id].sample_output_digests,
                alternate_batch.layers[layer_id].sample_output_digests,
            )
        self.assertEqual(
            baseline.attribution["rankings"],
            reordered.attribution["rankings"],
        )
        self.assertEqual(
            baseline.attribution["rankings"],
            alternate_batch.attribution["rankings"],
        )

    def test_rerun_is_byte_deterministic(self):
        plan = plan_from_dict(tiny_plan_dict())
        samples = tiny_samples()
        first = run_initial(plan, samples)
        second = run_initial(plan, samples)
        self.assertEqual(first.content_digest, second.content_digest)
        self.assertEqual(
            {key: value.output_digest for key, value in first.layers.items()},
            {key: value.output_digest for key, value in second.layers.items()},
        )

    def test_each_batch_has_actual_and_float_summaries(self):
        plan = plan_from_dict(tiny_plan_dict())
        run = run_initial(plan, tiny_samples())
        self.assertTrue(run.batches)
        self.assertEqual(set(run.batches[0].layer_output_digests), set(run.layers))
        self.assertEqual(set(run.batches[0].layer_float_digests), set(run.layers))
        for layer_id in run.layers:
            self.assertEqual(
                set(run.layers[layer_id].sample_float_digests),
                set(run.samples.sorted_ids),
            )

    def test_equal_error_bounds_use_stable_layer_id_tie_break(self):
        plan = plan_from_dict(tiny_plan_dict())
        run = run_initial(plan, tiny_samples())
        first = copy.deepcopy(run.layers["b"])
        second = copy.deepcopy(run.layers["c"])
        first.layer_id = "a_same"
        second.layer_id = "z_same"
        first.max_abs_error = second.max_abs_error = 0.5
        first.error_bound = second.error_bound = 1.0
        ranking = rank_attribution(
            {"a_same": first, "z_same": second}, ("a_same", "z_same")
        )
        self.assertEqual([item["layer_id"] for item in ranking["rankings"]], ["a_same", "z_same"])
        self.assertTrue(ranking["rankings"][0]["tie_affected"])
        self.assertTrue(ranking["top_decision_tie_affected"])


if __name__ == "__main__":
    unittest.main()
