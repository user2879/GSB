from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mixedmem.inference import (
    aggregate_layer_digest,
    compute_batch,
    error_statistics,
)
from mixedmem.ledger import Ledger
from mixedmem.quant import (
    Q_MAX,
    Q_MIN,
    fixed_forward_rows,
    float_forward_rows,
    quantize_scalar,
)
from mixedmem.runtime import InferenceRuntime
from mixedmem.schema import parse_plan
from mixedmem.samples import SampleFactory

from .helpers import plan_copy


class InferenceTests(unittest.TestCase):
    def test_rounding_and_saturation_policy(self) -> None:
        self.assertEqual(quantize_scalar(0.25, 2), 1)
        self.assertEqual(quantize_scalar(-0.25, 2), -1)
        self.assertEqual(quantize_scalar(1.0e30, 1024), Q_MAX)
        self.assertEqual(quantize_scalar(-1.0e30, 1024), Q_MIN)

    def test_fixed_path_is_repeatable(self) -> None:
        x = np.asarray([[0.1, -0.25], [0.3, 0.7]], dtype=np.float32)
        w = np.asarray([[0.2, -0.4], [0.5, 0.1]], dtype=np.float32)
        b = np.asarray([0.01, -0.02], dtype=np.float32)
        first = fixed_forward_rows(x, w, b, 128)
        second = fixed_forward_rows(x, w, b, 128)
        self.assertEqual(first.values.tobytes(), second.values.tobytes())
        self.assertEqual(float_forward_rows(x, w, b).dtype, np.float32)

    def test_layer_aggregate_digest_is_batch_size_invariant(self) -> None:
        plan = parse_plan(plan_copy())
        factory = SampleFactory(plan, 8)
        graph = __import__("mixedmem.graph", fromlist=["build_graph"]).build_graph(plan)
        versions = {layer.id: 1 for layer in plan.layers}
        whole = compute_batch(plan, graph, factory, factory.ids[:4], "a", versions)
        pieces = []
        for index, sample_id in enumerate(factory.ids[:4]):
            pieces.extend(
                compute_batch(
                    plan,
                    graph,
                    factory,
                    [sample_id],
                    f"p{index}",
                    versions,
                ).values()
            )
        for layer_id in graph.order:
            whole_digest = aggregate_layer_digest([whole[layer_id]])
            piece_digest = aggregate_layer_digest(
                [item for item in pieces if item.layer_id == layer_id]
            )
            self.assertEqual(whole_digest, piece_digest)

    def test_full_run_batch_size_invariance(self) -> None:
        plan = parse_plan(plan_copy())
        numeric = []
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            for directory, batch_size in ((left_dir, 1), (right_dir, 4)):
                ledger = Ledger(Path(directory) / "ledger.db")
                runtime = InferenceRuntime(plan, 8, batch_size=batch_size, ledger=ledger)
                report = runtime.run(worker_count=2)
                self.assertTrue(report["ok"])
                numeric.append(
                    {
                        layer_id: {
                            "float": value["float_digest"],
                            "quant": value["quant_digest"],
                            "combined": value["combined_digest"],
                            "error": value["errors"]["sum_abs_error"],
                            "bound": value["error_bound"],
                        }
                        for layer_id, value in report["layer_summaries"].items()
                    }
                )
                self.assertEqual(report["schedule"]["invariants"]["ok"], True)
                ledger.close()
        self.assertEqual(numeric[0], numeric[1])

    def test_error_statistics(self) -> None:
        reference = np.asarray([[1.0, -2.0]], dtype=np.float32)
        actual = np.asarray([[1.5, -1.0]], dtype=np.float32)
        stats = error_statistics(reference, actual)
        self.assertAlmostEqual(stats["max_abs_error"], 1.0)
        self.assertAlmostEqual(stats["mean_abs_error"], 0.75)


if __name__ == "__main__":
    unittest.main()
