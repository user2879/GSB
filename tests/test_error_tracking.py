from __future__ import annotations

import unittest

import numpy as np

from dp_grad_sim.error_tracking import aggregate_float64, track_tensor_errors
from dp_grad_sim.quantization import compute_common_quant_params, dequantize, quantize


class ErrorTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.arrays = {
            worker: rng.normal(0.0, 0.4, size=(4, 5)).astype(np.float32)
            for worker in range(3)
        }
        self.params8 = compute_common_quant_params(
            "tensor", self.arrays.values(), 8, "asymmetric"
        )
        self.params6 = compute_common_quant_params(
            "tensor", self.arrays.values(), 6, "asymmetric"
        )
        self.q8 = {worker: quantize(array, self.params8) for worker, array in self.arrays.items()}
        self.q6 = {worker: quantize(array, self.params6) for worker, array in self.arrays.items()}

    def _pack_error(self, params, q_by_worker) -> float:
        truth, _ = aggregate_float64(self.arrays, "mean")
        packed, _ = aggregate_float64(
            {worker: dequantize(q, params) for worker, q in q_by_worker.items()}, "mean"
        )
        return float(np.max(np.abs(packed.astype(np.float64) - truth.astype(np.float64))))

    def test_boundary_tensor_detects_different_lower_width_judgement(self) -> None:
        error8 = self._pack_error(self.params8, self.q8)
        error6 = self._pack_error(self.params6, self.q6)
        self.assertLess(error8, error6)
        tolerance = (error8 + error6) / 2.0
        record = track_tensor_errors(
            window=2,
            worker_id=0,
            tensor_name="tensor",
            originals_by_origin=self.arrays,
            source_q_by_origin=self.q8,
            received_q_by_origin=self.q8,
            params=self.params8,
            mode="mean",
            tolerance=tolerance,
            lower_params=self.params6,
        )
        self.assertIsNone(record["first_exceeded_stage"])
        boundary = record["boundary"]
        self.assertIsNotNone(boundary)
        self.assertFalse(boundary["current_exceeds"])
        self.assertTrue(boundary["lower_exceeds"])
        self.assertEqual(boundary["current_bits"], 8)
        self.assertEqual(boundary["lower_bits"], 6)
        self.assertIn(boundary["lower_error_source"], {"quantization", "fixed_point_aggregation"})

    def test_unpack_stage_can_be_first_exceeded_stage(self) -> None:
        truth, _ = aggregate_float64(self.arrays, "mean")
        packed, _ = aggregate_float64(
            {worker: dequantize(q, self.params8) for worker, q in self.q8.items()}, "mean"
        )
        received = dict(self.q8)
        received[1] = received[1] + 20
        unpacked, _ = aggregate_float64(
            {worker: dequantize(q, self.params8) for worker, q in received.items()}, "mean"
        )
        pack_error = float(np.max(np.abs(packed.astype(np.float64) - truth.astype(np.float64))))
        unpack_error = float(np.max(np.abs(unpacked.astype(np.float64) - truth.astype(np.float64))))
        self.assertLess(pack_error, unpack_error)
        tolerance = (pack_error + unpack_error) / 2.0
        record = track_tensor_errors(
            window=0,
            worker_id=0,
            tensor_name="tensor",
            originals_by_origin=self.arrays,
            source_q_by_origin=self.q8,
            received_q_by_origin=received,
            params=self.params8,
            mode="mean",
            tolerance=tolerance,
        )
        self.assertEqual(record["first_exceeded_stage"], "unpack")
        self.assertEqual(record["error_source"], "quantization")
        self.assertEqual([stage["stage"] for stage in record["stages"]], ["pack", "unpack", "aggregate"])


if __name__ == "__main__":
    unittest.main()

