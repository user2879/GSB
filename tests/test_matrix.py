from __future__ import annotations

import unittest

import numpy as np

from spectral_diag.config import RunConfig
from spectral_diag.errors import DiagnosticError, ErrorCode
from spectral_diag.matrix import quantize_xyz, solve_adaptation
from spectral_diag.scene import generate_scene


class MatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = generate_scene(RunConfig())

    def test_exact_linear_round_trip(self) -> None:
        known = np.array([[0.9, 0.1, 0.0], [0.0, 0.95, 0.05], [-0.02, 0.02, 0.88]])
        target = self.scene.source_xyz @ known
        solution = solve_adaptation(self.scene.source_xyz, target)
        self.assertTrue(solution.accepted)
        self.assertLess(float(np.max(np.abs(solution.matrix - known))), 1e-10)
        self.assertLess(solution.residual_rmse, 1e-10)

    def test_rank_deficient_is_rejected_without_identity(self) -> None:
        source = self.scene.source_xyz.copy()
        target = self.scene.target_xyz.copy()
        base_source = source[0].copy()
        base_target = target[0].copy()
        for index in range(source.shape[0]):
            source[index] = base_source * (0.5 + index * 0.03)
            target[index] = base_target * (0.5 + index * 0.03)
        solution = solve_adaptation(source, target, cond_threshold=1e4)
        self.assertFalse(solution.accepted)
        self.assertIsNone(solution.matrix)
        self.assertEqual(solution.error_code, str(ErrorCode.MATRIX_SINGULAR))
        self.assertTrue(set(solution.diagnostics.offending_indices) <= set(range(source.shape[0])))
        self.assertTrue(solution.diagnostics.collinear_pairs)

    def test_condition_pruning_is_deterministic(self) -> None:
        source = self.scene.source_xyz.copy()
        target = self.scene.target_xyz.copy()
        source[3] = [1.0e7, 2.0e7, 5.0e6]
        target[3] = [1.1e7, 1.9e7, 4.8e6]
        first = solve_adaptation(source, target, cond_threshold=1e4)
        second = solve_adaptation(source, target, cond_threshold=1e4)
        self.assertTrue(first.accepted)
        self.assertEqual(first.status, "accepted_after_pruning")
        self.assertEqual([step.removed_index for step in first.prune_trace], [3])
        self.assertLessEqual(first.condition_number, 1e4)
        self.assertEqual(first.to_dict()["solution_sha256"], second.to_dict()["solution_sha256"])

    def test_quantization_range_error_points_to_first_observation(self) -> None:
        values = self.scene.source_xyz.copy()
        values[5, 2] = 999.0
        with self.assertRaises(DiagnosticError) as context:
            quantize_xyz(values, quant_range=(0.0, 150.0), quant_bits=8)
        self.assertEqual(context.exception.code, ErrorCode.QUANT_RANGE)
        self.assertEqual(context.exception.details["sample_index"], 5)
        self.assertEqual(context.exception.details["component"], 2)


if __name__ == "__main__":
    unittest.main()
