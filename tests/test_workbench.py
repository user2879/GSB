from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mocap_workbench.constraints import FootContact, solve_constraints
from mocap_workbench.consistency import compare_pose_results
from mocap_workbench.demo import _error_checks
from mocap_workbench.errors import BusinessError, ErrorCode
from mocap_workbench.fusion import fuse_observations
from mocap_workbench.generator import (
    GenerationConfig,
    generate_scene,
    load_scene,
    save_scene,
    scene_file_hashes,
)
from mocap_workbench.incremental import IncrementalWorkbench
from mocap_workbench.pipeline import run_all_strategies, solve_scene
from mocap_workbench.retarget import inverse_retarget_pose, retarget_pose
from mocap_workbench.serialization import canonical_json_bytes
from mocap_workbench.strategies import BudgetConfig, select_strategy


class GeneratorTests(unittest.TestCase):
    def test_generation_and_files_are_deterministic(self) -> None:
        config = GenerationConfig(frames=30, seed=12345)
        first = generate_scene(config)
        second = generate_scene(config)
        self.assertEqual(first.digest(), second.digest())
        self.assertTrue(np.array_equal(first.visibility, second.visibility))
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            save_scene(first, left)
            save_scene(second, right)
            self.assertEqual(scene_file_hashes(left), scene_file_hashes(right))
            loaded = load_scene(left)
            self.assertEqual(loaded.digest(), first.digest())

    def test_visibility_manifest_encodes_every_frame_and_joint(self) -> None:
        scene = generate_scene(GenerationConfig(frames=7))
        bits = scene.visibility_bits()
        self.assertEqual(bits.shape, (7, 17))
        for frame in range(7):
            for joint in range(17):
                expected = sum(
                    int(scene.visibility[frame, joint, camera]) << camera for camera in range(3)
                )
                self.assertEqual(int(bits[frame, joint]), expected)


class FusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = generate_scene(GenerationConfig(frames=8))

    def test_three_camera_subsets_are_used_exactly(self) -> None:
        frame = 2
        observations = self.scene.observations[frame : frame + 1].copy()
        observations[0, 0] = np.asarray(
            ((2.0, 3.0, 4.0), (3.0, 4.0, 5.0), (4.0, 5.0, 6.0)), dtype=np.float64
        )
        sigma = np.full((1, 17, 3), 0.01, dtype=np.float64)
        truth = self.scene.truth[frame : frame + 1, 0]
        variants = []
        for cameras in ((0,), (0, 1), (0, 1, 2)):
            visibility = np.zeros((1, 17, 3), dtype=bool)
            visibility[0, 0, list(cameras)] = True
            observations[0, 0, 2] = 999.0
            if 2 not in cameras:
                observations[0, 0, 2] = 999.0
            result = fuse_observations(self.scene.source, observations, visibility, sigma)
            expected = np.mean([observations[0, 0, camera] for camera in cameras], axis=0)
            np.testing.assert_allclose(result.pose[0, 0], expected, atol=1e-12)
            variants.append(result.pose[0, 0].copy())
        self.assertGreater(np.linalg.norm(variants[0] - variants[1]), 1e-4)
        self.assertGreater(np.linalg.norm(variants[1] - variants[2]), 1e-4)

    def test_hidden_observations_are_never_used(self) -> None:
        observations = np.broadcast_to(
            self.scene.truth[0:1, :, None, :], (1, 17, 3, 3)
        ).copy()
        visibility = np.zeros((1, 17, 3), dtype=bool)
        visibility[0, :, 0] = True
        sigma = np.full((1, 17, 3), 0.01, dtype=np.float64)
        observations[0, :, 1:] = np.nan
        first = fuse_observations(self.scene.source, observations, visibility, sigma)
        observations[0, :, 1:] = 1e9
        second = fuse_observations(self.scene.source, observations, visibility, sigma)
        np.testing.assert_array_equal(first.pose, second.pose)

    def test_no_visible_observation_returns_business_code(self) -> None:
        hidden = np.zeros_like(self.scene.visibility)
        with self.assertRaises(BusinessError) as context:
            fuse_observations(
                self.scene.source,
                self.scene.observations,
                self.scene.visibility,
                self.scene.sigma,
                visibility_override=hidden,
            )
        self.assertEqual(context.exception.code, ErrorCode.E_OBS_MISSING)

    def test_missing_joint_completion_respects_source_bone_lengths(self) -> None:
        visibility = self.scene.visibility[0:1].copy()
        visibility[0, 6, :] = False
        result = fuse_observations(
            self.scene.source, self.scene.observations[0:1], visibility, self.scene.sigma[0:1]
        )
        self.assertTrue(result.completed[0, 6])
        self.assertLess(result.bone_length_error[0, 6], 1e-10)

    def test_isolated_occlusion_completion_remains_temporally_continuous(self) -> None:
        visibility = self.scene.visibility.copy()
        observations = self.scene.observations.copy()
        frame = 4
        joint = self.scene.source.index["r_ankle"]
        visibility[frame, joint, :] = False
        observations[frame, joint, :, :] = 1e9
        occluded = fuse_observations(self.scene.source, observations, visibility, self.scene.sigma)
        baseline = fuse_observations(
            self.scene.source, self.scene.observations, self.scene.visibility, self.scene.sigma
        )
        self.assertTrue(occluded.completed[frame, joint])
        self.assertLess(float(np.linalg.norm(occluded.pose[frame, joint] - baseline.pose[frame, joint])), 1e-2)
        self.assertLess(
            float(np.linalg.norm(occluded.pose[frame, joint] - occluded.pose[frame - 1, joint])),
            5e-2,
        )
        self.assertLess(
            float(np.linalg.norm(occluded.pose[frame, joint] - occluded.pose[frame + 1, joint])),
            5e-2,
        )
        self.assertLess(occluded.bone_length_error[frame, joint], 1e-10)

        observations[frame, joint, :, :] = -1e9
        mutated = fuse_observations(self.scene.source, observations, visibility, self.scene.sigma)
        np.testing.assert_array_equal(occluded.pose, mutated.pose)


class RetargetTests(unittest.TestCase):
    def test_retarget_roundtrip_and_target_bone_lengths(self) -> None:
        scene = generate_scene(GenerationConfig(frames=20))
        fused = fuse_observations(scene.source, scene.observations, scene.visibility, scene.sigma)
        result = retarget_pose(fused.pose, scene.source, scene.target)
        self.assertLess(float(np.max(result.roundtrip_residual)), 1e-9)
        self.assertLess(float(np.max(result.target_bone_length_error)), 1e-9)
        restored = inverse_retarget_pose(result.pose, scene.source, scene.target, result.mapping)
        np.testing.assert_allclose(restored, fused.pose, atol=1e-10)

    def test_rest_pose_maps_to_target_rest_pose(self) -> None:
        scene = generate_scene(GenerationConfig(frames=1))
        result = retarget_pose(scene.source.rest_world[None], scene.source, scene.target)
        np.testing.assert_allclose(result.pose[0], scene.target.rest_world, atol=1e-12)


class ConstraintTests(unittest.TestCase):
    def test_constraints_are_solved_without_relaxation(self) -> None:
        scene = generate_scene(GenerationConfig(frames=90))
        result = solve_scene(scene, strategy="full_segment")
        residuals = result.constraint_result["residuals"]
        self.assertLess(float(np.max(residuals)), 1e-3)
        self.assertLess(result.constraint_result["max_bone_length_error_m"], 1e-4)

    def test_unreachable_contact_returns_business_code(self) -> None:
        scene = generate_scene(GenerationConfig(frames=20))
        result = solve_scene(scene, strategy="full_segment")
        constraints = scene.constraints
        constraints.foot_contacts = [
            FootContact("l_ankle", (100.0, 100.0, 100.0), period=1, phase=0, duration=20),
            FootContact("r_ankle", (100.0, 100.0, 100.0), period=1, phase=0, duration=20),
        ]
        with self.assertRaises(BusinessError) as context:
            solve_constraints(result.strategy_pose, scene.target, constraints)
        self.assertEqual(context.exception.code, ErrorCode.E_CONSTRAINT_UNSAT)
        self.assertIn("conflict_constraints", context.exception.details)


class StrategyTests(unittest.TestCase):
    def test_budget_selects_expected_tiers(self) -> None:
        budget = BudgetConfig()
        self.assertEqual(select_strategy(300, 17, budget).strategy, "full_segment")
        self.assertEqual(select_strategy(3000, 17, budget).strategy, "windowed")
        self.assertEqual(select_strategy(30000, 17, budget).strategy, "sparse_anchor")

    def test_impossible_budget_returns_business_code(self) -> None:
        with self.assertRaises(BusinessError) as context:
            select_strategy(30000, 17, BudgetConfig(memory_mb=1e-6, time_budget_ms=1e-6))
        self.assertEqual(context.exception.code, ErrorCode.E_STRATEGY_BUDGET)

    def test_strategy_consistency_on_small_sequence(self) -> None:
        scene = generate_scene(GenerationConfig(frames=120))
        results, comparison = run_all_strategies(scene)
        self.assertEqual(len(results), 3)
        self.assertTrue(comparison["consistent"])
        for result in results.values():
            self.assertTrue(result.strategy_metadata["boundary_continuity_ok"])
            execution = result.report()["strategy_execution"]
            self.assertIn("boundary_root_seam_m", execution)
            self.assertIn("boundary_continuity_ok", execution)
        for pair in comparison["pairs"]:
            self.assertIsNone(pair["first_divergence_frame"])

    def test_consistency_locates_first_divergence_and_joint(self) -> None:
        left_pose = np.zeros((4, 3, 3), dtype=np.float64)
        right_pose = left_pose.copy()
        right_pose[2, 1, 0] = 0.2
        payload = {
            "left": {
                "pose": left_pose,
                "constraint_residuals": np.zeros((4, 1)),
                "constraint_residual_names": ["root_height"],
                "boundary_frames": [2],
                "joint_names": ["hips", "spine", "chest"],
            },
            "right": {
                "pose": right_pose,
                "constraint_residuals": np.zeros((4, 1)),
                "constraint_residual_names": ["root_height"],
                "boundary_frames": [2],
                "joint_names": ["hips", "spine", "chest"],
            },
        }
        report = compare_pose_results(payload, position_tolerance_m=0.01)
        pair = report["pairs"][0]
        self.assertFalse(report["consistent"])
        self.assertEqual(pair["first_divergence_frame"], 2)
        self.assertEqual(pair["first_divergence_joints"], ["spine"])
        self.assertEqual(pair["divergence_location"]["kind"], "window_or_anchor_boundary_region")


class IncrementalTests(unittest.TestCase):
    def test_closure_is_exact_and_outside_frames_are_reused(self) -> None:
        scene = generate_scene(GenerationConfig(frames=60))
        workbench = IncrementalWorkbench(scene)
        constraint_report = workbench.edit_frame_constraint(21, root_height=(1.07, 1.08))
        self.assertEqual(
            constraint_report["closure_counts"],
            {"fusion": 0, "retarget": 0, "strategy": 0, "constraints": 1},
        )
        self.assertEqual(constraint_report["reuse_counts"]["constraints"], 59)
        self.assertTrue(constraint_report["closure_equals_actual"])

        joint = "l_elbow"
        camera = int(np.flatnonzero(scene.visibility[30, scene.source.index[joint]])[0])
        weight_report = workbench.edit_camera_weight(30, joint, camera, 1.5)
        self.assertEqual(weight_report["closure_counts"]["fusion"], 1)
        self.assertEqual(weight_report["closure_counts"]["retarget"], 1)
        self.assertGreater(weight_report["closure_counts"]["strategy"], 0)
        self.assertTrue(weight_report["closure_equals_actual"])
        self.assertEqual(weight_report["reuse_counts"]["fusion"], 59)

        old_length = float(scene.target.bone_lengths[scene.target.index["l_knee"]])
        bone_report = workbench.edit_target_bone_length("l_knee", old_length * 1.001)
        self.assertEqual(bone_report["closure_counts"]["fusion"], 0)
        self.assertEqual(bone_report["reuse_counts"]["fusion"], 60)
        self.assertEqual(bone_report["closure_counts"]["retarget"], 60)
        self.assertTrue(bone_report["closure_equals_actual"])
        self.assertGreater(int(workbench.versions["retarget"].min()), 0)

    def test_closure_mismatch_returns_business_code(self) -> None:
        scene = generate_scene(GenerationConfig(frames=20))
        workbench = IncrementalWorkbench(scene)
        with self.assertRaises(BusinessError) as context:
            workbench._verify_closure({("fusion", 0)}, set())
        self.assertEqual(context.exception.code, ErrorCode.E_CLOSURE_MISMATCH)


class ErrorAndSerializationTests(unittest.TestCase):
    def test_all_declared_error_paths_are_exercised(self) -> None:
        checks = _error_checks(20261119)
        for code in (
            ErrorCode.E_OBS_MISSING,
            ErrorCode.E_RETARGET_TOL,
            ErrorCode.E_CONSTRAINT_UNSAT,
            ErrorCode.E_STRATEGY_BUDGET,
            ErrorCode.E_CLOSURE_MISMATCH,
        ):
            self.assertTrue(checks[code.value]["raised"], code)

    def test_canonical_json_is_stable_and_key_sorted(self) -> None:
        left = canonical_json_bytes({"b": 2.0, "a": [1, np.float64(2.5)]})
        right = canonical_json_bytes({"a": [1, 2.5], "b": 2.0})
        self.assertEqual(left, right)
        self.assertTrue(left.startswith(b'{"a"'))


if __name__ == "__main__":
    unittest.main()
