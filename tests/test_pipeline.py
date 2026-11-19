from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bone_retarget.binary_format import animation_to_binary, parse_binary
from bone_retarget.consistency import Tolerances, compare_animations
from bone_retarget.deterministic import generate_animation, generate_skeleton
from bone_retarget.errors import ErrorCode, PipelineError
from bone_retarget.hashing import animation_semantic_sha256, sha256_json
from bone_retarget.json_format import (
    animation_from_document,
    animation_to_document,
    migrate_json_document,
)
from bone_retarget.math3d import quat_angle_between, quat_from_axis_angle, quat_multiply
from bone_retarget.models import Animation, Skeleton
from bone_retarget.resample import resample_to_fps, sample_animation
from bone_retarget.retarget import retarget_motion
from bone_retarget.verify import run_verification


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = generate_skeleton(11, "source")
        cls.target = generate_skeleton(31, "target")
        cls.animation = generate_animation(cls.source, 12345)

    def test_json_roundtrip_all_schema_versions(self) -> None:
        assert_json_roundtrip(self, self.animation, self.source)

    def test_v1_migration_defaults_and_idempotence(self) -> None:
        document = animation_to_document(self.animation, schema_version=1)
        self.assertNotIn("joint_map", document)
        self.assertNotIn("quat_order", document)
        self.assertNotIn("interpolation", document)
        self.assertTrue(
            all("root_translation" not in frame for frame in document["frames"])
        )
        first = migrate_json_document(document)
        second = migrate_json_document(first.document)
        self.assertEqual(first.output_version, 3)
        self.assertEqual(first.document["quat_order"], "xyzw")
        self.assertEqual(first.document["interpolation"], "slerp")
        self.assertEqual(
            first.document["joint_map"],
            {name: index for index, name in enumerate(self.animation.joint_names)},
        )
        for frame in first.document["keyframes"]:
            self.assertEqual(frame["root_translation"], [0.0, 0.0, 0.0])
        self.assertEqual(first.after_sha256, second.before_sha256)
        self.assertEqual(second.before_sha256, second.after_sha256)
        self.assertTrue(first.document["joints"])
        self.assertEqual(
            first.document["keyframes"][0]["rotations"],
            document["frames"][0]["rotations"],
        )
        parsed = animation_from_document(document)
        self.assertEqual(parsed.joint_count, self.animation.joint_count)
        self.assertEqual(parsed.frame_count, self.animation.frame_count)
        self.assertTrue(np.array_equal(parsed.root_translation, np.zeros_like(parsed.root_translation)))

    def test_migration_errors(self) -> None:
        document = animation_to_document(self.animation)
        document["schema_version"] = 99
        with self.assertRaises(PipelineError) as caught:
            migrate_json_document(document)
        self.assertEqual(caught.exception.code, ErrorCode.E_SCHEMA_VERSION)
        latest_missing = animation_to_document(self.animation)
        del latest_missing["joint_map"]
        with self.assertRaises(PipelineError) as caught:
            animation_from_document(latest_missing)
        self.assertEqual(caught.exception.code, ErrorCode.E_FIELD_MISSING)
        self.assertEqual(caught.exception.field, "joint_map")

    def test_binary_roundtrip_all_endian_width_order_combinations(self) -> None:
        for endian in ("little", "big"):
            for index_width in (2, 4):
                for order in ("wxyz", "xyzw"):
                    with self.subTest(endian=endian, index_width=index_width, order=order):
                        permutation = tuple(range(1, self.source.joint_count)) + (0,)
                        data, metadata = animation_to_binary(
                            self.animation,
                            endian=endian,
                            index_width=index_width,
                            quat_order=order,
                            channel_joint_order=permutation,
                        )
                        parsed = parse_binary(data)
                        self.assertEqual(metadata.endian, endian)
                        self.assertEqual(parsed.metadata.endian, endian)
                        self.assertEqual(parsed.metadata.index_width, index_width)
                        self.assertEqual(parsed.metadata.quat_order, order)
                        self.assertEqual(
                            animation_semantic_sha256(self.animation),
                            animation_semantic_sha256(parsed.animation),
                        )

    def test_binary_business_errors(self) -> None:
        data, _ = animation_to_binary(self.animation, endian="little", index_width=4)
        bad_endian = bytearray(data)
        bad_endian[5] = ord("B")
        with self.assertRaises(PipelineError) as caught:
            parse_binary(bytes(bad_endian))
        self.assertEqual(caught.exception.code, ErrorCode.E_FORMAT_ENDIAN)

        overflow = bytearray(data)
        struct.pack_into("<I", overflow, 24, 0xFFFFFFFF)
        with self.assertRaises(PipelineError) as caught:
            parse_binary(bytes(overflow))
        self.assertEqual(caught.exception.code, ErrorCode.E_INDEX_OVERFLOW)

        with self.assertRaises(PipelineError) as caught:
            parse_binary(data[:-8])
        self.assertEqual(caught.exception.code, ErrorCode.E_QUAT_INVALID)
        self.assertIsNotNone(caught.exception.frame)

        unsupported = bytearray(data)
        unsupported[4] = 99
        with self.assertRaises(PipelineError) as caught:
            parse_binary(bytes(unsupported))
        self.assertEqual(caught.exception.code, ErrorCode.E_SCHEMA_VERSION)

        nan_data = bytearray(data)
        rotation_offset = 24 + self.animation.joint_count * 4
        for name in self.animation.joint_names:
            rotation_offset += 2 + len(name.encode("utf-8"))
        rotation_offset += self.animation.frame_count * 8
        rotation_offset += self.animation.frame_count * 3 * 8
        struct.pack_into("<d", nan_data, rotation_offset, float("nan"))
        with self.assertRaises(PipelineError) as caught:
            parse_binary(bytes(nan_data))
        self.assertEqual(caught.exception.code, ErrorCode.E_QUAT_INVALID)
        self.assertEqual(caught.exception.frame, 0)
        self.assertEqual(caught.exception.joint, self.animation.joint_names[0])

    def test_binary_nonmonotonic_timestamp(self) -> None:
        data, _ = animation_to_binary(
            self.animation, endian="little", index_width=4, quat_order="wxyz"
        )
        bytearray_data = bytearray(data)
        offset = 24 + self.animation.joint_count * 4
        for name in self.animation.joint_names:
            offset += 2 + len(name.encode("utf-8"))
        first_time = struct.unpack_from("<d", bytearray_data, offset)[0]
        struct.pack_into("<d", bytearray_data, offset + 8, first_time)
        with self.assertRaises(PipelineError) as caught:
            parse_binary(bytes(bytearray_data))
        self.assertEqual(caught.exception.code, ErrorCode.E_TIME_NONMONOTONIC)
        self.assertEqual(caught.exception.frame, 1)

    def test_quaternion_sign_independence(self) -> None:
        flipped = self.animation.rotations.copy()
        for joint_index in range(flipped.shape[1]):
            for frame_index in range(flipped.shape[0]):
                if (joint_index + frame_index) % 2 == 0:
                    flipped[frame_index, joint_index] *= -1.0
        alternate = Animation(
            joint_names=self.animation.joint_names,
            times=self.animation.times.copy(),
            rotations=flipped,
            root_translation=self.animation.root_translation.copy(),
            interpolation=self.animation.interpolation,
            fps=self.animation.fps,
        )
        self.assertEqual(
            animation_semantic_sha256(self.animation),
            animation_semantic_sha256(alternate),
        )
        sample_times = np.linspace(
            self.animation.times[0], self.animation.times[-1], 17
        )
        sampled_a = sample_animation(self.animation, sample_times)
        sampled_b = sample_animation(alternate, sample_times)
        maximum = float(
            np.max(quat_angle_between(sampled_a.rotations, sampled_b.rotations))
        )
        self.assertLess(maximum, 1e-12)

    def test_resample_boundaries_and_no_extrapolation(self) -> None:
        resampled, report = resample_to_fps(self.animation, 30.0)
        self.assertEqual(resampled.frame_count, 16)
        self.assertEqual(report.extrapolated, False)
        self.assertGreaterEqual(float(resampled.times[0]), float(self.animation.times[0]))
        self.assertLessEqual(float(resampled.times[-1]), float(self.animation.times[-1]))
        with self.assertRaises(PipelineError) as caught:
            sample_animation(self.animation, np.array([-0.001]))
        self.assertEqual(caught.exception.code, ErrorCode.E_TIME_NONMONOTONIC)

    def test_rest_pose_retarget_and_root_scale(self) -> None:
        rest_rotations = np.empty(
            (self.animation.frame_count, self.source.joint_count, 4), dtype=np.float64
        )
        for frame_index in range(self.animation.frame_count):
            for joint_index, joint in enumerate(self.source.joints):
                rest_rotations[frame_index, joint_index] = joint.rest_rotation
        rest_animation = Animation(
            joint_names=self.source.joint_names,
            times=self.animation.times.copy(),
            rotations=rest_rotations,
            root_translation=np.zeros((self.animation.frame_count, 3), dtype=np.float64),
            interpolation="slerp",
        )
        result = retarget_motion(rest_animation, self.source, self.target)
        expected = np.stack(
            [joint.rest_rotation for joint in self.target.joints], axis=0
        )
        errors = quat_angle_between(result.animation.rotations, expected[None, :, :])
        self.assertLess(float(np.max(errors)), 1e-12)
        self.assertLess(result.report["max_residual_rad"], 1e-12)
        self.assertLess(result.report["max_global_deviation_rad"], 1e-12)
        self.assertEqual(result.report["flagged_joints"], [])
        self.assertEqual(
            len(result.report["samples"]),
            result.animation.frame_count * result.animation.joint_count,
        )
        source_root = self.source.joints[self.source.joint_index(self.source.root_name)]
        target_root = self.target.joints[self.target.joint_index(self.target.root_name)]
        expected_scale = target_root.length / source_root.length
        self.assertAlmostEqual(result.report["root_translation_scale"], expected_scale, places=15)

        moving = self.animation.copy()
        moving_result = retarget_motion(moving, self.source, self.target)
        expected_roots = moving.root_translation * expected_scale
        self.assertTrue(np.allclose(moving_result.animation.root_translation, expected_roots, atol=1e-12))

    def test_forward_kinematics_matrices_match_reported_target_transforms(self) -> None:
        result = retarget_motion(self.animation, self.source, self.target)
        frame_index = 3
        matrices = result.animation.frame_global_matrices(self.target, frame_index)
        for joint_index, joint in enumerate(self.target.joints):
            sample = next(
                item
                for item in result.report["samples"]
                if item["frame"] == frame_index and item["joint"] == joint.name
            )
            self.assertTrue(
                np.allclose(
                    np.asarray(sample["target_global_matrix"], dtype=np.float64),
                    matrices[joint_index],
                    atol=1e-12,
                )
            )
        root_index = self.target.joint_index(self.target.root_name)
        expected_root_translation = (
            self.target.joints[root_index].rest_translation
            + result.animation.root_translation[frame_index]
        )
        self.assertTrue(
            np.allclose(matrices[root_index, :3, 3], expected_root_translation, atol=1e-12)
        )

    def test_target_missing_joint_and_consistency_failure_localization(self) -> None:
        reduced = Skeleton(name=self.target.name, joints=self.target.joints[:-1])
        with self.assertRaises(PipelineError) as caught:
            retarget_motion(self.animation, self.source, reduced)
        self.assertEqual(caught.exception.code, ErrorCode.E_FIELD_MISSING)
        self.assertEqual(caught.exception.joint, self.target.joints[-1].name)

        altered_rotations = self.animation.rotations.copy()
        altered_rotations[2, 3] = quat_multiply(
            altered_rotations[2, 3], quat_from_axis_angle([0.0, 1.0, 0.0], 0.1)
        )
        altered = Animation(
            joint_names=self.animation.joint_names,
            times=self.animation.times.copy(),
            rotations=altered_rotations,
            root_translation=self.animation.root_translation.copy(),
            interpolation=self.animation.interpolation,
        )
        with self.assertRaises(PipelineError) as caught:
            compare_animations(
                self.animation,
                altered,
                tolerances=Tolerances(),
                raise_on_divergence=True,
            )
        self.assertEqual(caught.exception.code, ErrorCode.E_RETARGET_DIVERGE)
        self.assertEqual(caught.exception.frame, 2)
        self.assertEqual(caught.exception.joint, self.animation.joint_names[3])

    def test_verification_suite_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            report_first = run_verification(first)
            report_second = run_verification(second)
            self.assertEqual(report_first["status"], "PASS")
            self.assertEqual(report_second["status"], "PASS")
            self.assertTrue(report_first["all_errors_covered"])
            self.assertEqual(report_first["manifest_sha256"], report_second["manifest_sha256"])
            first_root = Path(first)
            second_root = Path(second)
            self.assertEqual(
                (first_root / "report.json").read_bytes(),
                (second_root / "report.json").read_bytes(),
            )
            self.assertEqual(
                (first_root / "hashes.sha256").read_bytes(),
                (second_root / "hashes.sha256").read_bytes(),
            )


def assert_json_roundtrip(
    test_case: unittest.TestCase, animation: Animation, skeleton: Skeleton
) -> None:
    for schema_version in (1, 2, 3):
        with test_case.subTest(schema_version=schema_version):
            document = animation_to_document(
                animation, schema_version=schema_version, quat_order="xyzw"
            )
            parsed = animation_from_document(document)
            test_case.assertEqual(parsed.joint_names, animation.joint_names)
            test_case.assertTrue(np.array_equal(parsed.times, animation.times))
            rotation_error = float(
                np.max(quat_angle_between(animation.rotations, parsed.rotations))
            )
            test_case.assertLess(rotation_error, 1e-12)
            if schema_version == 3:
                test_case.assertTrue(
                    np.allclose(
                        parsed.root_translation,
                        animation.root_translation,
                        atol=1e-12,
                    )
                )
            else:
                test_case.assertTrue(
                    np.array_equal(
                        parsed.root_translation,
                        np.zeros_like(parsed.root_translation),
                    )
                )


if __name__ == "__main__":
    unittest.main()
