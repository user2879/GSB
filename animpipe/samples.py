"""Deterministic local generation of skeletons, source motion and BTRJ sample."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .animation import Animation, build_legacy_document, dump_json_animation
from .binary import BinarySpec, dump_binary
from .quat import multiply, normalize
from .skeleton import Skeleton, generate_skeleton

SOURCE_SEED = 20261120
TARGET_SEED = 20261121
MOTION_SEED = 20261122


@dataclass
class GeneratedSample:
    source_skel: Skeleton
    target_skel: Skeleton
    source_anim: Animation
    target_fps: float


def generate_source_animation(
    skeleton: Skeleton,
    *,
    frame_count: int = 24,
    fps: float = 12.0,
    interp: str = "slerp",
) -> Animation:
    rng = np.random.Generator(np.random.PCG64(MOTION_SEED))
    times = np.arange(frame_count, dtype=np.float64) / fps
    rotations = np.zeros((frame_count, skeleton.joint_count, 4), dtype=np.float64)
    roots = np.zeros((frame_count, 3), dtype=np.float64)

    phases = rng.uniform(0.0, 2.0 * np.pi, size=skeleton.joint_count)
    for frame in range(frame_count):
        for joint in range(skeleton.joint_count):
            angle = 0.12 * np.sin(2.0 * np.pi * frame / frame_count + phases[joint])
            axis = np.array(
                [
                    np.cos(phases[joint]),
                    np.sin(phases[joint]),
                    0.5 * np.cos(2.0 * phases[joint]),
                ]
            )
            axis /= np.linalg.norm(axis)
            half = angle * 0.5
            delta = np.concatenate([[np.cos(half)], axis * np.sin(half)])
            rotations[frame, joint] = normalize(
                multiply(skeleton.rest_rotations[joint], delta)
            )
        roots[frame] = np.array(
            [0.05 * frame, 0.02 * np.sin(frame * 0.5), 0.0], dtype=np.float64
        )

    return Animation(
        joint_names=skeleton.names,
        times=times,
        rotations=rotations,
        root_translations=roots,
        interp=interp,
        quat_order="wxyz",
        parameter_version="generated",
    )


def generate_all(
    output_dir: str,
    *,
    frame_count: int = 24,
    source_fps: float = 12.0,
    target_fps: float = 18.0,
) -> GeneratedSample:
    os.makedirs(output_dir, exist_ok=True)
    source_skel = generate_skeleton("source", SOURCE_SEED, length_scale=1.0)
    target_skel = generate_skeleton("target", TARGET_SEED, length_scale=1.35)
    source_anim = generate_source_animation(
        source_skel, frame_count=frame_count, fps=source_fps
    )

    dump_json_animation(source_anim, os.path.join(output_dir, "source_motion.json"))
    dump_binary(
        source_anim,
        os.path.join(output_dir, "sample_trajectory.btrj"),
        BinarySpec(endianness="<", index_width=2, quat_order="xyzw"),
    )
    legacy = build_legacy_document(source_anim)
    import json

    with open(os.path.join(output_dir, "legacy_v1.json"), "w", encoding="utf-8") as handle:
        json.dump(legacy, handle, sort_keys=True, indent=2)
        handle.write("\n")

    return GeneratedSample(
        source_skel=source_skel,
        target_skel=target_skel,
        source_anim=source_anim,
        target_fps=target_fps,
    )
