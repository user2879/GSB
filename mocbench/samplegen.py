import argparse
import json
import math
import os

import numpy as np

from . import quat

SEED = 20261119
FRAME_COUNT = 60
FPS = 30.0

BONE_DEFS = [
    ("root", None, 0.28, 0.0, 0.0),
    ("pelvis", "root", 0.92, 0.0, 0.0),
    ("l_knee", "pelvis", 0.85, -1.0, 0.0),
    ("l_foot", "l_knee", 0.78, -1.0, 0.0),
    ("r_knee", "pelvis", 0.85, 1.0, 0.0),
    ("r_foot", "r_knee", 0.78, 1.0, 0.0),
    ("spine", "pelvis", 0.74, 0.0, 0.0),
    ("head", "spine", 0.48, 0.0, 0.0),
    ("l_clavicle", "spine", 0.34, 0.0, -1.0),
    ("l_elbow", "l_clavicle", 0.52, 0.0, -1.0),
    ("l_hand", "l_elbow", 0.46, 0.0, -1.0),
    ("r_clavicle", "spine", 0.34, 0.0, 1.0),
    ("r_elbow", "r_clavicle", 0.52, 0.0, 1.0),
    ("r_hand", "r_elbow", 0.46, 0.0, 1.0),
]

GAPS = {
    "l_hand": [(0, 1)],
    "spine": [(10, 12)],
    "root": [(15, 16)],
    "l_elbow": [(20, 22)],
    "l_foot": [(30, 31)],
    "l_hand": [(40, 42)],
    "r_hand": [(55, 59)],
}

LIMITS = {
    "l_elbow": (0.0, 1.9),
    "r_elbow": (0.0, 1.9),
    "l_knee": (-1.75, 0.0),
    "r_knee": (-1.75, 0.0),
}

IK_FRAMES = [12, 24, 35, 48]
CONTACT_FRAMES = [0, 12, 24, 48]


def rest_rotation(sign_z, sign_x):
    direction = np.array([sign_x, sign_z, 0.0], dtype=np.float64)
    if np.linalg.norm(direction) < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    direction = direction / np.linalg.norm(direction)
    q = quat.rotation_between(np.array([0.0, 0.0, 1.0]), direction)
    return [float(x) for x in q]


def skeleton_definition():
    bones = []
    for name, parent, length, sign_z, sign_x in BONE_DEFS:
        bones.append(
            {
                "name": name,
                "parent": parent,
                "length": length,
                "rest_rotation": rest_rotation(sign_z, sign_x),
            }
        )
    return bones


def deterministic_curve(rng, name, frame_count):
    index = sum((ord(ch) for ch in name), 0)
    phase = (index % 7) * 0.37
    freq = 0.11 + (index % 5) * 0.017
    frames = []
    for frame in range(frame_count):
        angle_x = 0.18 * math.sin(frame * freq + phase) + 0.05 * math.sin(frame * 0.031 + index)
        angle_y = 0.14 * math.cos(frame * freq * 0.8 + phase * 1.7)
        angle_z = 0.12 * math.sin(frame * freq * 1.3 + phase * 0.6)
        q = quat.mul(quat.axis_angle([1.0, 0.0, 0.0], angle_x),
                     quat.mul(quat.axis_angle([0.0, 1.0, 0.0], angle_y),
                              quat.axis_angle([0.0, 0.0, 1.0], angle_z)))
        noise = rng.normal(0.0, 0.01, size=3)
        q = quat.normalize(quat.mul(quat.axis_angle(noise / max(np.linalg.norm(noise), 1e-9), float(np.linalg.norm(noise))), q))
        frames.append([float(x) for x in q])
    return frames


def build_sample(damping=0.9, tolerance=5e-4, max_iterations=90):
    rng = np.random.default_rng(SEED)
    names = [item[0] for item in BONE_DEFS]
    rotations = {name: deterministic_curve(rng, name, FRAME_COUNT) for name in names}

    non_unit = 0
    for name in names:
        for frame in range(FRAME_COUNT):
            marker = (frame * 13 + sum(ord(ch) for ch in name)) % 37
            if marker == 0:
                scale = 1.0 + 0.15 * ((frame % 3) - 1)
                rotations[name][frame] = [component * scale for component in rotations[name][frame]]
                non_unit += 1

    for name, spans in GAPS.items():
        for start, end in spans:
            for frame in range(start, end + 1):
                rotations[name][frame] = None

    skeleton = {"bones": skeleton_definition()}
    curves = {
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "rotations": rotations,
        "root_positions": [[0.0, 0.0, 0.0] for _ in range(FRAME_COUNT)],
    }

    lengths = {item[0]: item[2] for item in BONE_DEFS}
    ik_targets = []
    for frame in IK_FRAMES:
        if frame == 35:
            target = [1.4, 12.5, 0.2]
        else:
            target = [1.35 + 0.05 * math.sin(frame * 0.4), 1.05 + 0.04 * frame / FRAME_COUNT, 0.25]
        ik_targets.append(
            {
                "bone": "l_hand",
                "frames": [frame],
                "targets": [target],
            }
        )

    constraints = {
        "angle_limits": [
            {"bone": bone, "axis": [0.0, 0.0, 1.0], "lower": lower, "upper": upper}
            for bone, (lower, upper) in LIMITS.items()
        ],
        "ik_targets": ik_targets,
        "contact_frames": CONTACT_FRAMES,
        "contact_bones": ["l_foot", "r_foot"],
        "ground_height": 0.0,
    }
    params = {
        "max_iterations": max_iterations,
        "damping": damping,
        "tolerance": tolerance,
        "chunk_size": 8,
        "max_workers": 3,
        "memory_budget": 114688,
    }
    return {
        "format": "mocbench-sample/v1",
        "seed": SEED,
        "skeleton": skeleton,
        "curves": curves,
        "constraints": constraints,
        "params": params,
    }


def write_sample(path, damping=0.9):
    sample = build_sample(damping=damping)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sample, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate deterministic motion capture sample")
    parser.add_argument("--output", default="samples/sample.json")
    parser.add_argument("--damping", type=float, default=0.9)
    args = parser.parse_args(argv)
    path = write_sample(args.output, damping=args.damping)
    print(path)


if __name__ == "__main__":
    main()
