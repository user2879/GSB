import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import quat
from .errors import BenchError
from .skeleton import Skeleton, build_skeleton

REQUIRED_PARAMS = {
    "max_iterations": int,
    "damping": float,
    "tolerance": float,
    "chunk_size": int,
    "max_workers": int,
    "memory_budget": int,
}


@dataclass
class CurveData:
    frame_count: int
    fps: float
    rotations: Dict[str, List[Optional[np.ndarray]]]
    root_positions: List[np.ndarray]
    missing: Dict[str, List[int]]
    normalized_count: int


@dataclass
class Constraints:
    angle_limits: List[dict]
    ik_targets: List[dict]
    contact_frames: List[int]
    contact_bones: List[str]
    ground_height: float


@dataclass
class Params:
    max_iterations: int
    damping: float
    tolerance: float
    chunk_size: int
    max_workers: int
    memory_budget: int
    fps: float


@dataclass
class InputBundle:
    skeleton: Skeleton
    curves: CurveData
    constraints: Constraints
    params: Params
    raw: dict


def load_bundle(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise BenchError("E_INPUT", f"input file not found: {path}")
    except json.JSONDecodeError as exc:
        raise BenchError("E_INPUT", f"invalid JSON: {exc}", {"line": exc.lineno})
    return parse_bundle(data)


def parse_bundle(data):
    if not isinstance(data, dict):
        raise BenchError("E_INPUT", "bundle root must be an object")
    skeleton = build_skeleton(data.get("skeleton", {}).get("bones", []))
    curves = _parse_curves(data.get("curves", {}), skeleton)
    constraints = _parse_constraints(data.get("constraints", {}), skeleton, curves.frame_count)
    params = _parse_params(data.get("params", {}), curves.fps)
    return InputBundle(skeleton=skeleton, curves=curves, constraints=constraints, params=params, raw=data)


def _parse_curves(curves, skeleton):
    if not isinstance(curves, dict):
        raise BenchError("E_CURVE", "curves must be an object")
    frame_count = curves.get("frame_count")
    if not isinstance(frame_count, int) or frame_count <= 0:
        raise BenchError("E_CURVE", "curves.frame_count must be a positive integer")
    fps = curves.get("fps", 30.0)
    if not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        raise BenchError("E_PARAM", "fps must be a finite positive number")
    raw_rot = curves.get("rotations", {})
    if not isinstance(raw_rot, dict):
        raise BenchError("E_CURVE", "curves.rotations must map bone name to frame list")
    for name in skeleton.order:
        if name not in raw_rot:
            raise BenchError("E_CURVE", f"missing rotation curve for bone {name}", {"bone": name})
    extra = sorted(set(raw_rot) - set(skeleton.order))
    if extra:
        raise BenchError("E_CURVE", "rotation curves reference unknown bones", {"bones": extra})

    rotations = {}
    missing = {}
    normalized_count = 0
    for name in skeleton.order:
        seq = raw_rot[name]
        if not isinstance(seq, list) or len(seq) != frame_count:
            raise BenchError(
                "E_CURVE",
                f"curve frame count mismatch for bone {name}",
                {"bone": name, "expected": frame_count, "actual": len(seq) if isinstance(seq, list) else None},
            )
        frames = []
        gap = []
        for index, entry in enumerate(seq):
            if entry is None:
                frames.append(None)
                gap.append(index)
                continue
            if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                raise BenchError("E_QUATERNION", f"frame {index} of {name} is not a 4-vector")
            arr = np.asarray(entry, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                raise BenchError(
                    "E_QUATERNION",
                    f"non-finite quaternion at frame {index} of {name}",
                    {"bone": name, "frame": index},
                )
            if not quat.is_unit(arr):
                normalized_count += 1
                arr = quat.normalize(arr)
            frames.append(arr)
        rotations[name] = frames
        missing[name] = gap

    root_positions = []
    raw_positions = curves.get("root_positions")
    if raw_positions is not None:
        if not isinstance(raw_positions, list) or len(raw_positions) != frame_count:
            raise BenchError("E_CURVE", "root_positions length must equal frame_count")
        for index, entry in enumerate(raw_positions):
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise BenchError("E_CURVE", f"root position at frame {index} must be a 3-vector")
            arr = np.asarray(entry, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                raise BenchError("E_CURVE", f"non-finite root position at frame {index}")
            root_positions.append(arr)
    else:
        root_positions = [np.zeros(3, dtype=np.float64) for _ in range(frame_count)]

    return CurveData(
        frame_count=frame_count,
        fps=float(fps),
        rotations=rotations,
        root_positions=root_positions,
        missing=missing,
        normalized_count=normalized_count,
    )


def _parse_constraints(constraints, skeleton, frame_count):
    if not isinstance(constraints, dict):
        raise BenchError("E_INPUT", "constraints must be an object")

    angle_limits = []
    for item in constraints.get("angle_limits", []):
        bone = item.get("bone")
        if bone not in skeleton.bones:
            raise BenchError("E_LIMIT", "angle limit references unknown bone", {"bone": bone})
        axis = item.get("axis")
        if not isinstance(axis, (list, tuple)) or len(axis) != 3:
            raise BenchError("E_LIMIT", "angle limit axis must be a 3-vector", {"bone": bone})
        axis_arr = np.asarray(axis, dtype=np.float64)
        if not np.all(np.isfinite(axis_arr)) or np.linalg.norm(axis_arr) <= 0:
            raise BenchError("E_LIMIT", "angle limit axis invalid", {"bone": bone})
        lower = item.get("lower", -np.pi)
        upper = item.get("upper", np.pi)
        if not all(np.isfinite(v) for v in (lower, upper)) or lower > upper:
            raise BenchError(
                "E_LIMIT",
                f"angle limit lower must not exceed upper for {bone}",
                {"bone": bone, "lower": lower, "upper": upper},
            )
        angle_limits.append(
            {
                "bone": bone,
                "axis": quat.normalize_vec(axis_arr),
                "lower": float(lower),
                "upper": float(upper),
            }
        )

    ik_targets = []
    for item in constraints.get("ik_targets", []):
        bone = item.get("bone")
        if bone not in skeleton.bones:
            raise BenchError("E_TARGET", "IK target references unknown bone", {"bone": bone})
        target_frames = item.get("frames", [])
        targets = item.get("targets", [])
        if not isinstance(target_frames, list) or not isinstance(targets, list) or len(target_frames) != len(targets):
            raise BenchError("E_TARGET", "IK frames and targets length mismatch", {"bone": bone})
        for frame_index, position in zip(target_frames, targets):
            if not isinstance(frame_index, int) or not (0 <= frame_index < frame_count):
                raise BenchError(
                    "E_TARGET",
                    f"IK frame index out of range for {bone}",
                    {"bone": bone, "frame": frame_index, "frame_count": frame_count},
                )
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                raise BenchError("E_TARGET", "IK target must be a 3-vector", {"bone": bone})
            pos = np.asarray(position, dtype=np.float64)
            if not np.all(np.isfinite(pos)):
                raise BenchError("E_TARGET", "non-finite IK target", {"bone": bone, "frame": frame_index})
        ik_targets.append({"bone": bone, "frames": list(target_frames), "targets": [np.asarray(p, dtype=np.float64) for p in targets]})

    contact_frames = constraints.get("contact_frames", [])
    if not isinstance(contact_frames, list) or any(
        not isinstance(i, int) or not (0 <= i < frame_count) for i in contact_frames
    ):
        raise BenchError("E_CONTACT", "contact_frames must be valid frame indices")
    contact_bones = constraints.get("contact_bones", [])
    if not isinstance(contact_bones, list) or any(b not in skeleton.bones for b in contact_bones):
        raise BenchError("E_CONTACT", "contact_bones reference unknown bones")
    ground_height = constraints.get("ground_height", 0.0)
    if not isinstance(ground_height, (int, float)) or not np.isfinite(ground_height):
        raise BenchError("E_CONTACT", "ground_height must be finite")

    return Constraints(
        angle_limits=angle_limits,
        ik_targets=ik_targets,
        contact_frames=sorted(contact_frames),
        contact_bones=list(contact_bones),
        ground_height=float(ground_height),
    )


def _parse_params(params, fps):
    if not isinstance(params, dict):
        raise BenchError("E_PARAM", "params must be an object")
    values = {}
    for key, kind in REQUIRED_PARAMS.items():
        if key not in params:
            raise BenchError("E_PARAM", f"missing parameter: {key}", {"param": key})
        value = params[key]
        if kind is int:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise BenchError("E_PARAM", f"{key} must be a positive integer", {"param": key})
        else:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise BenchError("E_PARAM", f"{key} must be a finite positive number", {"param": key})
            if key == "damping" and not (0.0 < value <= 1.0):
                raise BenchError("E_PARAM", "damping must lie within (0, 1]", {"damping": value})
        values[key] = kind(value) if kind is float else value
    return Params(fps=float(fps), **values)
