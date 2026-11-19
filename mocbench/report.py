import json
import os

from .errors import BenchError
from .persistence import atomic_write_json
from .solver import SOLVE_ORDER


def build_report(frames, fill_result, bundle, version_digest):
    params = bundle.params
    per_constraint = {}
    for name in SOLVE_ORDER:
        per_constraint[name] = {"evaluated_frames": 0, "passed_frames": 0, "pass_rate": 0.0}

    unconverged = []
    passed_count = 0
    for frame in frames:
        frame_ok = frame["converged"]
        if frame_ok:
            passed_count += 1
        else:
            unconverged.append(
                {
                    "frame": frame["frame"],
                    "iterations_used": frame["iterations_used"],
                    "solve_order": list(frame["solve_order"]),
                    "final_residuals": frame["residuals"],
                }
            )
        for residual in frame["residuals"]:
            name = residual["constraint"]
            bucket = per_constraint[name]
            bucket["evaluated_frames"] += 1
            if residual["passed"]:
                bucket["passed_frames"] += 1

    for bucket in per_constraint.values():
        if bucket["evaluated_frames"]:
            bucket["pass_rate"] = bucket["passed_frames"] / bucket["evaluated_frames"]

    return {
        "version_digest": version_digest,
        "frame_count": bundle.curves.frame_count,
        "fps": bundle.curves.fps,
        "filled_interval_count": len(fill_result.intervals),
        "filled_frame_count": fill_result.filled_frame_count,
        "fill_intervals": fill_result.intervals,
        "normalized_quaternion_count": bundle.curves.normalized_count,
        "passed_frame_count": passed_count,
        "unconverged_frame_count": len(unconverged),
        "unconverged_frames": unconverged,
        "per_constraint_pass_rate": per_constraint,
        "solve_order": list(SOLVE_ORDER),
        "params": {
            "max_iterations": params.max_iterations,
            "damping": params.damping,
            "tolerance": params.tolerance,
            "chunk_size": params.chunk_size,
            "max_workers": params.max_workers,
            "memory_budget": params.memory_budget,
            "fps": params.fps,
        },
    }


def build_poses(frames, bundle, version_digest):
    return {
        "version_digest": version_digest,
        "frame_count": bundle.curves.frame_count,
        "fps": bundle.curves.fps,
        "bone_order": list(bundle.skeleton.order),
        "frames": [
            {
                "frame": frame["frame"],
                "timestamp": frame["timestamp"],
                "local_rotations": frame["local_rotations"],
                "root_position": frame["root_position"],
                "world_positions": frame["world_positions"],
            }
            for frame in frames
        ],
    }


def export_atomic(output_dir, report, poses):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.json")
    poses_path = os.path.join(output_dir, "poses.json")

    existing_version = _existing_version(report_path, poses_path)
    if existing_version is not None and existing_version == report["version_digest"]:
        old_poses = _read_json(poses_path)
        if old_poses.get("frame_count") == poses["frame_count"]:
            return {"replaced": False, "reason": "same_version_kept"}

    stage_dir = os.path.join(output_dir, ".export_tmp")
    os.makedirs(stage_dir, exist_ok=True)
    staged_report = os.path.join(stage_dir, "report.json")
    staged_poses = os.path.join(stage_dir, "poses.json")
    atomic_write_json(staged_report, report)
    atomic_write_json(staged_poses, poses)
    os.replace(staged_report, report_path)
    os.replace(staged_poses, poses_path)
    try:
        os.rmdir(stage_dir)
    except OSError:
        pass
    return {"replaced": True, "previous_version": existing_version}


def _existing_version(report_path, poses_path):
    if not (os.path.exists(report_path) and os.path.exists(poses_path)):
        return None
    report = _read_json(report_path)
    poses = _read_json(poses_path)
    report_version = report.get("version_digest")
    poses_version = poses.get("version_digest")
    if report_version != poses_version:
        raise BenchError(
            "E_DIGEST",
            "existing report and poses have different version digests",
            {"report": report_version, "poses": poses_version},
        )
    return report_version


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise BenchError("E_DIGEST", f"existing output is unreadable: {path}", {"detail": str(exc)})
