import numpy as np


def to_native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    return value


def frame_for_storage(frame_result):
    return {
        "frame": frame_result["frame"],
        "timestamp": float(frame_result["timestamp"]),
        "iterations_used": frame_result["iterations_used"],
        "converged": bool(frame_result["converged"]),
        "local_rotations": to_native(frame_result["local_rotations"]),
        "root_position": to_native(frame_result["root_position"]),
        "world_positions": to_native(frame_result["world_positions"]),
        "residuals": to_native(frame_result["residuals"]),
        "solve_order": list(frame_result["solve_order"]),
    }
