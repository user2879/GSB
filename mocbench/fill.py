from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import quat
from .parser import InputBundle


@dataclass
class FillResult:
    rotations: Dict[str, List[np.ndarray]]
    intervals: List[dict]
    filled_frame_count: int


def _runs(indices):
    runs = []
    if not indices:
        return runs
    start = prev = indices[0]
    for index in indices[1:]:
        if index == prev + 1:
            prev = index
            continue
        runs.append((start, prev))
        start = prev = index
    runs.append((start, prev))
    return runs


def fill_missing(bundle: InputBundle):
    curves = bundle.curves
    skeleton = bundle.skeleton
    rotations: Dict[str, List[Optional[np.ndarray]]] = {
        name: list(seq) for name, seq in curves.rotations.items()
    }
    intervals = []
    total_filled = 0

    for name in skeleton.order:
        seq = rotations[name]
        gaps = _runs(curves.missing[name])
        for start, end in gaps:
            frame_count = end - start + 1
            total_filled += frame_count
            left = start - 1
            right = end + 1
            is_root = name == skeleton.root
            if is_root or left < 0 or right >= curves.frame_count:
                method = "rest_pose"
                fallback = skeleton.bones[name].rest_rotation
                for frame_index in range(start, end + 1):
                    seq[frame_index] = fallback.copy()
            else:
                method = "slerp"
                q_left = seq[left]
                q_right = seq[right]
                span = right - left
                for frame_index in range(start, end + 1):
                    t = (frame_index - left) / span
                    seq[frame_index] = quat.slerp(q_left, q_right, t)
            intervals.append(
                {
                    "bone": name,
                    "start_frame": start,
                    "end_frame": end,
                    "frame_count": frame_count,
                    "method": method,
                }
            )

    intervals.sort(key=lambda item: (skeleton.order.index(item["bone"]), item["start_frame"]))
    return FillResult(rotations=rotations, intervals=intervals, filled_frame_count=total_filled)
