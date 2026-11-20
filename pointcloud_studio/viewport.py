"""Screen-space viewport box selection against an immutable scene snapshot."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .errors import ErrorCode, StudioError
from .camera import CameraPose, look_at, perspective


def camera_pose_from_dict(value: Mapping[str, Any]) -> CameraPose:
    return CameraPose(
        eye=tuple(float(item) for item in value.get("eye", (8.0, 4.5, 8.0))),
        target=tuple(float(item) for item in value.get("target", (0.0, 1.0, 0.0))),
        up=tuple(float(item) for item in value.get("up", (0.0, 1.0, 0.0))),
        fov_y_degrees=float(value.get("fov_y_degrees", 52.0)),
        near=float(value.get("near", 0.05)),
        far=float(value.get("far", 80.0)),
    )


def project_points(
    points: np.ndarray,
    camera: CameraPose,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(points, dtype=np.float64)
    mvp = perspective(camera.fov_y_degrees, width / max(height, 1), camera.near, camera.far) @ look_at(
        np.asarray(camera.eye, dtype=np.float64),
        np.asarray(camera.target, dtype=np.float64),
        np.asarray(camera.up, dtype=np.float64),
    )
    homogeneous = np.column_stack((source, np.ones(source.shape[0], dtype=np.float64)))
    clip = homogeneous @ mvp.T
    w = clip[:, 3]
    valid = w > 1.0e-12
    ndc = np.zeros((source.shape[0], 3), dtype=np.float64)
    ndc[valid] = clip[valid, :3] / w[valid, None]
    pixels = np.empty((source.shape[0], 2), dtype=np.float64)
    pixels[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
    pixels[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    visible = valid & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
    return pixels, ndc[:, 2], visible


def select_viewport_box(
    points: np.ndarray,
    camera: CameraPose,
    width: int,
    height: int,
    rect: Sequence[float],
) -> np.ndarray:
    if len(rect) != 4 or width <= 0 or height <= 0:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "viewport rectangle must be [x0,y0,x1,y1] with positive dimensions",
            context={"rect": list(rect), "width": width, "height": height},
        )
    x0, y0, x1, y1 = (float(item) for item in rect)
    if not np.isfinite([x0, y0, x1, y1]).all() or x0 > x1 or y0 > y1:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "viewport rectangle is not a finite ordered rectangle",
            context={"rect": [x0, y0, x1, y1]},
        )
    pixels, _, visible = project_points(points, camera, width, height)
    inside = (
        visible
        & (pixels[:, 0] >= x0)
        & (pixels[:, 0] <= x1)
        & (pixels[:, 1] >= y0)
        & (pixels[:, 1] <= y1)
    )
    return np.flatnonzero(inside)


def viewport_box_remove_operation(
    points: np.ndarray,
    *,
    camera: CameraPose,
    width: int,
    height: int,
    rect: Sequence[float],
    seq: int,
) -> dict[str, Any]:
    selected = select_viewport_box(points, camera, width, height, rect)
    if selected.size == 0:
        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "viewport box does not contain any point of the current snapshot",
            context={"rect": list(rect), "width": width, "height": height},
        )
    return {
        "seq": int(seq),
        "type": "remove_viewport_box",
        "params": {
            "camera": {
                "eye": list(camera.eye),
                "target": list(camera.target),
                "up": list(camera.up),
                "fov_y_degrees": camera.fov_y_degrees,
                "near": camera.near,
                "far": camera.far,
            },
            "width": int(width),
            "height": int(height),
            "rect": [float(item) for item in rect],
        },
    }
