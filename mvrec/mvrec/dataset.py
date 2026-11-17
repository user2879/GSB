"""Input sample scanning and view record construction."""

import json
import os
import re

from .config import load_cameras, validate_camera
from .errors import E_CAMERA_INVALID, E_VIEW_MISSING, PipelineError
from .pngio import read_png

VIEW_RE = re.compile(r"^view_(\d+)\.png$")


def scan_views(input_dir, injections=None):
    """Scan an input directory and build validated view records.

    Injections is an optional dict keyed by view id with entries such as
    {"corrupt_png": true, "missing_camera": true}. Failures are returned in
    the failure list instead of being raised, so the pipeline can continue
    with the remaining views.
    """
    injections = injections or {}
    image_dir = os.path.join(input_dir, "images")
    if not os.path.isdir(image_dir):
        image_dir = input_dir
    cameras_path = os.path.join(input_dir, "cameras.json")
    try:
        camera_payload = load_cameras(cameras_path)
    except PipelineError as exc:
        raise
    cameras_by_id = {}
    camera_failures = []
    for record in camera_payload:
        view_id = record.get("view_id")
        if isinstance(view_id, int) and view_id not in cameras_by_id:
            cameras_by_id[view_id] = record
        else:
            camera_failures.append(
                {"view_id": view_id, "stage": "load", "code": E_CAMERA_INVALID,
                 "message": "duplicate or invalid camera view_id"}
            )

    view_ids = []
    for name in sorted(os.listdir(image_dir)):
        match = VIEW_RE.match(name)
        if match:
            view_ids.append(int(match.group(1)))
    if len(set(view_ids)) != len(view_ids):
        raise PipelineError(E_VIEW_MISSING, "duplicate view serial numbers on disk")

    views, failures = [], list(camera_failures)
    reference_shape = None
    for view_id in sorted(view_ids):
        image_path = os.path.join(image_dir, "view_%03d.png" % view_id)
        inject = injections.get(str(view_id), injections.get(view_id, {}))
        if not os.path.exists(image_path):
            failures.append({"view_id": view_id, "stage": "load", "code": E_VIEW_MISSING,
                             "message": "image file missing"})
            continue
        try:
            if inject.get("corrupt_png"):
                from .pngio import PNG_SIGNATURE
                with open(image_path, "rb") as handle:
                    blob = handle.read()
                # Flip a payload byte while keeping the file present, so the
                # CRC/IDAT validation deterministically rejects the container.
                corrupted = bytearray(blob)
                corrupted[len(PNG_SIGNATURE) + 20] ^= 0xFF
                from .pngio import _read_chunks
                _read_chunks(bytes(corrupted))
            gray, rgb = read_png(image_path)
        except PipelineError as exc:
            failures.append({"view_id": view_id, "stage": "load", "code": exc.code,
                             "message": exc.message})
            continue
        if reference_shape is None:
            reference_shape = gray.shape
        elif gray.shape != reference_shape:
            failures.append({"view_id": view_id, "stage": "load", "code": E_VIEW_MISSING,
                             "message": "resolution inconsistent with reference view"})
            continue
        if inject.get("missing_camera") or view_id not in cameras_by_id:
            failures.append({"view_id": view_id, "stage": "load", "code": E_CAMERA_INVALID,
                             "message": "camera parameters missing"})
            continue
        try:
            K, R, t, fov_x, fov_y = validate_camera(cameras_by_id[view_id], view_id)
        except PipelineError as exc:
            failures.append({"view_id": view_id, "stage": "load", "code": exc.code,
                             "message": exc.message, **exc.context})
            continue
        views.append({
            "view_id": view_id,
            "image_path": image_path,
            "gray": gray,
            "rgb": rgb,
            "width": gray.shape[1],
            "height": gray.shape[0],
            "K": K,
            "R": R,
            "t": t,
            "fov_x": fov_x,
            "fov_y": fov_y,
        })
    return views, failures
