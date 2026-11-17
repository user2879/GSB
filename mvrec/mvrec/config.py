"""Configuration / camera loading, validation and stable hashing."""

import hashlib
import json

import numpy as np

from .errors import E_CAMERA_INVALID, E_CONFIG_INVALID, PipelineError

# Fields that may be overridden between suspend and resume without invalidating
# a snapshot; everything else is part of the structural configuration hash.
RESUME_OVERRIDABLE = {
    "slot_budget",
    "point_budget",
    "thread_count",
    "snapshot_commit_interval",
}

REQUIRED_FIELDS = {
    "feature_threshold",
    "min_inlier_matches",
    "min_parallax_deg",
    "depth_width",
    "depth_height",
    "depth_levels",
    "depth_smooth_passes",
    "voxel_size",
    "grouping_candidates",
    "slot_budget",
    "point_budget",
    "thread_count",
    "snapshot_commit_interval",
}


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(config):
    structural = {k: v for k, v in config.items() if k not in RESUME_OVERRIDABLE}
    return sha256_text(canonical_json(structural))


def camera_hash(cameras):
    payload = [
        {
            "view_id": cam["view_id"],
            "K": cam["K"],
            "R": cam["R"],
            "t": cam["t"],
        }
        for cam in sorted(cameras, key=lambda item: item["view_id"])
    ]
    return sha256_text(canonical_json(payload))


def _require_int(config, key, minimum):
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise PipelineError(
            E_CONFIG_INVALID, "%s must be an integer >= %r" % (key, minimum)
        )


def load_config(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(E_CONFIG_INVALID, "cannot parse config: %s" % exc)
    if not isinstance(config, dict):
        raise PipelineError(E_CONFIG_INVALID, "config root must be an object")
    missing = REQUIRED_FIELDS - set(config)
    if missing:
        raise PipelineError(
            E_CONFIG_INVALID, "missing config fields: %s" % sorted(missing)
        )
    _require_int(config, "feature_threshold", 0)
    _require_int(config, "min_inlier_matches", 1)
    if not isinstance(config["min_parallax_deg"], (int, float)) or not (
        0 < float(config["min_parallax_deg"]) < 90
    ):
        raise PipelineError(E_CONFIG_INVALID, "min_parallax_deg must be within (0, 90)")
    _require_int(config, "depth_width", 1)
    _require_int(config, "depth_height", 1)
    _require_int(config, "depth_levels", 2)
    _require_int(config, "depth_smooth_passes", 0)
    if not isinstance(config["voxel_size"], (int, float)) or config["voxel_size"] < 0:
        raise PipelineError(E_CONFIG_INVALID, "voxel_size must be a non-negative number")
    _require_int(config, "slot_budget", 0)
    _require_int(config, "point_budget", 0)
    _require_int(config, "thread_count", 1)
    _require_int(config, "snapshot_commit_interval", 1)
    candidates = config["grouping_candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise PipelineError(E_CONFIG_INVALID, "grouping_candidates must be a non-empty list")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise PipelineError(E_CONFIG_INVALID, "candidate %d must be an object" % index)
        if not isinstance(candidate.get("name"), str) or not candidate["name"]:
            raise PipelineError(E_CONFIG_INVALID, "candidate %d name invalid" % index)
        for field in ("group_size", "views_per_batch"):
            if not isinstance(candidate.get(field), int) or isinstance(candidate[field], bool) or candidate[field] < 1:
                raise PipelineError(
                    E_CONFIG_INVALID, "candidate %d field %s invalid" % (index, field)
                )
    names = [candidate["name"] for candidate in candidates]
    if len(set(names)) != len(names):
        raise PipelineError(E_CONFIG_INVALID, "duplicate grouping candidate names")
    if "loop_group_pairs" in config and config["loop_group_pairs"] is not None:
        if not isinstance(config["loop_group_pairs"], list):
            raise PipelineError(E_CONFIG_INVALID, "loop_group_pairs must be a list")
        for pair in config["loop_group_pairs"]:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in pair)
            ):
                raise PipelineError(E_CONFIG_INVALID, "loop pair must be [int, int]")
    return config


def validate_camera(camera, expected_view_id):
    """Return (K, R, t) float64 arrays or raise E_CAMERA_INVALID."""
    if not isinstance(camera, dict):
        raise PipelineError(E_CAMERA_INVALID, "camera record missing", {"view_id": expected_view_id})
    if camera.get("view_id") != expected_view_id:
        raise PipelineError(
            E_CAMERA_INVALID, "camera view_id mismatch", {"view_id": expected_view_id}
        )
    try:
        K = np.asarray(camera["K"], dtype=np.float64)
        R = np.asarray(camera["R"], dtype=np.float64)
        t = np.asarray(camera["t"], dtype=np.float64).reshape(3)
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(E_CAMERA_INVALID, "camera matrix malformed: %s" % exc,
                            {"view_id": expected_view_id})
    if K.shape != (3, 3) or R.shape != (3, 3):
        raise PipelineError(E_CAMERA_INVALID, "camera matrix shape invalid",
                            {"view_id": expected_view_id})
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
        raise PipelineError(E_CAMERA_INVALID, "camera contains non-finite values",
                            {"view_id": expected_view_id})
    try:
        np.linalg.inv(K)
    except np.linalg.LinAlgError:
        raise PipelineError(E_CAMERA_INVALID, "intrinsics matrix not invertible",
                            {"view_id": expected_view_id})
    fx, fy = K[0, 0], K[1, 1]
    width = int(camera.get("width", 0))
    height = int(camera.get("height", 0))
    if fx <= 0 or fy <= 0 or width <= 0 or height <= 0:
        raise PipelineError(E_CAMERA_INVALID, "focal length or image size non-positive",
                            {"view_id": expected_view_id})
    fov_x = np.degrees(2.0 * np.arctan(width / (2.0 * fx)))
    fov_y = np.degrees(2.0 * np.arctan(height / (2.0 * fy)))
    if not (1.0 <= fov_x <= 170.0) or not (1.0 <= fov_y <= 170.0):
        raise PipelineError(
            E_CAMERA_INVALID,
            "field of view out of legal range [1, 170] degrees",
            {"view_id": expected_view_id, "fov_x": float(fov_x), "fov_y": float(fov_y)},
        )
    ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    if ortho_err > 1e-6 or abs(float(np.linalg.det(R)) - 1.0) > 1e-6:
        raise PipelineError(E_CAMERA_INVALID, "rotation matrix not orthonormal",
                            {"view_id": expected_view_id})
    return K, R, t, float(fov_x), float(fov_y)


def load_cameras(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(E_CAMERA_INVALID, "cannot parse cameras file: %s" % exc)
    if not isinstance(payload, list):
        raise PipelineError(E_CAMERA_INVALID, "cameras file must be a list")
    return payload
