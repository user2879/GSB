"""Input discovery and validation for views, cameras, and observations."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import SolverConfig
from .errors import (
    ConfigError,
    E_CAMERA_INVALID,
    E_CFG_INVALID,
    E_OBS_INVALID,
    Failure,
)
from .hashing import combine_fingerprints, sha256_file
from .injections import InjectionItem, InjectionSet
from .models import Camera, InputBundle, Observation, PointSource


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
INSTANCE_SUFFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
ORTHOGONAL_TOLERANCE = 1e-6
K_DETERMINANT_TOLERANCE = 1e-12


def _failure(
    code: str,
    keypoint_id: str | None,
    stage: str,
    message: str,
    *,
    view_index: int | None = None,
    observation_index: int | None = None,
    details: dict[str, Any] | None = None,
) -> Failure:
    return Failure(
        code=code,
        keypoint_id=keypoint_id,
        stage=stage,
        message=message,
        view_index=view_index,
        observation_index=observation_index,
        details=details or {},
    )


def _resolve_image(
    input_dir: Path,
    cfg: SolverConfig,
    camera_path: Path,
    camera_raw: dict[str, Any],
    view_index: int,
    image_files: list[Path],
) -> Path | None:
    candidates: list[Path] = []
    declared = camera_raw.get("image") or camera_raw.get("image_file") or camera_raw.get("png")
    if isinstance(declared, str) and declared:
        declared_path = Path(declared)
        if declared_path.is_absolute():
            candidates.append(declared_path)
        else:
            candidates.extend(
                [
                    input_dir / declared_path,
                    camera_path.parent / declared_path,
                    input_dir / cfg.image_dir / declared_path,
                ]
            )
    candidates.extend(
        [
            input_dir / cfg.image_dir / f"view_{view_index:02d}.png",
            input_dir / cfg.image_dir / f"view_{view_index}.png",
            camera_path.with_suffix(".png"),
            input_dir / cfg.image_dir / camera_path.with_suffix(".png").name,
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    stem_matches = [p for p in image_files if p.stem == camera_path.stem]
    if len(stem_matches) == 1:
        return stem_matches[0]
    return None


def _parse_intrinsics(camera_raw: dict[str, Any]) -> np.ndarray:
    if "K" in camera_raw:
        K = np.asarray(camera_raw["K"], dtype=np.float64)
    else:
        intrinsic = camera_raw.get("intrinsics", camera_raw)
        if not isinstance(intrinsic, dict):
            raise ValueError("intrinsics must be an object")
        fx = float(intrinsic["fx"])
        fy = float(intrinsic["fy"])
        cx = float(intrinsic["cx"])
        cy = float(intrinsic["cy"])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("K must be a 3x3 matrix")
    if not np.all(np.isfinite(K)):
        raise ValueError("K contains non-finite values")
    det = float(np.linalg.det(K))
    if not math.isfinite(det) or abs(det) <= K_DETERMINANT_TOLERANCE:
        raise ValueError("K is singular or nearly singular")
    return K


def _parse_extrinsics(camera_raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    extrinsic = camera_raw.get("extrinsics", camera_raw)
    if not isinstance(extrinsic, dict):
        raise ValueError("extrinsics must be an object")
    R = np.asarray(extrinsic["R"], dtype=np.float64)
    t_raw = extrinsic.get("t", extrinsic.get("translation"))
    if t_raw is None:
        raise ValueError("extrinsics.t is required")
    t = np.asarray(t_raw, dtype=np.float64).reshape(3)
    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
        raise ValueError("extrinsics contain non-finite values")
    gram = R @ R.T
    if not np.allclose(gram, np.eye(3), atol=ORTHOGONAL_TOLERANCE, rtol=0.0):
        raise ValueError("R is not orthogonal")
    det = float(np.linalg.det(R))
    if not math.isfinite(det) or abs(det - 1.0) > 1e-5:
        raise ValueError("R must be a proper rotation with determinant +1")
    return R, t


def _decode_png(path: Path) -> tuple[np.ndarray, str]:
    with path.open("rb") as handle:
        header = handle.read(8)
    if header != PNG_SIGNATURE:
        raise ValueError("invalid PNG container header")
    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as image:
            image.load()
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"PNG decode failed: {exc}") from exc
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("decoded image is not a non-empty grayscale array")
    return gray, sha256_file(path)


def _load_cameras(
    input_dir: Path,
    cfg: SolverConfig,
    image_files: list[Path],
) -> tuple[dict[int, Camera], dict[int, np.ndarray], list[Failure], tuple[int, int] | None, str]:
    camera_dir = input_dir / cfg.camera_dir
    camera_files = sorted(camera_dir.glob("*.json")) if camera_dir.exists() else []
    failures: list[Failure] = []
    parsed: list[dict[str, Any]] = []
    seen_views: dict[int, int] = {}
    duplicate_views: set[int] = set()

    for camera_path in camera_files:
        try:
            raw = json.loads(camera_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    f"cannot parse camera JSON {camera_path.name}: {exc}",
                )
            )
            continue
        if not isinstance(raw, dict):
            failures.append(
                _failure(E_CAMERA_INVALID, None, "load", f"{camera_path.name} is not an object")
            )
            continue
        view_raw = raw.get("view_index", raw.get("view"))
        if isinstance(view_raw, bool) or not isinstance(view_raw, int):
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    f"{camera_path.name} has no integer view_index",
                )
            )
            continue
        view_index = int(view_raw)
        if view_index < 0:
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    f"{camera_path.name} has a negative view_index",
                    view_index=view_index,
                )
            )
            continue
        if view_index in seen_views:
            duplicate_views.add(view_index)
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    f"duplicate camera view_index {view_index}",
                    view_index=view_index,
                )
            )
            continue
        seen_views[view_index] = len(parsed)
        parsed.append(
            {
                "view_index": view_index,
                "camera_path": camera_path,
                "raw": raw,
            }
        )

    cameras: dict[int, Camera] = {}
    grays: dict[int, np.ndarray] = {}
    used_images: set[Path] = set()
    dimensions: dict[int, tuple[int, int]] = {}
    image_hashes: dict[int, str] = {}

    for entry in parsed:
        view_index = entry["view_index"]
        if view_index in duplicate_views:
            continue
        camera_path: Path = entry["camera_path"]
        raw: dict[str, Any] = entry["raw"]
        image_path = _resolve_image(
            input_dir, cfg, camera_path, raw, view_index, image_files
        )
        if image_path is None:
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    f"no image found for camera {camera_path.name}",
                    view_index=view_index,
                )
            )
            continue
        used_images.add(image_path)
        try:
            K = _parse_intrinsics(raw)
            R, t = _parse_extrinsics(raw)
            gray, image_sha = _decode_png(image_path)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    f"camera or image validation failed for view {view_index}: {exc}",
                    view_index=view_index,
                )
            )
            continue
        height, width = int(gray.shape[0]), int(gray.shape[1])
        declared_width = raw.get("width")
        declared_height = raw.get("height")
        try:
            declared_width_int = (
                None
                if declared_width is None
                else int(declared_width)
                if not isinstance(declared_width, bool)
                else None
            )
            declared_height_int = (
                None
                if declared_height is None
                else int(declared_height)
                if not isinstance(declared_height, bool)
                else None
            )
        except (TypeError, ValueError):
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    f"view {view_index} has non-integer declared dimensions",
                    view_index=view_index,
                )
            )
            continue
        if declared_width is not None and declared_width_int is None:
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    f"view {view_index} has invalid declared width",
                    view_index=view_index,
                )
            )
            continue
        if declared_height is not None and declared_height_int is None:
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    f"view {view_index} has invalid declared height",
                    view_index=view_index,
                )
            )
            continue
        if declared_width_int is not None and declared_width_int != width:
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    f"view {view_index} declared width does not match PNG",
                    view_index=view_index,
                )
            )
            continue
        if declared_height_int is not None and declared_height_int != height:
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    f"view {view_index} declared height does not match PNG",
                    view_index=view_index,
                )
            )
            continue
        cameras[view_index] = Camera(
            view_index=view_index,
            width=width,
            height=height,
            K=K,
            R=R,
            t=t,
            image_path=str(image_path),
            camera_path=str(camera_path),
            image_sha256=image_sha,
            source_image_sha256=image_sha,
        )
        grays[view_index] = gray
        dimensions[view_index] = (width, height)
        image_hashes[view_index] = image_sha

    expected_resolution: tuple[int, int] | None = None
    for view_index in sorted(dimensions):
        resolution = dimensions[view_index]
        if expected_resolution is None:
            expected_resolution = resolution
        elif resolution != expected_resolution:
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    "view resolution differs from the global fixed resolution",
                    view_index=view_index,
                    details={
                        "expected": list(expected_resolution),
                        "actual": list(resolution),
                    },
                )
            )
            cameras.pop(view_index, None)
            grays.pop(view_index, None)
            image_hashes.pop(view_index, None)

    for image_path in image_files:
        if image_path not in used_images:
            failures.append(
                _failure(
                    E_CAMERA_INVALID,
                    None,
                    "load",
                    f"unreferenced PNG view: {image_path.name}",
                )
            )

    if not cameras:
        failures.append(
            _failure(E_CAMERA_INVALID, None, "load", "no valid camera views found")
        )
    camera_hash = combine_fingerprints(camera.fingerprint for camera in cameras.values())
    return cameras, grays, failures, expected_resolution, camera_hash


def _split_keypoint_id(keypoint_id: str) -> tuple[str, str | None]:
    marker = "@instance-"
    if marker not in keypoint_id:
        if "@" in keypoint_id:
            raise ValueError("instance suffix must use the '@instance-' prefix")
        return keypoint_id, None
    if keypoint_id.count(marker) != 1:
        raise ValueError("keypoint id contains multiple instance suffixes")
    base, suffix = keypoint_id.rsplit(marker, 1)
    if not base or not INSTANCE_SUFFIX_RE.match(suffix):
        raise ValueError("invalid instance suffix")
    return base, suffix


def _apply_load_injections(
    keypoint_id: str,
    raw_observations: list[dict[str, Any]],
    items: list[InjectionItem],
    cameras: dict[int, Camera],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any] | None]:
    observations = [dict(obs) if isinstance(obs, dict) else obs for obs in raw_observations]
    hits: dict[str, str] = {}
    load_error: dict[str, Any] | None = None
    for item in items:
        if item.fault == "nonfinite_pixel":
            target_index = 0
            if item.view_index is not None:
                target_index = -1
                for idx, obs in enumerate(observations):
                    if isinstance(obs, dict) and obs.get("view_index") == item.view_index:
                        target_index = idx
                        break
            if target_index < 0 or not observations:
                continue
            if not isinstance(observations[target_index], dict):
                continue
            field_name = item.coordinate_field or "u"
            observations[target_index][field_name] = float("nan")
            observations[target_index]["__injection_id"] = item.id
            hits[item.id] = item.business_code
        elif item.fault == "nonorthogonal_rotation":
            view_index = item.view_index
            if view_index is None or view_index not in cameras:
                continue
            skew = float(item.value) if item.value is not None else 0.25
            camera = cameras[view_index]
            corrupted = camera.R.copy()
            corrupted[0, 1] += skew
            gram = corrupted @ corrupted.T
            orthogonal = bool(
                np.allclose(gram, np.eye(3), atol=ORTHOGONAL_TOLERANCE, rtol=0.0)
            )
            if not orthogonal:
                hits[item.id] = item.business_code
                load_error = _failure(
                    E_CAMERA_INVALID,
                    keypoint_id,
                    "load",
                    "injected camera rotation is not orthogonal",
                    view_index=view_index,
                    details={"injection_id": item.id},
                ).to_dict()
    return observations, hits, load_error


def _parse_observations(
    input_dir: Path,
    cfg: SolverConfig,
    cameras: dict[int, Camera],
    injections: InjectionSet,
) -> tuple[dict[str, PointSource], list[Failure]]:
    path = input_dir / cfg.observations_file
    failures: list[Failure] = []
    if not path.exists():
        return {}, [
            _failure(E_OBS_INVALID, None, "load", f"observation manifest not found: {path}")
        ]

    line_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(
                _failure(
                    E_OBS_INVALID,
                    None,
                    "load",
                    f"invalid JSONL at line {line_number}: {exc}",
                    details={"line_number": line_number},
                )
            )
            continue
        if not isinstance(record, dict):
            failures.append(
                _failure(
                    E_OBS_INVALID,
                    None,
                    "load",
                    f"observation line {line_number} is not an object",
                    details={"line_number": line_number},
                )
            )
            continue
        keypoint_id = record.get("keypoint_id", record.get("id"))
        if not isinstance(keypoint_id, str) or not keypoint_id.strip():
            failures.append(
                _failure(
                    E_OBS_INVALID,
                    None,
                    "load",
                    f"observation line {line_number} has no keypoint_id",
                    details={"line_number": line_number},
                )
            )
            continue
        keypoint_id = keypoint_id.strip()
        try:
            base_name, instance_suffix = _split_keypoint_id(keypoint_id)
        except ValueError as exc:
            failures.append(
                _failure(
                    E_OBS_INVALID,
                    keypoint_id,
                    "load",
                    f"ambiguous or malformed keypoint id: {exc}",
                    details={"line_number": line_number},
                )
            )
            continue
        if keypoint_id in seen_ids:
            failures.append(
                _failure(
                    E_OBS_INVALID,
                    keypoint_id,
                    "load",
                    "duplicate keypoint manifest line; lines are never merged",
                    details={"line_number": line_number},
                )
            )
            continue
        raw_observations = record.get("observations", record.get("obs"))
        if not isinstance(raw_observations, list):
            failures.append(
                _failure(
                    E_OBS_INVALID,
                    keypoint_id,
                    "load",
                    "observations must be a list",
                    details={"line_number": line_number},
                )
            )
            continue
        seen_ids.add(keypoint_id)
        line_records.append(
            {
                "keypoint_id": keypoint_id,
                "base_name": base_name,
                "instance_suffix": instance_suffix,
                "line_number": line_number,
                "raw_observations": [dict(v) if isinstance(v, dict) else v for v in raw_observations],
            }
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in line_records:
        groups.setdefault(record["base_name"], []).append(record)
    rejected_ids: set[str] = set()
    for base_name, records in sorted(groups.items()):
        if len(records) <= 1:
            continue
        suffixes = [record["instance_suffix"] for record in records]
        if any(suffix is None for suffix in suffixes) or len(set(suffixes)) != len(suffixes):
            for record in records:
                rejected_ids.add(record["keypoint_id"])
                failures.append(
                    _failure(
                        E_OBS_INVALID,
                        record["keypoint_id"],
                        "load",
                        "same-name keypoints are ambiguous; distinct '@instance-' suffixes are required",
                        details={"line_number": record["line_number"], "base_name": base_name},
                    )
                )

    points: dict[str, PointSource] = {}
    for record in sorted(line_records, key=lambda item: item["keypoint_id"]):
        keypoint_id = record["keypoint_id"]
        if keypoint_id in rejected_ids:
            continue
        raw_observations, injection_hits, load_error = _apply_load_injections(
            keypoint_id,
            record["raw_observations"],
            injections.for_keypoint(keypoint_id),
            cameras,
        )
        point_failures: list[Failure] = []
        valid: list[Observation] = []
        used_views: set[int] = set()
        for obs_index, raw_obs in enumerate(raw_observations):
            if not isinstance(raw_obs, dict):
                point_failures.append(
                    _failure(
                        E_OBS_INVALID,
                        keypoint_id,
                        "load",
                        "observation is not an object",
                        observation_index=obs_index,
                    )
                )
                continue
            view_index = raw_obs.get("view_index", raw_obs.get("view"))
            if isinstance(view_index, bool) or not isinstance(view_index, int):
                point_failures.append(
                    _failure(
                        E_OBS_INVALID,
                        keypoint_id,
                        "load",
                        "observation view_index is not an integer",
                        observation_index=obs_index,
                    )
                )
                continue
            view_index = int(view_index)
            if view_index in used_views:
                point_failures.append(
                    _failure(
                        E_OBS_INVALID,
                        keypoint_id,
                        "load",
                        "duplicate view_index within keypoint",
                        view_index=view_index,
                        observation_index=obs_index,
                    )
                )
                continue
            u_raw = raw_obs.get("u", raw_obs.get("x"))
            v_raw = raw_obs.get("v", raw_obs.get("y"))
            if (
                isinstance(u_raw, bool)
                or isinstance(v_raw, bool)
                or not isinstance(u_raw, (int, float))
                or not isinstance(v_raw, (int, float))
            ):
                point_failures.append(
                    _failure(
                        E_OBS_INVALID,
                        keypoint_id,
                        "load",
                        "pixel coordinates are not numeric",
                        view_index=view_index,
                        observation_index=obs_index,
                    )
                )
                continue
            u = float(u_raw)
            v = float(v_raw)
            if not math.isfinite(u) or not math.isfinite(v):
                point_failures.append(
                    _failure(
                        E_OBS_INVALID,
                        keypoint_id,
                        "load",
                        "pixel coordinates are non-finite",
                        view_index=view_index,
                        observation_index=obs_index,
                        details={
                            "injection_id": raw_obs.get("__injection_id"),
                        }
                        if raw_obs.get("__injection_id")
                        else {},
                    )
                )
                continue
            camera = cameras.get(view_index)
            if camera is None:
                point_failures.append(
                    _failure(
                        E_OBS_INVALID,
                        keypoint_id,
                        "load",
                        "observation references an unavailable view",
                        view_index=view_index,
                        observation_index=obs_index,
                    )
                )
                continue
            if not (0.0 <= u < float(camera.width) and 0.0 <= v < float(camera.height)):
                point_failures.append(
                    _failure(
                        E_OBS_INVALID,
                        keypoint_id,
                        "load",
                        "pixel coordinate is outside the image bounds",
                        view_index=view_index,
                        observation_index=obs_index,
                        details={
                            "u": u,
                            "v": v,
                            "width": camera.width,
                            "height": camera.height,
                        },
                    )
                )
                continue
            used_views.add(view_index)
            valid.append(Observation(keypoint_id=keypoint_id, view_index=view_index, u=u, v=v))
        valid.sort(key=lambda obs: obs.view_index)
        if len(valid) < cfg.min_views:
            point_failures.append(
                _failure(
                    E_OBS_INVALID,
                    keypoint_id,
                    "load",
                    f"only {len(valid)} valid observations; at least {cfg.min_views} are required",
                    details={"valid_observations": len(valid)},
                )
            )
        points[keypoint_id] = PointSource(
            keypoint_id=keypoint_id,
            line_number=record["line_number"],
            base_name=record["base_name"],
            instance_suffix=record["instance_suffix"],
            raw_observations=record["raw_observations"],
            observations=valid,
            failures=point_failures,
            injection_ids=[item.id for item in injections.for_keypoint(keypoint_id)],
            injection_hits=injection_hits,
            load_error=load_error,
        )
    return points, failures


def load_inputs(
    input_dir: str | Path,
    cfg: SolverConfig,
    *,
    config_path: str | Path | None = None,
    injections: InjectionSet | None = None,
) -> InputBundle:
    """Load and validate all inputs.

    Invalid observations are removed and reported; they do not abort the
    remaining keypoints.  Invalid cameras make observations that reference them
    invalid, while unrelated keypoints remain processable.
    """

    root = Path(input_dir)
    if not root.exists() or not root.is_dir():
        raise ConfigError(f"input directory does not exist: {root}")
    injection_set = injections or InjectionSet()
    image_dir = root / cfg.image_dir
    image_files = sorted(image_dir.rglob("*.png")) if image_dir.exists() else sorted(root.rglob("*.png"))
    cameras, grays, camera_failures, resolution, camera_hash = _load_cameras(
        root, cfg, image_files
    )
    points, observation_failures = _parse_observations(root, cfg, cameras, injection_set)
    failures = camera_failures + observation_failures

    version_path = root / cfg.version_file
    version_sha = ""
    input_versions: dict[str, Any] = {}
    if version_path.exists():
        version_sha = sha256_file(version_path)
        try:
            loaded_versions = json.loads(version_path.read_text(encoding="utf-8"))
            if isinstance(loaded_versions, dict):
                input_versions = loaded_versions
            else:
                failures.append(
                    _failure(
                        E_CFG_INVALID,
                        None,
                        "load",
                        "input version file must contain a JSON object",
                    )
                )
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(
                _failure(
                    E_CFG_INVALID,
                    None,
                    "load",
                    f"cannot parse input version file: {exc}",
                )
            )
    else:
        failures.append(
            _failure(
                E_CFG_INVALID,
                None,
                "load",
                f"input version file is missing: {cfg.version_file}",
            )
        )

    obs_path = root / cfg.observations_file
    observations_sha = sha256_file(obs_path) if obs_path.exists() else ""
    config_sha = sha256_file(config_path) if config_path is not None and Path(config_path).exists() else cfg.fingerprint

    return InputBundle(
        input_dir=root,
        cameras=cameras,
        points=points,
        global_failures=failures,
        input_versions=input_versions,
        observations_sha256=observations_sha,
        version_file_sha256=version_sha,
        config_sha256=config_sha,
        camera_hash=camera_hash,
        image_resolution=resolution,
        injection_report=injection_set.initial_report(),
        grayscale_images=grays,
    )
