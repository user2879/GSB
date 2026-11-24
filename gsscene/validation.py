"""Input loading and strict validation.

All validation problems are surfaced as business-coded failures.  A bad item is
excluded from the runnable set but is never silently dropped.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import codes
from .config import SceneConfig, validate_config_dict
from .models import Camera, Failure, LoadedProject, ViewRecord
from .util import atomic_write_json, file_fingerprint, hash_value, read_json

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
VIEW_STEM = re.compile(r"^view_(\d+)$", re.IGNORECASE)


def _failure(
    code: str,
    message: str,
    source: str,
    *,
    view_index: int | None = None,
    level: int | None = None,
    injection_id: str | None = None,
) -> Failure:
    return Failure(
        code=code,
        message=message,
        source=source,
        view_index=view_index,
        level=level,
        injection_id=injection_id,
    )


def _load_json_file(path: Path, source: str, code: str) -> tuple[Any | None, list[Failure]]:
    try:
        return read_json(path), []
    except FileNotFoundError:
        return None, [_failure(code, f"missing file: {path}", source)]
    except (OSError, json.JSONDecodeError) as exc:
        return None, [_failure(code, f"cannot parse {path.name}: {exc}", source)]


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _injection_lookup(hits: list[dict[str, Any]], kind: str, view_index: int | None = None, level: int | None = None) -> str | None:
    for hit in hits:
        if hit.get("kind") != kind:
            continue
        if view_index is not None and hit.get("view_index") not in (None, view_index):
            continue
        if level is not None and hit.get("level") not in (None, level):
            continue
        return str(hit.get("injection_id"))
    return None


def load_injections(sample_dir: Path, injections_path: Path | None = None) -> tuple[list[dict[str, Any]], list[Failure], list[dict[str, Any]]]:
    path = injections_path or (sample_dir / "injections.json")
    if not path.exists():
        return [], [], []
    value, failures = _load_json_file(path, "injection", codes.E_INJECTION_SPEC)
    if value is None:
        return [], failures, []
    if not isinstance(value, list):
        failures.append(_failure(codes.E_INJECTION_SPEC, "injections.json root must be a list", "injection"))
        return [], failures, []
    parsed: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            failures.append(_failure(codes.E_INJECTION_SPEC, f"injection #{index} must be an object", "injection"))
            continue
        injection_id = str(item.get("injection_id", f"INJ-{index + 1:03d}"))
        kind = item.get("kind")
        if injection_id in seen_ids:
            failures.append(_failure(codes.E_INJECTION_SPEC, f"duplicate injection id {injection_id}", "injection", injection_id=injection_id))
            continue
        seen_ids.add(injection_id)
        if kind not in codes.INJECTION_CODE_MAP:
            failures.append(_failure(codes.E_INJECTION_SPEC, f"unknown injection kind {kind!r}", "injection", injection_id=injection_id))
            continue
        normalized = dict(item)
        normalized["injection_id"] = injection_id
        parsed.append(normalized)
        hit = {
            "injection_id": injection_id,
            "kind": kind,
            "level": item.get("level"),
            "mapped_code": codes.INJECTION_CODE_MAP[kind],
            "view_index": item.get("view_index"),
        }
        hits.append(hit)

    marker_path = sample_dir / "injection_hits.json"
    if marker_path.exists():
        try:
            marker = read_json(marker_path)
            if isinstance(marker, list):
                by_id = {str(item.get("injection_id")): item for item in marker if isinstance(item, dict)}
                for hit in hits:
                    marker_item = by_id.get(hit["injection_id"])
                    if marker_item:
                        hit.update({key: value for key, value in marker_item.items() if key not in hit or hit[key] is None})
        except (OSError, json.JSONDecodeError):
            pass
    return parsed, failures, hits


def _camera_from_dict(value: Any, fallback_index: int) -> tuple[Camera | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(value, dict):
        return None, [_failure(codes.E_CAMERA_SHAPE, f"camera #{fallback_index} must be an object", "camera")]
    view_raw = value.get("view_index", fallback_index)
    view_index = _as_int(view_raw)
    if view_index is None or view_index < 0:
        return None, [_failure(codes.E_VIEW_SEQUENCE, f"invalid view_index {view_raw!r}", "camera")]
    try:
        intrinsic = np.asarray(value["intrinsic"], dtype=np.float64)
        rotation = np.asarray(value["rotation"], dtype=np.float64)
        translation = np.asarray(value["translation"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        return None, [_failure(codes.E_CAMERA_SHAPE, f"camera {view_index} has invalid fields: {exc}", "camera", view_index=view_index)]
    if intrinsic.shape != (3, 3):
        failures.append(_failure(codes.E_CAMERA_SHAPE, "intrinsic must be 3x3", "camera", view_index=view_index))
    if rotation.shape != (3, 3):
        failures.append(_failure(codes.E_CAMERA_SHAPE, "rotation must be 3x3", "camera", view_index=view_index))
    if translation.shape != (3,):
        failures.append(_failure(codes.E_CAMERA_SHAPE, "translation must have shape (3,)", "camera", view_index=view_index))
    if failures:
        return None, failures
    if not all(np.all(np.isfinite(item)) for item in (intrinsic, rotation, translation)):
        failures.append(_failure(codes.E_CAMERA_SHAPE, "camera contains non-finite values", "camera", view_index=view_index))
    if abs(float(np.linalg.det(intrinsic))) <= 1e-9:
        failures.append(_failure(codes.E_CAMERA_INTRINSIC, "intrinsic matrix is singular", "camera", view_index=view_index))
    ortho = rotation.T @ rotation
    if not np.allclose(ortho, np.eye(3), atol=1e-6, rtol=0.0) or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6:
        failures.append(_failure(codes.E_CAMERA_ROTATION, "rotation is not a proper orthogonal matrix", "camera", view_index=view_index))
    if not np.all(np.isfinite(translation)):
        failures.append(_failure(codes.E_CAMERA_TRANSLATION, "translation is not finite", "camera", view_index=view_index))
    if failures:
        return None, failures
    return Camera(view_index=view_index, intrinsic=intrinsic, rotation=rotation, translation=translation), []


def load_cameras(
    sample_dir: Path,
    hits: list[dict[str, Any]],
) -> tuple[dict[int, Camera], list[Failure], set[int], set[int]]:
    path = sample_dir / "cameras.json"
    value, failures = _load_json_file(path, "camera", codes.E_CAMERA_JSON)
    if value is None:
        return {}, failures, set(), set()
    if isinstance(value, dict):
        value = value.get("cameras")
    if not isinstance(value, list):
        failures.append(_failure(codes.E_CAMERA_JSON, "cameras.json root must be a list", "camera"))
        return {}, failures, set(), set()
    cameras: dict[int, Camera] = {}
    duplicate_ids: set[int] = set()
    all_ids: set[int] = set()
    for index, item in enumerate(value):
        camera, item_failures = _camera_from_dict(item, index)
        failures.extend(item_failures)
        if camera is None:
            raw_id = _as_int(item.get("view_index")) if isinstance(item, dict) else None
            if raw_id is not None:
                all_ids.add(raw_id)
                duplicate_ids.add(raw_id)
            continue
        all_ids.add(camera.view_index)
        if camera.view_index in cameras:
            duplicate_ids.add(camera.view_index)
            failures.append(
                _failure(codes.E_VIEW_DUPLICATE, f"duplicate camera view_index {camera.view_index}", "camera", view_index=camera.view_index)
            )
            continue
        cameras[camera.view_index] = camera
    for view_id in sorted(duplicate_ids):
        cameras.pop(view_id, None)
    return cameras, failures, all_ids, duplicate_ids


def _truncate_injection_for(hits: list[dict[str, Any]], view_index: int) -> str | None:
    return _injection_lookup(hits, "truncate_image", view_index=view_index)


def _camera_injection_for(hits: list[dict[str, Any]], view_index: int) -> str | None:
    return _injection_lookup(hits, "missing_camera", view_index=view_index)


def _scan_images(sample_dir: Path) -> tuple[dict[int, Path], list[Failure]]:
    images_dir = sample_dir / "images"
    failures: list[Failure] = []
    found: dict[int, Path] = {}
    if not images_dir.exists():
        return {}, [_failure(codes.E_VIEW_IMAGE_ORPHAN, f"missing images directory: {images_dir}", "view")]
    for path in sorted(images_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = VIEW_STEM.match(path.stem)
        if match is None:
            failures.append(_failure(codes.E_VIEW_SEQUENCE, f"image filename does not contain a view index: {path.name}", "view"))
            continue
        view_index = int(match.group(1))
        if view_index in found:
            failures.append(
                _failure(codes.E_VIEW_DUPLICATE, f"multiple images for view {view_index}", "view", view_index=view_index)
            )
            continue
        found[view_index] = path
    return found, failures


def _validate_image(
    path: Path,
    config: SceneConfig,
    view_index: int,
    injection_id: str | None,
) -> tuple[tuple[int, int] | None, list[Failure]]:
    failures: list[Failure] = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, [_failure(codes.E_IMAGE_HEADER, f"cannot read image {path.name}: {exc}", "image", view_index=view_index, injection_id=injection_id)]
    if len(raw) < 24:
        return None, [_failure(codes.E_IMAGE_TRUNCATED, f"image {path.name} is too short", "image", view_index=view_index, injection_id=injection_id)]
    suffix = path.suffix.lower()
    if suffix == ".png" and raw[:8] != b"\x89PNG\r\n\x1a\n":
        return None, [_failure(codes.E_IMAGE_HEADER, f"invalid PNG header: {path.name}", "image", view_index=view_index, injection_id=injection_id)]
    try:
        with Image.open(path) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        code = codes.E_IMAGE_TRUNCATED if injection_id or len(raw) < 128 else codes.E_IMAGE_DECODE
        return None, [_failure(code, f"image container verification failed for {path.name}: {exc}", "image", view_index=view_index, injection_id=injection_id)]
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != config.image_size:
                failures.append(
                    _failure(
                        codes.E_IMAGE_RESOLUTION,
                        f"expected {config.image_size}, got {image.size}",
                        "image",
                        view_index=view_index,
                        injection_id=injection_id,
                    )
                )
            if image.mode != config.pixel_mode:
                failures.append(
                    _failure(
                        codes.E_IMAGE_MODE,
                        f"expected pixel mode {config.pixel_mode}, got {image.mode}",
                        "image",
                        view_index=view_index,
                        injection_id=injection_id,
                    )
                )
            return image.size, failures
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, [_failure(codes.E_IMAGE_DECODE, f"cannot decode image {path.name}: {exc}", "image", view_index=view_index, injection_id=injection_id)]


def _load_depth(path: Path, image_size: tuple[int, int], view_index: int, injection_id: str | None) -> tuple[np.ndarray | None, list[Failure]]:
    if not path.exists():
        return None, [_failure(codes.E_DEPTH_MISSING, f"depth map missing: {path.name}", "depth", view_index=view_index, injection_id=injection_id)]
    try:
        depth = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        return None, [_failure(codes.E_DEPTH_SHAPE, f"cannot load depth map {path.name}: {exc}", "depth", view_index=view_index, injection_id=injection_id)]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape != image_size:
        return None, [_failure(codes.E_DEPTH_SHAPE, f"depth shape {depth.shape} != {image_size}", "depth", view_index=view_index, injection_id=injection_id)]
    depth = np.asarray(depth, dtype=np.float64)
    if not np.all(np.isfinite(depth)):
        return None, [_failure(codes.E_DEPTH_VALUE, "depth map contains non-finite values", "depth", view_index=view_index, injection_id=injection_id)]
    return depth, []


def load_views(
    sample_dir: Path,
    config: SceneConfig,
    cameras: dict[int, Camera],
    hits: list[dict[str, Any]],
) -> tuple[dict[int, ViewRecord], list[Failure], set[int], set[int], dict[int, Path]]:
    image_files, failures = _scan_images(sample_dir)
    views: dict[int, ViewRecord] = {}
    excluded: set[int] = set()
    duplicate_ids: set[int] = set()
    camera_ids = set(cameras)
    for view_index in sorted(set(image_files) | camera_ids):
        image_path = image_files.get(view_index)
        if image_path is None:
            failures.append(
                _failure(codes.E_VIEW_CAMERA_ORPHAN, f"camera exists without image for view {view_index}", "view", view_index=view_index)
            )
            excluded.add(view_index)
            continue
        camera = cameras.get(view_index)
        if camera is None:
            injection_id = _camera_injection_for(hits, view_index)
            failures.append(
                _failure(
                    codes.E_CAMERA_MISSING,
                    f"camera missing for image view {view_index}",
                    "camera",
                    view_index=view_index,
                    injection_id=injection_id,
                )
            )
            excluded.add(view_index)
            continue
        image_injection = _truncate_injection_for(hits, view_index)
        size, image_failures = _validate_image(image_path, config, view_index, image_injection)
        failures.extend(image_failures)
        if size is None or image_failures:
            excluded.add(view_index)
            continue
        depth_path = sample_dir / "depth" / f"view_{view_index:04d}.npy"
        if not depth_path.exists():
            alt = image_path.with_suffix(".npy")
            if alt.exists():
                depth_path = alt
        depth, depth_failures = _load_depth(depth_path, config.image_size, view_index, image_injection)
        failures.extend(depth_failures)
        if depth is None or depth_failures:
            excluded.add(view_index)
            continue
        views[view_index] = ViewRecord(
            view_index=view_index,
            image_path=image_path,
            depth_path=depth_path,
            camera=camera,
            image_fingerprint=file_fingerprint(image_path),
            depth_fingerprint=file_fingerprint(depth_path),
            image_size=config.image_size,
            injection_id=image_injection,
        )
    if not views and not failures:
        failures.append(_failure(codes.E_VIEW_IMAGE_ORPHAN, "no view images found", "view"))
    return views, failures, excluded, duplicate_ids, image_files


def load_edits(
    sample_dir: Path,
    config: SceneConfig,
    initial_views: set[int],
) -> tuple[list[dict[str, Any]], list[Failure]]:
    path = sample_dir / "edits.jsonl"
    if not path.exists():
        return [], []
    edits: list[dict[str, Any]] = []
    failures: list[Failure] = []
    seen_seq: set[int] = set()
    active_views = set(initial_views)
    basis_count = (config.sh_degree + 1) ** 2
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [_failure(codes.E_EDIT_JSON, f"cannot read edit plan: {exc}", "edit")]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            edit = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(_failure(codes.E_EDIT_JSON, f"line {line_number}: {exc}", "edit"))
            continue
        if not isinstance(edit, dict):
            failures.append(_failure(codes.E_EDIT_JSON, f"line {line_number}: edit must be an object", "edit"))
            continue
        seq = _as_int(edit.get("seq", line_number - 1))
        edit_type = edit.get("type")
        params = edit.get("params", {})
        if seq is None or seq < 0:
            failures.append(_failure(codes.E_EDIT_SEQUENCE, f"line {line_number}: invalid seq", "edit"))
            continue
        if seq in seen_seq:
            failures.append(_failure(codes.E_EDIT_SEQUENCE, f"duplicate edit seq {seq}", "edit"))
            continue
        seen_seq.add(seq)
        if edit_type not in codes.EDIT_TYPES:
            failures.append(_failure(codes.E_EDIT_TYPE, f"seq {seq}: unsupported edit type {edit_type!r}", "edit"))
            continue
        if not isinstance(params, dict):
            failures.append(_failure(codes.E_EDIT_JSON, f"seq {seq}: params must be an object", "edit"))
            continue
        view_index = _as_int(params.get("view_index"))
        if edit_type == "view_add":
            if view_index is None or view_index < 0:
                failures.append(_failure(codes.E_EDIT_VIEW_MISSING, f"seq {seq}: view_add requires a non-negative view_index", "edit"))
            elif view_index in active_views:
                failures.append(_failure(codes.E_EDIT_VIEW_DUPLICATE, f"seq {seq}: view {view_index} already exists", "edit", view_index=view_index))
            else:
                active_views.add(view_index)
        elif edit_type == "view_remove":
            if view_index is None or view_index not in active_views:
                failures.append(_failure(codes.E_EDIT_VIEW_MISSING, f"seq {seq}: view {view_index} does not exist", "edit", view_index=view_index))
            elif view_index is not None:
                active_views.remove(view_index)
        else:
            if view_index is not None and view_index not in active_views:
                failures.append(_failure(codes.E_EDIT_VIEW_MISSING, f"seq {seq}: view {view_index} does not exist", "edit", view_index=view_index))
            if edit_type in {"view_replace", "camera_tweak", "sh_modify"} and view_index is None:
                failures.append(_failure(codes.E_EDIT_VIEW_MISSING, f"seq {seq}: {edit_type} requires view_index", "edit"))
        if edit_type in {"level_scale", "level_density"}:
            level = _as_int(params.get("level"))
            if level is None or not 0 <= level < config.levels:
                failures.append(_failure(codes.E_EDIT_LEVEL, f"seq {seq}: invalid level {params.get('level')!r}", "edit", level=level))
            if edit_type == "level_scale" and (not isinstance(params.get("value"), (int, float)) or float(params.get("value")) <= 0):
                failures.append(_failure(codes.E_EDIT_LEVEL, f"seq {seq}: level_scale value must be positive", "edit", level=level))
            if edit_type == "level_density" and (not isinstance(params.get("value"), (int, float)) or float(params.get("value")) <= 0):
                failures.append(_failure(codes.E_EDIT_DENSITY, f"seq {seq}: level_density value must be positive", "edit", level=level))
        if edit_type == "sh_modify":
            level_values = params.get("levels", [])
            if not isinstance(level_values, list):
                failures.append(_failure(codes.E_EDIT_LEVEL, f"seq {seq}: sh_modify levels must be a list", "edit"))
            else:
                for raw_level in level_values:
                    level = _as_int(raw_level)
                    if level is None or not 0 <= level < config.levels:
                        failures.append(_failure(codes.E_EDIT_LEVEL, f"seq {seq}: invalid sh level {raw_level!r}", "edit", level=level))
            coefficient = _as_int(params.get("coefficient"))
            channel = _as_int(params.get("channel"))
            kernel_index = _as_int(params.get("kernel_index"))
            if coefficient is None or not 0 <= coefficient < basis_count:
                failures.append(_failure(codes.E_EDIT_SH_COEFF, f"seq {seq}: coefficient outside SH basis", "edit"))
            if channel is None or not 0 <= channel < 3:
                failures.append(_failure(codes.E_EDIT_SH_COEFF, f"seq {seq}: channel must be 0..2", "edit"))
            if kernel_index is None or kernel_index < 0:
                failures.append(_failure(codes.E_EDIT_SH_COEFF, f"seq {seq}: kernel_index must be non-negative", "edit"))
            if not isinstance(params.get("delta"), (int, float)):
                failures.append(_failure(codes.E_EDIT_SH_COEFF, f"seq {seq}: sh delta must be numeric", "edit"))
        if edit_type == "quant_switch":
            frac_bits = _as_int(params.get("frac_bits", config.frac_bits))
            if frac_bits not in codes.ALLOWED_FRAC_BITS:
                failures.append(_failure(codes.E_EDIT_QUANT, f"seq {seq}: invalid frac_bits", "edit"))
            if params.get("quant_gear", config.quant_gear) not in codes.QUANT_GEARS:
                failures.append(_failure(codes.E_EDIT_QUANT, f"seq {seq}: invalid quant_gear", "edit"))
        if edit_type in {"view_replace", "view_add"}:
            asset_dir = params.get("asset_dir")
            if not isinstance(asset_dir, str) or not asset_dir:
                failures.append(_failure(codes.E_ASSET_MISSING, f"seq {seq}: asset_dir is required", "asset", view_index=view_index))
            else:
                target = (sample_dir / asset_dir).resolve()
                try:
                    target.relative_to(sample_dir.resolve())
                except ValueError:
                    failures.append(_failure(codes.E_ASSET_MISSING, f"seq {seq}: asset_dir escapes sample root", "asset", view_index=view_index))
                else:
                    if not target.is_dir():
                        failures.append(_failure(codes.E_ASSET_MISSING, f"seq {seq}: asset_dir not found: {asset_dir}", "asset", view_index=view_index))
        normalized = {"params": dict(params), "seq": int(seq), "type": str(edit_type)}
        edits.append(normalized)
    edits.sort(key=lambda item: int(item["seq"]))
    return edits, failures


def load_project(sample_dir: Path, injections_path: Path | None = None) -> LoadedProject:
    sample_dir = Path(sample_dir).resolve()
    failures: list[Failure] = []
    injections, injection_failures, hits = load_injections(sample_dir, injections_path)
    failures.extend(injection_failures)
    injection_marker = sample_dir / "injection_hits.json"
    active_hits = hits if injection_marker.exists() else []

    raw_config, config_failures = _load_json_file(sample_dir / "config.json", "config", codes.E_CONFIG_JSON)
    config: SceneConfig | None = None
    if raw_config is not None:
        config, config_failures = validate_config_dict(raw_config)
        failures.extend(config_failures)
    else:
        failures.extend(config_failures)

    zero_decay_injection = _injection_lookup(active_hits, "zero_scale_decay")
    quant_injection = _injection_lookup(active_hits, "quant_gear_out_of_range")
    for index, failure in enumerate(failures):
        if failure.injection_id is not None:
            continue
        if failure.code == codes.E_CONFIG_SCALE_DECAY and zero_decay_injection:
            replacements = list(failures)
            replacements[index] = Failure(
                code=failure.code,
                message=failure.message,
                source=failure.source,
                view_index=failure.view_index,
                level=failure.level,
                injection_id=zero_decay_injection,
                severity=failure.severity,
            )
            failures = replacements
        elif failure.code == codes.E_CONFIG_QUANT_GEAR and quant_injection:
            replacements = list(failures)
            replacements[index] = Failure(
                code=failure.code,
                message=failure.message,
                source=failure.source,
                view_index=failure.view_index,
                level=failure.level,
                injection_id=quant_injection,
                severity=failure.severity,
            )
            failures = replacements

    cameras, camera_failures, _camera_ids, camera_duplicates = load_cameras(sample_dir, active_hits)
    failures.extend(camera_failures)
    views: dict[int, ViewRecord] = {}
    excluded_views: set[int] = set(camera_duplicates)
    excluded_levels: set[tuple[int, int]] = set()
    image_files: dict[int, Path] = {}
    edits: list[dict[str, Any]] = []
    if config is not None:
        views, view_failures, view_excluded, image_duplicates, image_files = load_views(sample_dir, config, cameras, active_hits)
        failures.extend(view_failures)
        excluded_views |= view_excluded | image_duplicates
        edits, edit_failures = load_edits(sample_dir, config, set(views))
        failures.extend(edit_failures)

        for hit in active_hits:
            if hit.get("kind") != "density_over_budget":
                continue
            level = _as_int(hit.get("level"))
            view_index = _as_int(hit.get("view_index"))
            if level is None or not 0 <= level < config.levels:
                continue
            from .config import kernel_grid_size

            grid_w, grid_h = kernel_grid_size(config, level)
            requested = grid_w * grid_h
            budget = config.kernel_budgets[level]
            if requested > budget:
                failures.append(
                    _failure(
                        codes.E_INJECTION_DENSITY_OVER_BUDGET,
                        f"injection {hit.get('injection_id')} requested {requested} kernels for level {level} budget {budget}",
                        "injection",
                        view_index=view_index,
                        level=level,
                        injection_id=str(hit.get("injection_id")),
                    )
                )
                if view_index is not None:
                    excluded_levels.add((view_index, level))
                else:
                    for view_id in views:
                        excluded_levels.add((view_id, level))
    else:
        _images, image_failures = _scan_images(sample_dir)
        failures.extend(image_failures)

    failures = sorted(_dedupe_failures(failures), key=lambda failure: failure.sort_key())
    return LoadedProject(
        sample_dir=sample_dir,
        config=config,
        views=views,
        edits=edits,
        injections=injections,
        failures=failures,
        injection_hits=active_hits,
        excluded_views=excluded_views,
        excluded_levels=excluded_levels,
        image_files=image_files,
    )


def _dedupe_failures(failures: list[Failure]) -> list[Failure]:
    result: list[Failure] = []
    seen: set[tuple[Any, ...]] = set()
    for failure in failures:
        key = (
            failure.code,
            failure.source,
            failure.view_index,
            failure.level,
            failure.injection_id,
            failure.message,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(failure)
    return result


def materialize_injected_sample(
    sample_dir: Path,
    out_dir: Path,
    injections_path: Path | None = None,
    selected_injection_ids: set[str] | None = None,
) -> Path:
    """Create an isolated copy with real on-disk input faults and a hit marker."""

    sample_dir = Path(sample_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(sample_dir, out_dir)
    injections, failures, hits = load_injections(sample_dir, injections_path)
    if failures:
        raise ValueError("; ".join(item.message for item in failures))
    if selected_injection_ids is not None:
        injections = [item for item in injections if str(item["injection_id"]) in selected_injection_ids]
        hits = [item for item in hits if str(item["injection_id"]) in selected_injection_ids]
    config_path = out_dir / "config.json"
    config = read_json(config_path)
    cameras_path = out_dir / "cameras.json"
    cameras = read_json(cameras_path) if cameras_path.exists() else []
    if isinstance(cameras, dict):
        cameras = cameras.get("cameras", [])
    for injection in injections:
        kind = injection["kind"]
        injection_id = str(injection["injection_id"])
        if kind == "truncate_image":
            view_index = int(injection["view_index"])
            image_path = out_dir / "images" / f"view_{view_index:04d}.png"
            raw = image_path.read_bytes()
            image_path.write_bytes(raw[: max(16, len(raw) // 3)])
        elif kind == "missing_camera":
            view_index = int(injection["view_index"])
            cameras = [item for item in cameras if int(item.get("view_index", -1)) != view_index]
        elif kind == "zero_scale_decay":
            level = injection.get("level")
            decay = config.get("scale_decay", 0.6)
            if isinstance(decay, list):
                if level is None:
                    config["scale_decay"] = [0.0 for _ in decay]
                else:
                    values = list(decay)
                    values[int(level)] = 0.0
                    config["scale_decay"] = values
            else:
                config["scale_decay"] = 0.0
        elif kind == "density_over_budget":
            level = int(injection["level"])
            density = float(injection.get("params", {}).get("density", 9000.0))
            values = list(config["level_density"])
            values[level] = density
            config["level_density"] = values
        elif kind == "quant_gear_out_of_range":
            config["quant_gear"] = str(injection.get("params", {}).get("gear", "BOGUS_GEAR"))
    atomic_write_json(config_path, config)
    atomic_write_json(cameras_path, cameras)
    atomic_write_json(out_dir / "injection_hits.json", hits)
    atomic_write_json(out_dir / "injections.json", injections)
    return out_dir
