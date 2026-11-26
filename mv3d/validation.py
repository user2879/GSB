"""Project, image, camera, and reconstruction-config validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .constants import (
    ALLOWED_FIXED_BITS,
    ALLOWED_QUANT_LEVELS,
    SCHEMA_VERSION,
    STRATEGY_TABLE,
)
from .errors import Failure, sorted_failures
from .fixed import bytes_hash, stable_hash
from .io import png_header, png_pixel_mode
from .models import (
    ImageSpec,
    ProjectState,
    ReconstructionConfig,
    ValidatedView,
    intrinsic_full_rank,
    parse_camera,
    rotation_is_orthogonal,
)


@dataclass
class ValidationResult:
    state: ProjectState
    image_spec: ImageSpec
    config: ReconstructionConfig | None
    views: dict[int, ValidatedView]
    pairs: tuple[tuple[int, int], ...]
    failures: list[Failure] = field(default_factory=list)
    runnable: bool = True

    @property
    def valid_view_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.views))


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _config_validation(
    state: ProjectState,
) -> tuple[ImageSpec, ReconstructionConfig | None, list[Failure]]:
    failures: list[Failure] = []
    raw_image = state.data.get("image")
    if not isinstance(raw_image, dict):
        failures.append(Failure("E_PROJECT_SCHEMA_INVALID", artifact="image"))
        raw_image = {}
    width = _as_int(raw_image.get("width"))
    height = _as_int(raw_image.get("height"))
    mode = raw_image.get("mode")
    if width is None or height is None or width <= 0 or height <= 0:
        failures.append(Failure("E_IMAGE_RESOLUTION_INVALID", artifact="image"))
        width = max(1, width or 1)
        height = max(1, height or 1)
    if mode != "RGB":
        failures.append(Failure("E_IMAGE_PIXEL_MODE_INVALID", artifact="image"))
    image_spec = ImageSpec(width=width, height=height, mode=str(mode))

    fixed = state.data.get("fixed")
    recon = state.data.get("reconstruction")
    if not isinstance(fixed, dict):
        failures.append(Failure("E_PROJECT_SCHEMA_INVALID", artifact="fixed"))
        fixed = {}
    if not isinstance(recon, dict):
        failures.append(Failure("E_PROJECT_SCHEMA_INVALID", artifact="reconstruction"))
        recon = {}
    bits = _as_int(fixed.get("bits"))
    quant = _as_int(fixed.get("quant_level"))
    bits_ok = bits is not None and bits in ALLOWED_FIXED_BITS
    quant_ok = quant is not None and quant in ALLOWED_QUANT_LEVELS
    if not bits_ok:
        failures.append(Failure("E_FIXED_BITS_INVALID", artifact="fixed.bits"))
    if not quant_ok:
        failures.append(Failure("E_QUANT_LEVEL_INVALID", artifact="fixed.quant_level"))

    min_disparity = _as_int(recon.get("min_disparity"))
    voxel_size = _as_int(recon.get("voxel_size"))
    if min_disparity is None or min_disparity <= 0:
        failures.append(Failure("E_MIN_DISPARITY_INVALID", artifact="reconstruction.min_disparity"))
    if voxel_size is None or voxel_size <= 0:
        failures.append(Failure("E_VOXEL_SIZE_INVALID", artifact="reconstruction.voxel_size"))

    residual = _as_int(recon.get("residual_threshold"))
    max_depth = _as_int(recon.get("max_subdiv_depth"))
    max_children = _as_int(recon.get("max_voxels_per_parent"))
    budget = _as_int(recon.get("global_point_budget"))
    quota = _as_int(recon.get("view_point_quota"))
    workers = _as_int(recon.get("workers"))
    tolerance = _as_int(recon.get("orthogonality_tolerance_ppm"))
    raw_bounds = recon.get("world_bounds")
    bounds = tuple(_as_int(value) for value in raw_bounds) if isinstance(raw_bounds, list) else ()
    if residual is None or residual <= 0:
        failures.append(Failure("E_RECONSTRUCTION_PARAM_INVALID", artifact="reconstruction.residual_threshold"))
        residual = 1
    if max_depth is None or max_depth < 0:
        failures.append(Failure("E_RECONSTRUCTION_PARAM_INVALID", artifact="reconstruction.max_subdiv_depth"))
        max_depth = 0
    if max_children is None or max_children < 1 or max_children > 8:
        failures.append(Failure("E_RECONSTRUCTION_PARAM_INVALID", artifact="reconstruction.max_voxels_per_parent"))
        max_children = max(1, min(8, max_children or 1))
    if budget is None or budget < 0:
        failures.append(Failure("E_RECONSTRUCTION_PARAM_INVALID", artifact="reconstruction.global_point_budget"))
        budget = 0
    if quota is None or quota <= 0:
        failures.append(Failure("E_RECONSTRUCTION_PARAM_INVALID", artifact="reconstruction.view_point_quota"))
        quota = 1
    if workers is None or workers <= 0:
        failures.append(Failure("E_RECONSTRUCTION_PARAM_INVALID", artifact="reconstruction.workers"))
        workers = 1
    if tolerance is None or tolerance < 0:
        failures.append(Failure("E_RECONSTRUCTION_PARAM_INVALID", artifact="reconstruction.orthogonality_tolerance_ppm"))
        tolerance = 10
    if len(bounds) != 6 or any(value is None for value in bounds):
        failures.append(Failure("E_RECONSTRUCTION_PARAM_INVALID", artifact="reconstruction.world_bounds"))
        bounds = (-32, -32, -32, 32, 32, 32)
    else:
        bounds = tuple(int(value) for value in bounds)
        if bounds[0] >= bounds[3] or bounds[1] >= bounds[4] or bounds[2] >= bounds[5]:
            failures.append(Failure("E_RECONSTRUCTION_PARAM_INVALID", artifact="reconstruction.world_bounds"))
    override = recon.get("strategy_override")
    allowed_strategies = {entry["name"] for entry in STRATEGY_TABLE}
    if override is not None and override not in allowed_strategies:
        failures.append(Failure("E_RECONSTRUCTION_PARAM_INVALID", artifact="reconstruction.strategy_override"))

    if (
        not bits_ok
        or not quant_ok
        or min_disparity is None
        or min_disparity <= 0
        or voxel_size is None
        or voxel_size <= 0
    ):
        return image_spec, None, failures
    config = ReconstructionConfig(
        min_disparity=min_disparity,
        voxel_size=voxel_size,
        residual_threshold=residual,
        max_subdiv_depth=max_depth,
        max_voxels_per_parent=max_children,
        world_bounds=bounds,
        global_point_budget=budget,
        view_point_quota=quota,
        workers=workers,
        bits=bits,
        quant_level=quant,
        strategy_override=override,
        orthogonality_tolerance_ppm=tolerance,
    )
    return image_spec, config, failures


def _paths_for_view(item: dict[str, Any], base_dir: Path) -> Path | None:
    raw = item.get("image")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path)


def validate_project(
    state: ProjectState,
    extra_failures: list[Failure] | None = None,
) -> ValidationResult:
    failures = list(extra_failures or [])
    if state.data.get("schema_version") != SCHEMA_VERSION:
        failures.append(Failure("E_PROJECT_SCHEMA_INVALID", artifact="schema_version"))
    image_spec, config, config_failures = _config_validation(state)
    failures.extend(config_failures)
    raw_views = state.data.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        failures.append(Failure("E_PROJECT_SCHEMA_INVALID", artifact="views"))
        raw_views = []

    seen_ids: dict[int, int] = {}
    seen_images: dict[str, list[int]] = {}
    parsed_views: list[tuple[int, dict[str, Any], Path | None]] = []
    for index, item in enumerate(raw_views):
        if not isinstance(item, dict):
            failures.append(Failure("E_PROJECT_SCHEMA_INVALID", artifact=f"views[{index}]"))
            continue
        view_id = _as_int(item.get("id"))
        if view_id is None or view_id <= 0:
            failures.append(Failure("E_PROJECT_SCHEMA_INVALID", artifact=f"views[{index}].id"))
            continue
        seen_ids[view_id] = seen_ids.get(view_id, 0) + 1
        path = _paths_for_view(item, state.base_dir)
        key = str(path.resolve()) if path is not None else f"missing:{index}"
        seen_images.setdefault(key, []).append(view_id)
        parsed_views.append((view_id, item, path))

    for view_id, count in sorted(seen_ids.items()):
        if count > 1:
            failures.append(Failure("E_VIEW_ID_DUPLICATE", view_id=view_id, artifact=f"v:{view_id:04d}"))
    for path, view_ids in sorted(seen_images.items()):
        if len(view_ids) > 1:
            for view_id in sorted(view_ids):
                failures.append(Failure("E_VIEW_IMAGE_UNIQUE", view_id=view_id, artifact=f"v:{view_id:04d}"))

    validated: dict[int, ValidatedView] = {}
    if config is not None:
        for view_id, item, path in parsed_views:
            if seen_ids[view_id] > 1:
                continue
            if path is None or not path.is_file():
                failures.append(Failure("E_VIEW_IMAGE_MISSING", view_id=view_id, artifact=f"v:{view_id:04d}"))
                continue
            try:
                header = png_header(path)
            except (OSError, ValueError) as error:
                failures.append(
                    Failure("E_PNG_HEADER_INVALID", view_id=view_id, artifact=f"v:{view_id:04d}", detail=str(error))
                )
                continue
            if header["width"] != image_spec.width or header["height"] != image_spec.height:
                failures.append(
                    Failure(
                        "E_IMAGE_RESOLUTION_INVALID",
                        view_id=view_id,
                        artifact=f"v:{view_id:04d}",
                        detail=f"{header['width']}x{header['height']}",
                    )
                )
                continue
            mode = png_pixel_mode(header)
            if mode != image_spec.mode:
                failures.append(
                    Failure(
                        "E_IMAGE_PIXEL_MODE_INVALID",
                        view_id=view_id,
                        artifact=f"v:{view_id:04d}",
                        detail=mode,
                    )
                )
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    if image.size != (image_spec.width, image_spec.height) or image.mode != image_spec.mode:
                        raise ValueError("decoded image does not match header")
            except (OSError, ValueError) as error:
                failures.append(
                    Failure("E_PNG_HEADER_INVALID", view_id=view_id, artifact=f"v:{view_id:04d}", detail=str(error))
                )
                continue
            try:
                camera = parse_camera(item["camera"])
            except (KeyError, TypeError, ValueError) as error:
                failures.append(
                    Failure("E_PROJECT_SCHEMA_INVALID", view_id=view_id, artifact=f"v:{view_id:04d}.camera", detail=str(error))
                )
                continue
            if not intrinsic_full_rank(camera):
                failures.append(
                    Failure("E_INTRINSIC_NONINVERTIBLE", view_id=view_id, artifact=f"v:{view_id:04d}.camera")
                )
                continue
            if not rotation_is_orthogonal(camera, config.orthogonality_tolerance_ppm):
                failures.append(
                    Failure("E_EXTRINSIC_NOT_ORTHOGONAL", view_id=view_id, artifact=f"v:{view_id:04d}.camera")
                )
                continue
            image_bytes = path.read_bytes()
            input_key = stable_hash(
                {
                    "image_sha256": bytes_hash(image_bytes),
                    "image_spec": image_spec.canonical(),
                    "camera": camera.canonical(),
                }
            )
            validated[view_id] = ValidatedView(
                view_id=view_id,
                image_path=path.resolve(),
                image_bytes=image_bytes,
                image_sha256=bytes_hash(image_bytes),
                camera=camera,
                input_key=input_key,
            )

    raw_pairs = state.data.get("pairs", [])
    pairs: list[tuple[int, int]] = []
    if not isinstance(raw_pairs, list):
        failures.append(Failure("E_PAIR_DECLARATION_INVALID", artifact="pairs"))
        raw_pairs = []
    pair_seen: set[tuple[int, int]] = set()
    for index, pair in enumerate(raw_pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            failures.append(Failure("E_PAIR_DECLARATION_INVALID", artifact=f"pairs[{index}]"))
            continue
        first = _as_int(pair[0])
        second = _as_int(pair[1])
        if first is None or second is None or first <= 0 or second <= 0 or first == second:
            failures.append(Failure("E_PAIR_DECLARATION_INVALID", artifact=f"pairs[{index}]"))
            continue
        canonical_pair = tuple(sorted((first, second)))
        if canonical_pair in pair_seen:
            failures.append(
                Failure("E_PAIR_DECLARATION_INVALID", artifact=f"p:{first:04d}:{second:04d}")
            )
            continue
        pair_seen.add(canonical_pair)
        pairs.append(canonical_pair)
        if first not in seen_ids or second not in seen_ids:
            failures.append(
                Failure(
                    "E_PAIR_UNKNOWN_VIEW",
                    artifact=f"p:{min(first, second):04d}:{max(first, second):04d}",
                    detail=f"{first},{second}",
                )
            )
    pairs = sorted(pairs)
    failures = sorted_failures(failures)
    runnable = config is not None and bool(validated) and image_spec.width > 0
    return ValidationResult(
        state=state,
        image_spec=image_spec,
        config=config,
        views=validated,
        pairs=tuple(pairs),
        failures=failures,
        runnable=runnable,
    )

