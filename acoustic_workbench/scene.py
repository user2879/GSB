"""Scene parameter normalization, validation, and version stamps."""

from __future__ import annotations

import copy
import math
from typing import Any

from .config import (
    CHANNEL_COUNT,
    DEFAULT_GRID_HEIGHT,
    DEFAULT_GRID_WIDTH,
    DEFAULT_NOISE_SEED,
    DEFAULT_SAMPLE_DENSITY,
    DEFAULT_SH_ORDER,
    DEFAULT_SOURCES,
    DEFAULT_TILE_COLS,
    DEFAULT_TILE_ROWS,
    DEFAULT_WALL_ALPHAS,
    DEFAULT_ROOM,
    DEFAULT_LISTENER,
    DURATION_SECONDS,
    MAX_SAMPLE_DENSITY,
    MAX_SH_ORDER,
    MIN_SAMPLE_DENSITY,
    PROJECTION_BOUNDS,
    SAMPLE_RATE,
    WALL_IDS,
    sph_order_bound,
)
from .errors import E_GRID_BOUNDS, E_PARAM_RANGE, E_SPH_ORDER, fail
from .hashing import canonical_json_bytes, sha256_bytes


def default_scene(*, revision: int = 1) -> dict[str, Any]:
    return {
        "revision": revision,
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "channels": CHANNEL_COUNT,
        "noise_seed": DEFAULT_NOISE_SEED,
        "room": copy.deepcopy(DEFAULT_ROOM),
        "listener": list(DEFAULT_LISTENER),
        "sources": copy.deepcopy(DEFAULT_SOURCES),
        "walls": copy.deepcopy(DEFAULT_WALL_ALPHAS),
        "receiver": {"sample_density": DEFAULT_SAMPLE_DENSITY},
        "reconstruction": {"order": DEFAULT_SH_ORDER},
        "grid": {
            "width": DEFAULT_GRID_WIDTH,
            "height": DEFAULT_GRID_HEIGHT,
            "tile_rows": DEFAULT_TILE_ROWS,
            "tile_cols": DEFAULT_TILE_COLS,
            "lon_min": PROJECTION_BOUNDS[0],
            "lon_max": PROJECTION_BOUNDS[1],
            "lat_min": PROJECTION_BOUNDS[2],
            "lat_max": PROJECTION_BOUNDS[3],
        },
    }


def initial_versions(scene: dict[str, Any], revision: int = 1) -> dict[str, int]:
    return {key: revision for key in flatten_params(scene)}


def flatten_params(scene: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {
        "noise.seed": scene["noise_seed"],
        "receiver.sample_density": scene["receiver"]["sample_density"],
        "reconstruction.order": scene["reconstruction"]["order"],
    }
    for source_id, position in scene["sources"].items():
        values[f"source.{source_id}.position"] = list(position)
    for wall_id in WALL_IDS:
        values[f"wall.{wall_id}.alpha"] = scene["walls"][wall_id]
    for key in ("width", "height", "tile_rows", "tile_cols", "lon_min", "lon_max", "lat_min", "lat_max"):
        values[f"grid.{key}"] = scene["grid"][key]
    return values


ALLOWED_PATCH_KEYS = frozenset(flatten_params(default_scene()))


def scene_hash(scene: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(scene))


def _position(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        fail(E_PARAM_RANGE, f"{label} must be a three-element position", key=label, value=value)
    try:
        result = [float(part) for part in value]
    except (TypeError, ValueError):
        fail(E_PARAM_RANGE, f"{label} contains a non-numeric coordinate", key=label, value=value)
    if not all(math.isfinite(part) for part in result):
        fail(E_PARAM_RANGE, f"{label} contains a non-finite coordinate", key=label, value=value)
    return result


def validate_scene(scene: dict[str, Any]) -> None:
    if not isinstance(scene, dict):
        fail(E_PARAM_RANGE, "scene must be an object")
    try:
        revision = scene["revision"]
        room = scene["room"]
        sources = scene["sources"]
        walls = scene["walls"]
        receiver = scene["receiver"]
        reconstruction = scene["reconstruction"]
        grid = scene["grid"]
        noise_seed = scene["noise_seed"]
    except (KeyError, TypeError):
        fail(E_PARAM_RANGE, "scene is missing required fields")
    if not isinstance(revision, int) or revision < 1:
        fail(E_PARAM_RANGE, "revision must be a positive integer", revision=revision)
    if scene.get("sample_rate") != SAMPLE_RATE or scene.get("channels") != CHANNEL_COUNT:
        fail(E_PARAM_RANGE, "sample rate and channel count are fixed", sample_rate=scene.get("sample_rate"))
    if not isinstance(noise_seed, int) or noise_seed < 0:
        fail(E_PARAM_RANGE, "noise seed must be a non-negative integer", noise_seed=noise_seed)

    room_dims = _position([room.get("width"), room.get("depth"), room.get("height")], "room.size")
    if any(dim <= 0.5 for dim in room_dims):
        fail(E_PARAM_RANGE, "room dimensions are too small", room=room)

    listener = _position(scene.get("listener"), "listener")
    if not _inside_room(listener, room_dims):
        fail(E_PARAM_RANGE, "listener is outside the room", listener=listener)
    if set(sources) != set(DEFAULT_SOURCES):
        fail(E_PARAM_RANGE, "source identifier set changed", sources=sorted(sources))
    for source_id, position in sources.items():
        value = _position(position, f"source.{source_id}.position")
        if not _inside_room(value, room_dims):
            fail(E_PARAM_RANGE, "source is outside the room", key=f"source.{source_id}.position", value=value)
    if set(walls) != set(WALL_IDS):
        fail(E_PARAM_RANGE, "wall identifier set changed", walls=sorted(walls))
    for wall_id, alpha in walls.items():
        try:
            value = float(alpha)
        except (TypeError, ValueError):
            fail(E_PARAM_RANGE, "absorption coefficient must be numeric", key=f"wall.{wall_id}.alpha")
        if not math.isfinite(value) or value < 0.0 or value > 0.99:
            fail(E_PARAM_RANGE, "absorption coefficient must be in [0, 0.99]", key=f"wall.{wall_id}.alpha", value=alpha)

    density = receiver.get("sample_density") if isinstance(receiver, dict) else None
    if not isinstance(density, int) or not (MIN_SAMPLE_DENSITY <= density <= MAX_SAMPLE_DENSITY):
        fail(E_PARAM_RANGE, "sample density is outside the supported range", value=density)
    order = reconstruction.get("order") if isinstance(reconstruction, dict) else None
    if not isinstance(order, int) or order < 0 or order > MAX_SH_ORDER:
        fail(E_SPH_ORDER, "spherical-harmonic order is outside [0, 8]", order=order)
    bound = sph_order_bound(density)
    if order > bound:
        fail(E_SPH_ORDER, "spherical-harmonic order exceeds the sampling-density bound", order=order, bound=bound, sample_density=density)

    _validate_grid(grid)


def _inside_room(position: list[float], room_dims: list[float]) -> bool:
    return all(0.05 <= position[i] <= room_dims[i] - 0.05 for i in range(3))


def _validate_grid(grid: Any) -> None:
    if not isinstance(grid, dict):
        fail(E_GRID_BOUNDS, "grid must be an object")
    try:
        width = grid["width"]
        height = grid["height"]
        tile_rows = grid["tile_rows"]
        tile_cols = grid["tile_cols"]
        bounds = (grid["lon_min"], grid["lon_max"], grid["lat_min"], grid["lat_max"])
    except KeyError:
        fail(E_GRID_BOUNDS, "grid is missing required fields")
    if not all(isinstance(value, int) for value in (width, height, tile_rows, tile_cols)):
        fail(E_GRID_BOUNDS, "grid dimensions and tile counts must be integers", grid=grid)
    if not (64 <= width <= 1024 and 32 <= height <= 512):
        fail(E_GRID_BOUNDS, "grid pixel dimensions are outside [64,1024] x [32,512]", width=width, height=height)
    if not (1 <= tile_rows <= 16 and 1 <= tile_cols <= 32):
        fail(E_GRID_BOUNDS, "tile counts are outside their supported range", tile_rows=tile_rows, tile_cols=tile_cols)
    if width % tile_cols or height % tile_rows:
        fail(E_GRID_BOUNDS, "tile grid must divide the projection exactly", width=width, height=height, tile_rows=tile_rows, tile_cols=tile_cols)
    expected = PROJECTION_BOUNDS
    try:
        numeric = tuple(float(value) for value in bounds)
    except (TypeError, ValueError):
        fail(E_GRID_BOUNDS, "grid bounds must be numeric", bounds=bounds)
    if numeric != expected:
        fail(E_GRID_BOUNDS, "equirectangular bounds must be the full sphere", bounds=numeric, expected=expected)


def validate_patch_keys(patch: dict[str, Any]) -> None:
    unknown = sorted(set(patch) - ALLOWED_PATCH_KEYS)
    if unknown:
        fail(E_PARAM_RANGE, "parameter key is not supported", keys=unknown, allowed=sorted(ALLOWED_PATCH_KEYS))


def apply_patch(
    scene: dict[str, Any],
    versions: dict[str, int],
    patch: dict[str, Any],
    *,
    revision: int,
) -> tuple[dict[str, Any], dict[str, int], list[str]]:
    validate_patch_keys(patch)
    updated = copy.deepcopy(scene)
    updated["revision"] = revision
    for key, value in patch.items():
        parts = key.split(".")
        if parts[0] == "source":
            updated["sources"][parts[1]] = list(value)
        elif parts[0] == "wall":
            updated["walls"][parts[1]] = value
        elif key == "receiver.sample_density":
            updated["receiver"]["sample_density"] = value
        elif key == "reconstruction.order":
            updated["reconstruction"]["order"] = value
        elif key == "noise.seed":
            updated["noise_seed"] = value
        elif parts[0] == "grid":
            updated["grid"][parts[1]] = value
    validate_scene(updated)
    old_flat = flatten_params(scene)
    new_flat = flatten_params(updated)
    changed = sorted(key for key in new_flat if old_flat.get(key) != new_flat[key])
    next_versions = dict(versions)
    for key in changed:
        next_versions[key] = revision
    for key in next_versions:
        next_versions.setdefault(key, 1)
    return updated, next_versions, changed


def parse_patch_value(key: str, raw: str) -> Any:
    if key.startswith("source.") or key == "listener":
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            fail(E_PARAM_RANGE, "position values require x,y,z", key=key, raw=raw)
        try:
            return [float(part) for part in parts]
        except ValueError:
            fail(E_PARAM_RANGE, "position contains a non-numeric coordinate", key=key, raw=raw)
    if key.startswith("wall."):
        try:
            return float(raw)
        except ValueError:
            fail(E_PARAM_RANGE, "absorption coefficient must be numeric", key=key, raw=raw)
    if key in {"receiver.sample_density", "reconstruction.order", "noise.seed", "grid.width", "grid.height", "grid.tile_rows", "grid.tile_cols"}:
        try:
            return int(raw)
        except ValueError:
            fail(E_PARAM_RANGE, "integer parameter expected", key=key, raw=raw)
    if key.startswith("grid."):
        try:
            return float(raw)
        except ValueError:
            fail(E_GRID_BOUNDS, "numeric grid bound expected", key=key, raw=raw)
    fail(E_PARAM_RANGE, "parameter key is not supported", key=key)
