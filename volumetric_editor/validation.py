"""Business validation with stable error codes."""

from __future__ import annotations

import math

from .errors import ErrorCode, fail
from .models import Scene

THREAD_MIN = 1
THREAD_MAX = 64
IOR_MIN = 1.0
IOR_MAX = 3.0


def validate_threads(threads: int) -> int:
    if isinstance(threads, bool) or not isinstance(threads, int):
        raise fail(
            ErrorCode.THREAD_BOUNDS,
            "threads must be an integer",
            received=repr(threads),
            minimum=THREAD_MIN,
            maximum=THREAD_MAX,
        )
    if threads < THREAD_MIN or threads > THREAD_MAX:
        raise fail(
            ErrorCode.THREAD_BOUNDS,
            "threads outside supported bounds",
            received=threads,
            minimum=THREAD_MIN,
            maximum=THREAD_MAX,
        )
    return threads


def validate_tile_size(tile_size: int, width: int, height: int) -> int:
    if isinstance(tile_size, bool) or not isinstance(tile_size, int):
        raise fail(
            ErrorCode.TILE_SIZE,
            "tile size must be an integer",
            received=repr(tile_size),
        )
    maximum = max(width, height)
    if tile_size < 1 or tile_size > maximum:
        raise fail(
            ErrorCode.TILE_SIZE,
            "tile size outside renderable bounds",
            received=tile_size,
            minimum=1,
            maximum=maximum,
        )
    return tile_size


def validate_ior(ior: float, surface_id: str | None = None) -> float:
    try:
        value = float(ior)
    except (TypeError, ValueError):
        raise fail(ErrorCode.IOR_RANGE, "refractive index must be numeric", received=repr(ior)) from None
    if not math.isfinite(value) or value < IOR_MIN or value > IOR_MAX:
        raise fail(
            ErrorCode.IOR_RANGE,
            "refractive index outside supported range",
            received=value,
            minimum=IOR_MIN,
            maximum=IOR_MAX,
            surface_id=surface_id,
        )
    return value


def validate_scatter(value: float) -> float:
    try:
        scatter = float(value)
    except (TypeError, ValueError):
        raise fail(ErrorCode.MEDIUM_COEFF, "scatter coefficient must be numeric", received=repr(value)) from None
    if not math.isfinite(scatter) or scatter < 0.0:
        raise fail(
            ErrorCode.MEDIUM_COEFF,
            "scatter coefficient must be finite and non-negative",
            received=scatter,
        )
    return scatter


def validate_scene(scene: Scene) -> Scene:
    for surface in scene.surfaces:
        validate_ior(surface.ior, surface.surface_id)
    validate_scatter(scene.light_cone.scatter)
    if scene.light_cone.absorption < 0.0 or not math.isfinite(scene.light_cone.absorption):
        raise fail(
            ErrorCode.MEDIUM_COEFF,
            "absorption coefficient must be finite and non-negative",
            received=scene.light_cone.absorption,
        )
    if scene.volume.base_step <= 0.0 or scene.volume.max_steps < 1:
        raise fail(ErrorCode.MEDIUM_COEFF, "invalid volume integration settings")
    return scene


def validate_budgets(queue_memory_budget_bytes: int, compute_budget_units: int) -> tuple[int, int]:
    if (
        isinstance(queue_memory_budget_bytes, bool)
        or not isinstance(queue_memory_budget_bytes, int)
        or queue_memory_budget_bytes <= 0
    ):
        raise fail(
            ErrorCode.BUDGET_DENIED,
            "tile queue memory budget must be a positive integer",
            received=repr(queue_memory_budget_bytes),
        )
    if (
        isinstance(compute_budget_units, bool)
        or not isinstance(compute_budget_units, int)
        or compute_budget_units <= 0
    ):
        raise fail(
            ErrorCode.BUDGET_DENIED,
            "frame compute budget must be a positive integer",
            received=repr(compute_budget_units),
        )
    return queue_memory_budget_bytes, compute_budget_units
