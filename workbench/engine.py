from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .commands import RuntimeState
from .configuration import WorkbenchConfig
from .errors import WorkbenchError
from .hashing import array_fingerprint, canonical_hash, stable_key_hash, stable_task_key
from .kernels import build_style_lut, ensure_finite_pixels, repair_tile, style_tile, validate_injected_rectangle
from .pyramid import (
    Pyramid,
    bilinear_index_vectors,
    build_pyramid,
    crop_tile,
    iter_tile_coordinates,
    referenced_coarse_tiles,
    tile_bounds,
    tile_identity,
    upsample_uint8_guide,
)

DONE_STATUSES = {"recomputed", "reused", "clean", "done"}


@dataclass(frozen=True)
class TilePlan:
    tile_id: str
    level: int
    row: int
    col: int
    bounds: tuple[int, int, int, int]
    dependencies: tuple[str, ...]
    input_fingerprint: str
    guide_fingerprint: str
    params_fingerprint: str
    quant_fingerprint: str
    cache_key: str
    layer_fingerprint: str
    lut_fingerprint: str
    task_keys: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "level": self.level,
            "row": self.row,
            "col": self.col,
            "bounds": list(self.bounds),
            "dependencies": list(self.dependencies),
            "input_fingerprint": self.input_fingerprint,
            "guide_fingerprint": self.guide_fingerprint,
            "params_fingerprint": self.params_fingerprint,
            "quant_fingerprint": self.quant_fingerprint,
            "cache_key": self.cache_key,
            "layer_fingerprint": self.layer_fingerprint,
            "lut_fingerprint": self.lut_fingerprint,
            "task_keys": self.task_keys,
        }


@dataclass
class PlannedTiles:
    pyramid: Pyramid
    plans: dict[str, TilePlan]
    order: list[str]


@dataclass
class TileRecord:
    tile_id: str
    level: int
    row: int
    col: int
    cache_key: str
    version: str
    dependencies: list[str]
    input_fingerprint: str
    guide_fingerprint: str
    params_fingerprint: str
    quant_fingerprint: str
    lut_fingerprint: str
    status: str = "pending"
    repair_fingerprint: str | None = None
    style_fingerprint: str | None = None
    repair_object: str | None = None
    style_object: str | None = None
    error: dict[str, Any] | None = None

    def to_manifest(self) -> dict[str, Any]:
        data = {
            "tile_id": self.tile_id,
            "level": self.level,
            "row": self.row,
            "col": self.col,
            "cache_key": self.cache_key,
            "version": self.version,
            "dependencies": self.dependencies,
            "input_fingerprint": self.input_fingerprint,
            "guide_fingerprint": self.guide_fingerprint,
            "params_fingerprint": self.params_fingerprint,
            "quant_fingerprint": self.quant_fingerprint,
            "lut_fingerprint": self.lut_fingerprint,
            "status": self.status,
            "repair_fingerprint": self.repair_fingerprint,
            "style_fingerprint": self.style_fingerprint,
            "repair_object": self.repair_object,
            "style_object": self.style_object,
        }
        if self.error:
            data["error"] = self.error
        return data


@dataclass
class ExecutionResult:
    repaired_levels: list[np.ndarray]
    styled_levels: list[np.ndarray]
    records: dict[str, TileRecord]
    recomputed: list[str] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    task_order: list[str] = field(default_factory=list)
    injection_hits: list[str] = field(default_factory=list)
    max_observed_workers: int = 1


def build_plan(
    config: WorkbenchConfig,
    pixels: np.ndarray,
    image_fingerprint: str,
    state: RuntimeState,
) -> PlannedTiles:
    pyramid = build_pyramid(config, pixels, image_fingerprint, state.rectangles)
    plans: dict[str, TilePlan] = {}
    order: list[str] = []

    for level in range(config.levels):
        iterations = int(state.iterations[level])
        guidance = float(state.guidance[level])
        strength = float(state.style_strength[level])
        weights = tuple(float(v) for v in state.style_weights[level])
        quant = int(state.quant_levels[level])
        bits = int(state.fractional_bits[level])
        params_fingerprint = canonical_hash({
            "iterations": iterations,
            "guidance_strength": guidance,
            "style_strength": strength,
            "style_weights": list(weights),
        })
        quant_fingerprint = canonical_hash({"quant_levels": quant, "fractional_bits": bits})
        _, lut_metadata = build_style_lut(
            quant_levels=quant,
            fractional_bits=bits,
            strength=strength,
            weights=weights,
            target=f"plan:L{level}",
        )

        for row, col in iter_tile_coordinates(config, level):
            tile_id = tile_identity(level, row, col)
            bounds = tile_bounds(config, level, row, col)
            source_tile = crop_tile(pyramid.image_levels[level], bounds)
            mask_tile = crop_tile(pyramid.mask_levels[level], bounds)
            input_fingerprint = canonical_hash({
                "source_fingerprint": array_fingerprint(source_tile),
                "mask_fingerprint": array_fingerprint(mask_tile),
            })
            dependencies = tuple(referenced_coarse_tiles(config, level, row, col))
            lower_keys = {dep: plans[dep].cache_key for dep in dependencies}
            guide_fingerprint = canonical_hash({
                "dependencies": list(dependencies),
                "lower_cache_keys": lower_keys,
            }) if dependencies else canonical_hash({"dependencies": []})
            cache_key = canonical_hash({
                "schema": "tile-cache-v1",
                "tile_id": tile_id,
                "layer_fingerprint": pyramid.layer_fingerprints[level],
                "input_fingerprint": input_fingerprint,
                "guide_fingerprint": guide_fingerprint,
                "params_fingerprint": params_fingerprint,
                "quant_fingerprint": quant_fingerprint,
            })
            task_keys = {
                "repair": stable_task_key("repair", level, row, col),
                "style": stable_task_key("style", level, row, col),
            }
            plans[tile_id] = TilePlan(
                tile_id=tile_id,
                level=level,
                row=row,
                col=col,
                bounds=bounds,
                dependencies=dependencies,
                input_fingerprint=input_fingerprint,
                guide_fingerprint=guide_fingerprint,
                params_fingerprint=params_fingerprint,
                quant_fingerprint=quant_fingerprint,
                cache_key=cache_key,
                layer_fingerprint=pyramid.layer_fingerprints[level],
                lut_fingerprint=lut_metadata["lut_key"],
                task_keys=task_keys,
            )
            order.append(tile_id)
    return PlannedTiles(pyramid=pyramid, plans=plans, order=order)


def empty_canvases(config: WorkbenchConfig, dtype=np.uint8) -> list[np.ndarray]:
    canvases = []
    for level in range(config.levels):
        width, height = config.level_dimensions[level]
        canvases.append(np.zeros((height, width, 3), dtype=dtype))
    return canvases


def _injections_for(injections: list[dict[str, Any]] | None, plan: TilePlan, stage: str) -> list[dict[str, Any]]:
    if not injections:
        return []
    result = []
    for injection in injections:
        if (
            int(injection.get("level", -1)) == plan.level
            and int(injection.get("row", -2)) == plan.row
            and int(injection.get("col", -2)) == plan.col
            and injection.get("stage", stage) == stage
        ):
            result.append(injection)
    return result


def _error_dict(exc: WorkbenchError, task_key: str, injection_id: str | None = None) -> dict[str, Any]:
    details = dict(exc.details)
    if injection_id:
        details["injection_id"] = injection_id
    return {
        "task_key": task_key,
        "code": exc.code,
        "message": exc.message,
        "details": details,
        "injection_id": injection_id,
    }


class ConcurrencyMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_observed = 0

    def entered(self) -> None:
        with self._lock:
            self.active += 1
            self.max_observed = max(self.max_observed, self.active)

    def left(self) -> None:
        with self._lock:
            self.active -= 1


def _repair_task(
    config: WorkbenchConfig,
    planned: PlannedTiles,
    plan: TilePlan,
    state: RuntimeState,
    guide_fixed_level: np.ndarray | None,
    injections: list[dict[str, Any]] | None,
    meter: ConcurrencyMeter | None,
) -> dict[str, Any]:
    task_key = plan.task_keys["repair"]
    injection_id = None
    if meter:
        meter.entered()
    try:
        source = crop_tile(planned.pyramid.image_levels[plan.level], plan.bounds)
        mask = crop_tile(planned.pyramid.mask_levels[plan.level], plan.bounds)
        guide_fp = None
        if guide_fixed_level is not None:
            y0, y1, x0, x1 = plan.bounds
            guide_fp = guide_fixed_level[y0:y1, x0:x1]

        for injection in _injections_for(injections, plan, "repair"):
            injection_id = str(injection["id"])
            kind = injection.get("kind")
            if kind == "nonfinite_pixels":
                bad = source.astype(np.float64, copy=True)
                bad[0, 0, 0] = np.nan
                ensure_finite_pixels(bad, task_key)
            if kind == "invalid_mask_rectangle":
                width, height = config.level_dimensions[plan.level]
                validate_injected_rectangle(injection.get("rect", {}), width, height, task_key)

        repaired = repair_tile(
            source,
            mask,
            level=plan.level,
            iterations=int(state.iterations[plan.level]),
            guidance_strength=float(state.guidance[plan.level]),
            fractional_bits=int(state.fractional_bits[plan.level]),
            guide_fixed=guide_fp,
        )
        return {"ok": True, "task_key": task_key, "tile_id": plan.tile_id, "array": repaired,
                "fingerprint": array_fingerprint(repaired), "injection_id": None}
    except WorkbenchError as exc:
        if injection_id and "injection_id" not in exc.details:
            exc.details["injection_id"] = injection_id
        return {"ok": False, "task_key": task_key, "tile_id": plan.tile_id,
                "error": _error_dict(exc, task_key, injection_id), "injection_id": injection_id}
    except Exception as exc:  # deterministic isolation boundary
        error = {"task_key": task_key, "code": "E_TASK_FAILED", "message": str(exc), "details": {},
                 "injection_id": injection_id}
        return {"ok": False, "task_key": task_key, "tile_id": plan.tile_id, "error": error,
                "injection_id": injection_id}
    finally:
        if meter:
            meter.left()


def _style_task(
    config: WorkbenchConfig,
    planned: PlannedTiles,
    plan: TilePlan,
    state: RuntimeState,
    repaired: np.ndarray,
    injections: list[dict[str, Any]] | None,
    meter: ConcurrencyMeter | None,
) -> dict[str, Any]:
    task_key = plan.task_keys["style"]
    injection_id = None
    if meter:
        meter.entered()
    try:
        bits = int(state.fractional_bits[plan.level])
        quant = int(state.quant_levels[plan.level])
        for injection in _injections_for(injections, plan, "style"):
            injection_id = str(injection["id"])
            kind = injection.get("kind")
            if kind == "quantization_out_of_range":
                quant = int(injection.get("levels", 7))
                bits = int(injection.get("fractional_bits", bits))
            if kind == "invalid_mask_rectangle":
                width, height = config.level_dimensions[plan.level]
                validate_injected_rectangle(injection.get("rect", {}), width, height, task_key)
            if kind == "nonfinite_pixels":
                bad = repaired.astype(np.float64, copy=True)
                bad[0, 0, 0] = np.inf
                ensure_finite_pixels(bad, task_key)

        scale = 1 << bits
        lut_fixed, lut_metadata = build_style_lut(
            quant_levels=quant,
            fractional_bits=bits,
            strength=float(state.style_strength[plan.level]),
            weights=tuple(float(v) for v in state.style_weights[plan.level]),
            target=task_key,
        )
        lut_bytes = np.clip((lut_fixed + scale // 2) // scale, 0, 255).astype(np.uint8)
        source = crop_tile(planned.pyramid.image_levels[plan.level], plan.bounds)
        mask = crop_tile(planned.pyramid.mask_levels[plan.level], plan.bounds)
        styled = style_tile(repaired, source, mask, lut_bytes)
        return {"ok": True, "task_key": task_key, "tile_id": plan.tile_id, "array": styled,
                "fingerprint": array_fingerprint(styled), "lut": lut_metadata, "injection_id": None}
    except WorkbenchError as exc:
        if injection_id and "injection_id" not in exc.details:
            exc.details["injection_id"] = injection_id
        return {"ok": False, "task_key": task_key, "tile_id": plan.tile_id,
                "error": _error_dict(exc, task_key, injection_id), "injection_id": injection_id}
    except Exception as exc:
        error = {"task_key": task_key, "code": "E_TASK_FAILED", "message": str(exc), "details": {},
                 "injection_id": injection_id}
        return {"ok": False, "task_key": task_key, "tile_id": plan.tile_id, "error": error,
                "injection_id": injection_id}
    finally:
        if meter:
            meter.left()


def execute_tiles(
    config: WorkbenchConfig,
    planned: PlannedTiles,
    state: RuntimeState,
    requested_ids: list[str],
    *,
    repaired_canvases: list[np.ndarray] | None = None,
    styled_canvases: list[np.ndarray] | None = None,
    existing_records: dict[str, TileRecord] | None = None,
    max_workers: int = 1,
    injections: list[dict[str, Any]] | None = None,
) -> ExecutionResult:
    repaired_levels = [canvas.copy() for canvas in (repaired_canvases or empty_canvases(config))]
    styled_levels = [canvas.copy() for canvas in (styled_canvases or empty_canvases(config))]
    existing_records = existing_records or {}
    records: dict[str, TileRecord] = {}
    recomputed: list[str] = []
    failures: list[dict[str, Any]] = []
    task_order: list[str] = []
    injection_hits: list[str] = []
    meter = ConcurrencyMeter()

    requested_set = set(requested_ids)
    requested_by_level: dict[int, list[str]] = {}
    for tile_id in requested_ids:
        plan = planned.plans[tile_id]
        requested_by_level.setdefault(plan.level, []).append(tile_id)

    worker_count = max(1, min(int(max_workers), int(config.max_workers)))

    for level in range(config.levels):
        level_ids = requested_by_level.get(level, [])
        if not level_ids:
            continue
        guide_fixed_for_level = None
        if level > 0:
            factor = config.upsample_factors_coarse_to_fine[level - 1]
            coarse_repaired = repaired_levels[level - 1]
            width, height = config.level_dimensions[level]
            guide_fixed_for_level = upsample_uint8_guide(
                coarse_repaired, factor, height, width, state.fractional_bits[level]
            )

        repair_outcomes: dict[str, dict[str, Any]] = {}
        repair_meter = meter if worker_count > 1 else None
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {}
            for tile_id in level_ids:
                plan = planned.plans[tile_id]
                future = executor.submit(
                    _repair_task, config, planned, plan, state, guide_fixed_for_level,
                    injections, repair_meter,
                )
                future_map[future] = tile_id
                task_order.append(plan.task_keys["repair"])
            for future in as_completed(future_map):
                outcome = future.result()
                repair_outcomes[outcome["tile_id"]] = outcome

        successful_repairs: dict[str, np.ndarray] = {}
        for tile_id in level_ids:
            outcome = repair_outcomes[tile_id]
            plan = planned.plans[tile_id]
            if outcome["ok"]:
                y0, y1, x0, x1 = plan.bounds
                repaired_levels[level][y0:y1, x0:x1] = outcome["array"]
                successful_repairs[tile_id] = outcome["array"]
            else:
                failures.append(outcome["error"])
                if outcome.get("injection_id"):
                    injection_hits.append(outcome["injection_id"])

        style_ids = [tile_id for tile_id in level_ids if tile_id in successful_repairs]
        style_outcomes: dict[str, dict[str, Any]] = {}
        style_meter = meter if worker_count > 1 else None
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {}
            for tile_id in style_ids:
                plan = planned.plans[tile_id]
                future = executor.submit(
                    _style_task, config, planned, plan, state, successful_repairs[tile_id],
                    injections, style_meter,
                )
                future_map[future] = tile_id
                task_order.append(plan.task_keys["style"])
            for future in as_completed(future_map):
                outcome = future.result()
                style_outcomes[outcome["tile_id"]] = outcome

        for tile_id in level_ids:
            plan = planned.plans[tile_id]
            old = existing_records.get(tile_id)
            repair_outcome = repair_outcomes[tile_id]
            style_outcome = style_outcomes.get(tile_id)
            if not repair_outcome["ok"]:
                records[tile_id] = TileRecord(
                    tile_id=tile_id, level=plan.level, row=plan.row, col=plan.col,
                    cache_key=plan.cache_key, version=plan.cache_key[:16],
                    dependencies=list(plan.dependencies), input_fingerprint=plan.input_fingerprint,
                    guide_fingerprint=plan.guide_fingerprint,
                    params_fingerprint=plan.params_fingerprint,
                    quant_fingerprint=plan.quant_fingerprint,
                    lut_fingerprint=plan.lut_fingerprint, status="failed",
                    error=repair_outcome["error"],
                )
                continue
            if style_outcome is None or not style_outcome["ok"]:
                error = style_outcome["error"] if style_outcome else {
                    "code": "E_STYLE_SKIPPED", "message": "style stage was not executed"
                }
                if style_outcome and style_outcome.get("injection_id"):
                    injection_hits.append(style_outcome["injection_id"])
                records[tile_id] = TileRecord(
                    tile_id=tile_id, level=plan.level, row=plan.row, col=plan.col,
                    cache_key=plan.cache_key, version=plan.cache_key[:16],
                    dependencies=list(plan.dependencies), input_fingerprint=plan.input_fingerprint,
                    guide_fingerprint=plan.guide_fingerprint,
                    params_fingerprint=plan.params_fingerprint,
                    quant_fingerprint=plan.quant_fingerprint,
                    lut_fingerprint=plan.lut_fingerprint, status="failed",
                    repair_fingerprint=repair_outcome["fingerprint"], error=error,
                )
                failures.append(error)
                continue

            if style_outcome.get("injection_id"):
                injection_hits.append(style_outcome["injection_id"])
            y0, y1, x0, x1 = plan.bounds
            styled_levels[level][y0:y1, x0:x1] = style_outcome["array"]
            records[tile_id] = TileRecord(
                tile_id=tile_id, level=plan.level, row=plan.row, col=plan.col,
                cache_key=plan.cache_key, version=plan.cache_key[:16],
                dependencies=list(plan.dependencies), input_fingerprint=plan.input_fingerprint,
                guide_fingerprint=plan.guide_fingerprint,
                params_fingerprint=plan.params_fingerprint,
                quant_fingerprint=plan.quant_fingerprint,
                lut_fingerprint=plan.lut_fingerprint, status="recomputed",
                repair_fingerprint=repair_outcome["fingerprint"],
                style_fingerprint=style_outcome["fingerprint"],
                repair_object=f"tiles/{tile_id}.repair.bin",
                style_object=f"tiles/{tile_id}.style.bin",
            )
            recomputed.append(tile_id)

    failures.sort(key=lambda item: item.get("task_key", ""))
    injection_hits = sorted(set(injection_hits))
    return ExecutionResult(
        repaired_levels=repaired_levels,
        styled_levels=styled_levels,
        records=records,
        recomputed=recomputed,
        failed=failures,
        task_order=task_order,
        injection_hits=injection_hits,
        max_observed_workers=meter.max_observed,
    )


def assemble_final(config: WorkbenchConfig, styled_levels: list[np.ndarray]) -> np.ndarray:
    return styled_levels[-1].copy()


def fingerprint_levels(levels: list[np.ndarray]) -> list[str]:
    return [array_fingerprint(level) for level in levels]
