from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import WorkbenchError
from .hashing import canonical_hash

ALLOWED_QUANT_LEVELS = (2, 4, 8, 16, 32, 64, 128, 256)
ALLOWED_FRACTIONAL_BITS = (4, 8, 12, 16)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_json(path: str | Path, code: str = "E_CONFIG_INVALID") -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise WorkbenchError(code, f"JSON file not found: {path}", {"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise WorkbenchError(
            code,
            f"invalid JSON in {path}: {exc.msg}",
            {"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkbenchError("E_CONFIG_INVALID", f"{name} must be a JSON object")
    return value


def _positive_int(value: Any, name: str, code: str = "E_CONFIG_INVALID") -> int:
    if not is_int(value) or value <= 0:
        raise WorkbenchError(code, f"{name} must be a positive integer", {"value": value})
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkbenchError("E_CONFIG_INVALID", f"{name} must be numeric", {"value": value})
    numeric = float(value)
    if not (minimum <= numeric <= maximum):
        raise WorkbenchError(
            "E_CONFIG_INVALID",
            f"{name} must be in [{minimum}, {maximum}]",
            {"value": value},
        )
    return numeric


def _as_level_list(value: Any, levels: int, name: str) -> list[Any]:
    if isinstance(value, list):
        if len(value) != levels:
            raise WorkbenchError(
                "E_CONFIG_INVALID",
                f"{name} must contain {levels} entries",
                {"actual": len(value), "expected": levels},
            )
        return list(value)
    return [value for _ in range(levels)]


@dataclass(frozen=True)
class WorkbenchConfig:
    source_path: Path
    config_dir: Path
    raw: dict[str, Any]
    config_hash: str
    global_version: int
    seed: int
    width: int
    height: int
    channels: int
    image_path: Path
    mask_path: Path
    levels: int
    factors_fine_to_coarse: tuple[int, ...]
    level_dimensions: tuple[tuple[int, int], ...]
    cumulative_downsample: tuple[int, ...]
    upsample_factors_coarse_to_fine: tuple[int, ...]
    tile_sizes: tuple[int, ...]
    fractional_bits: tuple[int, ...]
    quant_levels: tuple[int, ...]
    iterations: tuple[int, ...]
    guidance: tuple[float, ...]
    style_strength: tuple[float, ...]
    style_weights: tuple[tuple[float, float, float], ...]
    tile_budget: int
    max_workers: int
    snapshot_dir: Path

    @property
    def base_tile_rows(self) -> int:
        tile = self.tile_sizes[-1]
        return self.height // tile

    @property
    def base_tile_cols(self) -> int:
        tile = self.tile_sizes[-1]
        return self.width // tile

    def tile_shape(self, level: int) -> tuple[int, int]:
        return self.tile_sizes[level], self.tile_sizes[level]

    def tile_grid(self, level: int) -> tuple[int, int]:
        width, height = self.level_dimensions[level]
        tile = self.tile_sizes[level]
        return height // tile, width // tile


def load_config(path: str | Path) -> WorkbenchConfig:
    source_path = Path(path).resolve()
    raw = load_json(source_path, "E_CONFIG_INVALID")
    data = _require_mapping(raw, "configuration")

    allowed = {
        "schema_version", "global_version", "seed", "image", "mask", "pyramid",
        "fixed_point", "quantization", "inpaint", "style", "budget",
        "concurrency", "snapshot",
    }
    unknown = set(data) - allowed
    if unknown:
        raise WorkbenchError("E_CONFIG_INVALID", "unknown configuration keys", {"keys": sorted(unknown)})
    if data.get("schema_version") != 1:
        raise WorkbenchError("E_CONFIG_INVALID", "schema_version must be 1", {"value": data.get("schema_version")})

    global_version = _positive_int(data.get("global_version"), "global_version")
    seed = data.get("seed")
    if not is_int(seed):
        raise WorkbenchError("E_CONFIG_INVALID", "seed must be an integer", {"value": seed})

    image_cfg = _require_mapping(data.get("image"), "image")
    width = _positive_int(image_cfg.get("width"), "image.width")
    height = _positive_int(image_cfg.get("height"), "image.height")
    channels = _positive_int(image_cfg.get("channels", 3), "image.channels")
    if channels != 3:
        raise WorkbenchError("E_CONFIG_INVALID", "only 3-channel RGB samples are supported", {"channels": channels})
    image_rel = image_cfg.get("path")
    if not isinstance(image_rel, str) or not image_rel:
        raise WorkbenchError("E_CONFIG_INVALID", "image.path must be a non-empty string")
    image_path = (source_path.parent / image_rel).resolve()

    mask_cfg = _require_mapping(data.get("mask"), "mask")
    mask_rel = mask_cfg.get("path")
    if not isinstance(mask_rel, str) or not mask_rel:
        raise WorkbenchError("E_CONFIG_INVALID", "mask.path must be a non-empty string")
    mask_path = (source_path.parent / mask_rel).resolve()

    pyramid = _require_mapping(data.get("pyramid"), "pyramid")
    levels = _positive_int(pyramid.get("levels"), "pyramid.levels")
    if levels < 1 or levels > 12:
        raise WorkbenchError("E_CONFIG_INVALID", "pyramid.levels must be between 1 and 12", {"value": levels})
    factors = pyramid.get("downsample_factors")
    if levels == 1:
        factor_tuple: tuple[int, ...] = ()
    else:
        if not isinstance(factors, list) or len(factors) != levels - 1:
            raise WorkbenchError(
                "E_CONFIG_INVALID",
                "downsample_factors must contain levels-1 positive integers",
                {"levels": levels, "factors": factors},
            )
        for index, factor in enumerate(factors):
            _positive_int(factor, f"downsample_factors[{index}]")
        factor_tuple = tuple(int(f) for f in factors)
        if any(factor_tuple[i] > factor_tuple[i + 1] for i in range(len(factor_tuple) - 1)):
            raise WorkbenchError(
                "E_CONFIG_INVALID",
                "downsample_factors must be monotonically non-decreasing",
                {"factors": list(factor_tuple)},
            )

    dims_fine_to_coarse: list[tuple[int, int]] = [(width, height)]
    current_w, current_h = width, height
    for index, factor in enumerate(factor_tuple):
        if current_w % factor or current_h % factor:
            raise WorkbenchError(
                "E_CONFIG_INVALID",
                "downsample factor does not divide the preceding level dimensions",
                {"factor_index": index, "factor": factor, "width": current_w, "height": current_h},
            )
        current_w //= factor
        current_h //= factor
        dims_fine_to_coarse.append((current_w, current_h))
    level_dimensions = tuple(reversed(dims_fine_to_coarse))

    cumulative: list[int] = []
    for level in range(levels):
        count = levels - 1 - level
        product = 1
        for factor in factor_tuple[:count]:
            product *= factor
        cumulative.append(product)
    upsample_factors = tuple(
        factor_tuple[levels - 2 - level] if level < levels - 1 and levels >= 2 else 1
        for level in range(levels - 1)
    )

    tile_sizes_raw = pyramid.get("tile_sizes")
    if not isinstance(tile_sizes_raw, list) or len(tile_sizes_raw) != levels:
        raise WorkbenchError(
            "E_TILE_GEOMETRY",
            "pyramid.tile_sizes must be a list with one positive edge per coarse-to-fine level",
            {"levels": levels, "tile_sizes": tile_sizes_raw},
        )
    tile_sizes: list[int] = []
    for level, edge in enumerate(tile_sizes_raw):
        if not is_int(edge) or edge <= 0:
            raise WorkbenchError("E_TILE_GEOMETRY", "tile edge must be a positive integer", {"level": level, "value": edge})
        level_w, level_h = level_dimensions[level]
        if level_w % edge or level_h % edge or edge > min(level_w, level_h):
            raise WorkbenchError(
                "E_TILE_GEOMETRY",
                "tile edge must divide and fit the level dimensions",
                {"level": level, "tile_size": edge, "width": level_w, "height": level_h},
            )
        tile_sizes.append(int(edge))

    fp_cfg = _require_mapping(data.get("fixed_point", {}), "fixed_point")
    fbits_raw = fp_cfg.get("fractional_bits_by_level", fp_cfg.get("fractional_bits", 8))
    fbits_values = _as_level_list(fbits_raw, levels, "fixed_point.fractional_bits")
    fractional_bits: list[int] = []
    for level, bits in enumerate(fbits_values):
        if not is_int(bits) or bits not in ALLOWED_FRACTIONAL_BITS:
            raise WorkbenchError(
                "E_CONFIG_INVALID",
                "fractional bits must be one of " + repr(ALLOWED_FRACTIONAL_BITS),
                {"level": level, "value": bits},
            )
        fractional_bits.append(int(bits))

    quant_cfg = _require_mapping(data.get("quantization", {}), "quantization")
    quant_raw = quant_cfg.get("levels_by_level", quant_cfg.get("levels", 16))
    quant_values = _as_level_list(quant_raw, levels, "quantization.levels")
    quant_levels: list[int] = []
    for level, value in enumerate(quant_values):
        if not is_int(value) or value not in ALLOWED_QUANT_LEVELS:
            raise WorkbenchError(
                "E_CONFIG_INVALID",
                "quantization level must be one of " + repr(ALLOWED_QUANT_LEVELS),
                {"level": level, "value": value},
            )
        quant_levels.append(int(value))

    inpaint = _require_mapping(data.get("inpaint", {}), "inpaint")
    iter_raw = inpaint.get("iterations_by_level", inpaint.get("iterations", 3))
    iterations = [
        int(_positive_int(v, f"inpaint.iterations[{i}]"))
        for i, v in enumerate(_as_level_list(iter_raw, levels, "inpaint.iterations"))
    ]
    guidance_raw = inpaint.get("guidance_strength_by_level", inpaint.get("guidance_strength", 0.5))
    guidance = [
        _number(v, f"inpaint.guidance[{i}]", 0.0, 1.0)
        for i, v in enumerate(_as_level_list(guidance_raw, levels, "inpaint.guidance_strength"))
    ]

    style = _require_mapping(data.get("style", {}), "style")
    strength_raw = style.get("strength_by_level", style.get("strength", 0.5))
    style_strength = [
        _number(v, f"style.strength[{i}]", 0.0, 1.0)
        for i, v in enumerate(_as_level_list(strength_raw, levels, "style.strength"))
    ]
    weights_raw = style.get("weights_by_level", style.get("weights", [1.0, 1.0, 1.0]))
    weights_per_level = _as_level_list(weights_raw, levels, "style.weights")
    style_weights: list[tuple[float, float, float]] = []
    for level, item in enumerate(weights_per_level):
        if not isinstance(item, list) or len(item) != 3:
            raise WorkbenchError("E_CONFIG_INVALID", "each style weight entry must contain 3 numbers", {"level": level, "value": item})
        checked = tuple(_number(v, f"style.weights[{level}][{c}]", 0.0, 4.0) for c, v in enumerate(item))
        if sum(checked) <= 0:
            raise WorkbenchError("E_CONFIG_INVALID", "style weights must have a positive sum", {"level": level})
        style_weights.append(checked)  # type: ignore[arg-type]

    budget_cfg = _require_mapping(data.get("budget", {}), "budget")
    tile_budget = _positive_int(budget_cfg.get("tiles_per_batch", 1), "budget.tiles_per_batch")
    if tile_budget > 10_000_000:
        raise WorkbenchError("E_CONFIG_INVALID", "budget.tiles_per_batch is unreasonably large")

    concurrency = _require_mapping(data.get("concurrency", {}), "concurrency")
    max_workers = _positive_int(concurrency.get("max_workers", 1), "concurrency.max_workers")
    if max_workers > 32:
        raise WorkbenchError("E_CONFIG_INVALID", "concurrency.max_workers must not exceed 32")

    snapshot_cfg = _require_mapping(data.get("snapshot", {}), "snapshot") if data.get("snapshot") is not None else {}
    snapshot_rel = snapshot_cfg.get("dir", ".snapshots")
    if not isinstance(snapshot_rel, str):
        raise WorkbenchError("E_CONFIG_INVALID", "snapshot.dir must be a string")
    snapshot_dir = (source_path.parent / snapshot_rel).resolve()

    return WorkbenchConfig(
        source_path=source_path,
        config_dir=source_path.parent,
        raw=data,
        config_hash=canonical_hash(data),
        global_version=global_version,
        seed=int(seed),
        width=width,
        height=height,
        channels=channels,
        image_path=image_path,
        mask_path=mask_path,
        levels=levels,
        factors_fine_to_coarse=factor_tuple,
        level_dimensions=tuple(level_dimensions),
        cumulative_downsample=tuple(cumulative),
        upsample_factors_coarse_to_fine=tuple(upsample_factors),
        tile_sizes=tuple(tile_sizes),
        fractional_bits=tuple(fractional_bits),
        quant_levels=tuple(quant_levels),
        iterations=tuple(iterations),
        guidance=tuple(guidance),
        style_strength=tuple(style_strength),
        style_weights=tuple(style_weights),
        tile_budget=tile_budget,
        max_workers=max_workers,
        snapshot_dir=snapshot_dir,
    )
