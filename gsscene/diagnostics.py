"""High-precision reference diagnostics and edit attribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .cache import ArtifactCache
from .config import SceneConfig
from .util import atomic_write_json, read_json


def _region_assign(shape: tuple[int, int], tile_arrays: dict[tuple[int, int], dict[str, np.ndarray]], key: str) -> np.ndarray:
    height, width = shape
    tile_size = int(tile_arrays[next(iter(tile_arrays))]["pixels_q"].shape[0]) if tile_arrays else 1
    result = np.zeros((height, width, 4), dtype=np.float64)
    for (row, col), arrays in sorted(tile_arrays.items()):
        y0 = row * tile_size
        x0 = col * tile_size
        value = np.asarray(arrays[key], dtype=np.float64)
        result[y0 : y0 + value.shape[0], x0 : x0 + value.shape[1]] = value
    return result


def analyze_level(
    config: SceneConfig,
    kernels: dict[str, np.ndarray],
    tile_arrays: dict[tuple[int, int], dict[str, np.ndarray]],
    composite_arrays: dict[str, np.ndarray],
    view_index: int,
    level: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    height, width = config.image_size
    kernel_count = int(kernels["opacity"].shape[0])
    kernel_parameter_abs_diff = np.zeros((kernel_count,), dtype=np.float64)
    render_abs_diff = np.zeros((height, width, 4), dtype=np.float64)
    for (row, col), arrays in sorted(tile_arrays.items()):
        difference = np.abs(arrays["pixels"] - arrays["reference"])
        y0 = row * config.tile_size
        x0 = col * config.tile_size
        render_abs_diff[y0 : y0 + difference.shape[0], x0 : x0 + difference.shape[1]] = difference
    composite_abs_diff = np.abs(composite_arrays["pixels"] - composite_arrays["reference"])
    kernel_max = float(np.max(kernel_parameter_abs_diff)) if kernel_parameter_abs_diff.size else 0.0
    render_max = float(np.max(render_abs_diff[..., :3])) if render_abs_diff.size else 0.0
    composite_max = float(np.max(composite_abs_diff[..., :3])) if composite_abs_diff.size else 0.0
    values = {"kernel_init": kernel_max, "layer_render": render_max, "layer_composite": composite_max}
    tolerances = config.tolerance
    stages = [
        ("kernel_init", "kernel"),
        ("layer_render", "render"),
        ("layer_composite", "composite"),
    ]
    exceeded = [stage for stage, tolerance_key in stages if values[stage] > float(tolerances[tolerance_key])]
    first = exceeded[0] if exceeded else None
    summary = {
        "composite_max_abs_diff": composite_max,
        "composite_mean_abs_diff": float(np.mean(composite_abs_diff[..., :3])),
        "first_exceed_stage": first,
        "kernel_max_abs_diff": kernel_max,
        "kernel_mean_abs_diff": float(np.mean(kernel_parameter_abs_diff)),
        "level": int(level),
        "render_max_abs_diff": render_max,
        "render_mean_abs_diff": float(np.mean(render_abs_diff[..., :3])),
        "stage_values": values,
        "tolerance": {
            "composite": float(tolerances["composite"]),
            "kernel": float(tolerances["kernel"]),
            "render": float(tolerances["render"]),
        },
        "view_index": int(view_index),
    }
    arrays = {
        "composite_abs_diff": composite_abs_diff,
        "kernel_parameter_abs_diff": kernel_parameter_abs_diff,
        "render_abs_diff": render_abs_diff,
    }
    return summary, arrays


class DiagnosticHistory:
    def __init__(self, cache: ArtifactCache) -> None:
        self.path = cache.diagnostic_dir / "ranks.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            value = read_json(self.path)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def analyze_attributions(
        self,
        diagnostics: list[dict[str, Any]],
        config: SceneConfig,
        kernel_changed: set[tuple[int, int]],
        composite_changed: set[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        previous = self.load()
        previous_ranks = {
            (int(item["view_index"]), int(item["level"])): int(item["rank"])
            for item in previous.get("entries", [])
        }
        by_view: dict[int, list[dict[str, Any]]] = {}
        for item in diagnostics:
            by_view.setdefault(int(item["view_index"]), []).append(item)
        current_entries: list[dict[str, Any]] = []
        attributions: list[dict[str, Any]] = []
        previous_quant = previous.get("quant", {})
        quant_changed = (
            previous_quant.get("frac_bits") != config.frac_bits
            or previous_quant.get("quant_gear") != config.quant_gear
        )
        for view_index in sorted(by_view):
            ordered = sorted(
                by_view[view_index],
                key=lambda item: (-float(item["stage_values"]["layer_composite"]), int(item["level"])),
            )
            for rank, item in enumerate(ordered, start=1):
                level = int(item["level"])
                old_rank = previous_ranks.get((view_index, level))
                current_entries.append(
                    {
                        "composite_max_abs_diff": float(item["composite_max_abs_diff"]),
                        "level": level,
                        "rank": rank,
                        "view_index": view_index,
                    }
                )
                if old_rank is not None and old_rank != rank:
                    if quant_changed:
                        cause = "QUANTIZATION_CHANGE"
                    elif (view_index, level) in kernel_changed:
                        cause = "KERNEL_PARAMETER_CHANGE"
                    elif (view_index, level) in composite_changed:
                        cause = "COMPOSITE_ORDER_CHANGE"
                    else:
                        cause = "RANK_TIE_REORDER"
                    attributions.append(
                        {
                            "cause": cause,
                            "current_rank": rank,
                            "kernel_parameters_changed": (view_index, level) in kernel_changed,
                            "level": level,
                            "previous_rank": old_rank,
                            "quantization_changed": quant_changed,
                            "view_index": view_index,
                        }
                    )
        atomic_write_json(
            self.path,
            {
                "entries": current_entries,
                "quant": {"frac_bits": config.frac_bits, "quant_gear": config.quant_gear},
            },
        )
        return attributions
