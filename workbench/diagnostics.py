from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .configuration import WorkbenchConfig, is_int, load_json
from .errors import WorkbenchError
from .image_io import load_rgb_png
from .masks import load_mask_document
from .session import WorkbenchSession

ALLOWED_INJECTION_KINDS = {
    "nonfinite_pixels",
    "invalid_mask_rectangle",
    "quantization_out_of_range",
}


def load_injection_config(path: str | Path, config: WorkbenchConfig) -> list[dict[str, Any]]:
    data = load_json(path, "E_CONFIG_INVALID")
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise WorkbenchError("E_CONFIG_INVALID", "injection config must have schema_version=1")
    injections = data.get("injections")
    if not isinstance(injections, list) or not injections:
        raise WorkbenchError("E_CONFIG_INVALID", "injection config must contain a non-empty injections list")
    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(injections):
        if not isinstance(raw, dict):
            raise WorkbenchError("E_CONFIG_INVALID", "injection entry must be an object", {"index": index})
        injection_id = raw.get("id")
        kind = raw.get("kind")
        if not isinstance(injection_id, str) or not injection_id or injection_id in seen_ids:
            raise WorkbenchError("E_CONFIG_INVALID", "injection id must be a unique non-empty string", {"index": index})
        if kind not in ALLOWED_INJECTION_KINDS:
            raise WorkbenchError("E_CONFIG_INVALID", "unknown injection kind", {"kind": kind, "allowed": sorted(ALLOWED_INJECTION_KINDS)})
        level = raw.get("level")
        row = raw.get("row")
        col = raw.get("col")
        if not all(is_int(v) and not isinstance(v, bool) for v in (level, row, col)):
            raise WorkbenchError("E_CONFIG_INVALID", "injection level/row/col must be integers", {"id": injection_id})
        assert is_int(level) and is_int(row) and is_int(col)
        if not (0 <= level < config.levels):
            raise WorkbenchError("E_CONFIG_INVALID", "injection level is outside pyramid", {"id": injection_id, "level": level})
        rows, cols = config.tile_grid(level)
        if not (0 <= row < rows and 0 <= col < cols):
            raise WorkbenchError(
                "E_CONFIG_INVALID",
                "injection tile coordinate does not exist",
                {"id": injection_id, "row": row, "col": col, "rows": rows, "cols": cols},
            )
        if kind == "invalid_mask_rectangle" and not isinstance(raw.get("rect"), dict):
            raise WorkbenchError("E_CONFIG_INVALID", "invalid_mask_rectangle injection requires a rect object", {"id": injection_id})
        normalized.append(dict(raw))
        seen_ids.add(injection_id)
    return normalized


def _run(config: WorkbenchConfig, injections: list[dict[str, Any]], workers: int) -> WorkbenchSession:
    rectangles = load_mask_document(config.mask_path, config.width, config.height)
    pixels, fingerprint = load_rgb_png(config.image_path, config.width, config.height)
    session = WorkbenchSession.fresh(config, pixels, fingerprint, rectangles)
    total = sum(np.prod(config.tile_grid(level)) for level in range(config.levels))
    session.run_batch(budget=int(total), max_workers=workers, injections=injections)
    return session


def run_injection_diagnostic(
    config: WorkbenchConfig,
    injections: list[dict[str, Any]],
) -> dict[str, Any]:
    serial = _run(config, injections, 1)
    parallel = _run(config, injections, config.max_workers)

    serial_report = serial.latest_report
    parallel_report = parallel.latest_report
    serial_failed = serial_report["failed"]
    parallel_failed = parallel_report["failed"]

    pixel_equal = True
    layer_comparison = []
    for level in range(config.levels):
        repair_equal = np.array_equal(serial.repaired_levels[level], parallel.repaired_levels[level])
        style_equal = np.array_equal(serial.styled_levels[level], parallel.styled_levels[level])
        pixel_equal = pixel_equal and repair_equal and style_equal
        layer_comparison.append({"level": level, "repair_equal": repair_equal, "style_equal": style_equal})

    output_order_serial = list(serial.records.keys())
    output_order_parallel = list(parallel.records.keys())
    expected_hits = sorted(item["id"] for item in injections)
    result = {
        "serial_failure_count": len(serial_failed),
        "parallel_failure_count": len(parallel_failed),
        "failures_equal": serial_failed == parallel_failed,
        "budget_order_equal": serial_report["budget_charged"] == parallel_report["budget_charged"],
        "task_submission_order_equal": serial_report["task_order"] == parallel_report["task_order"],
        "output_order_equal": output_order_serial == output_order_parallel,
        "output_order": output_order_serial,
        "pixel_equal": pixel_equal,
        "layer_comparison": layer_comparison,
        "serial_budget_charged": serial_report["budget_charged"],
        "parallel_budget_charged": parallel_report["budget_charged"],
        "failures": serial_failed,
        "injection_hits": serial_report["injection_hits"],
        "expected_injection_hits": expected_hits,
        "all_injections_hit": serial_report["injection_hits"] == expected_hits,
        "parallel_max_observed_workers": parallel_report["max_observed_workers"],
    }
    result["consistent"] = all([
        result["failures_equal"],
        result["budget_order_equal"],
        result["task_submission_order_equal"],
        result["output_order_equal"],
        result["pixel_equal"],
        result["all_injections_hit"],
    ])
    return result
