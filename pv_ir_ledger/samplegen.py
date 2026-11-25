from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import DEFAULT_CONFIG, validate_config
from .imaging import write_png
from .truth import grid_intervals


SAMPLE_SEED = 274

PROFILES = [
    {"code": 0, "name": "NORMAL", "span": 120, "area_permille": 40},
    {"code": 1, "name": "WATCH", "span": 330, "area_permille": 80},
    {"code": 2, "name": "MINOR", "span": 590, "area_permille": 120},
    {"code": 3, "name": "MAJOR", "span": 720, "area_permille": 160},
    {"code": 4, "name": "CRITICAL", "span": 1000, "area_permille": 610},
]


def profile_for(component_id: int, variant: int) -> dict[str, int | str]:
    return PROFILES[(int(component_id) - 1 + int(variant) * 2) % len(PROFILES)]


def make_scene_heat(
    image_name: str,
    rows: int = 4,
    cols: int = 6,
    width: int = 320,
    height: int = 240,
    variant: int = 0,
    silent_component_id: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(SAMPLE_SEED + int(variant) * 1009 + len(image_name))
    heat = np.full((height, width), 72, dtype=np.int32)
    heat[::2, :] = np.maximum(0, heat[::2, :] - 8)
    intervals = grid_intervals(width, height, rows, cols)
    components: list[dict[str, Any]] = []

    for index, (x0, y0, x1, y1) in enumerate(intervals):
        component_id = index + 1
        cell_width = x1 - x0
        cell_height = y1 - y0
        margin_x = max(4, cell_width // 12)
        margin_y = max(4, cell_height // 12)
        bx1 = x0 + margin_x
        bx2 = x1 - margin_x
        by1 = y0 + margin_y
        by2 = y1 - margin_y
        base_heat = 260
        heat[by1:by2, bx1:bx2] = base_heat

        profile = profile_for(component_id, variant)
        span = int(profile["span"])
        target_area = int(profile["area_permille"]) * cell_width * cell_height // 1000
        hotspot_height = max(4, int(np.sqrt(target_area / 1.25)))
        hotspot_width = max(4, int(target_area / hotspot_height))
        hotspot_width = min(hotspot_width, cell_width - 2 * margin_x + 4)
        hotspot_height = min(hotspot_height, cell_height - 2 * margin_y + 4)
        center_x = x0 + cell_width // 2 + int(rng.integers(-2, 3))
        center_y = y0 + cell_height // 2 + int(rng.integers(-2, 3))
        hx1 = max(x0 + 1, center_x - hotspot_width // 2)
        hx2 = min(x1 - 1, hx1 + hotspot_width)
        hy1 = max(y0 + 1, center_y - hotspot_height // 2)
        hy2 = min(y1 - 1, hy1 + hotspot_height)
        peak_heat = min(1000, base_heat + span)
        heat[hy1:hy2, hx1:hx2] = peak_heat

        components.append(
            {
                "component_id": component_id,
                "bbox": [bx1, by1, bx2, by2],
                "expected_grade_code": profile["code"],
                "expected_grade": profile["name"],
                "design_thermal_span_heat": span,
                "design_hotspot_area_permille": int(profile["area_permille"]),
            }
        )

    if silent_component_id is not None:
        index = int(silent_component_id) - 1
        if 0 <= index < len(intervals):
            x0, y0, x1, y1 = intervals[index]
            heat[y0:y1, x0:x1] = 0

    return heat, {
        "schema": "pv-ir-truth-array-v1",
        "image": image_name,
        "width": width,
        "height": height,
        "grid_rows": rows,
        "grid_cols": cols,
        "components": components,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_replacement_image(samples_dir: str | Path, image_name: str, variant: int = 3) -> Path:
    config, _ = validate_config(DEFAULT_CONFIG)
    heat, _ = make_scene_heat(image_name, variant=variant)
    return write_png(Path(samples_dir) / image_name, heat, config["colormap_name"])


def generate_sample_tree(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    injections = root / "injections"
    root.mkdir(parents=True, exist_ok=True)
    injections.mkdir(parents=True, exist_ok=True)

    config, config_issues = validate_config({**DEFAULT_CONFIG, "sample_budget": 12_000_000})
    if config_issues:
        raise RuntimeError(config_issues)
    _write_json(root / "recognition_config.json", config)

    images = [
        ("IR_ARRAY_A.png", 0),
        ("IR_ARRAY_B.png", 1),
        ("IR_ARRAY_C.png", 2),
    ]
    truth_document: dict[str, Any] = {"schema": "pv-ir-truth-v1", "arrays": []}
    for image_name, variant in images:
        heat, truth_entry = make_scene_heat(image_name, variant=variant)
        write_png(root / image_name, heat, config["colormap_name"])
        truth_document["arrays"].append(truth_entry)
    _write_json(root / "array_truth.json", truth_document)

    commands = {
        "schema": "pv-ir-command-stream-v1",
        "commands": [
            {"op": "set_exchange_version", "version": "v2"},
            {"op": "set_grid", "grid_rows": 5, "grid_cols": 7},
            {"op": "set_percentiles", "percentile_low_permille": 560, "percentile_high_permille": 920},
            {"op": "set_thresholds", "grade_thresholds_permille": [220, 390, 570, 740]},
            {"op": "set_quantization", "fixed_decimals": 4, "quant_levels": 32},
            {"op": "replace_image", "image": "IR_ARRAY_A.png", "variant": 3},
            {"op": "delete_image", "image": "IR_ARRAY_C.png"},
        ],
    }
    _write_json(root / "command_stream.json", commands)

    source_png = (root / "IR_ARRAY_A.png").read_bytes()
    truncated = injections / "INJ01_truncated.png"
    truncated.write_bytes(source_png[: max(24, int(len(source_png) * 0.52))])

    Image.new("RGB", (320, 240), (0, 255, 0)).save(injections / "INJ02_unknown_pseudo_color.png", format="PNG")

    silent_heat, _ = make_scene_heat("INJ03_silent_cell.png", variant=0, silent_component_id=8)
    write_png(injections / "INJ03_silent_cell.png", silent_heat, config["colormap_name"])

    outside_grid_truth = {
        "schema": "pv-ir-truth-v1",
        "arrays": [
            {
                "image": "IR_ARRAY_A.png",
                "grid_rows": 4,
                "grid_cols": 6,
                "components": [
                    {"component_id": 1, "bbox": [42, 12, 96, 48], "expected_grade_code": 1},
                ],
            }
        ],
    }
    _write_json(injections / "INJ04_bbox_outside_grid_truth.json", outside_grid_truth)

    outside_image_truth = {
        "schema": "pv-ir-truth-v1",
        "arrays": [
            {
                "image": "IR_ARRAY_A.png",
                "grid_rows": 4,
                "grid_cols": 6,
                "components": [
                    {"component_id": 1, "bbox": [280, 210, 340, 260], "expected_grade_code": 4},
                ],
            }
        ],
    }
    _write_json(injections / "INJ05_bbox_outside_image_truth.json", outside_image_truth)

    bad_config = {
        "schema": "pv-ir-config-v1",
        "colormap_name": "MAP_CODE_999_OUT_OF_TABLE",
        "grid_rows": 0,
        "grid_cols": -2,
        "percentile_low_permille": 910,
        "percentile_high_permille": 120,
        "grade_thresholds_permille": [500, 400, 700, 800],
        "fixed_decimals": 9,
        "quant_levels": 13,
        "max_concurrency": 99,
    }
    _write_json(injections / "INJ06_invalid_config.json", bad_config)

    bad_v1 = (
        "IR_SHORT.png".ljust(32) + "     1\n"
        + "IR_BADCHAR.png".ljust(32)
        + "  X?  "
        + "     1" * 4
        + "   123"
        + "   1\n"
    )
    (injections / "INJ07_truncated_or_illegal_v1.txt").write_text(bad_v1, encoding="utf-8", newline="\n")

    injection_config = {**config}
    _write_json(injections / "INJ01_image_config.json", injection_config)
    _write_json(injections / "INJ03_silent_config.json", injection_config)
    _write_json(
        injections / "INJ02_bad_colormap_config.json",
        {**config, "colormap_name": "MAP_CODE_999_OUT_OF_TABLE", "config_hash": None},
    )

    manifest = {
        "schema": "pv-ir-injection-manifest-v1",
        "seed": SAMPLE_SEED,
        "injections": [
            {
                "id": "INJ01",
                "fault": "truncated PNG bytes",
                "input": "injections/INJ01_truncated.png",
                "config": "injections/INJ01_image_config.json",
                "expected_code": "E_IMAGE_FORMAT",
                "isolation": "other PNGs in the same directory continue processing",
            },
            {
                "id": "INJ02A",
                "fault": "configured pseudo-color mapping code outside allowed table",
                "config": "injections/INJ02_bad_colormap_config.json",
                "expected_code": "E_CONFIG_INVALID",
            },
            {
                "id": "INJ02B",
                "fault": "PNG pixel color absent from pseudo-color table",
                "input": "injections/INJ02_unknown_pseudo_color.png",
                "config": "recognition_config.json",
                "expected_code": "E_IMAGE_FORMAT",
            },
            {
                "id": "INJ03",
                "fault": "one grid cell is completely thermally silent",
                "input": "injections/INJ03_silent_cell.png",
                "config": "injections/INJ03_silent_config.json",
                "expected_code": "E_CELL_DEGRADED",
                "isolation": "other cells retain independent grades and statistics",
            },
            {
                "id": "INJ04",
                "fault": "truth bbox crosses its grid cell",
                "truth": "injections/INJ04_bbox_outside_grid_truth.json",
                "config": "recognition_config.json",
                "expected_code": "E_TRUTH_INVALID",
            },
            {
                "id": "INJ05",
                "fault": "truth bbox outside image",
                "truth": "injections/INJ05_bbox_outside_image_truth.json",
                "config": "recognition_config.json",
                "expected_code": "E_TRUTH_INVALID",
            },
            {
                "id": "INJ06",
                "fault": "multiple invalid configuration fields",
                "config": "injections/INJ06_invalid_config.json",
                "expected_code": "E_CONFIG_INVALID",
            },
            {
                "id": "INJ07",
                "fault": "short or illegal v1 fixed-width field",
                "input": "injections/INJ07_truncated_or_illegal_v1.txt",
                "expected_code": "E_FIELD_TRUNCATED",
            },
        ],
    }
    _write_json(root / "injection_manifest.json", manifest)
    return {"samples_dir": str(root), "truth": truth_document, "config": config, "manifest": manifest}
