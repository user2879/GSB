"""Fixed-point keypoint regression from connected components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import EffectiveConfig
from .models import ColumnTruth, ImageTruth


def assign_components_to_columns(
    components: list[dict[str, Any]], columns: list[ColumnTruth]
) -> dict[str, list[dict[str, Any]]]:
    assignment: dict[str, list[dict[str, Any]]] = {column.id: [] for column in columns}
    for component in components:
        center_x = (component["x0"] + component["x1"]) / 2.0
        containing = [column for column in columns if column.x0 <= center_x < column.x1]
        if containing:
            column = min(containing, key=lambda item: (item.order, item.id))
            basis = "x_interval"
        elif columns:
            column = min(
                columns,
                key=lambda item: (
                    min(abs(center_x - item.x0), abs(center_x - item.x1)),
                    item.order,
                    item.id,
                ),
            )
            basis = "nearest_interval"
        else:
            continue
        component = dict(component)
        component["column_assignment_basis"] = basis
        assignment.setdefault(column.id, []).append(component)
    for values in assignment.values():
        values.sort(key=lambda item: (item["y0"], item["x0"], item["component_id"]))
    return assignment


def run_keypoint_column(
    image_name: str,
    truth: ImageTruth,
    column: ColumnTruth,
    assigned_components: list[dict[str, Any]],
    effective: EffectiveConfig,
) -> dict[str, Any]:
    scale = 1 << effective.precision.fixed_bits
    quant_step = effective.precision.quant_step
    image_stem = Path(image_name).stem
    points: list[dict[str, Any]] = []
    for ordinal, component in enumerate(assigned_components):
        width = component["x1"] - component["x0"] + 1
        height = component["y1"] - component["y0"] + 1
        expand = max(1, int(round(max(width, height) * 0.06)))
        bx0, by0 = component["x0"] - expand, component["y0"] - expand
        bx1, by1 = component["x1"] + expand, component["y1"] + expand
        bbox_center_x = (bx0 + bx1) / 2.0
        bbox_center_y = (by0 + by1) / 2.0
        regressed_x = 0.65 * component["centroid_x"] + 0.35 * bbox_center_x
        regressed_y = 0.65 * component["centroid_y"] + 0.35 * bbox_center_y
        ideal_x_q = int(round(regressed_x * scale))
        ideal_y_q = int(round(regressed_y * scale))
        x_q = int(round(ideal_x_q / quant_step) * quant_step)
        y_q = int(round(ideal_y_q / quant_step) * quant_step)
        residual_x_q = x_q - ideal_x_q
        residual_y_q = y_q - ideal_y_q
        keypoint_id = f"kp-{image_stem}-{component['component_id']}"
        points.append(
            {
                "keypoint_id": keypoint_id,
                "image": image_name,
                "column": column.id,
                "column_ordinal": ordinal,
                "column_assignment_basis": component.get("column_assignment_basis", "x_interval"),
                "x_q": x_q,
                "y_q": y_q,
                "x_phys": round(x_q / scale, 9),
                "y_phys": round(y_q / scale, 9),
                "residual_x_q": residual_x_q,
                "residual_y_q": residual_y_q,
                "fixed_bits": effective.precision.fixed_bits,
                "fixed_scale": scale,
                "quant_step": quant_step,
                "regression_alpha": 0.65,
                "boundary_expand_px": expand,
                "bbox": [component["x0"], component["y0"], component["x1"], component["y1"]],
                "area": component["area"],
                "centroid": [round(component["centroid_x"], 6), round(component["centroid_y"], 6)],
                "source_component_id": component["component_id"],
                "source_component_ids": component["fragment_ids"],
                "source_window_id": component["window_id"],
                "source_candidate_id": component["candidate_id"],
            }
        )
    return {
        "image": image_name,
        "column": column.id,
        "column_order": column.order,
        "truth_point_count": len([point for point in truth.keypoints if point.column == column.id]),
        "detected_point_count": len(points),
        "keypoints": points,
    }


def regression_metadata(effective: EffectiveConfig) -> dict[str, Any]:
    return {
        "fixed_bits": effective.precision.fixed_bits,
        "fixed_scale": 1 << effective.precision.fixed_bits,
        "quant_step": effective.precision.quant_step,
        "alpha": 0.65,
        "boundary_expand_fraction": 0.06,
    }
